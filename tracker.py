"""Particle tracking across frames: simplified SORT.

Constant-velocity prediction (EMA) + Hungarian assignment + coasting
(tolerating brief disappearances) + minimum confirmation before counting.
No directional flow: counting is done via unique IDs, not a crossing
line, since particles float erratically.

Decoupled from the detector: it consumes any object exposing a `centroid`
(x, y) and a `label` string (e.g. detector.ClassifiedParticle), so the
class labels come straight from the Random Forest ("fiber"/"amorphous").
"""

from collections import Counter
from typing import Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import TrackerConfig


class Detection(Protocol):
    """What the tracker needs from each detection."""
    label: str

    @property
    def centroid(self) -> tuple[int, int]: ...


class Track:
    """State of a particle tracked across frames."""

    def __init__(self, detection: Detection, track_id: int):
        self.id = track_id
        self.position = np.array(detection.centroid, dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.frames_missing = 0
        self.hits = 1
        self.confirmed = False
        # Class the track was last counted under (set when confirmed);
        # lets the Tracker move the count if the majority vote later flips.
        self.counted_label: str | None = None
        self._class_history: Counter[str] = Counter([detection.label])
        # Last actually observed position (not predicted): velocity must be
        # measured between observations, never against the prediction.
        self._last_observed = self.position.copy()

    @property
    def label(self) -> str:
        """Most voted class over the track's history (robust to a single
        bad per-frame prediction)."""
        return self._class_history.most_common(1)[0][0]

    def predict(self) -> None:
        self.position = self.position + self.velocity

    def update(self, detection: Detection, velocity_alpha: float) -> None:
        new_position = np.array(detection.centroid, dtype=float)
        # frames_missing still carries the coasting frames: if it reappears
        # after k frames unseen, the displacement covers k+1 frames.
        elapsed_frames = self.frames_missing + 1
        observed_velocity = (new_position - self._last_observed) / elapsed_frames
        if self.hits == 1:
            # First real measurement: adopt it directly instead of
            # smoothing against the initial zero velocity.
            self.velocity = observed_velocity
        else:
            self.velocity = (velocity_alpha * observed_velocity
                             + (1 - velocity_alpha) * self.velocity)
        self.position = new_position
        self._last_observed = new_position.copy()
        self._class_history[detection.label] += 1
        self.frames_missing = 0
        self.hits += 1


class Tracker:
    """Assigns detections to tracks across frames and counts unique particles."""

    def __init__(self, config: TrackerConfig):
        self._cfg = config
        self._tracks: list[Track] = []
        self._counts: Counter[str] = Counter()
        self._next_id = 1

    @property
    def counts(self) -> dict[str, int]:
        """Confirmed unique particles counted per class label."""
        return dict(self._counts)

    @property
    def total_fibers(self) -> int:
        return self._counts["fiber"]

    @property
    def total_amorphous(self) -> int:
        return self._counts["amorphous"]

    @property
    def tracks(self) -> list[Track]:
        """All live tracks (confirmed or not) — read-only, for debugging/
        inspection. `update()` still only returns the confirmed ones."""
        return list(self._tracks)

    def update(self, detections: list[Detection]) -> list[Track]:
        for track in self._tracks:
            track.predict()

        matched, unmatched_tracks, unmatched_detections = self._assign(detections)

        for track_idx, detection_idx in matched:
            track = self._tracks[track_idx]
            track.update(detections[detection_idx], self._cfg.velocity_alpha)
            if not track.confirmed and track.hits >= self._cfg.min_hits:
                track.confirmed = True
                track.counted_label = track.label
                self._counts[track.label] += 1
            elif track.confirmed and track.label != track.counted_label:
                self._counts[track.counted_label] -= 1
                self._counts[track.label] += 1
                track.counted_label = track.label

        for track_idx in unmatched_tracks:
            self._tracks[track_idx].frames_missing += 1

        for detection_idx in unmatched_detections:
            self._tracks.append(Track(detections[detection_idx], self._next_id))
            self._next_id += 1

        self._tracks = [t for t in self._tracks
                        if t.frames_missing <= self._cfg.max_missed]

        return [t for t in self._tracks if t.confirmed]

    def _assign(
        self, detections: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not self._tracks or not detections:
            return [], list(range(len(self._tracks))), list(range(len(detections)))

        track_positions = np.array([t.position for t in self._tracks])
        detection_positions = np.array([d.centroid for d in detections], dtype=float)
        costs = np.linalg.norm(
            track_positions[:, None, :] - detection_positions[None, :, :], axis=2)

        rows, cols = linear_sum_assignment(costs)

        matched = []
        used_tracks = set()
        used_detections = set()
        for row, col in zip(rows, cols):
            if costs[row, col] <= self._cfg.max_distance:
                matched.append((row, col))
                used_tracks.add(row)
                used_detections.add(col)

        unmatched_tracks = [i for i in range(len(self._tracks)) if i not in used_tracks]
        unmatched_detections = [
            i for i in range(len(detections)) if i not in used_detections]

        return matched, unmatched_tracks, unmatched_detections
