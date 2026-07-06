"""Particle tracking across frames: simplified SORT.

Constant-velocity prediction (EMA) + Hungarian assignment + coasting
(tolerating brief disappearances) + minimum confirmation before counting.
No directional flow: counting is done via unique IDs, not a crossing
line, since particles float erratically.
"""

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import TrackerConfig
from detector import Detection, ParticleClass


class Track:
    """State of a particle tracked across frames."""

    _next_id = 1

    def __init__(self, detection: Detection):
        self.id = Track._next_id
        Track._next_id += 1
        self.position = np.array(detection.centroid, dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.frames_missing = 0
        self.hits = 1
        self.confirmed = False
        self._class_history: Counter[ParticleClass] = Counter([detection.particle_class])
        # Last actually observed position (not predicted): velocity must be
        # measured between observations, never against the prediction.
        self._last_observed = self.position.copy()

    @property
    def particle_class(self) -> ParticleClass:
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
        self._class_history[detection.particle_class] += 1
        self.frames_missing = 0
        self.hits += 1


class Tracker:
    """Assigns detections to tracks across frames and counts unique particles."""

    def __init__(self, config: TrackerConfig):
        self._cfg = config
        self._tracks: list[Track] = []
        self._total_fibers = 0
        self._total_amorphous = 0

    @property
    def total_fibers(self) -> int:
        return self._total_fibers

    @property
    def total_amorphous(self) -> int:
        return self._total_amorphous

    def update(self, detections: list[Detection]) -> list[Track]:
        for track in self._tracks:
            track.predict()

        matched, unmatched_tracks, unmatched_detections = self._assign(detections)

        for track_idx, detection_idx in matched:
            track = self._tracks[track_idx]
            track.update(detections[detection_idx], self._cfg.velocity_alpha)
            if not track.confirmed and track.hits >= self._cfg.min_hits:
                track.confirmed = True
                self._count(track.particle_class)

        for track_idx in unmatched_tracks:
            self._tracks[track_idx].frames_missing += 1

        for detection_idx in unmatched_detections:
            self._tracks.append(Track(detections[detection_idx]))

        self._tracks = [t for t in self._tracks
                        if t.frames_missing <= self._cfg.max_missed]

        return [t for t in self._tracks if t.confirmed]

    def _count(self, particle_class: ParticleClass) -> None:
        if particle_class is ParticleClass.FIBER:
            self._total_fibers += 1
        elif particle_class is ParticleClass.AMORPHOUS:
            self._total_amorphous += 1

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
