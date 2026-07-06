"""Detection and classification of microplastics via local-darkness segmentation.

Pipeline: gray -> blur -> local background (box blur) -> darkness
(background - frame) -> binary threshold -> morphology (open/close) ->
contours -> features -> classification.

Stateless per frame (no reference background), so it works live and on
still images. The legacy "background" mode (absdiff against a reference
frame) is kept behind config.segmentation_mode for comparison.
"""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from config import DetectionConfig


class ParticleClass(Enum):
    FIBER = "fiber"
    AMORPHOUS = "amorphous"
    BUBBLE = "bubble"


@dataclass(frozen=True)
class Detection:
    particle_class: ParticleClass
    contour: np.ndarray
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area: float
    aspect_ratio: float
    circularity: float
    intensity: float        # mean brightness over the filled blob
    core_intensity: float   # mean brightness of the eroded interior
    rim_intensity: float    # mean brightness of the outer band
    peak_darkness: float    # max value in the darkness image inside the blob
    mean_width: float       # 2*area/perimeter: mean thickness of the blob
    ring_spread: float      # p95 - p50 of the gray ring around the blob:
                            # high = bright glare on one side (bubble border)

    @property
    def centroid(self) -> tuple[int, int]:
        x, y, w, h = self.bbox
        return x + w // 2, y + h // 2


@dataclass(frozen=True)
class DetectionResult:
    particles: list[Detection]      # fibers and amorphous only
    discarded_bubbles: int
    mask: np.ndarray

    def count(self, particle_class: ParticleClass) -> int:
        return sum(1 for p in self.particles if p.particle_class == particle_class)


class ParticleClassifier:
    """Classification via geometric and intensity rules.

    If this is later replaced by a trained model, any class exposing the
    same `classify` method will work as a drop-in replacement.
    """

    def __init__(self, config: DetectionConfig):
        self._cfg = config

    def classify(self, area: float, aspect_ratio: float, circularity: float,
                 core_intensity: float, rim_intensity: float,
                 peak_darkness: float, mean_width: float,
                 ring_spread: float) -> ParticleClass | None:
        """Returns the particle class, or None if filtered out."""
        cfg = self._cfg
        if area < cfg.min_area or area > cfg.max_area:
            return None
        # Bubble, case 1 — closed ring: filling the contour covers the
        # bright interior -> bright core with clearly darker rim.
        if (core_intensity > cfg.bubble_intensity
                and core_intensity - rim_intensity > cfg.bubble_core_margin):
            return ParticleClass.BUBBLE
        # Bubble, case 2 — open border: the dark band touches the bubble's
        # bright interior (glare on one side, dark on the other). A solid
        # particle sits on uniform background, so its ring has low spread.
        if ring_spread > cfg.bubble_ring_spread:
            return ParticleClass.BUBBLE
        # Faint blobs (out of focus / noise) are only kept if they are a
        # true fiber: long, thin, very low circularity. A real non-fiber
        # particle in the focal plane is always sharply dark.
        is_elongated = aspect_ratio > cfg.fiber_aspect_ratio or aspect_ratio < 1 / cfg.fiber_aspect_ratio
        is_fiber_shaped = is_elongated or circularity < cfg.fiber_circularity
        if peak_darkness < cfg.min_peak_darkness:
            if not (circularity < cfg.faint_fiber_circularity
                    and mean_width <= cfg.max_faint_width):
                return None
        if is_fiber_shaped:
            return ParticleClass.FIBER
        return ParticleClass.AMORPHOUS


class MicroplasticDetector:
    """Segments particles against a reference background and classifies them."""

    def __init__(self, config: DetectionConfig,
                 classifier: ParticleClassifier | None = None):
        self._cfg = config
        self._classifier = classifier or ParticleClassifier(config)
        self._kernel = np.ones(
            (config.morphology_kernel, config.morphology_kernel), np.uint8)
        self._background_blur: np.ndarray | None = None

    def set_background(self, background_frame: np.ndarray) -> None:
        """Registers the reference frame (clean water, no particles)."""
        self._background_blur = self._preprocess(background_frame)

    @property
    def has_background(self) -> bool:
        return self._background_blur is not None

    def detect(self, frame: np.ndarray,
               stages: dict[str, np.ndarray] | None = None) -> DetectionResult:
        """Detects particles. If `stages` is given, each intermediate image
        of the pipeline is stored in it (for visual debugging)."""
        if (self._cfg.segmentation_mode == "background"
                and self._background_blur is None):
            raise RuntimeError(
                "Background not set: call set_background() first.")

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, self._cfg.blur_kernel, 0)
        if stages is not None:
            stages["01_gris"] = gray_frame
            stages["02_blur"] = blurred_frame
        mask, darkness, local_background = self._segment(blurred_frame, stages)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        particles: list[Detection] = []
        bubble_boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            detection = self._analyze_contour(
                contour, gray_frame, darkness, local_background)
            if detection is None:
                continue
            if detection.particle_class is ParticleClass.BUBBLE:
                bubble_boxes.append(detection.bbox)
            else:
                particles.append(detection)

        # A dark fragment next to a bubble belongs to the same border:
        # anything within the exclusion radius of a bubble is discarded.
        bubbles = len(bubble_boxes)
        if bubble_boxes:
            kept = [p for p in particles
                    if not self._near_any(p.bbox, bubble_boxes,
                                          self._cfg.bubble_exclusion_radius)]
            bubbles += len(particles) - len(kept)
            particles = kept

        return DetectionResult(particles, bubbles, mask)

    @staticmethod
    def _near_any(bbox: tuple[int, int, int, int],
                  boxes: list[tuple[int, int, int, int]],
                  radius: int) -> bool:
        x, y, w, h = bbox
        for bx, by, bw, bh in boxes:
            if (x < bx + bw + radius and bx < x + w + radius
                    and y < by + bh + radius and by < y + h + radius):
                return True
        return False

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, self._cfg.blur_kernel, 0)

    def _segment(self, blurred_frame: np.ndarray,
                 stages: dict[str, np.ndarray] | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (binary mask, darkness image, local background estimate)."""
        if self._cfg.segmentation_mode == "local":
            kernel_size = self._cfg.local_background_kernel
            local_background = cv2.blur(blurred_frame, (kernel_size, kernel_size))
            # Particle = darker than its local surroundings; cv2.subtract
            # saturates at 0, so brighter-than-background pixels drop out.
            difference = cv2.subtract(local_background, blurred_frame)
            threshold = self._cfg.local_darkness_threshold
            if stages is not None:
                stages["03_fondo_local"] = local_background
        else:
            local_background = self._background_blur
            difference = cv2.absdiff(blurred_frame, self._background_blur)
            threshold = self._cfg.subtraction_threshold
        _, thresholded = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
        opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, self._kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self._kernel)
        if stages is not None:
            stages["03_diferencia"] = difference
            stages["04_umbral"] = thresholded
            stages["05_apertura"] = opened
            stages["06_cierre"] = closed
        return closed, difference, local_background

    def _analyze_contour(self, contour: np.ndarray, gray_frame: np.ndarray,
                         darkness: np.ndarray,
                         local_background: np.ndarray) -> Detection | None:
        area = cv2.contourArea(contour)
        # Early discard: avoids computing expensive features over noise
        if area < self._cfg.min_area or area > self._cfg.max_area:
            return None

        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0.0
        circularity = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
        mean_width = (2 * area / perimeter) if perimeter > 0 else 0.0
        peak_darkness = float(darkness[y:y + h, x:x + w].max())
        intensity, core, rim = self._internal_intensities(
            contour, gray_frame, (x, y, w, h))
        spread = self._ring_spread(contour, gray_frame, (x, y, w, h))

        particle_class = self._classifier.classify(
            area, aspect_ratio, circularity, core, rim, peak_darkness,
            mean_width, spread)
        if particle_class is None:
            return None
        return Detection(particle_class, contour, (x, y, w, h),
                         area, aspect_ratio, circularity, intensity, core, rim,
                         peak_darkness, mean_width, spread)

    def _ring_spread(self, contour: np.ndarray, gray_frame: np.ndarray,
                     bbox: tuple[int, int, int, int]) -> float:
        """p95 - p50 of the gray values in a ring around the blob.

        A solid particle sits on uniform background -> low spread. A bubble
        border always touches the bubble's bright interior on one side ->
        the ring has a bright tail -> high spread. Robust against dark
        neighbors (they lower p50, not p95) and against the local dip the
        box blur introduces around dark blobs."""
        ring_width = self._cfg.bubble_halo_width
        x, y, w, h = bbox
        frame_h, frame_w = gray_frame.shape
        x0, y0 = max(0, x - ring_width), max(0, y - ring_width)
        x1, y1 = min(frame_w, x + w + ring_width), min(frame_h, y + h + ring_width)

        filled = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.drawContours(filled, [contour - (x0, y0)], -1, 255, -1)
        kernel = np.ones((ring_width, ring_width), np.uint8)
        ring = cv2.subtract(cv2.dilate(filled, kernel), filled)
        ring_values = gray_frame[y0:y1, x0:x1][ring > 0]
        if ring_values.size == 0:
            return 0.0
        p50, p95 = np.percentile(ring_values, (50, 95))
        return float(p95 - p50)

    @staticmethod
    def _internal_intensities(contour: np.ndarray, gray_frame: np.ndarray,
                              bbox: tuple[int, int, int, int]
                              ) -> tuple[float, float, float]:
        """Mean brightness of the filled blob, of its eroded core and of the
        outer band (blob minus core). Computed only over the bounding box
        (not the full frame) for performance on the Pi.

        A bubble segments as a dark ring: filling its outer contour covers
        the bright interior, so core >> rim exposes it."""
        x, y, w, h = bbox
        roi = gray_frame[y:y + h, x:x + w]
        filled = np.zeros((h, w), np.uint8)
        cv2.drawContours(filled, [contour - (x, y)], -1, 255, -1)
        mean = cv2.mean(roi, mask=filled)[0]

        # Erosion proportional to blob size, so the rim band scales with it
        erosion = max(3, min(w, h) // 4)
        kernel = np.ones((erosion, erosion), np.uint8)
        core_mask = cv2.erode(filled, kernel)
        if cv2.countNonZero(core_mask) == 0:
            # Blob too thin to erode: no distinguishable core
            return mean, mean, mean
        rim_mask = cv2.subtract(filled, core_mask)
        core = cv2.mean(roi, mask=core_mask)[0]
        rim = cv2.mean(roi, mask=rim_mask)[0] if cv2.countNonZero(rim_mask) else mean
        return mean, core, rim
