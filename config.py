"""Calibrated parameters for the microplastic detection pipeline.

Values obtained during calibration in Colab (June 2026) on frames from
the fixed-camera, controlled-lighting setup.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionConfig:
    # Segmentation: "local" (local darkness, stateless, live-capable)
    # or "background" (subtraction against a reference frame)
    segmentation_mode: str = "local"

    # Local darkness: the local background is estimated with a box blur
    # (O(1) per pixel regardless of window size, Pi-friendly). The window
    # must be larger than the biggest expected particle.
    local_background_kernel: int = 51
    local_darkness_threshold: int = 20

    # Preprocessing
    blur_kernel: tuple[int, int] = (5, 5)
    subtraction_threshold: int = 25
    morphology_kernel: int = 3

    # Size filter (px²) — provisional until the µm -> px calibration
    # over the debug frames is done
    min_area: float = 500.0
    max_area: float = 5000.0

    # Out-of-focus bubble edge filter: faint AND wide -> discarded.
    # Real particles are either very dark (in focus) or thin (fibers);
    # bubble edges are soft, low-contrast wide bands.
    min_peak_darkness: float = 60.0   # peak darkness below this = faint
    max_faint_width: float = 10.0     # faint blobs wider than this (px) = artifact
    # Faint blobs survive only as true fibers: long and thin. Real fibers
    # measure 0.08-0.13; bubble-edge slivers 0.20+ (calibrated on labeled
    # blobs, July 2026)
    faint_fiber_circularity: float = 0.16

    # Bubble filter. A bubble is bright inside with a dark border, so its
    # dark border blob always touches a bright region — even when the
    # bubble does not close inside the frame. Real particles are solid,
    # uniformly dark, and surrounded by plain background.
    bubble_intensity: float = 120.0     # min core brightness (closed ring)
    bubble_core_margin: float = 20.0    # min (core - rim) contrast (closed ring)
    bubble_halo_width: int = 9          # ring (px) inspected around the blob
    # p95 - p50 of the gray ring around the blob: a solid particle sits on
    # uniform background (low spread); a bubble border has a bright glare
    # side (high spread)
    bubble_ring_spread: float = 35.0
    # Detections closer than this (px) to a bubble blob are bubble
    # fragments (dark pieces of the same border) -> discarded
    bubble_exclusion_radius: int = 80

    # Fiber vs amorphous classification
    fiber_aspect_ratio: float = 2.5
    fiber_circularity: float = 0.3


@dataclass(frozen=True)
class TrackerConfig:
    # Max distance (px) between predicted position and detection to match
    max_distance: float = 80.0
    # Frames a track survives without being seen (~0.3 s at 30 FPS)
    max_missed: int = 10
    # Frames seen before a track is confirmed (and counted)
    min_hits: int = 2
    # EMA smoothing of the velocity
    velocity_alpha: float = 0.5
