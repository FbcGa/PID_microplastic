"""Worker thread: capture -> crop -> classify -> track -> annotate.

Runs on its own thread so the UI stays responsive; talks to the UI only
through a bounded, drop-oldest queue of immutable FrameResult snapshots.
All CV state (tracker, cached background) lives here, never in the UI.
Camera start/stop lifecycle belongs to the caller (main_live.py); this
thread only reads frames.
"""

import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np

from calibration import BACKGROUND_PATH, Calibration
from config import BackgroundSegmentationConfig, TrackerConfig
from cropping import crop_top_bottom_strips
from detector import classify_frame
from random_forest.rf_classifier import RandomForestParticleClassifier
from tracker import Tracker
from visualization import annotate

LINE_COLOR = (0, 255, 255)
COUNT_SEND_INTERVAL_S = 1.0  # ventana fuzzy en el Arduino es de 10 muestras


class Mode(Enum):
    IDLE = auto()
    PREVIEW = auto()
    RUNNING = auto()


@dataclass(frozen=True)
class FrameResult:
    frame_bgr: np.ndarray
    fibers: int
    amorphous: int
    mode: Mode


class PipelineWorker(threading.Thread):
    """Owns the frame-processing loop and all CV state. The UI only calls
    the public methods below and drains `results`."""

    def __init__(self, camera, classifier: RandomForestParticleClassifier,
                 calibration: Calibration, pump=None,
                 seg_cfg: BackgroundSegmentationConfig = BackgroundSegmentationConfig()):
        super().__init__(daemon=True)
        self._camera = camera
        self._classifier = classifier
        self._pump = pump
        self._seg_cfg = seg_cfg
        self.results: queue.Queue[FrameResult] = queue.Queue(maxsize=1)

        self._lock = threading.Lock()
        self._mode = Mode.IDLE
        self._crop_top = calibration.crop_top
        self._crop_bottom = calibration.crop_bottom

        self._tracker: Tracker | None = None
        self._background_bgr: np.ndarray | None = None
        self._background_g: np.ndarray | None = None
        self._stop_requested = False
        self._last_count_sent = 0.0

    # --- commands from the UI thread ---

    def set_mode(self, mode: Mode) -> None:
        with self._lock:
            self._mode = mode

    def set_crop(self, top: int, bottom: int) -> None:
        with self._lock:
            self._crop_top = top
            self._crop_bottom = bottom

    def start_counting(self, calibration: Calibration) -> None:
        background_bgr = cv2.imread(str(BACKGROUND_PATH))
        if background_bgr is None:
            raise IOError(f"No existe fondo de referencia: {BACKGROUND_PATH}")
        background_bgr = crop_top_bottom_strips(
            background_bgr, calibration.crop_top, calibration.crop_bottom)
        with self._lock:
            self._crop_top = calibration.crop_top
            self._crop_bottom = calibration.crop_bottom
            self._background_bgr = background_bgr
            self._background_g = background_bgr[:, :, 1]
            self._tracker = Tracker(TrackerConfig())
            self._mode = Mode.RUNNING

    def capture_background(self) -> None:
        self._camera.capture_background(BACKGROUND_PATH)

    def stop(self) -> None:
        self._stop_requested = True

    # --- worker loop ---

    def run(self) -> None:
        while not self._stop_requested:
            frame = self._camera.read()
            if frame is None:
                continue

            with self._lock:
                mode = self._mode
                top, bottom = self._crop_top, self._crop_bottom
                tracker = self._tracker
                background_bgr = self._background_bgr
                background_g = self._background_g

            if mode == Mode.RUNNING and tracker is not None:
                cropped = crop_top_bottom_strips(frame, top, bottom)
                particles = classify_frame(
                    cropped, background_bgr, self._classifier, self._seg_cfg, background_g)
                tracks = tracker.update(particles)
                annotated = annotate(cropped, particles, tracks, tracker)
                result = FrameResult(annotated, tracker.total_fibers,
                                     tracker.total_amorphous, mode)
                if self._pump is not None:
                    now = time.monotonic()
                    if now - self._last_count_sent >= COUNT_SEND_INTERVAL_S:
                        self._last_count_sent = now
                        self._pump.send_count(len(particles))
            elif mode == Mode.PREVIEW:
                result = FrameResult(self._draw_crop_lines(frame, top, bottom), 0, 0, mode)
            else:
                result = FrameResult(frame, 0, 0, mode)

            self._drop_oldest_put(result)

    @staticmethod
    def _draw_crop_lines(frame: np.ndarray, top: int, bottom: int) -> np.ndarray:
        preview = frame.copy()
        h, w = preview.shape[:2]
        cv2.line(preview, (0, top), (w, top), LINE_COLOR, 1)
        cv2.line(preview, (0, h - bottom), (w, h - bottom), LINE_COLOR, 1)
        return preview

    def _drop_oldest_put(self, result: FrameResult) -> None:
        try:
            self.results.get_nowait()
        except queue.Empty:
            pass
        self.results.put_nowait(result)
