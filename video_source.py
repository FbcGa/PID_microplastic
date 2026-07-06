"""Video sources for the detection system.

`VideoSource` is the abstraction consumed by `main.py`. For the
Raspberry Pi 5 with an HQ camera, it will suffice to add a
`PiCameraVideoSource(VideoSource)` class based on Picamera2, without
modifying the detector or the main loop.
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np


class VideoSource(ABC):
    """Source of BGR frames, whether a file, webcam, or Pi camera."""

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Returns the next BGR frame, or None if there are no more."""

    @abstractmethod
    def release(self) -> None:
        """Releases the source's resources."""

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class FileVideoSource(VideoSource):
    """Reads frames from a video file (development stage on PC)."""

    def __init__(self, path: str):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"Could not open video: {path}")

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def total_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()
