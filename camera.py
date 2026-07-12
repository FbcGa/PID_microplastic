"""Real-time frame capture: Picamera2 on the Raspberry Pi, or a looping
video file to develop the UI on a PC without a camera.

Both camera classes share a small duck-typed interface — start(), read()
-> np.ndarray | None, stop(), capture_background(path) -> np.ndarray — so
processing.py and main_live.py don't need to know which one they got.
Start/stop lifecycle is owned by the caller (main_live.py), not by the
camera itself or by whoever reads from it.
"""

from pathlib import Path

import cv2
import numpy as np

# Parametros calibrados en tools/grabar.py: iluminacion fija (sin AGC/AWB
# automatico) para que el fondo de referencia siga siendo valido frame a frame.
FPS = 30
SHUTTER_US = 500
GAIN = 8.0
AWB_GAINS = (3.0, 1.8)
RESOLUTION = (1280, 720)


class PiCamera:
    """Captura en vivo desde la camara de la Pi via Picamera2."""

    def __init__(self, resolution: tuple[int, int] = RESOLUTION):
        from picamera2 import Picamera2  # solo importable en la Pi
        from libcamera import controls

        self._picam2 = Picamera2()
        config = self._picam2.create_video_configuration(
            main={"size": resolution, "format": "RGB888"})
        self._picam2.configure(config)
        self._picam2.set_controls({
            "FrameRate": FPS,
            "ExposureTime": SHUTTER_US,
            "AnalogueGain": GAIN,
            "AwbEnable": False,
            "ColourGains": AWB_GAINS,
            "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Off,
        })

    def start(self) -> None:
        self._picam2.start()

    def read(self) -> np.ndarray:
        # RGB888 de Picamera2 ya es BGR en memoria: directo a OpenCV, sin
        # cvtColor (eso rompería la segmentación por canal verde en silencio).
        return self._picam2.capture_array("main")

    def capture_background(self, path: Path) -> np.ndarray:
        frame = self.read()
        cv2.imwrite(str(path), frame)
        return frame

    def stop(self) -> None:
        self._picam2.stop()


class VideoFileCamera:
    """Loopea un archivo de video, mismo interfaz que PiCamera — permite
    probar la UI completa en la PC Windows sin camara."""

    def __init__(self, path: str):
        self._path = path
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise IOError(f"No se pudo abrir el video: {self._path}")

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return frame if ok else None

    def capture_background(self, path: Path) -> np.ndarray:
        frame = self.read()
        cv2.imwrite(str(path), frame)
        return frame

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()


def open_camera(source: str | None):
    """source=None -> Picamera2 (Raspberry Pi); source=path -> video
    looped, para desarrollo en PC sin camara."""
    if source is None:
        return PiCamera()
    return VideoFileCamera(source)
