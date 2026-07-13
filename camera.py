"""Real-time frame capture: rpicam-vid on the Raspberry Pi, or a looping
video file to develop the UI on a PC without a camera.

Both camera classes share a small duck-typed interface — start(), read()
-> np.ndarray | None, stop(), capture_background(path) -> np.ndarray — so
processing.py and main_live.py don't need to know which one they got.
Start/stop lifecycle is owned by the caller (main_live.py), not by the
camera itself or by whoever reads from it.
"""

import subprocess
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


class RpicamCamera:
    """Captura en vivo desde la camara de la Pi via rpicam-vid, con los
    mismos controles calibrados en tools/grabar.py (--awb custom,
    --denoise cdn_off), pero emitiendo frames YUV420 crudos por stdout
    en vez de grabar a mp4."""

    def __init__(self, resolution: tuple[int, int] = RESOLUTION):
        self._resolution = resolution
        # YUV420: plano Y (w*h) + U y V submuestreados (w*h/4 cada uno).
        self._frame_bytes = resolution[0] * resolution[1] * 3 // 2
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        w, h = self._resolution
        cmd = [
            "rpicam-vid",
            "-t", "0",
            "--nopreview",
            "--width", str(w),
            "--height", str(h),
            "--framerate", str(FPS),
            "--shutter", str(SHUTTER_US),
            "--gain", str(GAIN),
            "--awb", "custom",
            "--awbgains", f"{AWB_GAINS[0]},{AWB_GAINS[1]}",
            "--denoise", "cdn_off",
            "--codec", "yuv420",
            "--flush",
            "-o", "-",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=self._frame_bytes)

    def read(self) -> np.ndarray | None:
        data = self._proc.stdout.read(self._frame_bytes)
        if data is None or len(data) < self._frame_bytes:
            return None  # rpicam-vid termino o pipe cortado
        w, h = self._resolution
        yuv = np.frombuffer(data, np.uint8).reshape(h * 3 // 2, w)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    def capture_background(self, path: Path) -> np.ndarray:
        frame = self.read()
        cv2.imwrite(str(path), frame)
        return frame

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


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
    """source=None -> rpicam-vid (Raspberry Pi); source=path -> video
    looped, para desarrollo en PC sin camara."""
    if source is None:
        return RpicamCamera()
    return VideoFileCamera(source)
