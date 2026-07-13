"""Entry point + controller for the live-capture app: wires camera, worker
thread and Tkinter UI together. Callbacks stay thin (state transitions
only) — all CV logic lives in processing.py, all capture in camera.py.

Usage:
    uv run main_live.py                      (Raspberry Pi: rpicam-vid)
    uv run main_live.py --source video.mp4   (PC dev: loops a video file)
"""

import argparse
import tkinter as tk
from pathlib import Path

from app_ui import App, Callbacks
from calibration import BACKGROUND_PATH, load_calibration, save_calibration
from camera import RESOLUTION, open_camera
from processing import Mode, PipelineWorker
from random_forest.rf_classifier import load_if_available

MIN_VISIBLE_HEIGHT = 20  # margen minimo entre las lineas de crop sup/inf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None,
                        help="Video para desarrollo en PC; si se omite usa rpicam-vid")
    parser.add_argument("--model", type=Path,
                        default=Path("random_forest/rf_model.joblib"),
                        help="Modelo Random Forest entrenado")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier = load_if_available(args.model)
    if classifier is None:
        raise FileNotFoundError(
            f"No existe el modelo {args.model}. Entrena primero con "
            "train_rf.py (necesita fibras y amorfas etiquetadas).")

    calibration = load_calibration()
    camera = open_camera(args.source)
    camera.start()

    worker = PipelineWorker(camera, classifier, calibration)
    worker.start()

    root = tk.Tk()

    def on_calibrate() -> None:
        worker.set_mode(Mode.PREVIEW)

    def on_crop_change(top: int, bottom: int) -> None:
        max_total = RESOLUTION[1] - MIN_VISIBLE_HEIGHT
        if top + bottom > max_total:
            top, bottom = calibration.crop_top, calibration.crop_bottom
        calibration.crop_top = top
        calibration.crop_bottom = bottom
        worker.set_crop(top, bottom)

    def on_capture_background() -> None:
        worker.capture_background()
        app.set_start_enabled(True)

    def on_save_calibration() -> None:
        save_calibration(calibration)
        worker.set_mode(Mode.IDLE)

    def on_start() -> None:
        worker.start_counting(calibration)

    def on_stop() -> None:
        worker.set_mode(Mode.IDLE)

    def on_close() -> None:
        worker.stop()
        camera.stop()
        root.destroy()

    callbacks = Callbacks(
        on_calibrate=on_calibrate,
        on_crop_change=on_crop_change,
        on_capture_background=on_capture_background,
        on_save_calibration=on_save_calibration,
        on_start=on_start,
        on_stop=on_stop,
        on_close=on_close,
    )

    app = App(root, worker.results, callbacks, calibration.crop_top,
              calibration.crop_bottom, background_exists=BACKGROUND_PATH.exists())

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
