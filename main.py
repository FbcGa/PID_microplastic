"""Entry point: detect, classify and track microplastics over a video.

Per frame: detector segments (resta contra una imagen de fondo) + the
Random Forest classifies each particle (fiber/amorphous), then the tracker
assigns stable IDs so each particle is counted once. Requires a trained
model (random_forest/rf_model.joblib); train it first with random_forest/train_rf.py.

Usage:
    uv run main.py video.mp4 --background fondo.jpg
    uv run main.py video.mp4 --background fondo.jpg --model random_forest/rf_model.joblib --no-display

Keys during playback:
    q / ESC  quit
    space    pause / resume
"""

import argparse
import time
from pathlib import Path

import cv2

from config import TrackerConfig
from detector import BackgroundSegmentationConfig, classify_frame_v3
from random_forest.rf_classifier import load_if_available
from tracker import Tracker
from utils import crop_top_bottom_strips
from visualization import annotate

WINDOW = "Microplasticos - deteccion / clasificacion / tracking"


def run(video_path: str, model_path: Path, background_bgr,
        cfg: BackgroundSegmentationConfig, display: bool = True) -> Tracker:
    classifier = load_if_available(model_path)
    if classifier is None:
        raise FileNotFoundError(
            f"No existe el modelo {model_path}. Entrena primero con "
            "train_rf.py (necesita fibras y amorfas etiquetadas).")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {video_path}")

    tracker = Tracker(TrackerConfig())
    background_bgr = crop_top_bottom_strips(background_bgr)
    background_g = background_bgr[:, :, 1]
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_budget_ms = 1000 / fps if fps > 0 else 33.3
    paused = False

    try:
        while True:
            if not paused:
                t0 = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break
                # Recorte de bordes ni bien se recibe el frame: esas areas
                # no deben entrar ni a la clasificacion ni al tracker.
                frame = crop_top_bottom_strips(frame)
                particles = classify_frame_v3(frame, background_bgr, classifier, cfg, background_g)
                tracks = tracker.update(particles)
                if display:
                    cv2.imshow(WINDOW, annotate(frame, particles, tracks, tracker))
                elapsed_ms = (time.perf_counter() - t0) * 1000
                wait = max(1, int(frame_budget_ms - elapsed_ms))

            if display:
                key = cv2.waitKey(wait) & 0xFF
                if key in (ord("q"), 27):  # 27 = ESC
                    break
                if key == ord(" "):
                    paused = not paused
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()

    print(f"Total fibras: {tracker.total_fibers}")
    print(f"Total amorfas: {tracker.total_amorphous}")
    return tracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deteccion + clasificacion RF + tracking sobre un video.")
    parser.add_argument("video", help="Ruta del video a procesar")
    parser.add_argument("--background", type=Path, required=True,
                        help="Imagen de fondo (agua limpia, sin particulas)")
    parser.add_argument("--model", type=Path,
                        default=Path("random_forest/rf_model.joblib"),
                        help="Modelo Random Forest entrenado")
    parser.add_argument("--no-display", action="store_true",
                        help="No mostrar ventana; solo procesar y contar")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    background = cv2.imread(str(args.background))
    if background is None:
        raise IOError(f"Could not load background: {args.background}")
    run(args.video, args.model, background, BackgroundSegmentationConfig(),
        display=not args.no_display)


if __name__ == "__main__":
    main()
