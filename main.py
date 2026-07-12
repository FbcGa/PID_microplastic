"""Entry point: detect, classify and track microplastics over a video.

Per frame: detector_v3 segments (resta contra una imagen de fondo) + the
Random Forest classifies each particle (fiber/amorphous), then the tracker
assigns stable IDs so each particle is counted once. Requires a trained
model (rf_model.joblib); train it first with train_rf.py.

Usage:
    uv run main.py video.mp4 --background fondo.jpg
    uv run main.py video.mp4 --background fondo.jpg --model rf_model.joblib --no-display

Keys during playback:
    q / ESC  quit
    space    pause / resume
"""

import argparse
import time
from pathlib import Path

import cv2

from config import TrackerConfig
from detector_v3 import (CLASS_COLORS, CLASS_LABELS, BackgroundSegmentationConfig,
                         ClassifiedParticle, classify_frame_v3)
from rf_classifier import load_if_available
from tracker import Track, Tracker
from utils import crop_top_bottom_strips

WINDOW = "Microplasticos - deteccion / clasificacion / tracking"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate(frame_bgr, particles: list[ClassifiedParticle],
             tracks: list[Track], tracker: Tracker):
    """Draws each particle's contour by class and each confirmed track's ID."""
    out = frame_bgr.copy()
    for p in particles:
        color = CLASS_COLORS.get(p.label, (200, 200, 200))
        cv2.drawContours(out, [p.contour], -1, color, 2)
    for t in tracks:
        x, y = int(t.position[0]), int(t.position[1])
        color = CLASS_COLORS.get(t.label, (200, 200, 200))
        cv2.circle(out, (x, y), 3, color, -1)
        cv2.putText(out, f"#{t.id} {CLASS_LABELS.get(t.label, t.label)}",
                    (x + 6, y - 6), FONT, 0.5, color, 1)
    _draw_counts(out, len(particles), tracker)
    return out


def _draw_counts(frame_bgr, n_particles: int, tracker: Tracker) -> None:
    """Semi-transparent panel with running unique-particle totals."""
    lines = [
        f"Frame: {n_particles} particulas",
        f"Total fibras: {tracker.total_fibers}",
        f"Total amorfas: {tracker.total_amorphous}",
    ]
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (260, 20 + 24 * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0, frame_bgr)
    for i, text in enumerate(lines):
        cv2.putText(frame_bgr, text, (10, 28 + 24 * i), FONT, 0.6,
                    (255, 255, 255), 1)


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
    parser.add_argument("--model", type=Path, default=Path("rf_model.joblib"),
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
