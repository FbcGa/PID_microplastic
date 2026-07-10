"""Frame-by-frame tracker debug log for a whole video.

Runs the same pipeline as main.py (detector_v2 segmentation -> RF
classification -> tracker) over every frame, and writes:
  - one CSV row per (frame, live track): positions, velocity, class votes,
    hits, coasting, confirmation, and the running unique-particle counts.
  - an annotated debug video: every live track drawn (gray/dashed while
    unconfirmed, colored by class once confirmed), with its id, hits and
    frames_missing, plus the frame number and running counts.

Usage:
    uv run debug_tracker.py amorfas.mp4
    uv run debug_tracker.py amorfas.mp4 --out debug_amorfas.csv --video-out debug_amorfas.mp4
    uv run debug_tracker.py amorfas.mp4 --no-video   (CSV only, faster)
"""

import argparse
import csv
from pathlib import Path

import cv2

from config import TrackerConfig
from detector_v2 import CLASS_COLORS, CLASS_LABELS, ChannelSegmentationConfig, classify_frame
from rf_classifier import load_if_available
from tracker import Track, Tracker

CSV_COLUMNS = [
    "frame", "n_detections", "track_id", "x", "y", "vx", "vy",
    "label", "hits", "frames_missing", "confirmed",
    "total_fibers_so_far", "total_amorphous_so_far",
]

FONT = cv2.FONT_HERSHEY_SIMPLEX
UNCONFIRMED_COLOR = (150, 150, 150)


def debug_annotate(frame_bgr, frame_idx: int, particles, tracker: Tracker):
    """Every live track (confirmed or not), plus frame/count overlay."""
    out = frame_bgr.copy()
    for p in particles:
        color = CLASS_COLORS.get(p.label, (200, 200, 200))
        cv2.drawContours(out, [p.contour], -1, color, 1)

    for t in tracker.tracks:
        x, y = int(t.position[0]), int(t.position[1])
        color = CLASS_COLORS.get(t.label, (200, 200, 200)) if t.confirmed else UNCONFIRMED_COLOR
        radius = 4 if t.confirmed else 2
        cv2.circle(out, (x, y), radius, color, -1)
        tag = CLASS_LABELS.get(t.label, t.label) if t.confirmed else "?"
        cv2.putText(out, f"#{t.id} {tag} h{t.hits} m{t.frames_missing}",
                    (x + 6, y - 6), FONT, 0.4, color, 1)

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (230, 66), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)
    cv2.putText(out, f"Frame {frame_idx} | detecciones: {len(particles)}",
                (8, 18), FONT, 0.45, (255, 255, 255), 1)
    cv2.putText(out, f"Fibras: {tracker.total_fibers}  Amorfas: {tracker.total_amorphous}",
                (8, 38), FONT, 0.45, (255, 255, 255), 1)
    cv2.putText(out, f"Tracks vivos: {len(tracker.tracks)}",
                (8, 58), FONT, 0.45, (255, 255, 255), 1)
    return out


def run(video_path: Path, model_path: Path, out_path: Path,
        video_out_path: Path | None, cfg: ChannelSegmentationConfig) -> None:
    classifier = load_if_available(model_path)
    if classifier is None:
        raise FileNotFoundError(
            f"No existe el modelo {model_path}. Entrena primero con train_rf.py.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = None
    if video_out_path is not None:
        video_out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))

    Track._next_id = 1
    tracker = Tracker(TrackerConfig())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    try:
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                particles = classify_frame(frame, classifier, cfg)
                tracker.update(particles)

                for t in tracker.tracks:
                    writer.writerow([
                        frame_idx, len(particles), t.id,
                        round(float(t.position[0]), 2), round(float(t.position[1]), 2),
                        round(float(t.velocity[0]), 3), round(float(t.velocity[1]), 3),
                        t.label, t.hits, t.frames_missing, t.confirmed,
                        tracker.total_fibers, tracker.total_amorphous,
                    ])

                if video_writer is not None:
                    video_writer.write(debug_annotate(frame, frame_idx, particles, tracker))

                frame_idx += 1
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()

    print(f"{frame_idx} frames procesados -> {out_path}")
    if video_writer is not None:
        print(f"Video anotado -> {video_out_path}")
    print(f"Total fibras: {tracker.total_fibers} | "
          f"Total amorfas: {tracker.total_amorphous}")
    print(f"Tracks vivos al final: {len(tracker.tracks)} "
          f"({sum(1 for t in tracker.tracks if t.confirmed)} confirmados)")


def parse_args() -> argparse.Namespace:
    defaults = ChannelSegmentationConfig()
    parser = argparse.ArgumentParser(
        description="Exporta un CSV y un video anotado del comportamiento del tracker.")
    parser.add_argument("video", type=Path, help="Ruta del video a procesar")
    parser.add_argument("--model", type=Path, default=Path("rf_model.joblib"))
    parser.add_argument("--out", type=Path, default=None,
                        help="CSV de salida (default: debug_<video>.csv)")
    parser.add_argument("--video-out", type=Path, default=None,
                        help="Video anotado de salida (default: debug_<video>.mp4)")
    parser.add_argument("--no-video", action="store_true",
                        help="No generar el video anotado, solo el CSV")
    parser.add_argument("--green-thresh", type=int, default=defaults.green_thresh)
    parser.add_argument("--min-area", type=float, default=defaults.min_area)
    parser.add_argument("--max-area", type=float, default=defaults.max_area)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out or Path(f"debug_{args.video.stem}.csv")
    video_out_path = None
    if not args.no_video:
        video_out_path = args.video_out or Path(f"debug_{args.video.stem}.mp4")
    cfg = ChannelSegmentationConfig(
        green_thresh=args.green_thresh,
        min_area=args.min_area,
        max_area=args.max_area)
    run(args.video, args.model, out_path, video_out_path, cfg)


if __name__ == "__main__":
    main()
