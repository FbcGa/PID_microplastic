"""Trim a video to a [start, end] time range.

Usage:
    uv run cut_video.py video.mp4 --start 10 --end 25
    uv run cut_video.py video.mp4 --start 1:30 --end 2:00 --out recorte.mp4
    uv run cut_video.py video.mp4 --start 5          (hasta el final)
"""

import argparse
from pathlib import Path

import cv2


def parse_time(value: str) -> float:
    """Accepts seconds ("12.5") or "mm:ss" / "hh:mm:ss"."""
    if ":" not in value:
        return float(value)
    parts = [float(p) for p in value.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def cut_video(video_path: Path, start: float, end: float | None,
             out_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, round(start * fps))
    end_frame = total_frames if end is None else min(total_frames, round(end * fps))
    if start_frame >= end_frame:
        cap.release()
        raise ValueError(
            f"Rango invalido: start={start}s (frame {start_frame}) >= "
            f"end={end}s (frame {end_frame})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    try:
        for _ in range(end_frame - start_frame):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        cap.release()
        writer.release()

    duration = written / fps
    print(f"{video_path.name}: frames {start_frame}-{start_frame + written} "
          f"({duration:.2f}s a {fps:.1f} fps) -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recorta un video a un rango de tiempo [start, end].")
    parser.add_argument("video", type=Path, help="Ruta del video de entrada")
    parser.add_argument("--start", type=parse_time, default=0.0,
                        help="Inicio: segundos (12.5) o mm:ss / hh:mm:ss")
    parser.add_argument("--end", type=parse_time, default=None,
                        help="Fin (default: hasta el final del video)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Ruta de salida (default: <video>_cut.mp4)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out or args.video.with_name(f"{args.video.stem}_cut.mp4")
    cut_video(args.video, args.start, args.end, out_path)


if __name__ == "__main__":
    main()
