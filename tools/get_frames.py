"""Extract frames from a video (or a folder of videos) into images for PDI testing.

Reads a single video, or every video in a folder, and writes frames as JPGs
into a per-video subfolder of the output folder (frames/<video_name>/frame_*.jpg).
By default it samples one frame per second; use --every to sample by frame
count, or --max-per-video to cap how many are written.

Usage:
    uv run get_frames.py                       (videos/ -> frames/, 1 fps)
    uv run get_frames.py mi_video.mp4          (un solo video -> frames/)
    uv run get_frames.py mi_video.mp4 --out salida/
    uv run get_frames.py --every 30            (one frame every 30 frames)
    uv run get_frames.py --fps 2               (two frames per second)
    uv run get_frames.py --max-per-video 20    (at most 20 frames per video)
    uv run get_frames.py --videos otra/ --out salida/
"""

import argparse
from pathlib import Path

import cv2

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


def frame_step(cap: cv2.VideoCapture, every: int | None, fps: float) -> int:
    """Number of frames to skip between saved frames."""
    if every is not None:
        return max(1, every)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        return 1  # unknown fps: save every frame
    return max(1, round(video_fps / fps))


def extract_video(path: Path, out_dir: Path, every: int | None,
                  fps: float, max_per_video: int | None) -> int:
    """Extracts frames from one video; returns how many were written."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  ! no se pudo abrir: {path.name}")
        return 0

    video_dir = out_dir / path.stem
    video_dir.mkdir(parents=True, exist_ok=True)

    step = frame_step(cap, every, fps)
    saved = 0
    index = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if index % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            out_path = video_dir / f"frame_{index:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
            if max_per_video is not None and saved >= max_per_video:
                break
        index += 1

    cap.release()
    print(f"  {path.name}: {saved} frames (1 de cada {step}) -> {video_dir}")
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae frames de un video o de una carpeta de videos.")
    parser.add_argument("video", type=Path, nargs="?", default=None,
                        help="Ruta a un solo video a procesar (ignora --videos)")
    parser.add_argument("--videos", type=Path, default=Path("videos"),
                        help="Carpeta con los videos (default: videos/)")
    parser.add_argument("--out", type=Path, default=Path("frames"),
                        help="Carpeta de salida (default: frames/)")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--fps", type=float, default=1.0,
                          help="Frames por segundo a extraer (default: 1)")
    sampling.add_argument("--every", type=int, default=None,
                          help="Guardar 1 frame cada N frames (ignora --fps)")
    parser.add_argument("--max-per-video", type=int, default=None,
                        help="Maximo de frames a guardar por video")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.video is not None:
        if not args.video.is_file():
            raise IOError(f"No existe el video: {args.video}")
        videos = [args.video]
        origen = args.video
    else:
        if not args.videos.is_dir():
            raise IOError(f"No existe la carpeta de videos: {args.videos}")
        videos = sorted(p for p in args.videos.iterdir()
                        if p.suffix.lower() in VIDEO_EXTENSIONS)
        if not videos:
            raise IOError(f"No se encontraron videos en: {args.videos}")
        origen = args.videos

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(videos)} video(s) en {origen} -> {args.out}")

    total = 0
    for path in videos:
        total += extract_video(path, args.out, args.every,
                               args.fps, args.max_per_video)
    print(f"Total: {total} frames extraidos en {args.out}")


if __name__ == "__main__":
    main()
