"""Visual debugging of the detection pipeline over still frames.

Processes every image in frames/ and, for each one, writes the output of
each pipeline stage to frames/debug/<name>/:

    00_original      input frame
    01_gris          grayscale
    02_blur          Gaussian blur
    03_fondo_local   local background (box blur)        [local mode]
    03_diferencia    darkness / absdiff
    03b_dif_amplificada  same, contrast-stretched (visual aid only)
    04_umbral        binary threshold
    05_apertura      morphological open
    06_cierre        morphological close (final mask)
    07_resultado     classified blobs drawn over the original

Colors in 07_resultado: green = fiber, orange = amorphous, red = bubble
(discarded); with --todos, size-rejected blobs are drawn in gray with
their area, to calibrate min_area/max_area (µm -> px).

Usage:
    uv run debug_frames.py
    uv run debug_frames.py --todos
    uv run debug_frames.py --background fondo.png   (background mode only)
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from config import DetectionConfig
from detector import Detection, MicroplasticDetector, ParticleClass

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

CLASS_COLORS = {
    ParticleClass.FIBER: (0, 255, 0),        # green
    ParticleClass.AMORPHOUS: (0, 165, 255),  # orange
    ParticleClass.BUBBLE: (0, 0, 255),       # red (discarded)
}
REJECTED_COLOR = (160, 160, 160)             # gray: filtered out by size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runs the detector over frames/ saving every stage.")
    parser.add_argument("--frames", default="frames",
                        help="Folder with the frames to process")
    parser.add_argument("--background", default=None,
                        help="Background image (background mode only)")
    parser.add_argument("--todos", action="store_true",
                        help="Also draw size-rejected blobs with their area")
    return parser.parse_args()


def annotate(frame: np.ndarray, gray: np.ndarray, darkness: np.ndarray,
             local_bg: np.ndarray, mask: np.ndarray,
             detector: MicroplasticDetector, show_rejected: bool,
             frame_name: str, csv_rows: list[dict]) -> np.ndarray:
    """Draws every blob in the final mask with its ID number (features go
    to the CSV), classified like the detector does (including bubbles,
    which detect() discards)."""
    out = frame.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    analyzed = [(contour,
                 detector._analyze_contour(contour, gray, darkness, local_bg))
                for contour in contours]
    bubble_boxes = [d.bbox for _, d in analyzed
                    if d is not None and d.particle_class is ParticleClass.BUBBLE]
    radius = detector._cfg.bubble_exclusion_radius

    blob_id = 0
    for contour, detection in analyzed:
        x, y, w, h = cv2.boundingRect(contour)
        if detection is None and not show_rejected:
            continue
        blob_id += 1
        if detection is not None:
            particle_class = detection.particle_class.value
            if (detection.particle_class is not ParticleClass.BUBBLE
                    and detector._near_any(detection.bbox, bubble_boxes, radius)):
                particle_class = "bubble_vecina"
            color = (CLASS_COLORS[ParticleClass.BUBBLE]
                     if particle_class == "bubble_vecina"
                     else CLASS_COLORS[detection.particle_class])
            csv_rows.append({
                "frame": frame_name, "blob": blob_id,
                "clase": particle_class,
                "area": round(detection.area),
                "aspect_ratio": round(detection.aspect_ratio, 2),
                "circularidad": round(detection.circularity, 2),
                "oscuridad_pico": round(detection.peak_darkness),
                "ancho_medio": round(detection.mean_width, 1),
                "dispersion_anillo": round(detection.ring_spread),
                "x": x, "y": y,
            })
        elif detection is None:
            color = REJECTED_COLOR
            csv_rows.append({
                "frame": frame_name, "blob": blob_id,
                "clase": "descartado",
                "area": round(cv2.contourArea(contour)),
                "aspect_ratio": "", "circularidad": "",
                "oscuridad_pico": "", "ancho_medio": "",
                "dispersion_anillo": "", "x": x, "y": y,
            })
        cv2.drawContours(out, [contour], -1, color, 2)
        cv2.putText(out, str(blob_id), (x, max(24, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return out


def process_frame(detector: MicroplasticDetector, image_path: Path,
                  output_dir: Path, show_rejected: bool,
                  csv_rows: list[dict]) -> float:
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"  [skipped] could not read {image_path.name}")
        return 0.0

    stages: dict[str, np.ndarray] = {}
    start = time.perf_counter()
    result = detector.detect(frame, stages)
    elapsed_ms = (time.perf_counter() - start) * 1000

    stage_dir = output_dir / image_path.stem
    stage_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(stage_dir / "00_original.png"), frame)
    for name, image in stages.items():
        cv2.imwrite(str(stage_dir / f"{name}.png"), image)
    amplified = cv2.normalize(stages["03_diferencia"], None, 0, 255,
                              cv2.NORM_MINMAX)
    cv2.imwrite(str(stage_dir / "03b_dif_amplificada.png"), amplified)
    cv2.imwrite(str(stage_dir / "07_resultado.png"),
                annotate(frame, stages["01_gris"], stages["03_diferencia"],
                         stages["03_fondo_local"], result.mask,
                         detector, show_rejected, image_path.stem, csv_rows))

    fibers = result.count(ParticleClass.FIBER)
    amorphous = result.count(ParticleClass.AMORPHOUS)
    print(f"  {image_path.name}: {fibers} fibers, {amorphous} amorphous, "
          f"{result.discarded_bubbles} bubbles  ({elapsed_ms:.1f} ms)")
    return elapsed_ms


def main() -> None:
    args = parse_args()
    frames_dir = Path(args.frames)
    if not frames_dir.is_dir():
        raise SystemExit(f"Folder not found: {frames_dir}")

    config = DetectionConfig()
    detector = MicroplasticDetector(config)
    print(f"Segmentation mode: {config.segmentation_mode}")

    background_path = None
    if args.background is not None:
        background_path = Path(args.background)
        background = cv2.imread(str(background_path))
        if background is None:
            raise SystemExit(f"Could not read background: {background_path}")
        detector.set_background(background)

    output_dir = frames_dir / "debug"
    frame_paths = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
        and p != background_path
        and p.stem.lower() not in ("background", "fondo"))
    if not frame_paths:
        raise SystemExit(f"No images found in {frames_dir}")

    csv_rows: list[dict] = []
    times = [process_frame(detector, p, output_dir, args.todos, csv_rows)
             for p in frame_paths]

    csv_path = output_dir / "blobs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame", "blob", "clase", "area", "aspect_ratio", "circularidad",
            "oscuridad_pico", "ancho_medio", "dispersion_anillo", "x", "y"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Features de {len(csv_rows)} blobs -> {csv_path}")

    times = [t for t in times if t > 0]
    if times:
        print(f"detect(): mean {np.mean(times):.1f} ms/frame "
              f"(max {max(times):.1f} ms) on this PC")


if __name__ == "__main__":
    main()
