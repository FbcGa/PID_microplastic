"""Interactive per-particle labeling tool to build the training dataset.

Runs detector's segmentation (resta contra una imagen de fondo) over
every frame under a folder, then for each detected particle shows it
highlighted and waits for a keypress to label it. Each labeled particle
becomes one row (its feature vector + label) in dataset.csv.

Keys:
    f   fiber (fibra)
    a   amorphous (amorfa)
    s   skip this particle (not written)
    x   crossed / merged particles (excluded from training, crop saved to
        merged/ and counted, since its descriptors are not trustworthy)
    q   quit and save

Usage:
    uv run tools/label_particles.py --background frames/fondo.jpg
    uv run tools/label_particles.py --background frames/fondo.jpg --frames otra/ --out mi_dataset.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector import (BackgroundSegmentationConfig, extract_contours,
                      filter_by_area, segment_against_background)
from features import FEATURE_NAMES, extract_features, feature_vector
from cropping import crop_top_bottom_strips

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
LABEL_KEYS = {ord("f"): "fiber", ord("a"): "amorphous"}
CSV_COLUMNS = FEATURE_NAMES + ["label", "source_frame", "blob_id"]

# Colors (BGR)
HIGHLIGHT = (0, 255, 255)   # current particle: yellow
OTHER = (120, 120, 120)     # the rest: dim gray


def find_frames(frames_dir: Path) -> list[Path]:
    """All image files under frames_dir (recursive, for get_frames subfolders)."""
    return sorted(p for p in frames_dir.rglob("*")
                  if p.suffix.lower() in IMAGE_EXTENSIONS)


def detect_particles(frame_bgr: np.ndarray, background_bgr: np.ndarray,
                     cfg: BackgroundSegmentationConfig) -> list[np.ndarray]:
    """Kept contours for one frame, via detector."""
    stages = segment_against_background(frame_bgr, background_bgr, cfg)
    kept, _discarded = filter_by_area(
        extract_contours(stages["mascara_g"]), cfg.min_area, cfg.max_area,
        cfg.max_circularity)
    return kept


def already_labeled(csv_path: Path) -> set[tuple[str, int]]:
    """(source_frame, blob_id) pairs already in the CSV, to skip on resume.
    Per-particle rather than per-frame: quitting mid-frame must not lose
    the remaining unlabeled particles of that frame."""
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="") as f:
        return {(row["source_frame"], int(row["blob_id"]))
                for row in csv.DictReader(f)}


def _open_writer(csv_path: Path):
    """Opens dataset.csv in append mode, writing the header if new."""
    is_new = not csv_path.exists() or csv_path.stat().st_size == 0
    handle = csv_path.open("a", newline="")
    writer = csv.writer(handle)
    if is_new:
        writer.writerow(CSV_COLUMNS)
    return handle, writer


def _render(frame_bgr: np.ndarray, contours: list[np.ndarray], current: int,
            max_width: int = 1100) -> np.ndarray:
    """Full frame with the current particle highlighted next to a zoomed crop."""
    canvas = frame_bgr.copy()
    cv2.drawContours(canvas, contours, -1, OTHER, 1)
    cv2.drawContours(canvas, [contours[current]], -1, HIGHLIGHT, 2)
    x, y, w, h = cv2.boundingRect(contours[current])
    cv2.rectangle(canvas, (x - 4, y - 4), (x + w + 4, y + h + 4), HIGHLIGHT, 2)

    pad = 30
    fh, fw = frame_bgr.shape[:2]
    cx0, cy0 = max(0, x - pad), max(0, y - pad)
    cx1, cy1 = min(fw, x + w + pad), min(fh, y + h + pad)
    crop = frame_bgr[cy0:cy1, cx0:cx1]

    if canvas.shape[1] > max_width:
        scale = max_width / canvas.shape[1]
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale)
    # Zoom the crop to a fixed height so small particles are visible
    crop_h = canvas.shape[0]
    crop_scale = crop_h / crop.shape[0] if crop.shape[0] > 0 else 1.0
    crop = cv2.resize(crop, None, fx=crop_scale, fy=crop_scale)
    return np.hstack([canvas, crop])


def run(frames_dir: Path, csv_path: Path, merged_dir: Path,
        background_bgr: np.ndarray, cfg: BackgroundSegmentationConfig) -> None:
    frames = find_frames(frames_dir)
    if not frames:
        raise IOError(f"No se encontraron imagenes en: {frames_dir}")

    done = already_labeled(csv_path)
    handle, writer = _open_writer(csv_path)
    window = "Etiquetado (f=fibra a=amorfa s=saltar x=cruzado q=salir)"
    counts = {"fiber": 0, "amorphous": 0, "skipped": 0, "merged": 0}

    background_shape = background_bgr.shape
    background_bgr = crop_top_bottom_strips(background_bgr)

    try:
        for frame_path in frames:
            source = frame_path.relative_to(frames_dir).as_posix()
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"  ! no se pudo leer: {source}")
                continue
            if frame.shape != background_shape:
                print(f"  ! tamano distinto al fondo, se salta: {source}")
                continue
            # Recorte de bordes ni bien se recibe el frame: mismas areas
            # que descarta el pipeline de deteccion en produccion.
            frame = crop_top_bottom_strips(frame)

            contours = detect_particles(frame, background_bgr, cfg)
            for blob_id, contour in enumerate(contours):
                if (source, blob_id) in done:
                    continue
                cv2.imshow(window, _render(frame, contours, blob_id))
                key = cv2.waitKey(0) & 0xFF

                if key == ord("q"):
                    print("Saliendo...")
                    return
                if key in LABEL_KEYS:
                    label = LABEL_KEYS[key]
                    feats = extract_features(contour)
                    writer.writerow(
                        feature_vector(feats) + [label, source, blob_id])
                    handle.flush()
                    counts[label] += 1
                elif key == ord("x"):
                    _save_merged(frame, contour, merged_dir, frame_path, blob_id)
                    counts["merged"] += 1
                else:  # s or any other key: skip
                    counts["skipped"] += 1
    finally:
        handle.close()
        cv2.destroyAllWindows()
        print(f"Fibras: {counts['fiber']} | Amorfas: {counts['amorphous']} | "
              f"Saltados: {counts['skipped']} | Cruzados (merged): {counts['merged']}")
        print(f"Dataset: {csv_path}")


def _save_merged(frame_bgr: np.ndarray, contour: np.ndarray, merged_dir: Path,
                 frame_path: Path, blob_id: int) -> None:
    """Saves the crop of a merged/crossed particle for later inspection."""
    merged_dir.mkdir(parents=True, exist_ok=True)
    x, y, w, h = cv2.boundingRect(contour)
    pad = 10
    fh, fw = frame_bgr.shape[:2]
    crop = frame_bgr[max(0, y - pad):min(fh, y + h + pad),
                     max(0, x - pad):min(fw, x + w + pad)]
    out = merged_dir / f"{frame_path.stem}_blob{blob_id}.jpg"
    cv2.imwrite(str(out), crop)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Etiquetado por particula para el dataset de Random Forest.")
    parser.add_argument("--background", type=Path, required=True,
                        help="Imagen de fondo (agua limpia, sin particulas)")
    parser.add_argument("--frames", type=Path, default=Path("frames"),
                        help="Carpeta con los frames (default: frames/)")
    parser.add_argument("--out", type=Path,
                        default=Path("random_forest/dataset.csv"),
                        help="CSV de salida (default: random_forest/dataset.csv)")
    parser.add_argument("--merged", type=Path, default=Path("merged"),
                        help="Carpeta para recortes cruzados/fusionados")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    background = cv2.imread(str(args.background))
    if background is None:
        raise IOError(f"Could not load background: {args.background}")
    run(args.frames, args.out, args.merged, background, BackgroundSegmentationConfig())


if __name__ == "__main__":
    main()
