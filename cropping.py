"""Small standalone helpers shared across scripts.

Usage (CLI):
    uv run utils.py fondo.jpg --out fondo_recortado.jpg
"""

import argparse
from enum import IntEnum
from pathlib import Path

import cv2
import numpy as np


class EdgeCropPx(IntEnum):
    """Px recortados de los bordes superior/inferior antes de segmentar."""
    TOP = 50
    BOTTOM = 100


def crop_top_bottom_strips(frame: np.ndarray) -> np.ndarray:
    """Frame with its top and bottom edges removed (EdgeCropPx.TOP /
    EdgeCropPx.BOTTOM), full width, returning the remaining middle band."""
    h = frame.shape[0]
    top = min(EdgeCropPx.TOP, h)
    bottom = min(EdgeCropPx.BOTTOM, h - top)
    return frame[top:h - bottom, :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recorta una franja superior e inferior de una imagen "
                     f"({EdgeCropPx.TOP}px arriba, {EdgeCropPx.BOTTOM}px abajo).")
    parser.add_argument("image", type=Path, help="Imagen de entrada (p.ej. fondo.jpg)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Ruta de salida (default: <nombre>_cropped.<ext>)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise IOError(f"No se pudo leer la imagen: {args.image}")

    cropped = crop_top_bottom_strips(image)

    out_path = args.out or args.image.with_stem(args.image.stem + "_cropped")
    cv2.imwrite(str(out_path), cropped)
    print(f"{args.image} {image.shape[:2]} -> {out_path} {cropped.shape[:2]}")


if __name__ == "__main__":
    main()
