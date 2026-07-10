"""Pixel-length reference grid over a frame.

Draws grid lines every N pixels (labeled) on top of a real frame, so
distances (particle displacement, sizes) can be read directly off the
image instead of guessed. Useful to calibrate tracker.max_distance, the
area filter, etc.

Usage:
    uv run show_grid.py frames/<sub>/frame_000000.jpg
    uv run show_grid.py frames/<sub>/frame_000000.jpg --step 25
"""

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_grid(frame_bgr: np.ndarray, step: int) -> np.ndarray:
    """Overlays a step-px grid with axis labels on a copy of the frame."""
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()
    color = (0, 255, 255)

    for x in range(0, w + 1, step):
        cv2.line(out, (x, 0), (x, h), color, 1)
        cv2.putText(out, str(x), (min(x + 2, w - 25), 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    for y in range(0, h + 1, step):
        cv2.line(out, (0, y), (w, y), color, 1)
        cv2.putText(out, str(y), (2, min(y + 12, h - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


def show(frame_bgr: np.ndarray, step: int) -> None:
    h, w = frame_bgr.shape[:2]
    print(f"Dimensiones del frame: {w}x{h} px | cuadricula cada {step} px")

    gridded = draw_grid(frame_bgr, step)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    gridded_rgb = cv2.cvtColor(gridded, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.canvas.manager.set_window_title(f"Cuadricula de referencia ({w}x{h} px)")
    axes[0].imshow(frame_rgb)
    axes[0].set_title("Original")
    axes[1].imshow(gridded_rgb)
    axes[1].set_title(f"Cuadricula cada {step} px")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Superpone una cuadricula en px sobre un frame real.")
    parser.add_argument("frame", help="Ruta de la imagen")
    parser.add_argument("--step", type=int, default=50,
                        help="Espaciado de la cuadricula en px (default: 50)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = cv2.imread(args.frame)
    if frame is None:
        raise IOError(f"No se pudo cargar el frame: {args.frame}")
    show(frame, args.step)


if __name__ == "__main__":
    main()
