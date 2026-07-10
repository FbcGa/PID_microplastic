"""Alternative segmentation: green-channel threshold (ported from a
validated MATLAB script), for visual comparison against the local-darkness
method in detector.py.

MATLAB reference:
    G = I(:,:,2);
    Z(G<umbral) = 255;

Usage:
    uv run detector_v2.py frames/frame2.jpg
"""

import argparse
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np

from features import extract_features
from rf_classifier import RandomForestParticleClassifier, load_if_available


@dataclass(frozen=True)
class ChannelSegmentationConfig:
    # Umbral sobre el canal verde: pixel con G por debajo = particula
    green_thresh: int = 100
    # Filtro de tamano (px^2) — provisional hasta la calibracion um -> px
    min_area: float = 200.0
    max_area: float = 8000.0


# Colores por clase (BGR): fibra naranja, amorfa verde
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "fiber": (0, 165, 255),
    "amorphous": (0, 255, 0),
}
CLASS_LABELS = {"fiber": "fibra", "amorphous": "amorfa"}


def segment_channels(frame_bgr: np.ndarray,
                      green_thresh: int = ChannelSegmentationConfig.green_thresh
                      ) -> dict[str, np.ndarray]:
    """Segments dark particles via a green-channel threshold: a pixel is
    particle if its G value is below green_thresh."""
    b, g, r = cv2.split(frame_bgr)
    mask_g = np.where(g < green_thresh, 255, 0).astype(np.uint8)
    return {
        "canal_r": r,
        "canal_g": g,
        "canal_b": b,
        "mascara_g": mask_g,
    }


def extract_contours(mask: np.ndarray) -> list[np.ndarray]:
    # RETR_CCOMP + top-level filter: keeps particles inside ring holes
    # (e.g. inside a bubble halo) that RETR_EXTERNAL would drop, without
    # duplicating the inner edge of the rings like RETR_LIST would.
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    return [c for c, h in zip(contours, hierarchy[0]) if h[3] == -1]


def filter_by_area(contours: list[np.ndarray], min_area: float,
                   max_area: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Splits contours into (kept, discarded) by area."""
    kept, discarded = [], []
    for contour in contours:
        if min_area <= cv2.contourArea(contour) <= max_area:
            kept.append(contour)
        else:
            discarded.append(contour)
    return kept, discarded


def draw_contours(frame_bgr: np.ndarray, mask: np.ndarray,
                  min_area: float = ChannelSegmentationConfig.min_area,
                  max_area: float = ChannelSegmentationConfig.max_area
                  ) -> np.ndarray:
    """Green = kept detections; red = discarded by the area filter
    (drawn thin, with their area, to help tune min/max)."""
    kept, discarded = filter_by_area(extract_contours(mask), min_area, max_area)
    output = frame_bgr.copy()
    cv2.drawContours(output, discarded, -1, (0, 0, 255), 1)
    for contour in discarded:
        x, y, _w, _h = cv2.boundingRect(contour)
        cv2.putText(output, f"{cv2.contourArea(contour):.0f}", (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.drawContours(output, kept, -1, (0, 255, 0), 2)
    print(f"Blobs: {len(kept)} detectados, {len(discarded)} descartados por area")
    return output


def classify_particles(kept: list[np.ndarray], gray: np.ndarray,
                       classifier: RandomForestParticleClassifier
                       ) -> list[str]:
    """Predicted class for each kept contour, using the RF model."""
    return [classifier.classify(extract_features(contour, gray))
            for contour in kept]


def draw_classified(frame_bgr: np.ndarray, kept: list[np.ndarray],
                    labels: list[str]) -> np.ndarray:
    """Each contour drawn in its class color with a text label."""
    output = frame_bgr.copy()
    for contour, label in zip(kept, labels):
        color = CLASS_COLORS.get(label, (200, 200, 200))
        cv2.drawContours(output, [contour], -1, color, 2)
        x, y, _w, _h = cv2.boundingRect(contour)
        cv2.putText(output, CLASS_LABELS.get(label, label), (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return output


def show_stages(frame_bgr: np.ndarray, stages: dict[str, np.ndarray],
                green_thresh: int,
                min_area: float = ChannelSegmentationConfig.min_area,
                max_area: float = ChannelSegmentationConfig.max_area,
                classifier: RandomForestParticleClassifier | None = None
                ) -> None:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if classifier is not None:
        kept, _ = filter_by_area(
            extract_contours(stages["mascara_g"]), min_area, max_area)
        labels = classify_particles(kept, stages["canal_g"], classifier)
        result = draw_classified(frame_bgr, kept, labels)
        n_fiber = labels.count("fiber")
        n_amorf = labels.count("amorphous")
        title = f"Clasificacion RF: {n_fiber} fibra, {n_amorf} amorfa"
        print(f"Clasificados: {n_fiber} fibra, {n_amorf} amorfa")
    else:
        result = draw_contours(frame_bgr, stages["mascara_g"], min_area, max_area)
        title = f"Contornos (G<{green_thresh})"
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(frame_rgb)
    axes[0].set_title("Original")
    axes[1].imshow(result_rgb)
    axes[1].set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    plt.show()


def show_original(frame_bgr: np.ndarray) -> None:
    """Standalone, larger figure with just the original image, so pixels
    can be inspected (hover, zoom, pan) without other panels in the way."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.canvas.manager.set_window_title("Original - inspeccion de pixeles")
    ax.imshow(frame_rgb)
    ax.set_title("Original")
    fig.tight_layout()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-channel threshold segmentation (MATLAB-style) "
                     "with stage-by-stage visualization.")
    defaults = ChannelSegmentationConfig()
    parser.add_argument("frame", help="Path to the test frame (image file)")
    parser.add_argument("--green-thresh", type=int, default=defaults.green_thresh)
    parser.add_argument("--min-area", type=float, default=defaults.min_area,
                        help="Area minima (px^2) para aceptar un blob")
    parser.add_argument("--max-area", type=float, default=defaults.max_area,
                        help="Area maxima (px^2) para aceptar un blob")
    parser.add_argument("--no-classify", action="store_true",
                        help="No usar el modelo RF aunque exista; solo contornos")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = cv2.imread(args.frame)
    if frame is None:
        raise IOError(f"Could not load frame: {args.frame}")

    show_original(frame)

    # Si existe un modelo entrenado, se clasifican los contornos; si no,
    # se muestran los contornos sin clasificar (comportamiento previo).
    classifier = None if args.no_classify else load_if_available()
    if classifier is not None:
        print("Modelo RF cargado: mostrando clasificacion fibra/amorfa")

    stages = segment_channels(frame, args.green_thresh)
    show_stages(frame, stages, args.green_thresh, args.min_area, args.max_area,
                classifier)


if __name__ == "__main__":
    main()
