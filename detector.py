"""Background-reference segmentation: absdiff against a captured clean-water
image on the green channel, for cases where a fixed global threshold can't
separate particle from background (e.g. thin fibers whose G value overlaps
the background's).

Usage:
    uv run detector.py frames/frame2.jpg --background frames/fondo.jpg
"""

import argparse
from dataclasses import dataclass

import cv2
import numpy as np

from config import BackgroundSegmentationConfig
from features import extract_features
from random_forest.rf_classifier import RandomForestParticleClassifier, load_if_available
from utils import crop_top_bottom_strips

# Colores por clase (BGR): fibra naranja, amorfa verde
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "fiber": (0, 165, 255),
    "amorphous": (0, 255, 0),
}
CLASS_LABELS = {"fiber": "fibra", "amorphous": "amorfa"}


@dataclass(frozen=True)
class ClassifiedParticle:
    """One detected+classified particle. Exposes the centroid and label the
    tracker needs, plus the contour for drawing."""
    contour: np.ndarray
    label: str  # "fiber" | "amorphous"
    bbox: tuple[int, int, int, int]  # x, y, w, h

    @property
    def centroid(self) -> tuple[int, int]:
        x, y, w, h = self.bbox
        return x + w // 2, y + h // 2

def extract_contours(mask: np.ndarray) -> list[np.ndarray]:
    # RETR_CCOMP + top-level filter: keeps particles inside ring holes
    # (e.g. inside a bubble halo) that RETR_EXTERNAL would drop, without
    # duplicating the inner edge of the rings like RETR_LIST would.
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    return [c for c, h in zip(contours, hierarchy[0]) if h[3] == -1]


def circularity(contour: np.ndarray) -> float:
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return 4 * np.pi * cv2.contourArea(contour) / (perimeter ** 2)


def filter_by_area(contours: list[np.ndarray], min_area: float,
                   max_area: float, max_circularity: float
                   ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Splits contours into (kept, discarded) by area and circularity.
    Blobs near-perfectly circular (circularity >= max_circularity) are
    discarded as bubbles, regardless of area."""
    kept, discarded = [], []
    for contour in contours:
        area_ok = min_area <= cv2.contourArea(contour) <= max_area
        round_ok = circularity(contour) < max_circularity
        if area_ok and round_ok:
            kept.append(contour)
        else:
            discarded.append(contour)
    return kept, discarded


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


def show_original(frame_bgr: np.ndarray) -> None:
    """Standalone, larger figure with just the original image, so pixels
    can be inspected (hover, zoom, pan) without other panels in the way."""
    import matplotlib.pyplot as plt
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.canvas.manager.set_window_title("Original - inspeccion de pixeles")
    ax.imshow(frame_rgb)
    ax.set_title("Original")
    fig.tight_layout()


def segment_against_background(frame_bgr: np.ndarray, background_bgr: np.ndarray,
                               cfg: BackgroundSegmentationConfig = BackgroundSegmentationConfig(),
                               background_g: np.ndarray | None = None
                               ) -> dict[str, np.ndarray]:
    """canal_g crudo (para features) + mascara_g (via absdiff contra el
    fondo, blur, threshold, apertura+cierre).

    frame_bgr y background_bgr deben llegar ya recortados (ver
    crop_top_bottom_strips) por el llamador, antes de entrar al pipeline:
    el recorte de bordes es una decision de que region analizar, no un
    detalle interno de la segmentacion.

    background_g: canal verde del fondo ya extraido, para evitar repetir
    el split del fondo (que no cambia) en cada frame de un video. Si no
    se pasa, se calcula aqui."""
    g = frame_bgr[:, :, 1]
    bg_g = background_g if background_g is not None else background_bgr[:, :, 1]
    diff = cv2.absdiff(g, bg_g)
    blurred = cv2.GaussianBlur(diff, (cfg.blur_ksize, cfg.blur_ksize), 0)
    _, binary = cv2.threshold(blurred, cfg.diff_thresh, 255, cv2.THRESH_BINARY)
    if cfg.open_ksize > 1:
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_ksize,) * 2)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    else:
        opened = binary
    if cfg.close_ksize > 1:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_ksize,) * 2)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)
    else:
        closed = opened
    return {
        "canal_g": g,
        "fondo_g": bg_g,
        "diferencia_g": diff,
        "diferencia_blur": blurred,
        "umbral": binary,
        "apertura": opened,
        "mascara_g": closed,
    }


def show_morphology_stages(frame_bgr: np.ndarray, stages: dict[str, np.ndarray]) -> None:
    """Grid with every step of the pipeline (original, canal G, fondo G,
    diferencia, blur, umbral, apertura, cierre final), para ver en que paso
    se pierde una fibra."""
    import matplotlib.pyplot as plt
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    panels = [
        ("Original", frame_rgb),
        ("Canal G (frame)", stages["canal_g"]),
        ("Canal G (fondo)", stages["fondo_g"]),
        ("Diferencia |frame-fondo|", stages["diferencia_g"]),
        ("Diferencia + blur", stages["diferencia_blur"]),
        ("Umbral (binaria)", stages["umbral"]),
        ("Apertura (quita ruido)", stages["apertura"]),
        ("Cierre (mascara final)", stages["mascara_g"]),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (title, img) in zip(axes.flat, panels):
        cmap = None if img.ndim == 3 else "gray"
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def draw_contours_debug(frame_bgr: np.ndarray, mask: np.ndarray,
                        min_area: float, max_area: float,
                        max_circularity: float) -> np.ndarray:
    """Draws contours color-coded by whether they passed the filter, and
    labels area/circularity on both
    the accepted (green) and discarded (red) contours."""
    kept, discarded = filter_by_area(
        extract_contours(mask), min_area, max_area, max_circularity)
    output = frame_bgr.copy()
    for contour, color, thickness in ([(c, (0, 0, 255), 1) for c in discarded]
                                       + [(c, (0, 255, 0), 2) for c in kept]):
        cv2.drawContours(output, [contour], -1, color, thickness)
        x, y, _w, _h = cv2.boundingRect(contour)
        cv2.putText(output, f"{cv2.contourArea(contour):.0f} c={circularity(contour):.2f}",
                    (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    print(f"Blobs: {len(kept)} detectados, {len(discarded)} descartados "
          f"por area/circularidad")
    return output


def classify_frame_v3(frame_bgr: np.ndarray, background_bgr: np.ndarray,
                      classifier: RandomForestParticleClassifier,
                      cfg: BackgroundSegmentationConfig = BackgroundSegmentationConfig(),
                      background_g: np.ndarray | None = None
                      ) -> list[ClassifiedParticle]:

    stages = segment_against_background(frame_bgr, background_bgr, cfg, background_g)
    kept, _ = filter_by_area(
        extract_contours(stages["mascara_g"]), cfg.min_area, cfg.max_area,
        cfg.max_circularity)
    features_list = [extract_features(contour) for contour in kept]
    labels = classifier.classify_batch(features_list)

    return [
        ClassifiedParticle(contour, label, cv2.boundingRect(contour))
        for contour, label in zip(kept, labels)
    ]


def show_stages_v3(frame_bgr: np.ndarray, stages: dict[str, np.ndarray],
                   min_area: float = BackgroundSegmentationConfig.min_area,
                   max_area: float = BackgroundSegmentationConfig.max_area,
                   classifier: RandomForestParticleClassifier | None = None,
                   max_circularity: float = BackgroundSegmentationConfig.max_circularity
                   ) -> None:
    import matplotlib.pyplot as plt
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if classifier is not None:
        kept, _ = filter_by_area(
            extract_contours(stages["mascara_g"]), min_area, max_area,
            max_circularity)
        features_list = [extract_features(contour) for contour in kept]
        labels = classifier.classify_batch(features_list)
        result = draw_classified(frame_bgr, kept, labels)
        n_fiber = labels.count("fiber")
        n_amorf = labels.count("amorphous")
        title = f"Clasificacion RF: {n_fiber} fibra, {n_amorf} amorfa"
        print(f"Clasificados: {n_fiber} fibra, {n_amorf} amorfa")
    else:
        result = draw_contours_debug(frame_bgr, stages["mascara_g"], min_area, max_area,
                                     max_circularity)
        title = "Contornos (diff vs fondo)"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segmentacion por resta contra una imagen de fondo "
                     "(canal verde), con visualizacion. Los parametros del "
                     "pipeline (umbrales, kernels, area, circularidad) se "
                     "ajustan editando BackgroundSegmentationConfig, no por args.")
    parser.add_argument("frame", help="Path al frame de prueba (imagen)")
    parser.add_argument("--background", required=True,
                        help="Path a la imagen de fondo (agua limpia, sin particulas)")
    parser.add_argument("--no-classify", action="store_true",
                        help="No usar el modelo RF aunque exista; solo contornos")
    parser.add_argument("--debug-stages", action="store_true",
                        help="Muestra cada paso del pipeline (diff, blur, "
                             "umbral, apertura, cierre) en una grilla, para "
                             "ver en que paso se pierde una fibra")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = cv2.imread(args.frame)
    if frame is None:
        raise IOError(f"Could not load frame: {args.frame}")
    background = cv2.imread(args.background)
    if background is None:
        raise IOError(f"Could not load background: {args.background}")
    if frame.shape != background.shape:
        raise IOError(
            f"Frame y background tienen tamano distinto: "
            f"{frame.shape} vs {background.shape}. Deben venir de la misma "
            f"camara/resolucion/crop.")

    frame = crop_top_bottom_strips(frame)
    background = crop_top_bottom_strips(background)

    show_original(frame)

    classifier = None if args.no_classify else load_if_available()
    if classifier is not None:
        print("Modelo RF cargado: mostrando clasificacion fibra/amorfa")

    cfg = BackgroundSegmentationConfig()
    stages = segment_against_background(frame, background, cfg)
    if args.debug_stages:
        show_morphology_stages(frame, stages)
    show_stages_v3(frame, stages, cfg.min_area, cfg.max_area, classifier,
                   cfg.max_circularity)


if __name__ == "__main__":
    main()
