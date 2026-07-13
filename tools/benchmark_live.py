"""Mide el costo por etapa del pipeline en vivo (captura, segmentacion,
clasificacion RF, tracking, dibujo) sin abrir la UI Tkinter, para
diagnosticar cuellos de botella reales en la Pi 5 en vez de adivinarlos.

Reutiliza los mismos modulos que main_live.py/processing.py (misma
camara, misma config, mismo modelo); solo agrega cronometraje por etapa
y, opcionalmente, una aproximacion del costo de render de la UI.

Usage:
    uv run tools/benchmark_live.py --seconds 30                        (Pi, camara real)
    uv run tools/benchmark_live.py --source videos/x.mp4 --seconds 10  (PC, sin camara)
    uv run tools/benchmark_live.py --seconds 30 --with-ui-cost         (+ costo aproximado de pintar la UI)
"""

import argparse
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration import BACKGROUND_PATH, load_calibration
from camera import open_camera
from config import BackgroundSegmentationConfig, TrackerConfig
from cropping import crop_top_bottom_strips
from detector import (ClassifiedParticle, extract_contours, filter_by_area,
                      segment_against_background)
from features import extract_features
from random_forest.rf_classifier import load_if_available
from tracker import Tracker
from visualization import annotate

STAGE_ORDER = [
    "read", "crop", "segmentacion", "contornos", "features", "rf",
    "tracker", "annotate", "ciclo_total",
    "ui_cvtColor", "ui_resize_bicubic", "ui_resize_nearest",
]

# read() p50 por debajo de este umbral significa que el consumidor nunca
# espera un frame nuevo: el pipe de rpicam-vid ya tiene frames en cola.
READ_BLOCKED_THRESHOLD_MS = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=None,
                        help="Video para desarrollo en PC; si se omite usa rpicam-vid (Pi)")
    parser.add_argument("--seconds", type=float, default=30.0,
                        help="Duracion de la medicion, en segundos")
    parser.add_argument("--background", type=Path, default=BACKGROUND_PATH,
                        help="Imagen de fondo de referencia (default: la de la app en vivo)")
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).resolve().parent.parent
                                / "random_forest" / "rf_model.joblib",
                        help="Modelo Random Forest entrenado")
    parser.add_argument("--with-ui-cost", action="store_true",
                        help="Aproxima el costo de pintar la UI (cvtColor + "
                             "resize PIL, con BICUBIC y NEAREST) sobre el "
                             "frame anotado")
    return parser.parse_args()


def classify_frame_timed(cropped, background_bgr, background_g, classifier,
                         seg_cfg: BackgroundSegmentationConfig,
                         timings: dict[str, list[float]]) -> list[ClassifiedParticle]:
    """Mismo pipeline que detector.classify_frame, pero con cronometraje
    por sub-etapa (segmentacion vs contornos vs features vs RF)."""
    t0 = time.perf_counter()
    stages = segment_against_background(cropped, background_bgr, seg_cfg,
                                        background_g, full_stages=False)
    timings["segmentacion"].append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    kept, _ = filter_by_area(extract_contours(stages["mascara_g"]),
                             seg_cfg.min_area, seg_cfg.max_area,
                             seg_cfg.max_circularity)
    timings["contornos"].append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    features_list = [extract_features(c) for c in kept]
    timings["features"].append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    labels = classifier.classify_batch(features_list)
    timings["rf"].append((time.perf_counter() - t0) * 1000)

    return [ClassifiedParticle(c, label, cv2.boundingRect(c))
            for c, label in zip(kept, labels)]


def read_cpu_times() -> dict[str, tuple[int, int]]:
    """{'cpu0': (idle, total), ...} desde /proc/stat; {} si no existe (PC)."""
    try:
        lines = Path("/proc/stat").read_text().splitlines()
    except OSError:
        return {}
    times = {}
    for line in lines:
        if not line.startswith("cpu") or line.startswith("cpu "):
            continue
        parts = line.split()
        nums = [int(n) for n in parts[1:]]
        idle = nums[3] + nums[4]  # idle + iowait
        times[parts[0]] = (idle, sum(nums))
    return times


def cpu_percent(before: dict[str, tuple[int, int]],
               after: dict[str, tuple[int, int]]) -> dict[str, float]:
    result = {}
    for label, (idle0, total0) in before.items():
        if label not in after:
            continue
        idle1, total1 = after[label]
        d_total = total1 - total0
        result[label] = 0.0 if d_total <= 0 else (1 - (idle1 - idle0) / d_total) * 100
    return result


def read_vcgencmd(*args: str) -> str | None:
    try:
        result = subprocess.run(["vcgencmd", *args], capture_output=True,
                                text=True, timeout=2)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None  # no es una Pi (PC de desarrollo): se omite
    return result.stdout.strip()


def read_temp_c() -> float | None:
    raw = read_vcgencmd("measure_temp")  # "temp=53.1'C"
    if raw is None:
        return None
    try:
        return float(raw.split("=")[1].rstrip("'C"))
    except (IndexError, ValueError):
        return None


THROTTLED_BITS = {
    0: "bajo_voltaje_ahora", 1: "freq_limitada_ahora",
    2: "throttled_ahora", 3: "temp_limite_ahora",
    16: "bajo_voltaje_alguna_vez", 17: "freq_limitada_alguna_vez",
    18: "throttled_alguna_vez", 19: "temp_limite_alguna_vez",
}


def read_throttled() -> str | None:
    raw = read_vcgencmd("get_throttled")  # "throttled=0x50000"
    if raw is None:
        return None
    try:
        value = int(raw.split("=")[1], 16)
    except (IndexError, ValueError):
        return None
    active = [name for bit, name in THROTTLED_BITS.items() if value & (1 << bit)]
    return f"{raw} ({', '.join(active) if active else 'ok'})"


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(len(sorted_values) * pct))
    return sorted_values[idx]


def print_report(timings: dict[str, list[float]], frames_processed: int,
                 frames_dropped_none: int, elapsed: float,
                 cpu_before: dict[str, tuple[int, int]],
                 cpu_after: dict[str, tuple[int, int]]) -> None:
    print(f"\n{'Etapa':<20}{'p50 (ms)':>12}{'p95 (ms)':>12}{'max (ms)':>12}{'n':>8}")
    for stage in STAGE_ORDER:
        samples = timings.get(stage)
        if not samples:
            continue
        arr = sorted(samples)
        print(f"{stage:<20}{percentile(arr, 0.50):>12.2f}"
              f"{percentile(arr, 0.95):>12.2f}{arr[-1]:>12.2f}{len(arr):>8}")

    fps = frames_processed / elapsed if elapsed > 0 else 0.0
    print(f"\nFrames procesados: {frames_processed} "
          f"(read() devolvio None {frames_dropped_none} veces)")
    print(f"FPS efectivo: {fps:.1f} en {elapsed:.1f}s")

    read_samples = timings.get("read", [])
    if read_samples:
        read_p50 = statistics.median(read_samples)
        if read_p50 < READ_BLOCKED_THRESHOLD_MS:
            print(f"Diagnostico pipe: read() p50={read_p50:.2f}ms "
                  f"(<{READ_BLOCKED_THRESHOLD_MS:.0f}ms) -> el consumidor va "
                  f"atrasado, hay frames encolados en rpicam-vid.")
        else:
            print(f"Diagnostico pipe: read() p50={read_p50:.2f}ms -> ok, "
                  f"el consumidor espera al productor (sin cola acumulada).")

    cpu = cpu_percent(cpu_before, cpu_after)
    if cpu:
        print("\nCPU por core:")
        for label in sorted(cpu, key=lambda l: int(l.removeprefix("cpu"))):
            print(f"  {label}: {cpu[label]:.0f}%")

    temp = read_temp_c()
    if temp is not None:
        print(f"Temperatura: {temp:.1f} C")
    throttled = read_throttled()
    if throttled is not None:
        print(f"Throttled: {throttled}")


def main() -> None:
    args = parse_args()

    background_full = cv2.imread(str(args.background))
    if background_full is None:
        raise IOError(f"No existe fondo de referencia: {args.background}")
    classifier = load_if_available(args.model)
    if classifier is None:
        raise FileNotFoundError(
            f"No existe el modelo {args.model}. Entrena primero con "
            "train_rf.py (necesita fibras y amorfas etiquetadas).")

    calibration = load_calibration()
    background_bgr = crop_top_bottom_strips(
        background_full, calibration.crop_top, calibration.crop_bottom)
    background_g = background_bgr[:, :, 1]

    seg_cfg = BackgroundSegmentationConfig()
    tracker = Tracker(TrackerConfig())

    if args.with_ui_cost:
        from PIL import Image
        from app_ui import PANEL_WIDTH, WINDOW_H, WINDOW_W
        video_w, video_h = WINDOW_W - PANEL_WIDTH, WINDOW_H

    camera = open_camera(args.source)
    camera.start()

    timings: dict[str, list[float]] = defaultdict(list)
    frames_processed = 0
    frames_dropped_none = 0

    cpu_before = read_cpu_times()
    start = time.perf_counter()
    deadline = start + args.seconds

    try:
        while time.perf_counter() < deadline:
            t_cycle0 = time.perf_counter()

            t0 = time.perf_counter()
            frame = camera.read()
            timings["read"].append((time.perf_counter() - t0) * 1000)
            if frame is None:
                frames_dropped_none += 1
                continue

            t0 = time.perf_counter()
            cropped = crop_top_bottom_strips(
                frame, calibration.crop_top, calibration.crop_bottom)
            timings["crop"].append((time.perf_counter() - t0) * 1000)

            particles = classify_frame_timed(
                cropped, background_bgr, background_g, classifier, seg_cfg, timings)

            t0 = time.perf_counter()
            tracks = tracker.update(particles)
            timings["tracker"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            annotated = annotate(cropped, particles, tracks, tracker)
            timings["annotate"].append((time.perf_counter() - t0) * 1000)

            if args.with_ui_cost:
                t0 = time.perf_counter()
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                timings["ui_cvtColor"].append((time.perf_counter() - t0) * 1000)

                image = Image.fromarray(rgb)
                scale = min(video_w / image.width, video_h / image.height)
                new_size = (max(1, int(image.width * scale)),
                           max(1, int(image.height * scale)))

                t0 = time.perf_counter()
                image.resize(new_size, Image.Resampling.BICUBIC)
                timings["ui_resize_bicubic"].append((time.perf_counter() - t0) * 1000)

                t0 = time.perf_counter()
                image.resize(new_size, Image.Resampling.NEAREST)
                timings["ui_resize_nearest"].append((time.perf_counter() - t0) * 1000)

            timings["ciclo_total"].append((time.perf_counter() - t_cycle0) * 1000)
            frames_processed += 1
    finally:
        camera.stop()

    elapsed = time.perf_counter() - start
    cpu_after = read_cpu_times()

    print_report(timings, frames_processed, frames_dropped_none, elapsed,
                cpu_before, cpu_after)


if __name__ == "__main__":
    main()
