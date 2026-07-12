import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Parámetros que tienes que ajustar ---
CSV_PATH = "debug_amorfas.csv"
FPS = 30                      # el framerate con el que se grabó el CSV
ESCALA_UM_POR_PX = 6.67       # de tu calibración con el simulante de 300 um

# --- Ojo con las unidades de vx, vy ---
VX_VY_EN_PX_POR_FRAME = True

BLUR_MAXIMO_PX = 1.0          # tolerancia de borrón en píxeles
UMBRAL_OUTLIER_UM_S = 25000   # donde viste el hueco en el histograma


def main():
    df = pd.read_csv(CSV_PATH)

    df["velocidad_px"] = np.sqrt(df["vx"]**2 + df["vy"]**2)

    if VX_VY_EN_PX_POR_FRAME:
        df["velocidad_px_s"] = df["velocidad_px"] * FPS
    else:
        df["velocidad_px_s"] = df["velocidad_px"]

    df["velocidad_um_s"] = df["velocidad_px_s"] * ESCALA_UM_POR_PX

    v_p50 = df["velocidad_um_s"].median()
    v_p90 = df["velocidad_um_s"].quantile(0.90)
    v_p95 = df["velocidad_um_s"].quantile(0.95)
    v_p99 = df["velocidad_um_s"].quantile(0.99)
    v_max = df["velocidad_um_s"].max()

    print(f"Velocidad mediana:      {v_p50:.1f} um/s")
    print(f"Velocidad percentil90:  {v_p90:.1f} um/s")
    print(f"Velocidad percentil95:  {v_p95:.1f} um/s")
    print(f"Velocidad percentil99:  {v_p99:.1f} um/s")
    print(f"Velocidad máxima:       {v_max:.1f} um/s")
    print()

    blur_maximo_um = BLUR_MAXIMO_PX * ESCALA_UM_POR_PX

    resultados = {}
    for nombre, v in [
        ("mediana", v_p50),
        ("percentil90", v_p90),
        ("percentil95", v_p95),
        ("percentil99", v_p99),
        ("máximo", v_max),
    ]:
        shutter_us = (blur_maximo_um / v) * 1_000_000
        resultados[nombre] = shutter_us
        print(f"Shutter máximo (según {nombre}): {shutter_us:.0f} us")

    # --- Verificación de outliers (cruce con frames_missing y hits) ---
    outliers = df[df["velocidad_um_s"] > UMBRAL_OUTLIER_UM_S]
    print(f"\nRegistros por encima de {UMBRAL_OUTLIER_UM_S} um/s: {len(outliers)}")
    print(
        outliers[["frame", "track_id", "velocidad_um_s", "frames_missing", "hits"]]
        .sort_values("velocidad_um_s", ascending=False)
    )
   # --- Recalcular velocidad desde x,y, exigiendo frames consecutivos reales ---
    df_sorted = df.sort_values(["track_id", "frame"]).copy()
    df_sorted["frame_diff"] = df_sorted.groupby("track_id")["frame"].diff()
    df_sorted["dx"] = df_sorted.groupby("track_id")["x"].diff()
    df_sorted["dy"] = df_sorted.groupby("track_id")["y"].diff()

    # --- Recalcular velocidad usando SOLO detecciones reales (frames_missing == 0) ---
    df_det = df[df["frames_missing"] == 0].sort_values(["track_id", "frame"]).copy()

    df_det["frame_diff"] = df_det.groupby("track_id")["frame"].diff()
    df_det["dx"] = df_det.groupby("track_id")["x"].diff()
    df_det["dy"] = df_det.groupby("track_id")["y"].diff()

    df_det_valid = df_det.dropna(subset=["frame_diff"]).copy()

    # velocidad promedio en el intervalo real entre dos detecciones (no asume frame_diff=1)
    df_det_valid["velocidad_px_recalc"] = (
        np.sqrt(df_det_valid["dx"]**2 + df_det_valid["dy"]**2) / df_det_valid["frame_diff"]
    )
    df_det_valid["velocidad_px_s_recalc"] = df_det_valid["velocidad_px_recalc"] * FPS
    df_det_valid["velocidad_um_s_recalc"] = (
        df_det_valid["velocidad_px_s_recalc"] * ESCALA_UM_POR_PX
    )

    v_p90_d = df_det_valid["velocidad_um_s_recalc"].quantile(0.90)
    v_p95_d = df_det_valid["velocidad_um_s_recalc"].quantile(0.95)
    v_p99_d = df_det_valid["velocidad_um_s_recalc"].quantile(0.99)
    v_max_d = df_det_valid["velocidad_um_s_recalc"].max()

    print(f"\nRegistros con detección real consecutiva: {len(df_det_valid)}")
    print(f"Con velocidad recalculada (solo detecciones reales, normalizado por intervalo):")
    print(f"  p90: {v_p90_d:.1f} um/s -> shutter {(blur_maximo_um / v_p90_d) * 1e6:.0f} us")
    print(f"  p95: {v_p95_d:.1f} um/s -> shutter {(blur_maximo_um / v_p95_d) * 1e6:.0f} us")
    print(f"  p99: {v_p99_d:.1f} um/s -> shutter {(blur_maximo_um / v_p99_d) * 1e6:.0f} us")
    print(f"  max: {v_max_d:.1f} um/s -> shutter {(blur_maximo_um / v_max_d) * 1e6:.0f} us")
    # --- Histograma ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].hist(df["velocidad_um_s"], bins=60, color="#4C72B0", edgecolor="black", alpha=0.8)
    axes[0].axvline(v_p50, color="green", linestyle="--", label=f"mediana ({v_p50:.0f})")
    axes[0].axvline(v_p90, color="gold", linestyle="--", label=f"p90 ({v_p90:.0f})")
    axes[0].axvline(v_p95, color="orange", linestyle="--", label=f"p95 ({v_p95:.0f})")
    axes[0].axvline(v_p99, color="red", linestyle="--", label=f"p99 ({v_p99:.0f})")
    axes[0].axvline(v_max, color="black", linestyle=":", label=f"max ({v_max:.0f})")
    axes[0].set_xlabel("velocidad (um/s)")
    axes[0].set_ylabel("frecuencia")
    axes[0].set_title("Distribución de velocidades")
    axes[0].legend(fontsize=8)

    axes[1].hist(df["velocidad_um_s"], bins=60, color="#4C72B0", edgecolor="black", alpha=0.8)
    axes[1].set_yscale("log")
    axes[1].axvline(v_p95, color="orange", linestyle="--", label="p95")
    axes[1].axvline(v_max, color="black", linestyle=":", label="max")
    axes[1].axvline(UMBRAL_OUTLIER_UM_S, color="purple", linestyle="-.", label="umbral outlier")
    axes[1].set_xlabel("velocidad (um/s)")
    axes[1].set_ylabel("frecuencia (escala log)")
    axes[1].set_title("Misma distribución, eje Y logarítmico\n(para ver si la cola es ruido aislado)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("hist_velocidad.png", dpi=150)
    print("\nHistograma guardado en: hist_velocidad.png")

    return df, resultados


if __name__ == "__main__":
    main()