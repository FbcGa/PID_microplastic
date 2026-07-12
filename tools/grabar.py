"""Graba video desde la camara de la Raspberry Pi via rpicam-vid.

Usage:
    uv run tools/grabar.py
    uv run tools/grabar.py --width 1280 --height 720   (resolucion explicita,
        para que coincida con main_live.py y el fondo capturado para el)
"""

import argparse
import subprocess
from datetime import datetime

FPS = 30
AWBGAINS = "3.0,1.8"

SHUTTER_US = 500
GAIN = 8.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=None,
                        help="Ancho de captura (default: resolucion nativa del sensor)")
    parser.add_argument("--height", type=int, default=None,
                        help="Alto de captura (default: resolucion nativa del sensor)")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"{FPS}fps_sh{SHUTTER_US}_g{GAIN}_{timestamp}.mp4"

    command = [
        "rpicam-vid",
        "-t", "0",
        "--framerate", str(FPS),
        "--shutter", str(SHUTTER_US),
        "--gain", str(GAIN),
        "--awb", "custom",
        "--awbgains", AWBGAINS,
        "--denoise", "cdn_off",
        "--codec", "libav",
        "--libav-format", "mp4",
        "-o", out,
    ]
    if args.width and args.height:
        command += ["--width", str(args.width), "--height", str(args.height)]

    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(
            f"rpicam-vid fallo (codigo {result.returncode}); no se guardo nada.")
    print(f"Guardado en: {out}")


if __name__ == "__main__":
    main()