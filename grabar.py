import argparse
import subprocess
from datetime import datetime

FPS = 30
AWBGAINS = "3.0,1.8"
SHUTTER = 10000  # microsegundos (10000us = 1/100s). None = automatico
GAIN = 1.0       # ganancia analoga (IMX477: util hasta ~4-8x). None = automatico


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graba video con rpicam-vid.")
    parser.add_argument("--shutter", type=int, default=SHUTTER,
                        help=f"Tiempo de obturador en microsegundos (default: {SHUTTER})")
    parser.add_argument("--gain", type=float, default=GAIN,
                        help=f"Ganancia analoga (default: {GAIN})")
    return parser.parse_args()


def main():
    args = parse_args()
    out = datetime.now().strftime("grabacion_%Y%m%d_%H%M%S.mp4")
    cmd = [
        "rpicam-vid",
        "-t", "0",
        "--framerate", str(FPS),
        "--awb", "custom",
        "--awbgains", AWBGAINS,
        "--codec", "libav",
        "--libav-format", "mp4",
        "-o", out,
    ]
    if args.shutter is not None:
        cmd += ["--shutter", str(args.shutter)]
    if args.gain is not None:
        cmd += ["--gain", str(args.gain)]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(
            f"rpicam-vid fallo (codigo {result.returncode}); no se guardo nada.")
    print(f"Guardado en: {out}")


if __name__ == "__main__":
    main()
