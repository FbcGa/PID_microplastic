import subprocess
from datetime import datetime

FPS = 30
AWBGAINS = "3.0,1.8"

SHUTTER_US = 500   
GAIN = 8.0         


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"{FPS}fps_sh{SHUTTER_US}_g{GAIN}_{timestamp}.mp4"

    result = subprocess.run([
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
    ])
    if result.returncode != 0:
        raise SystemExit(
            f"rpicam-vid fallo (codigo {result.returncode}); no se guardo nada.")
    print(f"Guardado en: {out}")


if __name__ == "__main__":
    main()