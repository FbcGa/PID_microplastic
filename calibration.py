"""Persistence for live-capture calibration: crop top/bottom strips.

The reference background photo (fondo_live.jpg) lives alongside
calibration.json but is read/written directly by processing.py, since it's
a plain image file, not JSON state.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from cropping import EdgeCropPx

CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration.json"
BACKGROUND_PATH = Path(__file__).resolve().parent / "fondo_live.jpg"


@dataclass
class Calibration:
    crop_top: int = int(EdgeCropPx.TOP)
    crop_bottom: int = int(EdgeCropPx.BOTTOM)


def load_calibration(path: Path = CALIBRATION_PATH) -> Calibration:
    """Returns saved calibration, or defaults if missing/corrupt."""
    try:
        data = json.loads(path.read_text())
        return Calibration(crop_top=int(data["crop_top"]),
                           crop_bottom=int(data["crop_bottom"]))
    except (OSError, ValueError, KeyError):
        return Calibration()


def save_calibration(calibration: Calibration, path: Path = CALIBRATION_PATH) -> None:
    path.write_text(json.dumps(asdict(calibration)))
