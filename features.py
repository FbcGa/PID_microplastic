"""Descriptor extraction for particle classification.

Single source of truth for the feature vector: the same `extract_features`
is used both to build the training dataset (label_particles.py) and at
inference time (rf_classifier.py / detector.py), so features can never
be computed differently between training and prediction.

Pure shape descriptors: area, equivalent_diameter, hu_3/5/6/7 and the
intensity descriptors (mean_intensity, std_intensity, peak_darkness) were
dropped for near-zero importance in the trained Random Forest. Random
Forest splits on thresholds, so no scaling/normalization is needed.
"""

import cv2
import numpy as np

# Canonical order of the feature vector. Any consumer (training, inference)
# must build the vector in exactly this order.
FEATURE_NAMES: list[str] = [
    "perimeter",
    "aspect_ratio",
    "circularity",
    "solidity",
    "extent",
    "mean_width",
    "eccentricity",
    "elongation",
    "hu_1", "hu_2", "hu_4",
]


def extract_features(contour: np.ndarray) -> dict[str, float]:
    """Shape descriptors for one contour. Returns a dict keyed by FEATURE_NAMES."""
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)

    # Rotation-invariant aspect ratio: longer side / shorter side (>= 1).
    (_cx, _cy), (rw, rh), _angle = cv2.minAreaRect(contour)
    long_side, short_side = max(rw, rh), min(rw, rh)
    aspect_ratio = long_side / short_side if short_side > 0 else 0.0

    circularity = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
    mean_width = (2 * area / perimeter) if perimeter > 0 else 0.0

    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = area / hull_area if hull_area > 0 else 0.0
    extent = area / float(w * h) if w > 0 and h > 0 else 0.0

    eccentricity, elongation = _ellipse_features(contour)

    hu = _hu_moments(contour)

    return {
        "perimeter": perimeter,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "solidity": solidity,
        "extent": extent,
        "mean_width": mean_width,
        "eccentricity": eccentricity,
        "elongation": elongation,
        "hu_1": hu[0], "hu_2": hu[1], "hu_4": hu[3],
    }


def feature_vector(features: dict[str, float]) -> list[float]:
    """Flattens a feature dict into a list in FEATURE_NAMES order."""
    return [features[name] for name in FEATURE_NAMES]


def _ellipse_features(contour: np.ndarray) -> tuple[float, float]:
    """(eccentricity, elongation) from a fitted ellipse.

    fitEllipse needs >= 5 points; returns (0, 1) as a neutral fallback for
    tiny/degenerate contours (elongation 1 = not elongated)."""
    if len(contour) < 5:
        return 0.0, 1.0
    (_c, (axis_a, axis_b), _angle) = cv2.fitEllipse(contour)
    major, minor = max(axis_a, axis_b), min(axis_a, axis_b)
    if major <= 0 or minor <= 0:
        return 0.0, 1.0
    ratio = minor / major
    eccentricity = float(np.sqrt(max(0.0, 1 - ratio ** 2)))
    elongation = float(major / minor)
    return eccentricity, elongation


def _hu_moments(contour: np.ndarray) -> list[float]:
    """7 Hu moments, log-transformed to a comparable scale.

    Raw Hu moments span many orders of magnitude; the sign-preserving log
    (-sign*log10|hu|) compresses them, matching common practice."""
    hu = cv2.HuMoments(cv2.moments(contour)).flatten()
    out: list[float] = []
    for value in hu:
        if value == 0:
            out.append(0.0)
        else:
            out.append(float(-np.sign(value) * np.log10(abs(value))))
    return out
