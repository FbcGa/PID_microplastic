"""Inference-time Random Forest particle classifier.

Loads the model trained by train_rf.py and classifies a particle from its
feature dict. The feature vector is rebuilt in the exact order stored with
the model, so it stays consistent even if FEATURE_NAMES changes later.
"""

from pathlib import Path

import joblib
import numpy as np

DEFAULT_MODEL_PATH = Path("rf_model.joblib")


class RandomForestParticleClassifier:
    """Wraps a trained Random Forest. `classify` takes the feature dict from
    features.extract_features and returns the predicted class string."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        bundle = joblib.load(Path(model_path))
        self._model = bundle["model"]
        self._feature_names = bundle["feature_names"]
        # Trained with n_jobs=-1 (parallelizes across trees, worth it for a
        # big training batch); at inference we predict 1-few rows at a time,
        # where spinning up the parallel backend costs far more than the
        # single-threaded prediction itself.
        self._model.n_jobs = 1

    def classify(self, features: dict[str, float]) -> str:
        vector = np.array(
            [[features[name] for name in self._feature_names]], dtype=float)
        return str(self._model.predict(vector)[0])

    def classify_batch(self, features_list: list[dict[str, float]]) -> list[str]:
        """Like classify(), but predicts all rows in a single model call —
        avoids paying the per-tree dispatch overhead once per particle."""
        if not features_list:
            return []
        matrix = np.array(
            [[features[name] for name in self._feature_names]
             for features in features_list], dtype=float)
        return [str(label) for label in self._model.predict(matrix)]

    def classify_proba(self, features: dict[str, float]) -> dict[str, float]:
        """Per-class probabilities, e.g. to flag low-confidence predictions."""
        vector = np.array(
            [[features[name] for name in self._feature_names]], dtype=float)
        probs = self._model.predict_proba(vector)[0]
        return {str(cls): float(p)
                for cls, p in zip(self._model.classes_, probs)}


def load_if_available(model_path: Path = DEFAULT_MODEL_PATH
                      ) -> RandomForestParticleClassifier | None:
    """Returns the classifier, or None if no model file exists yet."""
    model_path = Path(model_path)
    if not model_path.exists():
        return None
    return RandomForestParticleClassifier(model_path)
