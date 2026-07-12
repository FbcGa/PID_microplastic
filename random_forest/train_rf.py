"""Train and evaluate the Random Forest particle classifier.

Reads the labeled dataset.csv (produced by label_particles.py), trains a
RandomForestClassifier, reports cross-validated performance, a confusion
matrix and feature importances, and saves the model to rf_model.joblib.

Random Forest splits on thresholds, so features are used as-is (no scaling).
class_weight="balanced" compensates the likely fiber/amorphous imbalance.

Usage:
    uv run random_forest/train_rf.py                       (dataset.csv -> rf_model.joblib)
    uv run random_forest/train_rf.py --dataset otro.csv --out modelo.joblib --test-size 0.25
"""

import argparse
import csv
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import FEATURE_NAMES

THIS_DIR = Path(__file__).resolve().parent


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads (X, y, groups) from dataset.csv, X columns in FEATURE_NAMES
    order. groups = source_frame, so CV can split by frame instead of by
    particle (particles from the same frame are correlated, not independent
    samples)."""
    features, labels, groups = [], [], []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            features.append([float(row[name]) for name in FEATURE_NAMES])
            labels.append(row["label"])
            groups.append(row["source_frame"])
    if not features:
        raise ValueError(f"Dataset vacio: {csv_path}")
    return np.asarray(features, dtype=float), np.asarray(labels), np.asarray(groups)


def build_model(random_state: int = 0) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def report_class_balance(y: np.ndarray) -> None:
    classes, counts = np.unique(y, return_counts=True)
    print("Distribucion de clases:")
    for cls, n in zip(classes, counts):
        print(f"  {cls:<12} {n}")


def cross_validate(model: RandomForestClassifier, X: np.ndarray, y: np.ndarray,
                   groups: np.ndarray) -> None:
    """Grouped k-fold CV: folds split by source_frame, so no particle from a
    frame in the training fold also appears in the validation fold. Without
    this, a fold can score high just by memorizing a specific frame's
    background/lighting quirks rather than generalizing to new frames."""
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(5, n_groups))
    if n_groups < 2:
        print("CV omitida: el dataset tiene un solo frame de origen.")
        return
    cv = GroupKFold(n_splits=n_splits)
    scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring="f1_macro")
    print(f"F1-macro CV agrupada por frame ({n_splits}-fold): "
          f"{scores.mean():.3f} +/- {scores.std():.3f}")


def show_importances(model: RandomForestClassifier) -> None:
    order = np.argsort(model.feature_importances_)[::-1]
    print("Importancia de features (mayor a menor):")
    for i in order:
        print(f"  {FEATURE_NAMES[i]:<20} {model.feature_importances_[i]:.4f}")


def train(csv_path: Path, model_path: Path, test_size: float) -> None:
    X, y, groups = load_dataset(csv_path)
    print(f"Dataset: {len(y)} muestras, {X.shape[1]} features")
    report_class_balance(y)

    # Un clasificador supervisado necesita >= 2 clases. Con una sola clase
    # el modelo no aprende nada (accuracy 1.0 trivial, importancias en 0):
    # se aborta con un mensaje claro en vez de guardar un modelo inutil.
    n_classes = len(np.unique(y))
    if n_classes < 2:
        print(f"\nERROR: el dataset tiene una sola clase ('{y[0]}'). "
              "Random Forest necesita ejemplos de AMBAS clases (fibra y "
              "amorfa) para aprender a distinguirlas.")
        print("Etiqueta mas particulas de la otra clase con "
              "label_particles.py y vuelve a entrenar. No se guardo modelo.")
        return

    model = build_model()
    cross_validate(model, X, y, groups)

    # Held-out split for a confusion matrix (stratified when possible).
    min_class = int(np.unique(y, return_counts=True)[1].min())
    if min_class >= 2 and len(y) >= 5:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=0)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        labels = sorted(np.unique(y))
        print("\nMatriz de confusion (filas=real, cols=predicho):")
        print("labels:", labels)
        print(confusion_matrix(y_te, y_pred, labels=labels))
        print("\nReporte por clase:")
        print(classification_report(y_te, y_pred, zero_division=0))
    else:
        print("\nPocas muestras para un split held-out; se omite la matriz.")

    # Final model trained on ALL data before saving.
    model.fit(X, y)
    show_importances(model)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, model_path)
    print(f"\nModelo guardado en: {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena el Random Forest fibra vs amorfa.")
    parser.add_argument("--dataset", type=Path, default=THIS_DIR / "dataset.csv")
    parser.add_argument("--out", type=Path, default=THIS_DIR / "rf_model.joblib")
    parser.add_argument("--test-size", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args.dataset, args.out, args.test_size)


if __name__ == "__main__":
    main()
