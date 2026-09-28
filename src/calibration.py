"""Probability-only calibration and uncertainty; never logs input documents."""

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, f1_score, log_loss

Reason = Literal["short_input", "low_confidence", "small_margin"] | None


def validate_probabilities(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2 or not np.isfinite(p).all():
        raise ValueError("Invalid probability matrix")
    if (
        np.any(p < 0)
        or np.any(p > 1)
        or not np.allclose(p.sum(axis=1), 1, atol=1e-9, rtol=0)
    ):
        raise ValueError("Probabilities must be normalized")
    return p


class ProbabilityAdapter(ClassifierMixin, BaseEstimator):
    """Identity adapter over REAL RF responses, not a surrogate text model.

    This permits reusing exactly the same fold-local RF probabilities for both
    sklearn calibration methods. It learns no features or labels. The caller
    must supply independent out-of-fold probabilities to fit_calibrator.
    """

    def __init__(self, classes: tuple[str, ...]):
        self.classes = classes

    def fit(self, X, y=None):
        p = validate_probabilities(X)
        if p.shape[1] != len(self.classes):
            raise ValueError("Class order mismatch")
        self.classes_ = np.asarray(self.classes)
        self.n_features_in_ = len(self.classes)
        return self

    def predict_proba(self, X):
        p = validate_probabilities(X)
        if p.shape[1] != self.n_features_in_:
            raise ValueError("Class count mismatch")
        return p

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def fit_calibrator(probabilities, labels, classes, method):
    """Fit sklearn OvR mappings to supplied TRAIN-only OOF RF responses.

    CV is performed explicitly by the experiment runner, with fold-local TF-IDF
    and nested evaluation. FrozenEstimator freezes only this identity adapter;
    it does NOT fit a model on calibration labels. sklearn performs class-wise
    sigmoid/isotonic fitting and multiclass renormalization.
    """
    adapter = ProbabilityAdapter(tuple(classes)).fit(probabilities)
    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(adapter), method=method, ensemble=False
    )
    return calibrated.fit(probabilities, labels)


def uncertainty(probabilities, word_count: int, t1: float, t2: float) -> Reason:
    p = validate_probabilities(np.asarray(probabilities).reshape(1, -1))[0]
    if not 0 <= t1 <= 1 or not 0 <= t2 <= 1 or word_count < 0:
        raise ValueError("Invalid uncertainty policy")
    ordered = np.sort(p)
    # Priority is deliberate and documented. Both numerical rules use strict <.
    if word_count < 50:
        return "short_input"
    if ordered[-1] < t1:
        return "low_confidence"
    if ordered[-1] - ordered[-2] < t2:
        return "small_margin"
    return None


def answered_mask(probabilities, word_counts, t1, t2):
    p = validate_probabilities(probabilities)
    ordered = np.sort(p, axis=1)
    return (
        (np.asarray(word_counts) >= 50)
        & (ordered[:, -1] >= t1)
        & (ordered[:, -1] - ordered[:, -2] >= t2)
    )


def reliability_bins(labels, probabilities, classes):
    p = validate_probabilities(probabilities)
    confidence = p.max(axis=1)
    correct = np.asarray(classes)[p.argmax(axis=1)] == np.asarray(labels)
    indices = np.minimum((confidence * 10).astype(int), 9)
    rows = []
    for index in range(10):
        mask = indices == index
        rows.append(
            {
                "bin": index,
                "lower": index / 10,
                "upper": (index + 1) / 10,
                "count": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean())
                if mask.any()
                else None,
                "accuracy": float(correct[mask].mean()) if mask.any() else None,
            }
        )
    return rows


def probability_metrics(labels, probabilities, classes):
    p = validate_probabilities(probabilities)
    y = np.asarray(labels)
    predicted = np.asarray(classes)[p.argmax(axis=1)]
    one_hot = y[:, None] == np.asarray(classes)[None, :]
    bins = reliability_bins(y, p, classes)
    return {
        "log_loss": float(log_loss(y, p, labels=classes)),
        # Multiclass Brier = mean over rows of SUM across classes; range [0,2].
        "brier_score": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
        "ece_10_bins": float(
            sum(
                b["count"] / len(y) * abs(b["accuracy"] - b["mean_confidence"])
                for b in bins
                if b["count"]
            )
        ),
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(
            f1_score(y, predicted, labels=classes, average="macro", zero_division=0)
        ),
    }


def selective_metrics(labels, probabilities, classes, word_counts, t1, t2):
    mask = answered_mask(probabilities, word_counts, t1, t2)
    correct = np.asarray(classes)[np.argmax(probabilities, axis=1)] == np.asarray(
        labels
    )
    count = int(mask.sum())
    return {
        "t1": float(t1),
        "t2": float(t2),
        "answered": count,
        "total": len(mask),
        "coverage": float(mask.mean()),
        "accuracy_on_answered": float(correct[mask].mean()) if count else None,
        "correct_answered": int((correct & mask).sum()),
        "wrong_answered": int((~correct & mask).sum()),
        "utility": float(np.where(mask, np.where(correct, 1, -1), 0).mean()),
    }
