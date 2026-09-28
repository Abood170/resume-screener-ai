"""Held-out metrics and confusion matrices for fitted estimators."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def evaluate(model, texts, labels, classes: list[str], output: Path, name: str) -> dict:
    predicted = model.predict(texts)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="macro", zero_division=0
    )
    matrix = confusion_matrix(labels, predicted, labels=classes)
    fig, ax = plt.subplots(figsize=(15, 13))
    sns.heatmap(
        matrix,
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
        annot=True,
        fmt="d",
        ax=ax,
    )
    ax.set(
        xlabel="Predicted", ylabel="Actual", title=f"Held-out confusion matrix: {name}"
    )
    fig.tight_layout()
    fig.savefig(output / f"confusion_matrix_{name}.png", dpi=160)
    plt.close(fig)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": matrix.tolist(),
    }


def write_results(results: dict, path: Path) -> None:
    path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
