"""Nested training-only calibration selection; one frozen held-out evaluation.

Run with --run-dir pointing to a new experiment directory containing a before/
snapshot. Training OOF caches contain probabilities and row IDs, never resumes.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.calibration import (
    fit_calibrator,
    probability_metrics,
    reliability_bins,
    selective_metrics,
)
from src.data import ROOT, sha256


def write_json(path, value):
    if path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive = path.parent / "revisions" / stamp
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, archive / path.name)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main(run: Path):
    if not (run / "before/results.json").exists():
        raise ValueError("Archive the current results/artifacts before running")
    if (run / "test_results.json").exists():
        raise ValueError("Held-out results already exist: do not evaluate twice")
    baseline = json.loads((run / "before/results.json").read_text())
    manifest = json.loads((run / "before/models/manifest.json").read_text())
    split = json.loads((ROOT / "reports/split.json").read_text())
    for filename in ["model.joblib", "vectorizer.joblib"]:
        assert sha256(ROOT / "models" / filename) == manifest["artifacts"][filename]
    assert sha256(ROOT / "reports/split.json") == sha256(
        run / "before/reports/split.json"
    )
    assert sha256(ROOT / "data/resumes.csv") == manifest["dataset_sha256"]
    protocol = {
        "seed": 42,
        "outer_cv": 5,
        "inner_cv": 5,
        "methods": ["sigmoid", "isotonic"],
        "selection": "Among methods improving pooled nested-OOF log loss AND Brier versus raw, with accuracy and macro F1 no more than 0.01 lower, minimize pooled OOF log loss; Brier breaks ties, then sigmoid. Otherwise retain raw.",
        "material_degradation_tolerance": 0.01,
        "threshold_t1_grid": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "threshold_t2_grid": [0.05, 0.1, 0.15, 0.2],
        "threshold_selection": "Maximize pooled training OOF utility: correct +1, wrong -1, abstain 0; ties higher coverage then lower T1 then lower T2. Compare no numerical abstention (0,0), retaining short-input abstention. No target accuracy or coverage.",
        "short_input_words": 50,
        "uncertainty": "short_input (<50 words) OR top_probability<T1 OR top1_minus_top2<T2; reason priority short_input, low_confidence, small_margin",
        "brier_definition": "mean(sum_class((probability - one_hot_label)^2)); range [0,2]",
        "ece_definition": "Top-label ECE, 10 equal-width bins [lower,upper), final bin includes 1; weighted absolute bin accuracy-minus-mean-confidence; empty bins contribute zero.",
        "data_hash": manifest["dataset_sha256"],
        "split_hash": sha256(ROOT / "reports/split.json"),
    }
    if (run / "protocol.json").exists():
        assert json.loads((run / "protocol.json").read_text()) == protocol
    else:
        write_json(run / "protocol.json", protocol)
    data = pd.read_csv(ROOT / "data/resumes.csv", keep_default_na=False)
    # No test features or predictions are accessed until selection.json exists.
    train = data.loc[split["train_row_indices"]]
    x, y = train.Resume.to_numpy(), train.Category.to_numpy()
    classes = manifest["classes"]
    model = joblib.load(ROOT / "models/model.joblib")
    vectorizer = joblib.load(ROOT / "models/vectorizer.joblib")
    estimator = Pipeline([("tfidf", clone(vectorizer)), ("classifier", clone(model))])
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probabilities = {
        method: np.zeros((len(train), len(classes)))
        for method in ["raw", "sigmoid", "isotonic"]
    }
    fold_rows = []
    folds = []
    started = time.perf_counter()
    for fold, (fit_idx, val_idx) in enumerate(outer.split(x, y)):
        cache = run / f"outer_fold_{fold}.npz"
        folds.append(
            {
                "fold": fold,
                "fit_row_indices": train.index[fit_idx].tolist(),
                "validation_row_indices": train.index[val_idx].tolist(),
            }
        )
        if cache.exists():
            saved = np.load(cache, allow_pickle=False)
            assert np.array_equal(saved["validation_row_indices"], train.index[val_idx])
            for method in probabilities:
                probabilities[method][val_idx] = saved[method]
            print(f"Loaded archived training fold {fold + 1}/5", flush=True)
        else:
            inner_probs = np.zeros((len(fit_idx), len(classes)))
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for inner_fold, (inner_fit, inner_val) in enumerate(
                inner.split(x[fit_idx], y[fit_idx])
            ):
                fitted = clone(estimator).fit(
                    x[fit_idx[inner_fit]], y[fit_idx[inner_fit]]
                )
                assert fitted.classes_.tolist() == classes
                inner_probs[inner_val] = fitted.predict_proba(x[fit_idx[inner_val]])
                print(
                    f"Training outer {fold + 1}/5, inner {inner_fold + 1}/5 complete",
                    flush=True,
                )
            outer_model = clone(estimator).fit(x[fit_idx], y[fit_idx])
            raw = outer_model.predict_proba(x[val_idx])
            probabilities["raw"][val_idx] = raw
            for method in ["sigmoid", "isotonic"]:
                calibrator = fit_calibrator(inner_probs, y[fit_idx], classes, method)
                probabilities[method][val_idx] = calibrator.predict_proba(raw)
            np.savez_compressed(
                cache,
                validation_row_indices=train.index[val_idx].to_numpy(),
                **{m: p[val_idx] for m, p in probabilities.items()},
            )
            print(f"Training outer fold {fold + 1}/5 complete", flush=True)
        for method, p in probabilities.items():
            fold_rows.append(
                {
                    "fold": fold,
                    "method": method,
                    **probability_metrics(y[val_idx], p[val_idx], classes),
                }
            )
    if not (run / "folds.json").exists():
        write_json(run / "folds.json", folds)
    summaries = {
        method: probability_metrics(y, p, classes)
        for method, p in probabilities.items()
    }
    raw = summaries["raw"]
    eligible = [
        m
        for m in ["sigmoid", "isotonic"]
        if summaries[m]["log_loss"] < raw["log_loss"]
        and summaries[m]["brier_score"] < raw["brier_score"]
        and summaries[m]["accuracy"] >= raw["accuracy"] - 0.01
        and summaries[m]["macro_f1"] >= raw["macro_f1"] - 0.01
    ]
    selected = (
        min(
            eligible,
            key=lambda m: (
                summaries[m]["log_loss"],
                summaries[m]["brier_score"],
                m != "sigmoid",
            ),
        )
        if eligible
        else "raw"
    )
    training_results = {
        "pooled_nested_oof": summaries,
        "fold_metrics": fold_rows,
        "selected_method": selected,
        "eligible_methods": eligible,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(run / "training_results.json", training_results)
    pd.DataFrame(fold_rows).to_csv(run / "cv_fold_metrics.csv", index=False)
    np.savez_compressed(
        run / "training_oof.npz",
        row_indices=train.index.to_numpy(),
        labels=y.astype(str),
        **probabilities,
    )
    print(json.dumps(training_results["pooled_nested_oof"], indent=2), flush=True)
    if selected == "raw":
        raise RuntimeError(
            "Neither calibration improves the declared training criteria; raw retained, test not evaluated"
        )
    word_counts = np.array([len(text.split()) for text in x])
    policy_rows = [
        selective_metrics(y, probabilities[selected], classes, word_counts, t1, t2)
        for t1 in protocol["threshold_t1_grid"]
        for t2 in protocol["threshold_t2_grid"]
    ]
    policy_rows.append(
        selective_metrics(y, probabilities[selected], classes, word_counts, 0, 0)
    )
    chosen = min(
        policy_rows,
        key=lambda row: (-row["utility"], -row["coverage"], row["t1"], row["t2"]),
    )
    selection = {
        "method": selected,
        "t1": chosen["t1"],
        "t2": chosen["t2"],
        "short_input_words": 50,
        "training_policy": chosen,
        "protocol_sha256": sha256(run / "protocol.json"),
        "selection_data": "Training partition nested OOF predictions only; tuning scores are not independent estimates after selection.",
    }
    write_json(run / "selection.json", selection)  # Frozen BEFORE any test evaluation.
    pd.DataFrame(policy_rows).to_csv(run / "threshold_grid.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    for t2 in protocol["threshold_t2_grid"]:
        rows = sorted(
            [
                r
                for r in policy_rows
                if r["t2"] == t2 and r["accuracy_on_answered"] is not None
            ],
            key=lambda r: r["coverage"],
        )
        ax.plot(
            [r["coverage"] for r in rows],
            [r["accuracy_on_answered"] for r in rows],
            "o-",
            label=f"Margin threshold {t2}",
        )
    ax.scatter(
        chosen["coverage"],
        chosen["accuracy_on_answered"],
        marker="*",
        s=260,
        color="black",
        label="Chosen on training OOF",
        zorder=5,
    )
    ax.set(
        xlabel="Coverage (answered / all training OOF resumes)",
        ylabel="Accuracy among answered",
        xlim=(0, 1.03),
        ylim=(0, 1.03),
        title="Training-only accuracy–coverage tradeoff (short-input rule included)",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(run / "coverage_vs_accuracy.png", dpi=180)
    plt.close(fig)
    # Outer raw responses are honest 5-fold OOF scores for ALL training rows.
    # Fit the final mapping on these; reuse the original full-training RF unchanged.
    final_calibrator = fit_calibrator(probabilities["raw"], y, classes, selected)
    joblib.dump(final_calibrator, run / "calibrator.joblib")
    test = data.loc[split["test_row_indices"]]
    before = model.predict_proba(vectorizer.transform(test.Resume))
    after = final_calibrator.predict_proba(before)
    before_scores = probability_metrics(test.Category, before, classes)
    assert (
        before_scores["accuracy"]
        == baseline["models"][baseline["selected_model"]]["accuracy"]
    )
    assert (
        before_scores["macro_f1"]
        == baseline["models"][baseline["selected_model"]]["macro_f1"]
    )
    after_scores = probability_metrics(test.Category, after, classes)
    delta = {key: after_scores[key] - before_scores[key] for key in before_scores}
    evaluation = {
        "before": before_scores,
        "after": after_scores,
        "delta_after_minus_before": delta,
        "material_accuracy_or_f1_degradation": delta["accuracy"] < -0.01
        or delta["macro_f1"] < -0.01,
        "policy": selective_metrics(
            test.Category,
            after,
            classes,
            test.Resume.map(lambda t: len(t.split())),
            chosen["t1"],
            chosen["t2"],
        ),
        "reliability_bins": {
            "before": reliability_bins(test.Category, before, classes),
            "after": reliability_bins(test.Category, after, classes),
        },
        "selection_sha256_before_test": sha256(run / "selection.json"),
        "held_out_evaluations": 1,
    }
    write_json(run / "test_results.json", evaluation)
    np.savez_compressed(
        run / "test_probabilities.npz",
        row_indices=test.index.to_numpy(),
        labels=test.Category.to_numpy().astype(str),
        before=before,
        after=after,
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1]}
    )
    for phase, color in [("before", "#64748b"), ("after", "#087f5b")]:
        bins = evaluation["reliability_bins"][phase]
        occupied = [b for b in bins if b["count"]]
        axes[0].plot(
            [b["mean_confidence"] for b in occupied],
            [b["accuracy"] for b in occupied],
            "o-",
            color=color,
            label=phase,
        )
        offset = -0.018 if phase == "before" else 0.018
        axes[1].bar(
            [(b["lower"] + b["upper"]) / 2 + offset for b in bins],
            [b["count"] for b in bins],
            width=0.036,
            color=color,
            label=phase,
        )
    axes[0].plot([0, 1], [0, 1], "--", color="black", alpha=0.5)
    axes[0].set(
        xlabel="Mean top-label confidence",
        ylabel="Observed top-label accuracy",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Held-out reliability (10 equal-width bins)",
    )
    axes[1].set(
        xlabel="Top-label confidence bin",
        ylabel="Test resumes",
        title="Bin support; empty bins omitted from curve",
    )
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(run / "reliability_before_after.png", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {
                "selection": selection,
                "held_out": {
                    k: v for k, v in evaluation.items() if k != "reliability_bins"
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.run_dir.resolve())
