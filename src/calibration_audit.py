"""Audit saved predictions and fold membership without rerunning the holdout."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from src.data import ROOT, sha256


def main(run: Path):
    split = json.loads((ROOT / "reports/split.json").read_text())
    manifest = json.loads((ROOT / "models/manifest.json").read_text())
    original = json.loads((run / "before/models/manifest.json").read_text())
    train = np.load(run / "training_oof.npz", allow_pickle=False)
    test = np.load(run / "test_probabilities.npz", allow_pickle=False)
    classes = np.asarray(manifest["classes"])
    calibrator = joblib.load(run / "calibrator.joblib")
    assert np.array_equal(classes, calibrator.classes_)
    assert np.array_equal(train["row_indices"], split["train_row_indices"])
    assert np.array_equal(test["row_indices"], split["test_row_indices"])
    assert set(train["row_indices"]).isdisjoint(test["row_indices"])
    assert sha256(ROOT / "models/model.joblib") == original["artifacts"]["model.joblib"]
    assert (
        sha256(ROOT / "models/vectorizer.joblib")
        == original["artifacts"]["vectorizer.joblib"]
    )
    covered = []
    for fold in json.loads((run / "folds.json").read_text()):
        fit_ids = np.asarray(fold["fit_row_indices"])
        val_ids = np.asarray(fold["validation_row_indices"])
        assert set(fit_ids).isdisjoint(val_ids)
        assert (set(fit_ids) | set(val_ids)) == set(train["row_indices"])
        row_to_label = dict(zip(train["row_indices"], train["labels"]))
        fit_labels = np.array([row_to_label[i] for i in fit_ids])
        for fitting, calibration in StratifiedKFold(
            5, shuffle=True, random_state=42
        ).split(fit_ids, fit_labels):
            assert set(fit_ids[fitting]).isdisjoint(fit_ids[calibration])
            assert set(fit_ids[fitting]).isdisjoint(val_ids)
            assert set(fit_ids[calibration]).isdisjoint(val_ids)
        covered.extend(val_ids.tolist())
    assert sorted(covered) == sorted(train["row_indices"].tolist())
    before = classes[test["before"].argmax(axis=1)]
    after = classes[test["after"].argmax(axis=1)]
    true = test["labels"]
    report = {
        "source": "Saved once-only test probabilities; no new inference or selection",
        "class_order_matches": True,
        "train_test_and_nested_folds_disjoint": True,
        "base_model_and_vectorizer_unchanged": True,
        "changed_category_count": int((before != after).sum()),
        "wrong_to_correct": int(((before != true) & (after == true)).sum()),
        "correct_to_wrong": int(((before == true) & (after != true)).sum()),
        "changed_but_both_wrong": int(
            ((before != after) & (before != true) & (after != true)).sum()
        ),
        "class_report_before": classification_report(
            true, before, labels=classes, output_dict=True, zero_division=0
        ),
        "class_report_after": classification_report(
            true, after, labels=classes, output_dict=True, zero_division=0
        ),
        "interpretation": "Class-specific OvR sigmoid mappings can reorder multiclass winners; this is not a monotonic transformation of the top score alone. The gain is accompanied by some regressions. Original title/template shortcut risks remain; no independent external validation is claimed.",
    }
    with (run / "calibration_audit.json").open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(
        json.dumps(
            {k: v for k, v in report.items() if not k.startswith("class_report")},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    main(parser.parse_args().run_dir.resolve())
