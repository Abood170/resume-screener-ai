"""Label-aware diagnostic ablation; never alters production artifacts.

Run: python -m src.leakage_impact
Own-label masking requires ground truth and is NOT deployable preprocessing.
The fixed calibrated baseline protocol is repeated, without method selection.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.calibration import fit_calibrator
from src.data import ROOT, sha256
from src.error_analysis import normalized, label_present, VARIANTS
from src.experiment_log import append_row


def mask_own_label(text: str, category: str) -> str:
    """Remove boundary-aware variants with exactly the audit's normalization.

    Map casefolded characters back to original offsets, so punctuation/case in
    nonmatching content is preserved. Remove all overlapping matches at once.
    Broad terms (e.g. financial/design) include legitimate occupational content;
    this ablation cannot isolate spurious titles from useful semantic evidence.
    """
    folded, offsets = [], []
    for index, char in enumerate(text):
        for lower in char.casefold():
            folded.append(lower)
            offsets.append(index)
    tokens = list(re.finditer(r"[^\W_]+", "".join(folded)))
    view, mapping = [], []
    for token in tokens:
        if view:
            view.append(" ")
            mapping.append(offsets[token.start()])
        view.extend(token.group())
        mapping.extend(offsets[token.start():token.end()])
    view = "".join(view)
    assert view == normalized(text)
    removed = np.zeros(len(text), dtype=bool)
    for variant in VARIANTS[category]:
        pattern = r"(?<!\S)" + re.escape(variant) + r"(?!\S)"
        for match in re.finditer(pattern, view):
            removed[mapping[match.start()]:mapping[match.end()-1]+1] = True
    return "".join(" " if removed[i] else char for i, char in enumerate(text))


def save(path: Path, obj: object) -> None:
    # Exclusive creation protects all previous diagnostic evidence.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)


def scores(y, prediction, classes) -> dict:
    report = classification_report(y, prediction, labels=classes, output_dict=True,
                                   zero_division=0)
    return {"macro_f1": f1_score(y, prediction, average="macro"),
            "accuracy": accuracy_score(y, prediction), "per_class": report}


def main() -> None:
    baseline = ROOT / "experiments/baseline_v1"
    out = ROOT / "experiments/leakage_impact"
    out.mkdir(exist_ok=False)  # A rerun must use a new directory, never overwrite.
    frozen = json.loads((baseline / "results.json").read_text())
    checks = json.loads((baseline / "checksums.json").read_text())["sha256"]
    for name, digest in checks.items():
        assert sha256(baseline / name) == digest
    assert sha256(ROOT / "reports/split.json") == checks["reports/split.json"]
    assert sha256(ROOT / "data/resumes.csv") == frozen["dataset"]["csv_sha256"]
    variants = json.loads((ROOT / "reports/error_analysis/label_variants.json").read_text())
    assert variants == VARIANTS
    protected = {str(p.relative_to(ROOT)): sha256(p) for folder in
                 ["models", "api", "tests", "experiments/baseline_v1"]
                 for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in str(p)}
    for name in ["results.json", "reports/split.json", "src/train.py", "src/preprocess.py", "src/predict.py"]:
        protected[name] = sha256(ROOT / name)
    split = json.loads((baseline / "reports/split.json").read_text())
    assert not set(split["train_row_indices"]) & set(split["test_row_indices"])
    data = pd.read_csv(ROOT / "data/resumes.csv", keep_default_na=False)
    train = data.loc[split["train_row_indices"]]
    x = np.array([mask_own_label(t, c) for t, c in zip(train.Resume, train.Category)])
    y = train.Category.to_numpy()
    classes = frozen["classes"]
    method = frozen["calibration"]["method"]
    model = joblib.load(baseline / "models/model.joblib")
    vectorizer = joblib.load(baseline / "models/vectorizer.joblib")
    calibrator = joblib.load(baseline / "models/calibrator.joblib")
    estimator = Pipeline([("tfidf", clone(vectorizer)), ("classifier", clone(model))])
    protocol = {"purpose": "Diagnostic only, not model selection", "seed": 42,
                "outer_folds": 5, "inner_folds": 5, "calibration_method_fixed": method,
                "mask": "Ground-truth own-category variants removed before vectorization in train and test",
                "variants": variants, "baseline_checksums": checks,
                "production_checksums_before": protected,
                "hyperparameters": {k: str(v) for k, v in estimator.get_params().items()},
                "policy": "No thresholds tuned; all-row classification before abstention. No permanent adoption decision."}
    save(out / "protocol.json", protocol)
    raw_oof = np.zeros((len(y), len(classes)))
    calibrated_oof = np.zeros_like(raw_oof)
    fold_metrics, folds = [], []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    started = time.perf_counter()
    for fold, (fit, val) in enumerate(cv.split(x, y)):
        inner_oof = np.zeros((len(fit), len(classes)))
        for inner, (a, b) in enumerate(cv.split(x[fit], y[fit])):
            fitted = clone(estimator).fit(x[fit[a]], y[fit[a]])
            assert fitted.classes_.tolist() == classes
            inner_oof[b] = fitted.predict_proba(x[fit[b]])
            print(f"Outer {fold+1}/5 inner {inner+1}/5 complete", flush=True)
        outer_model = clone(estimator).fit(x[fit], y[fit])
        raw_oof[val] = outer_model.predict_proba(x[val])
        fold_calibrator = fit_calibrator(inner_oof, y[fit], classes, method)
        calibrated_oof[val] = fold_calibrator.predict_proba(raw_oof[val])
        fold_metrics.append(float(f1_score(y[val], np.array(classes)[calibrated_oof[val].argmax(1)], average="macro")))
        folds.append({"fold": fold, "fit_rows": train.index[fit].tolist(), "validation_rows": train.index[val].tolist()})
        np.savez_compressed(out / f"fold_{fold}.npz", row_indices=train.index[val], raw=raw_oof[val], calibrated=calibrated_oof[val])
    cv_seconds = time.perf_counter() - started
    save(out / "folds.json", folds)
    start_fit = time.perf_counter()
    final = clone(estimator).fit(x, y)
    final_calibrator = fit_calibrator(raw_oof, y, classes, method)
    fit_seconds = time.perf_counter() - start_fit
    joblib.dump(final, out / "masked_pipeline.joblib")
    joblib.dump(final_calibrator, out / "masked_calibrator.joblib")
    # All training is finished before accessing held-out texts/predictions.
    save(out / "training_results.json", {"cv_folds": fold_metrics, "cv_mean": float(np.mean(fold_metrics)),
         "cv_std": float(np.std(fold_metrics)), "cv_seconds": cv_seconds, "fit_seconds": fit_seconds})
    test = data.loc[split["test_row_indices"]]
    masked_test = [mask_own_label(t, c) for t, c in zip(test.Resume, test.Category)]
    baseline_p = calibrator.predict_proba(model.predict_proba(vectorizer.transform(test.Resume)))
    masked_p = final_calibrator.predict_proba(final.predict_proba(masked_test))
    # Supplemental distribution-shift diagnostic, distinct from retrained ablation.
    frozen_masked_p = calibrator.predict_proba(model.predict_proba(vectorizer.transform(masked_test)))
    labels = np.array(classes)
    before = labels[baseline_p.argmax(1)]
    after = labels[masked_p.argmax(1)]
    evaluation_only = labels[frozen_masked_p.argmax(1)]
    baseline_scores = scores(test.Category, before, classes)
    masked_scores = scores(test.Category, after, classes)
    assert baseline_scores["macro_f1"] == frozen["calibration"]["test"]["after"]["macro_f1"]
    assert baseline_scores["accuracy"] == frozen["calibration"]["test"]["after"]["accuracy"]
    available = np.array([label_present(normalized(t), c) for t,c in zip(test.Resume, test.Category)])
    raw_count = sum(label_present(normalized(t), c) for t,c in zip(data.Resume, data.Category))
    assert all(not label_present(normalized(t), c) for t,c in zip(masked_test, test.Category))
    correct = before == test.Category.to_numpy()
    changed_wrong = correct & (after != test.Category.to_numpy())
    fixed_wrong = correct & (evaluation_only != test.Category.to_numpy())
    rows = [{"category": c, "support": baseline_scores["per_class"][c]["support"],
             "baseline_f1": baseline_scores["per_class"][c]["f1-score"],
             "masked_f1": masked_scores["per_class"][c]["f1-score"],
             "delta": masked_scores["per_class"][c]["f1-score"]-baseline_scores["per_class"][c]["f1-score"]} for c in classes]
    rows.sort(key=lambda row: (row["delta"], row["category"]))
    result = {"diagnostic_only": True, "baseline": {**baseline_scores,
              "cv_mean": frozen["calibration"]["cv_fold_macro_f1"][method]["mean"],
              "cv_std": frozen["calibration"]["cv_fold_macro_f1"][method]["std"]},
              "masked": {**masked_scores, "cv_mean": float(np.mean(fold_metrics)), "cv_std": float(np.std(fold_metrics)), "cv_folds": fold_metrics},
              "delta": {k: masked_scores[k]-baseline_scores[k] for k in ["macro_f1", "accuracy"]},
              "per_class_deltas": rows,
              "fractions": {"baseline_correct": int(correct.sum()), "correct_to_incorrect": int(changed_wrong.sum()),
                  "correct_to_incorrect_fraction": float(changed_wrong.sum()/correct.sum()),
                  "incorrect_to_correct": int(((~correct)&(after==test.Category.to_numpy())).sum()),
                  "test_shortcut_count": int(available.sum()), "test_count": len(test),
                  "test_shortcut_fraction": float(available.mean()), "raw_shortcut_count": raw_count,
                  "raw_count": len(data), "raw_shortcut_fraction": raw_count/len(data)},
              "evaluation_only_frozen_model": {**scores(test.Category, evaluation_only, classes),
                  "correct_to_incorrect": int(fixed_wrong.sum()), "correct_to_incorrect_fraction": float(fixed_wrong.sum()/correct.sum())},
              "timing": {"cv_seconds": cv_seconds, "fit_seconds": fit_seconds},
              "production_unchanged": all(sha256(ROOT / p)==digest for p,digest in protected.items())}
    assert result["production_unchanged"]
    np.savez_compressed(out / "test_predictions.npz", row_indices=test.index, labels=test.Category.to_numpy().astype(str), baseline=baseline_p, masked=masked_p, evaluation_only=frozen_masked_p)
    save(out / "results.json", result)
    pd.DataFrame(rows).to_csv(out / "per_class_deltas.csv", index=False)
    render(out)
    log_result = {"selected_model": "masked_random_forest_sigmoid", "models": {
        "masked_random_forest_sigmoid": {"cv_macro_f1_mean": result["masked"]["cv_mean"], "cv_macro_f1_std": result["masked"]["cv_std"],
             "macro_f1": masked_scores["macro_f1"], "accuracy": masked_scores["accuracy"], "fit_seconds": fit_seconds}}}
    save(out / "log_metrics.json", log_result)
    append_row("L1", "Own-label masking diagnostic; RF + fixed sigmoid", log_result, False,
               "Measurement only, not model selection or adoption. Production unchanged. Time includes full-training RF and calibration fit. See leakage_impact/SUMMARY.md.")
    print("Diagnostic results saved; production hashes unchanged.", flush=True)


def render(out: Path) -> None:
    r = json.loads((out / "results.json").read_text())
    lines = ["# Label-shortcut impact diagnostic", "", "All numbers below come from [results.json](results.json).", "",
             "| System | CV macro F1 (mean ± std) | Test macro F1 | Test accuracy |", "|---|---|---|---|"]
    for key, name in [("baseline", "Frozen baseline v1 (RF + sigmoid)"), ("masked", "Own-label masked train and test (RF + sigmoid)")]:
        m = r[key]
        lines.append(f"| {name} | {m['cv_mean']} ± {m['cv_std']} | {m['macro_f1']} | {m['accuracy']} |")
    f = r["fractions"]
    lines += ["", f"Masked minus baseline: test macro F1 {r['delta']['macro_f1']}; accuracy {r['delta']['accuracy']}.", "",
        f"Of {f['baseline_correct']} baseline-correct test predictions, {f['correct_to_incorrect']} became incorrect after masked retraining (fraction {f['correct_to_incorrect_fraction']}); {f['incorrect_to_correct']} previously incorrect predictions became correct.",
        f"Own-label variants were available in {f['test_shortcut_count']}/{f['test_count']} test resumes (fraction {f['test_shortcut_fraction']}) versus {f['raw_shortcut_count']}/{f['raw_count']} raw resumes (fraction {f['raw_shortcut_fraction']}). The test subset and raw corpus have different denominators; the saved split excludes cleaned empty/duplicate rows and samples within categories.", "",
        "| Category (largest drop first) | Support | Baseline F1 | Masked F1 | Delta |", "|---|---|---|---|---|"]
    for row in r["per_class_deltas"]:
        lines.append(f"| {row['category']} | {row['support']} | {row['baseline_f1']} | {row['masked_f1']} | {row['delta']} |")
    e = r["evaluation_only_frozen_model"]
    lines += ["", "## Interpretation and limits", "",
        "Verdict: substantial dependence on own-label vocabulary, with residual predictive signal after removal. The evidence does not justify calling the model either pure label lookup or mostly genuine content understanding.",
        f"Removing the recorded own-label variants changes test macro F1 from {r['baseline']['macro_f1']} to {r['masked']['macro_f1']}. The remaining accuracy is {r['masked']['accuracy']}. This measures sensitivity to the listed vocabulary, not a causal partition into genuine content versus lookup.",
        "A binary 'mostly genuine' versus 'mostly label lookup' conclusion is not identified by this experiment alone. The category-specific deletion uses ground truth, can create label-dependent missing-word patterns, and removes useful domain content as well as titles (e.g. financial, design, accounting). Other label variants, templates and near-duplicates remain. Consequently this is neither a clean content-only benchmark nor an honest performance ceiling.",
        f"The three largest F1 decreases are {', '.join(row['category']+' ('+str(row['delta'])+')' for row in r['per_class_deltas'][:3])}. These are the most sensitive classes in this diagnostic, not proof that their original predictions only read labels.",
        f"Supplemental evaluation-only masking of the frozen model gives macro F1 {e['macro_f1']} and accuracy {e['accuracy']}; {e['correct_to_incorrect']}/{f['baseline_correct']} original correct answers become wrong (fraction {e['correct_to_incorrect_fraction']}). This includes distribution shift and must not be confused with the retrained comparison above.",
        "The same frozen TF-IDF/RF hyperparameters, seed, split, outer CV and nested calibration folds were used; sigmoid was fixed from baseline, not selected anew. TF-IDF was fitted independently inside every training fold. All-row metrics are before abstention, with no threshold tuning. Baseline scores were reproduced from frozen artifacts. No production artifacts, preprocessing, API, or existing tests were changed; no permanent fix is recommended or selected here."]
    (out / "SUMMARY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
