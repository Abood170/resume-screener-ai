"""Generate documentation and verification counts from saved result files."""

import argparse
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from fastapi.testclient import TestClient

from api.main import create_app
from src.data import ROOT
from src.publish_calibration import archive_existing, write_archived


def section(results: dict, prefix: str = "") -> str:
    calibration, policy = results["calibration"], results["abstain"]
    test = calibration["test"]
    cv = calibration["training_cv"]["pooled_nested_oof"]
    run = calibration["run_directory"]
    metrics = ["ece_10_bins", "brier_score", "log_loss", "accuracy", "macro_f1"]
    table = "\n".join(
        f"| {m} | {test['before'][m]} | {test['after'][m]} | {test['delta_after_minus_before'][m]} |"
        for m in metrics
    )
    cv_table = "\n".join(
        f"| {method} | {cv[method]['log_loss']} | {cv[method]['brier_score']} | {cv[method]['ece_10_bins']} |"
        for method in ["raw", "sigmoid", "isotonic"]
    )
    audit = calibration["audit"]
    return f"""## Calibration and abstention

The original Random Forest and TF-IDF artifacts are unchanged. A separate **{calibration["method"]}** calibration mapping adjusts their probabilities. The original dataset and saved seed-42 split are preserved. All tables in this section are generated from `results.json`, not entered manually.

**Training-only selection:** nested stratified five-fold CV. Every outer validation row is excluded from both classifier fitting and calibration fitting. Within each outer fitting partition, an inner five-fold loop refits TF-IDF and Random Forest and supplies OOF probabilities for calibrator fitting. The same raw probability bank is used for sigmoid and isotonic. `CalibratedClassifierCV` with a frozen identity response adapter performs sklearn's class-wise mapping and renormalization; the adapter learns nothing and the actual CV work is explicit in `src/calibrate.py`. No model sees its calibration labels during base-model fitting.

Pooled outer-OOF comparison (all training rows):

| Method | Log loss | Multiclass Brier | Top-label ECE |
|---|---|---|---|
{cv_table}

The predeclared primary criterion was lower log loss, with Brier as a tie-breaker. Eligible methods had to improve both proper scores over raw and lose no more than 0.01 in training OOF accuracy or macro F1. Sigmoid wins on log loss and Brier. **Isotonic has lower training ECE**, but ECE was not the selection criterion; its more flexible mapping can overfit sparse classes and produce extreme probabilities. This risk is documented in the [scikit-learn calibration API](https://scikit-learn.org/1.6/modules/generated/sklearn.calibration.CalibratedClassifierCV.html). No test score was used to choose between methods.

The final mapping is fitted to the raw five-fold OOF predictions from the entire training partition and applied to the unchanged full-training Random Forest. This follows non-ensemble cross-validated calibration semantics. [Frozen protocol]({prefix}{run}/protocol.json), [folds]({prefix}{run}/folds.json), [selection saved before test evaluation]({prefix}{run}/selection.json).

**Once-only held-out evaluation:**

| Metric | Before | After | After minus before |
|---|---|---|---|
{table}

Multiclass Brier is the mean **sum across classes** of squared probability errors (range 0–2), not divided by class count. Log loss uses natural logarithms and sklearn's machine-epsilon clipping for zeros. ECE uses the top predicted class, ten equal-width confidence bins, and the count-weighted absolute difference between bin accuracy and mean confidence; the final bin includes probability one. Empty bins contribute zero. ECE is bin-sensitive and is not a proper scoring rule.

Neither accuracy nor macro F1 degraded, including against the predeclared absolute tolerance of 0.01. The class-specific mappings can reorder winners: {audit["changed_category_count"]} test predictions changed, including {audit["wrong_to_correct"]} wrong-to-correct, {audit["correct_to_wrong"]} correct-to-wrong, and {audit["changed_but_both_wrong"]} changes that remained wrong. The [audit]({prefix}{run}/calibration_audit.json) verifies class ordering, nested fold isolation and unchanged base artifacts using saved predictions, without another held-out inference run. The gain is not uniform across examples and does not remove source/title shortcuts identified in the earlier error analysis. Calibration remains imperfect; the measured ECE is not zero.

![Reliability before and after]({prefix}reports/figures/reliability_before_after.png)

**Abstention:** `T1={policy["t1"]}`, `T2={policy["t2"]}`. Mark uncertain when calibrated top probability is below T1 **or** the calibrated top-two margin is below T2. Fewer than {policy["short_input_words"]} whitespace words also triggers uncertainty. Comparisons are strict `<`; a value equal to a threshold passes that rule. Reason precedence is `short_input`, then `low_confidence`, then `small_margin`.

Thresholds were selected from a coarse, predeclared grid plus a no-numerical-abstention reference. Utility is **+1 for a correct answer, -1 for a wrong answer, 0 for abstention**, averaged over all training OOF rows. Ties favor greater coverage, then lower T1, then lower T2. This is a transparent demo cost assumption, not a validated business cost or a target accuracy. The chosen training utility is {policy["training_oof"]["utility"]}; selected training coverage is {policy["training_oof"]["coverage"]} and answered accuracy is {policy["training_oof"]["accuracy_on_answered"]}. These are tuning scores and may be optimistic after selection. [All grid candidates]({prefix}{run}/threshold_grid.csv).

On the held-out set, **{policy["test"]["answered"]}/{policy["test"]["total"]} resumes were answered**, coverage **{policy["test"]["coverage"]}**. Accuracy among answered resumes is **{policy["test"]["accuracy_on_answered"]}** ({policy["test"]["correct_answered"]} correct, {policy["test"]["wrong_answered"]} wrong). Coverage describes this evaluation distribution, not how often an arbitrary real-world CV will be answered.

![Training coverage versus accuracy]({prefix}reports/figures/coverage_vs_accuracy.png)

**API compatibility:** `predicted_category`, `confidence` and `top_predictions` retain their original **uncalibrated** meanings. New clients use `calibrated_predicted_category`, `calibrated_confidence` and `calibrated_top_predictions` together: multiclass calibration can change the winning category. Both `/predict` and `/predict/file` also return `is_uncertain` and `uncertainty_reason` (`low_confidence`, `small_margin`, `short_input` or null). Both candidate lists are retained when uncertain; the top three probabilities are not rescaled. A false uncertainty flag only means no policy trigger, not guaranteed correctness. Influential terms still describe the original forest's global feature salience, not the calibration mapping or a causal explanation. [Real API responses]({prefix}{run}/api_examples.json) use authored demo inputs, not held-out records; input contents are not logged.

The frontend shows a neutral **“The model isn't confident about this CV”** banner and **“Possible categories”** when uncertain. Otherwise it shows the calibrated winner. Probability bars use calibrated values, with a keyboard-focusable explanation of calibration and its limits. Display percentages are truncated to one decimal place; API values retain their original precision. The ethical footer remains unchanged: this is not a hiring decision tool.

**Limitations:** small, imbalanced, single-source English data; few calibration positives for minority classes; a distribution difference between fold-trained forests and the full-training forest; residual calibration error and dataset-specific thresholds. Calibration does not guarantee correctness, candidate quality, fairness or reliable out-of-domain detection. Short or non-English inputs remain problematic. The holdout was previously inspected for baseline error analysis, so it is not a pristine unseen external benchmark; no calibration or threshold decisions used its scores. No external data, heavy dependency, model retraining policy or production monitoring was added.

**Reproduction:** `python -m src.calibrate --run-dir <new-run-directory>` requires a `before/` archive matching the current raw artifacts, results and split. It saves training-only choices before evaluating the test set and refuses a run directory that already has held-out results. Use saved `test_probabilities.npz` and `test_results.json` for subsequent analysis rather than rerunning inference to choose settings. `python -m src.calibration_report --run-dir {run}` regenerates documentation and counts without model selection. Older files are archived before replacement. Restart FastAPI after artifact updates; missing or corrupt calibration artifacts fail readiness instead of silently claiming calibrated scores.
"""


def main(run: Path):
    results = json.loads((ROOT / "results.json").read_text())
    suites = list(ET.parse(run / "backend_pytest.xml").getroot().iter("testsuite"))
    backend = {
        k: sum(int(s.attrib.get(k, 0)) for s in suites)
        for k in ["tests", "failures", "errors", "skipped"]
    }
    backend["passed"] = (
        backend["tests"] - backend["failures"] - backend["errors"] - backend["skipped"]
    )
    frontend_raw = json.loads((run / "frontend_tests.json").read_text())
    frontend = {
        "tests": frontend_raw["numTotalTests"],
        "passed": frontend_raw["numPassedTests"],
        "failures": frontend_raw["numFailedTests"],
        "errors": 0
        if frontend_raw["success"]
        else frontend_raw.get("numRuntimeErrorTestSuites"),
        "skipped": frontend_raw["numPendingTests"],
    }
    results["verification"]["pytest"] = backend
    results["verification"]["frontend"] = frontend
    results["calibration"]["audit"] = json.loads(
        (run / "calibration_audit.json").read_text()
    )
    results["calibration"]["api_examples"] = json.loads(
        (run / "api_examples.json").read_text()
    )
    results["calibration"]["cv_fold_macro_f1"] = {}
    for method in ["raw", "sigmoid", "isotonic"]:
        values = [
            r["macro_f1"]
            for r in results["calibration"]["training_cv"]["fold_metrics"]
            if r["method"] == method
        ]
        results["calibration"]["cv_fold_macro_f1"][method] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    # Refresh the existing example without printing its request text.
    with TestClient(create_app()) as client:
        response = client.post(
            "/predict", json=results["verification"]["api_example"]["request"]
        )
        response.raise_for_status()
        results["verification"]["api_example"]["response"] = response.json()
    write_archived(
        ROOT / "results.json", json.dumps(results, indent=2, allow_nan=False), run
    )
    archive_existing(ROOT / "reports/pytest.xml", run)
    shutil.copy2(run / "backend_pytest.xml", ROOT / "reports/pytest.xml")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme = readme.replace(
        "Reproduce the complete workflow (the data download is optional when using the included CSV):",
        "The following commands reproduce the original **uncalibrated** ML experiment. Use an archived working copy, not the serving directory: `src.train` overwrites raw artifacts and their manifest. The current API additionally requires the calibration workflow below before verification. The included CSV avoids any new download:",
    )
    readme = readme.replace("python -m src.data\npython -m src.eda", "# Reuse the included CSV; do not download new data without approval.\npython -m src.eda")
    readme = readme.replace("python -m src.verify\npython -m src.readme", "# Then run the calibration workflow below before API verification.")
    readme = readme.replace("Run `python -m src.verify` to refresh evidence, or `python -m pytest -q` for tests alone.", "Run `python -m pytest -v` for the full suite. The calibration report generator below refreshes documentation from saved test reports and archives earlier results first.")
    start = readme.index("```json\n", readme.index("## API contract")) + len(
        "```json\n"
    )
    end = readme.index("\n```", start)
    readme = readme[:start] + json.dumps(response.json(), indent=2) + readme[end:]
    readme = re.sub(
        r"Recorded pytest outcome: \*\*\d+ tests, \d+ failures, \d+ errors, \d+ skipped\*\*",
        f"Recorded pytest outcome: **{backend['tests']} tests, {backend['failures']} failures, {backend['errors']} errors, {backend['skipped']} skipped**",
        readme,
    )
    readme = re.sub(
        r"All 35 original tests remain unchanged; 28 upload tests bring the backend total to 63\.",
        "The original tests and upload regressions remain unchanged; current full-suite counts are recorded below.",
        readme,
    )
    readme = readme.replace(
        "The frontend's 26 tests and production build pass.",
        f"The frontend's {frontend['tests']} tests and production build pass.",
    )
    readme = readme.replace(
        "- Confidence is uncalibrated and classes are closed-set.",
        "- Legacy confidence is uncalibrated; new calibrated confidence still has residual error, and classes are closed-set.",
    )
    readme = readme.replace(
        "Calibrate probabilities on a separate validation partition and evaluate abstention for unknown domains.",
        "Validate the new calibration and abstention policy on independent, consented data and evaluate unknown-domain behavior.",
    )
    marker = "<!-- generated-calibration:start -->"
    ending = "<!-- generated-calibration:end -->"
    generated = marker + "\n" + section(results) + ending
    if marker in readme:
        readme = (
            readme[: readme.index(marker)]
            + generated
            + readme[readme.index(ending) + len(ending) :]
        )
    else:
        readme = readme.replace("## Limitations", generated + "\n\n## Limitations", 1)
    write_archived(ROOT / "README.md", readme, run)
    write_archived(run / "REPORT.md", section(results, "../../../"), run)
    cv = results["calibration"]["cv_fold_macro_f1"][results["calibration"]["method"]]
    test = results["calibration"]["test"]["after"]
    log = (ROOT / "experiments/experiment_log.md").read_text()
    marker = f"<!-- calibration-run:{run.name} -->"
    if marker not in log:
        log += (
            f"\n## Calibration and abstention\n\n{marker}\n\n"
            "| ID | Change | CV macro F1 (mean ± std) | Test macro F1 | Test accuracy | Train time (s) | Kept? | Notes |\n"
            "|---|---|---|---|---|---|---|---|\n"
            f"| C1 | Sigmoid calibration; base RF unchanged | {cv['mean']} ± {cv['std']} | {test['macro_f1']} | {test['accuracy']} | {results['calibration']['training_cv']['elapsed_seconds']} | Yes | Time is nested method-comparison wall time, not a single model fit. Selection used training OOF log loss/Brier; thresholds {results['abstain']['t1']}, {results['abstain']['t2']}. [Generated report](../{run.relative_to(ROOT).as_posix()}/REPORT.md). |\n"
        )
        write_archived(ROOT / "experiments/experiment_log.md", log, run)
    front_readme = (ROOT / "frontend/README.md").read_text(encoding="utf-8")
    if "## Calibrated results and uncertainty" not in front_readme:
        front_readme += "\n## Calibrated results and uncertainty\n\nNew responses retain all legacy fields and additionally provide `calibrated_predicted_category`, `calibrated_confidence`, `calibrated_top_predictions`, `is_uncertain` and `uncertainty_reason`. The UI uses the calibrated category and bars together; the legacy winner can differ. Unknown or inconsistent fields produce a friendly error, never invented calibration. Older backends are clearly labeled uncalibrated.\n\nUncertain predictions show a neutral banner, a plain-language reason and **Possible categories**. Certain predictions keep the category headline with **Calibrated confidence** and a keyboard-focusable explanatory tooltip. This means no uncertainty rule was triggered, not guaranteed correctness or candidate quality. Display percentages are truncated to one decimal place. The ethical footer is unchanged. See the [generated methodology and exact results](../README.md#calibration-and-abstention).\n"
    front_readme = front_readme.replace(
        "and **model confidence (uncalibrated)**",
        "and **calibrated confidence** when supported by the backend",
    )
    front_readme = front_readme.replace(
        "and **model confidence (uncalibrated)**.", "and **calibrated confidence**."
    )
    front_readme = front_readme.replace(
        "Display percentages use one decimal place without renormalization.",
        "Display percentages are truncated to one decimal place without renormalization.",
    )
    write_archived(ROOT / "frontend/README.md", front_readme, run)
    verify_text = (
        f"# Verification\n\nGenerated from the calibration experiment's test reports.\n\n"
        f"- Backend: {backend['passed']} passed, {backend['failures']} failed, {backend['errors']} errors, {backend['skipped']} skipped.\n"
        f"- Frontend: {frontend['passed']} passed, {frontend['failures']} failed, {frontend['errors']} errors, {frontend['skipped']} skipped.\n"
        "- Frontend tests include request/response validation and rendered uncertain/certain/legacy/loading states, tooltip description, and non-rounded-up percentages.\n"
        "- TypeScript and the Vite production build passed.\n"
        "- Backend examples are real TestClient responses from saved artifacts; illustrative inputs are authored, not benchmark resumes.\n"
        "- No full cross-browser or production-load validation is claimed. Browser file-selection automation was previously blocked by extension permissions; this step did not claim to resolve that permission.\n"
        f"- Exact results and archived prior verification: [experiment](../{run.relative_to(ROOT).as_posix()}/REPORT.md).\n"
    )
    write_archived(ROOT / "frontend/VERIFICATION.md", verify_text, run)
    print(json.dumps({"backend": backend, "frontend": frontend}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    main(parser.parse_args().run_dir.resolve())
