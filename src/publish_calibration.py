"""Publish a frozen calibration experiment without re-evaluating the holdout."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from src.data import ROOT, sha256


def archive_existing(path: Path, run: Path) -> None:
    if path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = run / "revisions" / stamp / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def write_archived(path: Path, value: str, run: Path) -> None:
    archive_existing(path, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def publish(run: Path) -> None:
    if (run / "published.json").exists():
        raise ValueError("This experiment has already been published")
    selection = json.loads((run / "selection.json").read_text())
    training = json.loads((run / "training_results.json").read_text())
    test = json.loads((run / "test_results.json").read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    assert test["selection_sha256_before_test"] == sha256(run / "selection.json")
    assert selection["protocol_sha256"] == sha256(run / "protocol.json")
    original = json.loads((run / "before/results.json").read_text())
    assert sha256(ROOT / "results.json") == sha256(run / "before/results.json")
    manifest = json.loads((run / "before/models/manifest.json").read_text())
    for name in ["model.joblib", "vectorizer.joblib"]:
        assert sha256(ROOT / "models" / name) == manifest["artifacts"][name]
    assert sha256(ROOT / "reports/split.json") == protocol["split_hash"]
    relative_run = run.relative_to(ROOT).as_posix()
    calibration = {
        "method": selection["method"],
        "run_directory": relative_run,
        "artifact": "calibrator.joblib",
        "protocol": protocol,
        "training_cv": training,
        "test": test,
        "legacy_contract": "predicted_category, confidence and top_predictions retain raw RF semantics; calibrated_predicted_category, calibrated_confidence and calibrated_top_predictions form a separate coherent view.",
        "limitations": "Small, single-source English dataset. Isotonic can overfit with few per-class positives. Calibration may change class ranking and does not detect every out-of-domain input. Holdout previously inspected during baseline error analysis; no calibration decisions use its scores.",
    }
    abstain = {
        "t1": selection["t1"],
        "t2": selection["t2"],
        "short_input_words": 50,
        "rule": protocol["uncertainty"],
        "selection": protocol["threshold_selection"],
        "training_oof": selection["training_policy"],
        "test": test["policy"],
        "threshold_grid": f"{relative_run}/threshold_grid.csv",
        "limitations": "Utility is an explicit demo assumption, not a validated operational cost. Thresholds are dataset-specific; tuned training coverage/accuracy are optimistic after selection.",
    }
    original["calibration"] = calibration
    original["abstain"] = abstain
    manifest["artifacts"]["calibrator.joblib"] = sha256(run / "calibrator.joblib")
    manifest["calibration"] = {
        "artifact": "calibrator.joblib",
        "method": selection["method"],
        "run_directory": relative_run,
        "protocol_sha256": selection["protocol_sha256"],
    }
    manifest["abstain"] = {
        "t1": selection["t1"],
        "t2": selection["t2"],
        "short_input_words": 50,
    }
    original["artifacts"] = manifest
    for source, destination in [
        (run / "calibrator.joblib", ROOT / "models/calibrator.joblib"),
        (
            run / "coverage_vs_accuracy.png",
            ROOT / "reports/figures/coverage_vs_accuracy.png",
        ),
        (
            run / "reliability_before_after.png",
            ROOT / "reports/figures/reliability_before_after.png",
        ),
    ]:
        archive_existing(destination, run)
        shutil.copy2(source, destination)
    write_archived(ROOT / "models/manifest.json", json.dumps(manifest, indent=2), run)
    write_archived(
        ROOT / "results.json", json.dumps(original, indent=2, allow_nan=False), run
    )
    (run / "published.json").write_text(
        json.dumps(
            {
                "results_sha256": sha256(ROOT / "results.json"),
                "manifest_sha256": sha256(ROOT / "models/manifest.json"),
            },
            indent=2,
        )
    )
    print(
        "Published calibrated mapping and policy; original RF/vectorizer/split unchanged."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    publish(args.run_dir.resolve())
