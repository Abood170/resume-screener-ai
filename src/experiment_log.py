"""Append reproducible experiment rows without changing existing evidence.

Pass a full results.json dictionary. Active calibration metrics take precedence
when present; otherwise use the selected base model. No evaluation is performed.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

LOG_PATH = Path(__file__).resolve().parents[1] / "experiments" / "experiment_log.md"
HEADER = "| ID | Change | CV macro F1 (mean ± std) | Test macro F1 | Test accuracy | Train time (s) | Kept? | Notes |"
SEPARATOR = "|---|---|---|---|---|---|---|---|"


def metrics_from_results(results: Mapping[str, Any]) -> dict[str, Any]:
    """Extract recorded metrics, including the meaning of the timing field."""
    model = results["selected_model"]
    base = results["models"][model]
    calibration = results.get("calibration")
    if calibration and calibration["method"] in ("sigmoid", "isotonic"):
        method = calibration["method"]
        cv = calibration["cv_fold_macro_f1"][method]
        test = calibration["test"]["after"]
        return dict(model=f"{model} + {method}", mean=cv["mean"], std=cv["std"],
                    f1=test["macro_f1"], accuracy=test["accuracy"],
                    seconds=calibration["training_cv"]["elapsed_seconds"],
                    timing=calibration["training_cv"].get(
                        "timing_description",
                        "Time is nested calibration method-comparison wall time, not a single fit."))
    return dict(model=model, mean=base["cv_macro_f1_mean"], std=base["cv_macro_f1_std"],
                f1=base["macro_f1"], accuracy=base["accuracy"],
                seconds=base["fit_seconds"], timing=base.get(
                    "timing_description", "Time is selected base-model fit time."))


def _cell(value: object) -> str:
    return str(value).replace("|", "&#124;").replace("\r", " ").replace("\n", " ")


def append_row(id: str, change: str, results_dict: Mapping[str, Any],
               kept: bool, notes: str) -> str:
    """Append to the first experiment table; reject duplicate IDs.

    Existing bytes are archived before editing. Numeric values use Python's
    round-trip representation, without display rounding. Returns the written row.
    """
    from datetime import datetime, timezone
    import shutil

    if not isinstance(id, str) or not id.strip():
        raise ValueError("Experiment ID must be a nonempty string")
    m = metrics_from_results(results_dict)
    cells = [id, change, f"{m['mean']} ± {m['std']}", m["f1"], m["accuracy"],
             m["seconds"], "Yes" if kept else "No", f"{notes} {m['timing']}"]
    row = "| " + " | ".join(_cell(v) for v in cells) + " |"
    previous = LOG_PATH.read_text(encoding="utf-8") if LOG_PATH.exists() else ""
    lines = previous.splitlines(keepends=True)
    if any(line.startswith(f"| {_cell(id)} |") for line in lines):
        raise ValueError(f"Experiment ID already exists: {id}")
    if previous and HEADER not in previous:
        raise ValueError("Existing log has no compatible table header")
    if previous:
        archive = LOG_PATH.parent / "archive" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive.mkdir(parents=True, exist_ok=False)
        shutil.copy2(LOG_PATH, archive / LOG_PATH.name)
        start = next(i for i, line in enumerate(lines) if line.rstrip() == HEADER)
        end = start + 2
        while end < len(lines) and lines[end].startswith("|"):
            end += 1
        lines.insert(end, row + "\n")
        content = "".join(lines)
    else:
        content = f"# Experiment log\n\n{HEADER}\n{SEPARATOR}\n{row}\n"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(content, encoding="utf-8", newline="\n")
    return row
