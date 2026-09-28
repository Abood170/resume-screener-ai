"""Descriptive baseline audit; never fits, selects, or writes model artifacts.

Raw resume text is used only in memory. Outputs contain aggregate statistics and
dataset row identifiers, never excerpts, names, contact details or free text.
Reruns archive earlier reports and experiment notes before replacing anything.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.data import ROOT, sha256
from src.predict import Predictor
from src.preprocess import clean_text


# Explicit, conservative lexical variants, not a learned dictionary. A match is
# an occurrence anywhere in the document, not proof of a candidate's job title.
VARIANTS = {
    "ACCOUNTANT": ["accountant", "accountants", "accounting"],
    "ADVOCATE": ["advocate", "advocates", "advocacy"],
    "AGRICULTURE": ["agriculture", "agricultural"],
    "APPAREL": ["apparel"],
    "ARTS": ["arts", "artist", "artists", "art"],
    "AUTOMOBILE": ["automobile", "automobiles", "automotive"],
    "AVIATION": ["aviation"],
    "BANKING": ["banking", "banker", "bankers"],
    "BPO": ["bpo", "business process outsourcing"],
    "BUSINESS-DEVELOPMENT": ["business development"],
    "CHEF": ["chef", "chefs"],
    "CONSTRUCTION": ["construction"],
    "CONSULTANT": ["consultant", "consultants", "consulting", "consultancy"],
    "DESIGNER": ["designer", "designers", "design"],
    "DIGITAL-MEDIA": ["digital media"],
    "ENGINEERING": ["engineering", "engineer", "engineers"],
    "FINANCE": ["finance", "financial"],
    "FITNESS": ["fitness"],
    "HEALTHCARE": ["healthcare", "health care"],
    "HR": ["hr", "human resources", "human resource"],
    "INFORMATION-TECHNOLOGY": ["information technology"],
    "PUBLIC-RELATIONS": ["public relations"],
    "SALES": ["sales", "salesperson"],
    "TEACHER": ["teacher", "teachers", "teaching"],
}

# Fixed domain vocabulary permits aggregate inspection without publishing any
# resume fragments. These indicators are descriptive, not annotated causes.
DOMAINS = {
    "customer_operations": [
        "customer service",
        "call center",
        "call centre",
        "outsourcing",
        "operations",
        "process",
    ],
    "sales_business": [
        "sales",
        "marketing",
        "business development",
        "account management",
        "retail",
    ],
    "technology": [
        "software",
        "programming",
        "database",
        "network",
        "technical support",
        "information technology",
    ],
    "creative_design": ["design", "designer", "artist", "art", "graphic", "creative"],
    "education": [
        "teacher",
        "teaching",
        "student",
        "students",
        "curriculum",
        "classroom",
    ],
    "finance_accounting": ["financial", "accounting", "banking", "budget", "audit"],
    "engineering_automotive": [
        "engineer",
        "engineering",
        "automotive",
        "automobile",
        "mechanical",
        "manufacturing",
    ],
    "healthcare": ["patient", "clinical", "medical", "healthcare", "nursing"],
}

HEADINGS = [
    "summary",
    "experience",
    "education",
    "skills",
    "employment",
    "qualifications",
]


def normalized(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.casefold()).strip()


def contains_phrase(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def label_present(text: str, category: str) -> bool:
    return any(contains_phrase(text, term) for term in VARIANTS[category])


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def archive_previous(output: Path, stamp: str) -> None:
    """Copy a prior run in full before writing any replacement report."""
    if output.exists():
        destination = ROOT / "reports/error_analysis_archive" / stamp
        shutil.copytree(output, destination)
    output.mkdir(parents=True, exist_ok=True)


def render_summary(output: Path) -> None:
    """Render every reported number from generated files, never manual metrics."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    r = json.loads((output / "analysis_results.json").read_text())
    # CSV dtype=str preserves the exact serialized decimals, without a second
    # binary floating-point parse/format cycle changing their last digit.
    pairs = pd.read_csv(output / "top_confusions.csv", dtype=str)
    gaps = r["cv_vs_test"]
    labels = r["label_shortcuts"]
    near, rest = r["near_duplicates"]
    weakest = "\n".join(
        f"| {x['category']} | {x['train_support']} | {x['support']} | {x['precision']} | {x['recall']} | {x['f1']} |"
        for x in r["weakest_classes"]
    )
    pair_lines = "\n".join(
        f"| {x.true_category} -> {x.predicted_category} | {x['count']} | {x.true_support} | {x.fraction_of_true_class} |"
        for _, x in pairs.head(3).iterrows()
    )
    tied = pairs[pairs["count"].eq(pairs.iloc[2]["count"])]
    tie_text = "; ".join(
        f"{x.true_category} -> {x.predicted_category}" for _, x in tied.iterrows()
    )
    reviews = []
    for x in r["review"]:
        domains = ", ".join(
            f"{k.replace('_', ' ')} {v}/{x['reviewed_count']}"
            for k, v in x["domain_presence_counts"].items()
            if v
        )
        destinations = ", ".join(
            f"{k}: {v}" for k, v in sorted(x["predicted_counts"].items())
        )
        reviews.append(
            f"- **{x['category']}**: inspected {x['reviewed_count']} of {x['total_errors']} errors "
            f"({x['test_support']} test resumes); word counts {x['word_count_min']}–{x['word_count_max']}. "
            f"Own-label variants occur in {x['own_label_present_count']}/{x['reviewed_count']}; "
            f"predicted-label variants in {x['predicted_label_present_count']}/{x['reviewed_count']}. "
            f"Domain indicators: {domains}. Predictions: {destinations}. "
            f"Under-50-word inputs: {x['short_count']}; fewer than two recognized section markers: {x['fewer_than_two_headings_count']}."
        )
    verification_path = output / "verification.json"
    verification = None
    if (output / "pytest.xml").exists() and (output / "frontend_tests.json").exists():
        suites = list(ET.parse(output / "pytest.xml").getroot().iter("testsuite"))
        counts = {
            k: sum(int(s.attrib.get(k, 0)) for s in suites)
            for k in ["tests", "failures", "errors", "skipped"]
        }
        counts["passed"] = (
            counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
        )
        front = json.loads((output / "frontend_tests.json").read_text())
        verification = {
            "backend": counts,
            "frontend": {
                "tests": front["numTotalTests"],
                "passed": front["numPassedTests"],
                "failures": front["numFailedTests"],
                "skipped": front["numPendingTests"],
                "errors": 0
                if front["success"]
                else front.get("numRuntimeErrorTestSuites"),
                "runner_success": front["success"],
            },
        }
        if verification_path.exists():
            destination = ROOT / "reports/error_analysis_archive" / stamp
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(verification_path, destination / verification_path.name)
        save_json(verification_path, verification)
    verification_text = (
        "Full-suite verification has not yet been recorded for this report."
    )
    if verification is not None:
        b, f = verification["backend"], verification["frontend"]
        verification_text = (
            f"Backend: {b['tests']} tests, {b['passed']} passed, {b['failures']} failed, {b['errors']} errors, {b['skipped']} skipped. "
            f"Frontend: {f['tests']} tests, {f['passed']} passed, {f['failures']} failed, {f['errors']} errors, {f['skipped']} skipped; runner success: {f['runner_success']}. "
            "See [verification.json](verification.json), [pytest.xml](pytest.xml) and [frontend_tests.json](frontend_tests.json)."
        )
    summary = f"""# Baseline error analysis

Analysis only: the saved Random Forest, preprocessing, API and split are unchanged. The prior freeze stopped because this is not a Git repository; no `baseline-v1` tag or frozen archive exists. This audit identifies the current artifacts by hashes in [analysis_results.json](analysis_results.json).

## Scope and reproduced baseline

The existing split contains {r["train_rows"]} training and {r["test_rows"]} test resumes. Its confusion matrix reproduces the recorded baseline exactly. Test accuracy: {r["test_accuracy"]}; macro F1: {gaps["test_macro_f1"]}. Training CV macro F1: {gaps["cv_macro_f1_mean"]} ± {gaps["cv_macro_f1_std"]}. Test minus CV: {gaps["test_minus_cv_macro_f1"]}. This is a descriptive gap, not evidence of significance or improvement; CV models also use smaller fitting partitions. No model was fitted or selected. Source: [cv_vs_test_gap.csv](cv_vs_test_gap.csv).

These held-out diagnostics are final descriptive reporting. Recommendations below are hypotheses to investigate on training-only out-of-fold predictions, not settings chosen from test performance. Repeatedly consulting this holdout would compromise its independence for future improvement claims.

## Weakest classes

| Category | Train support | Test support | Precision | Recall | F1 |
|---|---|---|---|---|---|
{weakest}

Training support and class F1 have Pearson correlation {r["support_f1_pearson"]} and Spearman correlation {r["support_f1_spearman"]}. Small classes are among the weakest, but ARTS shows that support alone is insufficient. These are descriptive associations across categories, not causal effects. Sources: [per_class_report.csv](per_class_report.csv), [support plot](class_support_vs_f1.png).

## Most common confusions

| True -> predicted | Count | True-class support | Fraction of true class |
|---|---|---|---|
{pair_lines}

Third place is tied at {pairs.iloc[2]["count"]}: {tie_text}. Ties are ordered alphabetically, not selectively. The requested top-pair file has a fixed cutoff; additional tied pairs at its boundary can be omitted. Sources: [top_confusions.csv](top_confusions.csv), [normalized matrix](confusion_matrix_normalized.png). Plot annotations are truncated to two decimals; [matrix CSV](confusion_matrix_normalized.csv) preserves the underlying values.

## Leakage and shortcuts

- **Label wording:** own-label variants appear in {labels["own_variant_count"]}/{labels["total_raw_rows"]} raw resumes (share {labels["own_variant_share"]}); {labels["categories_at_least_half_own_variant"]}/{labels["category_count"]} categories have shares at least one half. In the first eighty words, matches occur in {labels["own_variant_first_80_words_count"]}/{labels["total_raw_rows"]} (share {labels["own_variant_first_80_words_share"]}). This benchmark partly permits job-title lookup. These frequencies do not establish how much the model relies on it, and a label mention can describe education or past work rather than the target occupation.
- **Literal-match caveat:** [label_shortcuts.csv](label_shortcuts.csv) reports every category separately for raw, training and test data. Exact case-insensitive substring matching can count `arts` inside `parts`, or miss space-separated versions of hyphenated labels. The separate boundary-aware variant measure normalizes punctuation/whitespace and uses the explicit [variant dictionary](label_variants.json). Variants can still be broad; these are lexical indicators, not human annotations of job titles.
- **Near duplicates:** using the saved training-fitted TF-IDF space, {near["count"]}/{r["test_rows"]} test resumes have nearest-training cosine similarity strictly above 0.9. Accuracy is {near["accuracy"]} ({near["correct"]}/{near["count"]}) versus {rest["accuracy"]} ({rest["correct"]}/{rest["count"]}) for the remainder. All {near["same_label_as_nearest_train"]} high-similarity matches share the training neighbor's category. The higher accuracy is descriptive and the group is small; similarity in a reduced vocabulary is not proof of a duplicate or a causal estimate of score inflation. No rows were removed. Source: [near_duplicate_summary.csv](near_duplicate_summary.csv), [nearest neighbors](nearest_train_similarity.csv).
- **Exact overlap:** {r["exact_cleaned_test_train_overlap"]} test resumes share an exact preprocessing-normalized text hash with training. This does not exclude template, author or source overlap. The overall macro F1 did not cross the requested suspicious-gain threshold, and no improvement is claimed.

## Structured review of misclassified examples

Whole documents were read in memory by fixed lexical/structure checks; only sanitized counts and dataset row IDs are retained. This is a structured evidence review, not a human semantic annotation or causal attribution. Up to five errors per class were sampled without replacement using seed 42. BPO has fewer than five test examples, so all available errors were inspected. Sources and indicator definitions: [misclassification_review.json](misclassification_review.json).

{chr(10).join(reviews)}

Multiple domain signals and predicted-label wording support overlapping vocabulary as a plausible source of confusion. The inspected errors provide no evidence that very short text is the explanation. Section markers do not show obvious missing structure, but cannot rule out unusual formatting or distinguish real employment from education/history mentions. No resume excerpts or personal data are published.

## Next investigations — training data only

1. **Compare unweighted and class-weighted baselines within training CV**, keeping the existing test split untouched. The low-support BPO/AUTOMOBILE failures motivate the comparison; ARTS prevents assuming imbalance explains every weakness. Inspect per-class out-of-fold recall as well as macro F1. [Evidence](per_class_report.csv)
2. **Run a predeclared title/label-masking diagnostic inside training CV.** Compare with unchanged text to measure dependence on explicit title wording. Do not adopt masking from this held-out analysis or claim it must improve accuracy. [Evidence](label_shortcuts.csv)
3. **Audit training-only near-duplicate/template clusters and use grouped training CV as a sensitivity check.** Keep the held-out split fixed and report the different CV protocol separately. Do not remove difficult or similar test rows. [Evidence](near_duplicate_summary.csv)
4. **Review training labels and an annotation rubric for overlapping occupations**, especially ARTS/TEACHER and FINANCE/ACCOUNTANT, before changing label definitions. Inspect analogous training examples rather than relabeling test errors. [Evidence](top_confusions.csv)
5. **Stratify training out-of-fold errors by label presence, domain indicators and length before changing text cleaning.** The sampled errors contain substantial text and mixed-domain wording, so a short-input or formatting fix is not supported by this review. Any feature/preprocessing experiment must earn adoption through training CV. [Evidence](misclassification_review.json)

No fixes, new dependencies or external downloads were made. Any later data acquisition still requires approval.

## Verification and reproducibility

{verification_text}

Run `python -m src.error_analysis` to regenerate the analysis. Existing report directories are copied to timestamped `reports/error_analysis_archive/` before replacement, and an existing experiment log is archived before notes are appended. `python -m src.error_analysis --render-only` regenerates this summary from saved results. Original `results.json`, artifacts and split are never written. All numeric findings above are rendered from generated result files.
"""
    summary_path = output / "SUMMARY.md"
    if summary_path.exists():
        destination = ROOT / "reports/error_analysis_archive" / stamp
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary_path, destination / summary_path.name)
    summary_path.write_text(summary, encoding="utf-8")
    log = ROOT / "experiments/experiment_log.md"
    log.parent.mkdir(exist_ok=True)
    existing = (
        log.read_text(encoding="utf-8")
        if log.exists()
        else "# Experiment log\n\nThe requested baseline freeze was not completed because this is not a Git repository. No baseline model row is fabricated.\n"
    )
    marker = f"### Error analysis {r['created_utc']}"
    if marker not in existing:
        if log.exists():
            destination = ROOT / "experiments/archive" / stamp
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log, destination / log.name)
        notes = (
            f"\n## Error analysis\n\n{marker}\n\n"
            "| Note | Finding |\n|---|---|\n"
            f"| Scope | Descriptive audit of current saved artifacts; no model fitted, selected or changed. |\n"
            f"| Weakest classes | {', '.join(x['category'] for x in r['weakest_classes'])} |\n"
            f"| Label shortcuts | Own-label variants in {labels['own_variant_count']}/{labels['total_raw_rows']}; share {labels['own_variant_share']}. |\n"
            f"| Near duplicates | {near['count']} test rows above 0.9 cosine; accuracy {near['accuracy']} versus {rest['accuracy']} for the rest. |\n"
            "| Report | [Generated summary](../reports/error_analysis/SUMMARY.md); all values come from its generated result files. This is a notes group, not a model-comparison row. |\n"
            "| Selection guardrail | Future selection uses training-only CV; keep the existing held-out split unchanged. |\n"
        )
        log.write_text(existing + notes, encoding="utf-8")


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "reports/error_analysis"
    archive_previous(output, stamp)
    original_results = ROOT / "results.json"
    results = json.loads(original_results.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "models/manifest.json").read_text())
    provenance = json.loads((ROOT / "data/provenance.json").read_text())
    split = json.loads((ROOT / "reports/split.json").read_text())
    dataset_hash = sha256(ROOT / "data/resumes.csv")
    assert dataset_hash == provenance["csv_sha256"] == manifest["dataset_sha256"]
    train_ids, test_ids = split["train_row_indices"], split["test_row_indices"]
    assert len(set(train_ids)) == len(train_ids)
    assert len(set(test_ids)) == len(test_ids)
    assert not set(train_ids).intersection(test_ids)
    data = pd.read_csv(ROOT / "data/resumes.csv", keep_default_na=False)
    train, test = data.loc[train_ids], data.loc[test_ids]
    predictor = Predictor()
    classes = predictor.model.classes_.tolist()
    assert set(classes) == set(VARIANTS)
    assert classes == results["classes"]
    assert predictor.model_name == results["selected_model"]
    assert len(train) == results["methodology"]["train_rows"]
    assert len(test) == results["methodology"]["test_rows"]
    # Transform with the already-fitted TRAIN-only vectorizer. No fit calls.
    x_train = predictor.vectorizer.transform(train.Resume)
    x_test = predictor.vectorizer.transform(test.Resume)
    predicted = predictor.model.predict(x_test)
    true = test.Category.to_numpy()
    correct = predicted == true
    report = classification_report(
        true, predicted, labels=classes, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(true, predicted, labels=classes)
    recorded = results["models"][predictor.model_name]
    assert matrix.tolist() == recorded["confusion_matrix"], (
        "Baseline predictions changed"
    )
    assert float(accuracy_score(true, predicted)) == recorded["accuracy"]
    assert np.isclose(
        report["macro avg"]["f1-score"], recorded["macro_f1"], rtol=0, atol=1e-15
    )
    train_counts = train.Category.value_counts()
    per_class = pd.DataFrame(
        [
            {
                "category": c,
                "precision": report[c]["precision"],
                "recall": report[c]["recall"],
                "f1": report[c]["f1-score"],
                "support": int(report[c]["support"]),
                "train_support": int(train_counts[c]),
            }
            for c in classes
        ]
    ).sort_values(["f1", "category"], kind="stable")
    per_class.to_csv(output / "per_class_report.csv", index=False)
    confusions = sorted(
        [
            {
                "true_category": a,
                "predicted_category": b,
                "count": int(matrix[i, j]),
                "true_support": int(matrix[i].sum()),
                "fraction_of_true_class": float(matrix[i, j] / matrix[i].sum()),
            }
            for i, a in enumerate(classes)
            for j, b in enumerate(classes)
            if i != j and matrix[i, j] > 0
        ],
        key=lambda r: (-r["count"], r["true_category"], r["predicted_category"]),
    )
    pd.DataFrame(confusions[:15]).to_csv(output / "top_confusions.csv", index=False)
    normalized_matrix = matrix / matrix.sum(axis=1, keepdims=True)
    pd.DataFrame(normalized_matrix, index=classes, columns=classes).to_csv(
        output / "confusion_matrix_normalized.csv", index_label="true_category"
    )
    fig, ax = plt.subplots(figsize=(20, 17))
    sns.heatmap(
        normalized_matrix,
        ax=ax,
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
        vmin=0,
        vmax=1,
        annot=np.floor(normalized_matrix * 100) / 100,
        fmt=".2f",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "Fraction of actual class"},
    )
    ax.set(
        xlabel="Predicted category",
        ylabel="Actual category",
        title="Baseline held-out confusion matrix (row-normalized)",
    )
    ax.tick_params(axis="both", labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix_normalized.png", dpi=180)
    plt.close(fig)
    # Label gutters avoid overlap in the dense high-support cluster, using only
    # matplotlib. Connecting lines identify the actual point coordinates.
    fig, ax = plt.subplots(figsize=(19, 12))
    ax.scatter(per_class.train_support, per_class.f1, s=65, color="#2563eb")
    ordered = per_class.sort_values(["train_support", "f1", "category"])
    left, right = ordered.iloc[: len(ordered) // 2], ordered.iloc[len(ordered) // 2 :]
    for group, x_pos, alignment in [(left, 0.01, "left"), (right, 0.99, "right")]:
        for (_, row), y_pos in zip(
            group.sort_values("f1").iterrows(), np.linspace(0.06, 0.96, len(group))
        ):
            ax.annotate(
                row.category,
                (row.train_support, row.f1),
                xytext=(x_pos, y_pos),
                textcoords="axes fraction",
                ha=alignment,
                va="center",
                fontsize=10,
                arrowprops={"arrowstyle": "-", "color": "#94a3b8", "lw": 0.7},
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
            )
    ax.set(
        xlim=(-15, 135),
        ylim=(-0.06, 1.08),
        xlabel="Training resumes per class",
        ylabel="Held-out class F1",
        title="Training support versus baseline held-out F1 (descriptive association)",
    )
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output / "class_support_vs_f1.png", dpi=160)
    plt.close(fig)
    gap = float(report["macro avg"]["f1-score"] - recorded["cv_macro_f1_mean"])
    gap_row = {
        "cv_macro_f1_mean": recorded["cv_macro_f1_mean"],
        "cv_macro_f1_std": recorded["cv_macro_f1_std"],
        "test_macro_f1": report["macro avg"]["f1-score"],
        "test_minus_cv_macro_f1": gap,
        "interpretation": f"Test macro F1 exceeds mean CV macro F1 by {gap!r}; different data and fit sizes, not evidence of significance or improvement.",
    }
    pd.DataFrame([gap_row]).to_csv(output / "cv_vs_test_gap.csv", index=False)
    label_rows = []
    for scope, frame in [("all_raw", data), ("train", train), ("test", test)]:
        for category in classes:
            texts = frame.loc[frame.Category.eq(category), "Resume"]
            canonical = sum(category.casefold() in text.casefold() for text in texts)
            variants = sum(label_present(normalized(text), category) for text in texts)
            leading = sum(
                label_present(normalized(" ".join(text.split()[:80])), category)
                for text in texts
            )
            label_rows.append(
                {
                    "scope": scope,
                    "category": category,
                    "support": len(texts),
                    "literal_substring_count": canonical,
                    "literal_substring_share": canonical / len(texts),
                    "boundary_variant_count": variants,
                    "boundary_variant_share": variants / len(texts),
                    "first_80_words_variant_count": leading,
                    "first_80_words_variant_share": leading / len(texts),
                }
            )
    pd.DataFrame(label_rows).to_csv(output / "label_shortcuts.csv", index=False)
    save_json(output / "label_variants.json", VARIANTS)
    nearest = []
    for offset in range(0, len(test), 64):
        similarities = cosine_similarity(x_test[offset : offset + 64], x_train)
        for within, values in enumerate(similarities):
            position = offset + within
            index = int(np.argmax(values))
            nearest.append(
                {
                    "test_row_index": int(test_ids[position]),
                    "nearest_train_row_index": int(train_ids[index]),
                    "cosine_similarity": float(values[index]),
                    "above_0_9": bool(values[index] > 0.9),
                    "true_category": str(true[position]),
                    "predicted_category": str(predicted[position]),
                    "nearest_train_category": str(train.iloc[index].Category),
                    "correct": bool(correct[position]),
                }
            )
    pd.DataFrame(nearest).to_csv(output / "nearest_train_similarity.csv", index=False)
    similarity_groups = []
    for flag in [True, False]:
        group = [r for r in nearest if r["above_0_9"] == flag]
        similarity_groups.append(
            {
                "group": "cosine > 0.9" if flag else "cosine <= 0.9",
                "count": len(group),
                "correct": sum(r["correct"] for r in group),
                "accuracy": sum(r["correct"] for r in group) / len(group)
                if group
                else None,
                "same_label_as_nearest_train": sum(
                    r["true_category"] == r["nearest_train_category"] for r in group
                ),
            }
        )
    pd.DataFrame(similarity_groups).to_csv(
        output / "near_duplicate_summary.csv", index=False
    )
    # Hashes only: the normalized strings never leave memory.
    clean_train = {
        hashlib.sha256(clean_text(t).encode()).hexdigest() for t in train.Resume
    }
    exact_overlap = sum(
        hashlib.sha256(clean_text(t).encode()).hexdigest() in clean_train
        for t in test.Resume
    )
    weak = per_class.head(3).category.tolist()
    inspections = []
    for category in weak:
        errors = np.flatnonzero((true == category) & ~correct)
        # Fixed random sampling, independent of scores or individual content.
        chosen = np.random.default_rng(42).choice(
            errors, size=min(5, len(errors)), replace=False
        )
        profiles = []
        for position in sorted(chosen, key=lambda p: test_ids[p]):
            text = test.iloc[position].Resume
            norm = normalized(text)
            domain_counts = {
                domain: sum(contains_phrase(norm, term) for term in terms)
                for domain, terms in DOMAINS.items()
            }
            profiles.append(
                {
                    "row_index": int(test_ids[position]),
                    "predicted_category": str(predicted[position]),
                    "word_count": len(text.split()),
                    "short_under_50": len(text.split()) < 50,
                    "own_label_variant_present": label_present(norm, category),
                    "predicted_label_variant_present": label_present(
                        norm, str(predicted[position])
                    ),
                    "recognized_section_heading_count": sum(
                        contains_phrase(norm, heading) for heading in HEADINGS
                    ),
                    "domain_distinct_term_counts": domain_counts,
                }
            )
        inspections.append(
            {
                "category": category,
                "test_support": int(report[category]["support"]),
                "total_errors": len(errors),
                "reviewed_count": len(profiles),
                "profiles": profiles,
                "predicted_counts": dict(
                    Counter(p["predicted_category"] for p in profiles)
                ),
                "word_count_min": min(p["word_count"] for p in profiles),
                "word_count_max": max(p["word_count"] for p in profiles),
                "short_count": sum(p["short_under_50"] for p in profiles),
                "own_label_present_count": sum(
                    p["own_label_variant_present"] for p in profiles
                ),
                "predicted_label_present_count": sum(
                    p["predicted_label_variant_present"] for p in profiles
                ),
                "fewer_than_two_headings_count": sum(
                    p["recognized_section_heading_count"] < 2 for p in profiles
                ),
                "domain_presence_counts": {
                    d: sum(p["domain_distinct_term_counts"][d] > 0 for p in profiles)
                    for d in DOMAINS
                },
            }
        )
    save_json(
        output / "misclassification_review.json",
        {
            "sampling": "Up to five errors per weakest class, numpy default_rng seed 42, no replacement; all errors when fewer available.",
            "method": "In-memory whole-document structured reading using fixed lexical domains, label variants, length and section markers. No raw text or personal data emitted; indicators are not human-annotated causal explanations.",
            "domains": DOMAINS,
            "headings": HEADINGS,
            "classes": inspections,
        },
    )
    raw_rows = [row for row in label_rows if row["scope"] == "all_raw"]
    summary = {
        "created_utc": stamp,
        "python": platform.python_version(),
        "baseline_status": "Current saved artifacts; baseline-v1 tag/archive was not created because the directory is not a Git repository.",
        "model": predictor.model_name,
        "dataset_sha256": dataset_hash,
        "source_hashes": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [
                original_results,
                ROOT / "reports/split.json",
                ROOT / "models/model.joblib",
                ROOT / "models/vectorizer.joblib",
                ROOT / "models/manifest.json",
            ]
        },
        "train_rows": len(train),
        "test_rows": len(test),
        "test_accuracy": float(correct.mean()),
        "cv_vs_test": gap_row,
        "weakest_classes": per_class.head(3).to_dict("records"),
        "top_confusions": confusions[:3],
        "support_f1_pearson": float(
            np.corrcoef(per_class.train_support, per_class.f1)[0, 1]
        ),
        "support_f1_spearman": float(
            per_class.train_support.corr(per_class.f1, method="spearman")
        ),
        "label_shortcuts": {
            "total_raw_rows": len(data),
            "own_variant_count": sum(row["boundary_variant_count"] for row in raw_rows),
            "own_variant_share": sum(row["boundary_variant_count"] for row in raw_rows)
            / len(data),
            "own_variant_first_80_words_count": sum(
                row["first_80_words_variant_count"] for row in raw_rows
            ),
            "own_variant_first_80_words_share": sum(
                row["first_80_words_variant_count"] for row in raw_rows
            )
            / len(data),
            "categories_at_least_half_own_variant": sum(
                row["boundary_variant_share"] >= 0.5 for row in raw_rows
            ),
            "category_count": len(classes),
        },
        "near_duplicates": similarity_groups,
        "exact_cleaned_test_train_overlap": exact_overlap,
        "review": [
            {k: v for k, v in review.items() if k != "profiles"}
            for review in inspections
        ],
        "leakage_stop_triggered": report["macro avg"]["f1-score"] > 0.85,
        "no_refit_or_selection": True,
    }
    save_json(output / "analysis_results.json", summary)
    render_summary(output)
    print(json.dumps(summary, indent=2))  # Aggregate results only, never resumes.


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Render prose from existing generated analysis files",
    )
    args = parser.parse_args()
    if args.render_only:
        render_summary(ROOT / "reports/error_analysis")
    else:
        main()
