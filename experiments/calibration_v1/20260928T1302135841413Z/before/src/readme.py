"""Generate documentation from actual recorded results, never typed-in scores."""

import json
from src.data import ROOT


def render(r: dict) -> str:
    d, c, m = r["dataset"], r["cleaning"], r["methodology"]
    selected = r["selected_model"]
    rows = []
    for name, scores in r["models"].items():
        rows.append(
            "| "
            + " | ".join(
                [name]
                + [
                    str(scores[k])
                    for k in [
                        "cv_macro_f1_mean",
                        "cv_macro_f1_std",
                        "accuracy",
                        "macro_precision",
                        "macro_recall",
                        "macro_f1",
                    ]
                ]
            )
            + " |"
        )
    table = "\n".join(rows)
    timing = "\n".join(
        f"| {name} | {s['cv_seconds']} | {s['fit_seconds']} |"
        for name, s in r["models"].items()
    )
    rationale = {
        "logistic_regression": "Logistic Regression combines the best observed CV macro F1 with class-specific linear coefficients that can be inspected for influential terms. Its sparse linear form is a practical fit for TF-IDF. Coefficients describe associations, not causal explanations.",
        "multinomial_nb": "Multinomial Naive Bayes achieved the best observed CV macro F1. Its simple class-conditional term statistics are inspectable and its fit is inexpensive; conditional independence is a strong assumption and probability calibration remains unverified.",
        "random_forest": "Random Forest achieved the best observed CV macro F1. It can capture nonlinear feature interactions, but trees are harder to interpret as class-specific term evidence and cost more than a sparse linear model. The measured CV advantage is the reason to accept that complexity here, not evidence that forests are universally superior for text.",
    }[selected]
    example = r["verification"]["api_example"]
    recalls = r["models"][selected]["per_class_recall"]
    lowest = min(recalls, key=recalls.get)
    return f"""# Resume Screener AI

A locally trained resume **job-category classifier** with a FastAPI REST service. It demonstrates an auditable ML workflow for a junior engineering portfolio: source provenance, duplicate handling, leakage-safe model comparison, persisted artifacts, and tested inference. It organizes resume text; it does not assess candidate merit, rank people, or automate hiring decisions.

## Dataset and disclosure

Real public [Kaggle Resume Dataset]({d["source"]}) by Snehaan Bhawal, downloaded directly from Kaggle. **No synthetic training data was used.** The first attempted `gauravduttakiit/resume-dataset` endpoint returned HTTP Forbidden; the successful dataset uses broader occupational labels rather than the narrower developer categories in that variant.

The archive contains website-derived resume examples associated with LiveCareer. Source schema `Resume_str` is renamed to `Resume`; only `Resume` and `Category` are retained in `data/resumes.csv`. Raw size: **{d["raw_rows"]} resumes, {d["raw_categories"]} categories**. These are source-provided labels, not independently verified annotations. See `data/provenance.json` for URLs, archive member and hashes. CSV SHA-256: `{d["csv_sha256"]}`.

The dataset may contain personal information. Public availability does not imply consent for unrelated reuse; avoid logging resume bodies, review upstream terms before redistribution, and do not commit private resumes. The project makes no independent licensing or anonymization guarantee about source records.

## EDA

The executed [notebook](notebooks/eda.ipynb) and `src/eda.py` produce category counts, whitespace word-length statistics, and frequent cleaned terms per category. Raw mean word length is **{r["eda"]["word_length_statistics"]["mean"]}**; median is **{r["eda"]["word_length_statistics"]["50%"]}**. Counts and terms are recorded in `results.json` under `eda`.

![Category distribution](reports/figures/category_distribution.png)

![Resume lengths](reports/figures/text_lengths.png)

[Frequent terms by category](reports/figures/frequent_terms.png)

## Methodology and leakage controls

- Normalize case, remove URLs and nonalphabetic characters, tokenize, remove NLTK English stopwords, and noun-lemmatize with WordNet. This deterministic function is reused by the saved vectorizer at inference.
- Remove {c["empty_rows_removed"]} empty cleaned rows, {c["conflicting_rows_removed"]} rows from {c["conflicting_text_groups_removed"]} conflicting-label text groups, and {c["duplicate_rows_removed"]} additional identical cleaned-text rows. This leaves **{c["retained_rows"]} unique cleaned resumes**. Exact cleaned duplicates cannot cross the split. Near-duplicate templates and shared source authors are not resolved.
- Stratified train/test split: test fraction **{m["test_fraction"]}**, seed **{m["seed"]}**, yielding **{m["train_rows"]} training** and **{m["test_rows"]} test** rows. Original CSV indices are saved in `reports/split.json`; category support is in `results.json`.
- TF-IDF: `max_features={m["max_features"]}`, `ngram_range={tuple(m["ngram_range"])}`. It is fitted inside a scikit-learn Pipeline in each of **{m["cv_folds"]} shuffled stratified folds**, preventing vocabulary/IDF leakage. Corpus-wide EDA does not set the vocabulary.
- Compare Logistic Regression, Multinomial Naive Bayes and Random Forest using mean training-fold macro F1. Fixed baseline hyperparameters live in `src/train.py`; there is no hidden search or test-set tuning. Macro F1 weights minority categories equally.
- Freeze model selection before held-out evaluation. Fit each model on the training partition, then evaluate all on the same test partition. Persist only the winner and its paired vectorizer. Do not refit on the test set, so the saved artifact reproduces the reported held-out score.

## Actual results

Values below are unrounded decimal values read directly from `results.json`. CV standard deviation describes fold variation, not a confidence interval. Held-out scores describe this split only.

| Model | CV macro F1 mean | CV std | Test accuracy | Test macro precision | Test macro recall | Test macro F1 |
|---|---|---|---|---|---|---|
{table}

**Selected: `{selected}`.** {rationale} Naive Bayes provides a cheap, inspectable baseline but assumes conditional independence. Logistic Regression offers direct class-term weights. Random Forest provides nonlinear interactions at greater complexity. These tradeoffs are considered alongside the predeclared CV selection rule; the test scores do not choose the winner. No statistical significance claim is made about the ordering.

Measured wall-clock training costs on the environment recorded in `results.json` (not serving latency benchmarks):

| Model | CV seconds | Full training-partition fit seconds |
|---|---|---|
{timing}

![Selected model confusion matrix](reports/figures/confusion_matrix_{selected}.png)

All model confusion matrices are saved under `reports/figures/`; raw matrices and ordered class labels are in `results.json`.

## Run locally

Tested Python version: `{r["environment"]["python"]}`. Run from the project root. `requirements.txt` pins the actual tested environment, including transitive dependencies. Install Python, then:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The included dataset and trained artifacts let you serve without retraining. NLTK corpus downloads are explicit setup steps; inference does not access the network. Interactive API documentation: `http://127.0.0.1:8000/docs`. `GET /health` returns model readiness; unavailable artifacts or corpora produce HTTP service unavailable.

Reproduce the complete workflow (the data download is optional when using the included CSV):

```bash
python -m src.data
python -m src.eda
python -m src.train
python -m src.notebook
python -m src.verify
python -m src.readme
```

`src.data` reuses `data/source.zip` if present; otherwise it downloads the current upstream archive. Compare the resulting hash with the included provenance before treating a rerun as the same dataset. Upstream changes can change results. Training overwrites model artifacts and results; verification and README generation should follow it. Timing varies by hardware, and numeric libraries can introduce platform-dependent floating point differences. NLTK corpus contents are external dependencies and are not package-pinned.

## API contract and real example

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{json.dumps(example["request"])}'
```

On Windows use `curl.exe` if `curl` is a PowerShell alias. Actual response from the saved model, captured by `src.verify` in `results.json`:

```json
{json.dumps(example["response"], indent=2)}
```

`confidence` is the winning class's **uncalibrated** `predict_proba` value. It is not a validated likelihood of correctness or a candidate suitability score. The API rejects missing/empty/whitespace text, non-string values, oversized text, unknown fields and inputs with no vocabulary overlap. Weakly related inputs with some known words can still receive a label; no general out-of-domain detector is claimed. Artifacts load once at startup; synchronous inference runs through FastAPI's thread pool. Missing or mismatched artifacts produce an unavailable response, and unexpected prediction failures return a generic server error without exposing internals. `MODEL_DIR` optionally selects another trusted artifact directory.

## Verification and repository map

Recorded pytest outcome: **{r["verification"]["pytest"]["tests"]} tests, {r["verification"]["pytest"]["failures"]} failures, {r["verification"]["pytest"]["errors"]} errors, {r["verification"]["pytest"]["skipped"]} skipped**. `reports/pytest.xml` contains the actual test run. Coverage includes preprocessing edge cases, artifact integrity, prediction/probability consistency, exact cleaned train/test disjointness, reproduction of the held-out score from saved artifacts, API validation, readiness failures and generic internal-error handling. Run `python -m src.verify` to refresh evidence, or `python -m pytest -q` for tests alone.

| Path | Purpose |
|---|---|
| `data/resumes.csv`, `data/provenance.json` | Original text/label projection and source audit |
| `src/preprocess.py` | Shared NLTK normalization |
| `src/train.py`, `src/evaluate.py` | Fold-local features, selection, metrics and plots |
| `src/predict.py` | Local artifact loading and inference |
| `api/main.py` | Validated REST interface and health check |
| `models/` | Trained model, vectorizer and checksum manifest |
| `notebooks/eda.ipynb`, `reports/` | Executed exploration, figures, split and test evidence |
| `results.json`, `src/readme.py` | Numeric evidence and generated documentation |
| `tests/` | Unit, artifact integration and API tests |

## Limitations

- This is a tested portfolio service, not a demonstrated production hiring system. There is no production monitoring, authentication, rate limiting, load testing or deployment validation. Add infrastructure-level request byte limits before public exposure; the app's text-length validation occurs after JSON parsing.
- The small, imbalanced, single-source English dataset has occupational and website/template bias. Labels overlap (for example banking and finance), and source job titles inside resumes can make classification artificially easy. No external, temporal or employer-held-out benchmark is available.
- Exact cleaned duplicates are removed; near duplicates and shared authors may still inflate results. Global duplicate filtering is a predefined data-integrity step, not a learned feature transformation.
- Scores are the real measured baselines, with substantial remaining errors. The lowest held-out class recall is **{lowest}: {recalls[lowest]}**; this is a concrete weakness hidden by aggregate accuracy. The model cannot infer job fit, seniority or applicant quality. No fairness evaluation or hiring-outcome validation was performed.
- Alphabetic cleaning loses numeric experience, non-English letters and distinctions such as C++/C#. Noun-only lemmatization does not fully normalize verbs. Short snippets differ from the longer training resumes.
- Confidence is uncalibrated and classes are closed-set. Inputs outside those classes can be misclassified confidently. No uncertainty interval, robustness or latency guarantee is asserted.
- Joblib is pickle-based: load trusted artifacts only. Checksums detect accidental mismatches; they do not authenticate artifacts against a malicious replacement of both files and manifest. A model/version change requires retraining and reevaluation.

## What I'd improve with more time

Collect consented, more diverse and independently labeled resumes; detect near-duplicate/source groups before splitting; validate on a genuinely external corpus. Audit errors by class and writing style, preserve technical skill tokens, and tune regularization within nested validation. Compare class weighting and a linear SVM without consulting the holdout. Calibrate probabilities on a separate validation partition and evaluate abstention for unknown domains. Add privacy-aware drift/error monitoring, controlled retraining and API load measurements before deployment.

## Evidence policy

This README is generated from `results.json` by `python -m src.readme`; metric cells are not manually copied or rounded. That file records dataset/cleaning counts, EDA, split configuration, fold scores, held-out metrics, timing, environment, hashes and the actual tested example. Re-running training invalidates prior verification until tests run again.
"""


def main() -> None:
    results = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    (ROOT / "README.md").write_text(render(results), encoding="utf-8")
    print("Generated README.md from results.json")


if __name__ == "__main__":
    main()
