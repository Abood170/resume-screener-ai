---
title: Resume Screener AI
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Resume Screener AI

## Live Demo

Hugging Face Spaces deployment is pending. See [DEPLOY_HF.md](DEPLOY_HF.md); `/health` checks readiness and `/docs` provides the API schema.

A locally trained job-category classifier using scikit-learn, FastAPI and React. It demonstrates reproducible training, calibration and uncertainty handling. **This is not a hiring decision tool, a candidate ranking system or a measure of applicant quality.**

## Dataset and disclosure

The real public [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) contains 2484 rows and 24 categories; no synthetic training data was used. Source-provided labels and website-derived examples were not independently verified. The retained dataset has 2481 rows. Source and dataset hash are recorded in [data/provenance.json](data/provenance.json); current CSV SHA-256: `76275a0d8e029e4fb46250296c0a25661bd44534ccdec034f4a19f42bbc753a0`.

Public availability does not establish consent for unrelated reuse. Do not log private resumes. No independent licensing or anonymization guarantee is made for source data.

## Methodology

- Preserve the original stratified seed-42 split: 1984 training and 497 held-out rows in `reports/split.json`. No rows were resplit after masking.
- Apply a **fixed universal dictionary** containing every category name and the earlier audit's variants to every text, without using its label. Lowercase, remove URLs/nonletters, remove NLTK stopwords, noun-lemmatize, and remove canonical lemma forms of the same dictionary. This removes useful domain vocabulary too.
- The saved v2 vectorizer calls `clean_text_universal` during both fitting and API inference. `clean_text_legacy` preserves old behavior; the `clean_text` compatibility entry point keeps frozen baseline/L1 pickles reproducible. Do not substitute the legacy function in new production fits.
- Keep baseline TF-IDF parameters (`max_features=5000`, `ngram_range=(1, 2)`) and all Random Forest hyperparameters unchanged. Fit vocabulary/IDF inside each training fold. See [protocol.json](experiments/production_v2/protocol.json).
- Fixed sigmoid calibration is refitted from scratch. Outer and inner stratified CV each use 5 folds; calibrators see out-of-fold base-model probabilities. Final calibration uses training OOF predictions; the final forest fits only the full training partition. No method search was repeated.
- Select abstention thresholds using only training OOF predictions and the original grid/utility rule. Freeze choices before the one final held-out experiment evaluation. Later metric audits use saved predictions; the regression suite also verifies saved-artifact reproducibility, without tuning settings.

## Actual results

Generated from [experiments/production_v2/results.json](experiments/production_v2/results.json). CV standard deviation describes fold variation, not a confidence interval. Every row below uses calibrated predictions before abstention.

| System | CV macro F1 (mean ± std) | Test macro F1 | Test accuracy |
|---|---|---|---|
| baseline_v1 | 0.710961947058129 ± 0.014977244812944244 | 0.7783811932565733 | 0.8048289738430584 |
| L1 | 0.5317317225299487 ± 0.009913121069946657 | 0.5580907457690872 | 0.5955734406438632 |
| production_v2 | 0.4861169105304429 ± 0.014420560374723257 | 0.520205602976937 | 0.5633802816901409 |

The earlier audit found own-label variants in 2199/2484 raw resumes (share 0.8852657004830918). Baseline v1 therefore included a known shortcut. L1 removed only each resume's own true category variants, an informative diagnostic that cannot be used on unlabeled inputs. Production v2 removes all variants regardless of category.

Production v2 minus L1: test macro F1 -0.03788514279215027; training CV macro F1 -0.04561481199950579. The drop is present in training CV too; it is not solely a held-out anomaly.

Universal masking removes an average of 15.551307847082494 whitespace-delimited words per test resume, versus 7.334004024144869 for L1. Including canonical lemma masking, it removes 19.80482897384306 cleaned tokens on average, versus 8.50503018108652 for L1. On training resumes the corresponding raw-word averages are 15.145665322580646 versus 7.120463709677419. Raw-word counts and cleaned-token counts use different tokenization and should not be subtracted from one another.

This broader removal includes useful domain evidence as well as titles. L1 conditions its removal on the true category; v2 does not. The additional canonical lemma pass also removes singular forms such as sale. These differences are consistent with a lower score, but do not establish a causal decomposition. Neither a runtime masking bug nor altered CV folds is indicated by the integrity audit. There are 0 empty test texts after masking, 0 exact masked cross-split overlaps, and 0 forbidden vocabulary features.

The hypothesis that v2 might fall between baseline and L1 was not a guaranteed bound and was not borne out. No preprocessing, model parameter, calibration method or threshold was selected using this held-out result. Promotion implements the requested label-independent methodology, not a claim of predictive improvement. This is materially lower than the old internal benchmark, but more defensible against the known label-word shortcut. It is not proof that all remaining predictions are genuine content understanding or that the model is ready for hiring decisions.


[Per-class F1 for all three systems](experiments/production_v2/comparison.md). Historical results and artifacts remain in `experiments/baseline_v1/` and `experiments/leakage_impact/`; current serving metrics and checksums are in root `results.json` and `models/manifest.json`.

## Calibration and abstention

| Metric | Raw v2 | Calibrated v2 |
|---|---|---|
| brier_score | 0.6901419517102615 | 0.6029392062638212 |
| log_loss | 1.969160540109228 | 1.6418886069471392 |
| ece_10_bins | 0.23144869215291752 | 0.09569966890724542 |
| macro_f1 | 0.4599307596698923 | 0.520205602976937 |
| accuracy | 0.5352112676056338 | 0.5633802816901409 |

Brier is the mean sum across classes of squared probability error. Log loss uses natural logarithms. ECE uses top-label confidence and ten equal-width bins; it depends on binning and is not a correctness guarantee.

Current thresholds: **T1=0.4, T2=0.05** (previously 0.3 and 0.15). Select the maximum training OOF utility: correct answer +1, wrong answer -1, abstention 0; ties favor coverage, then lower thresholds. This is a declared demo cost assumption, not a validated operational cost. Inputs shorter than 50 whitespace words also trigger uncertainty. Reason priority is short input, low top probability, small top-two margin. Numerical comparisons are strict `<`.

Held-out coverage: **0.6156941649899397**, answering **306/497** resumes. Accuracy on answered resumes: **0.7189542483660131**. This selective accuracy must not be confused with all-row accuracy. Training selection scores are optimistic after threshold tuning.

![Training threshold tradeoff](experiments/production_v2/coverage_vs_accuracy.png)
![Held-out reliability](experiments/production_v2/reliability_before_after.png)

## Run locally

Use Python 3.12.14 and the pinned `requirements.txt`. From the project root:

```bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another terminal: `cd frontend`, `npm install`, `npm run dev`. Open the displayed Vite URL. Included artifacts serve offline; NLTK downloads are explicit setup steps. The Dockerfile downloads corpora during build, serves on `PORT` (default 7860), and runs as a non-root user. An actual Docker build has not been verified locally.

## API contract and real example

`GET /health`, `POST /predict` with JSON `{"text": "..."}`, and `POST /predict/file` with multipart field `file` keep their existing schemas. Create `request.json` locally with your text; never commit personal CVs:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary @request.json
curl -X POST http://127.0.0.1:8000/predict/file -F "file=@cv.pdf"
```

Use `curl.exe` in PowerShell if needed. This response was captured from the promoted artifacts through FastAPI TestClient using an authored accounting demo, not a held-out record. Input content is not logged:

```json
{
  "predicted_category": "ACCOUNTANT",
  "confidence": 0.675,
  "calibrated_predicted_category": "ACCOUNTANT",
  "calibrated_confidence": 0.6463547658669152,
  "calibrated_top_predictions": [
    {
      "category": "ACCOUNTANT",
      "probability": 0.6463547658669152
    },
    {
      "category": "FINANCE",
      "probability": 0.22120984156348816
    },
    {
      "category": "CONSULTANT",
      "probability": 0.010829865275572736
    }
  ],
  "is_uncertain": false,
  "uncertainty_reason": null,
  "top_predictions": [
    {
      "category": "ACCOUNTANT",
      "probability": 0.675
    },
    {
      "category": "FINANCE",
      "probability": 0.205
    },
    {
      "category": "BANKING",
      "probability": 0.03
    }
  ],
  "top_terms": [
    "statement",
    "account",
    "reconciliation",
    "bank",
    "ledger",
    "general ledger",
    "tax",
    "report"
  ],
  "text_stats": {
    "word_count": 76,
    "short_input": false
  },
  "source": "text",
  "text_preview": null
}
```

Legacy `predicted_category`, `confidence` and `top_predictions` still describe raw RF probabilities. The calibrated category/confidence/top-three fields form a separate coherent view. Predictions may change with the new model; field names, types, validation and raw-versus-calibrated semantics are unchanged. `is_uncertain` and `uncertainty_reason` remain explicit. The frontend shows possible categories when uncertain and preserves the ethical footer.

`top_terms` uses present TF-IDF features multiplied by global RF impurity importance. These are influential terms, not class-specific or causal explanations. The full vocabulary and a real API example were checked for forbidden label phrases and lemma forms. File responses return a preview of original extracted text to the requesting browser; the model uses the universally masked representation.

Uploads accept PDF, DOCX and TXT, validate signatures and enforce bounded in-memory processing. No uploaded file is permanently stored. Scanned/image-only and encrypted PDFs are unsupported; no OCR is performed. Detailed existing upload bounds and errors are enforced in `src/extract.py` and `api/uploads.py`. Missing artifacts fail readiness; validation errors and unexpected failures return friendly responses without resume content or stack traces.

## Verification and reproduction

Backend: **104 passed, 0 failed, 0 errors, 0 skipped**. Frontend: **46 passed, 0 failed, 0 errors, 0 skipped**. Raw test evidence and schema comparison are in `experiments/production_v2/`. Run `python -m pytest -v` and, from `frontend`, `npm test`.

`python -m src.production_v2 --run-dir experiments/<new-run>` reproduces the fixed training protocol with the included dataset and baseline snapshot. It refuses existing experiment evidence and does not promote automatically. Do not rerun to optimize against the holdout. Root `src/train.py` is the historical unmasked workflow and must not be used to regenerate current production artifacts. `python -m src.production_v2_report --final` regenerates this README from saved evidence, archiving previous documentation. Root results are synchronized from the production experiment; older results are archived before replacement.

## Limitations

- Universal masking addresses a specific known lexical shortcut; near-duplicate templates, related occupation words, website bias and source-label errors remain. It is a more defensible measurement of this masking policy, not proof of an unbiased or leakage-free model.
- The lower number is visible rather than hidden: identifying and addressing an inflated internal benchmark is a positive methodological finding, not evidence of improved accuracy. The test split has been inspected across earlier experiments and is not a pristine external benchmark.
- Small, imbalanced, single-source English data limits calibration and minority-class estimates. Non-English CVs, short inputs and out-of-domain texts are unreliable. Thresholds are dataset-specific; confidence does not measure candidate quality.
- Broad variants remove genuine occupational content; alphabetic cleaning loses numeric experience and distinctions such as C++ versus C#. Scanned PDFs are unsupported and extraction can misorder columns.
- There is no production monitoring, fairness/hiring-outcome validation, rate limiting, authentication, load testing or public-deployment validation. CORS currently allows all origins; restrict it before public deployment.
- Joblib loads trusted artifacts only. Checksums detect mismatches, not malicious replacement of artifacts and their manifest. Restart the API after artifact changes; already running processes retain their loaded model until restarted.

## What I'd improve with more time

Obtain consented, independently labeled external data; validate source/template-separated generalization; inspect minority-class errors and domain shift. Compare alternative models using training-only validation, and validate calibration and abstention on independent data. Add privacy-aware monitoring and deployment load tests. No candidate-quality scoring is proposed.
