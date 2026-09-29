"""Render production documentation exclusively from generated experiment evidence."""
import argparse
import json
from pathlib import Path
from src.data import ROOT
from src.production_v2 import archive


def write(path: Path, text: str, run: Path):
    archive(path,run)
    path.write_text(text,encoding='utf-8')


def main(final=False):
    run=ROOT/'experiments/production_v2'
    r=json.loads((run/'results.json').read_text())
    audit=json.loads((run/'investigation_review.json').read_text())
    b=json.loads((ROOT/'experiments/baseline_v1/results.json').read_text())
    shortcut=json.loads((ROOT/'reports/error_analysis/analysis_results.json').read_text())
    comparison=['| System | CV macro F1 (mean ± std) | Test macro F1 | Test accuracy |','|---|---|---|---|']
    for name,m in r['leakage_comparison'].items():
        comparison.append(f"| {name} | {m['cv_mean']} ± {m['cv_std']} | {m['macro_f1']} | {m['accuracy']} |")
    table='\n'.join(comparison)
    per=['| Category | Baseline v1 F1 | L1 F1 | Production v2 F1 |','|---|---|---|---|']
    for c in r['classes']:
        values=[str(r['leakage_comparison'][key]['per_class'][c]['f1-score']) for key in ['baseline_v1','L1','production_v2']]
        per.append('| '+c+' | '+' | '.join(values)+' |')
    inv=r['investigation']; rem=audit['removal_comparison']; current=r['calibration']['test']['after']
    investigation=f"""Production v2 minus L1: test macro F1 {inv['test_macro_f1_delta_vs_l1']}; training CV macro F1 {inv['cv_macro_f1_delta_vs_l1']}. The drop is present in training CV too; it is not solely a held-out anomaly.

Universal masking removes an average of {rem['test']['universal']['mean_raw_words_removed']} whitespace-delimited words per test resume, versus {rem['test']['l1_mean_raw_words_removed']} for L1. Including canonical lemma masking, it removes {rem['test']['universal']['mean_clean_tokens_removed']} cleaned tokens on average, versus {rem['test']['l1_mean_clean_tokens_removed']} for L1. On training resumes the corresponding raw-word averages are {rem['train']['universal']['mean_raw_words_removed']} versus {rem['train']['l1_mean_raw_words_removed']}. Raw-word counts and cleaned-token counts use different tokenization and should not be subtracted from one another.

This broader removal includes useful domain evidence as well as titles. L1 conditions its removal on the true category; v2 does not. The additional canonical lemma pass also removes singular forms such as sale. These differences are consistent with a lower score, but do not establish a causal decomposition. Neither a runtime masking bug nor altered CV folds is indicated by the integrity audit. There are {r['masking_audit']['test']['overall']['empty_after']} empty test texts after masking, {r['masking_audit']['masked_exact_cross_split_overlap']} exact masked cross-split overlaps, and {len(r['masking_audit']['forbidden_features'])} forbidden vocabulary features.

The hypothesis that v2 might fall between baseline and L1 was not a guaranteed bound and was not borne out. No preprocessing, model parameter, calibration method or threshold was selected using this held-out result. Promotion implements the requested label-independent methodology, not a claim of predictive improvement. This is materially lower than the old internal benchmark, but more defensible against the known label-word shortcut. It is not proof that all remaining predictions are genuine content understanding or that the model is ready for hiring decisions.
"""
    write(run/'comparison.md','# Production v2 comparison\n\nGenerated from results.json and investigation_review.json. All-row calibrated predictions, before abstention; no numeric rounding.\n\n'+table+'\n\n'+'\n'.join(per)+'\n\n## Investigation before promotion\n\n'+investigation,run)
    if not final: return
    verify=json.loads((run/'verification.json').read_text())
    examples=json.loads((run/'api_examples.json').read_text())
    e=examples['clear_accounting_demo']['response']
    calibration=['| Metric | Raw v2 | Calibrated v2 |','|---|---|---|']
    for k in ['brier_score','log_loss','ece_10_bins','macro_f1','accuracy']:
        calibration.append(f"| {k} | {r['calibration']['test']['before'][k]} | {current[k]} |")
    thresholds=r['abstain']; policy=thresholds['test']; meth=r['methodology']
    # Retain deployment metadata; the rest of the README describes current v2.
    original=(run/'before/README.md').read_text(encoding='utf-8')
    metadata='---'+original.split('---',2)[1]+'---\n' if original.startswith('---') else ''
    # The earlier audit stores the label-shortcut summary under this key.
    labels=shortcut['label_shortcuts']
    text=metadata+f"""
# Resume Screener AI

## Live Demo

Hugging Face Spaces deployment is pending. See [DEPLOY_HF.md](DEPLOY_HF.md); `/health` checks readiness and `/docs` provides the API schema.

A locally trained job-category classifier using scikit-learn, FastAPI and React. It demonstrates reproducible training, calibration and uncertainty handling. **This is not a hiring decision tool, a candidate ranking system or a measure of applicant quality.**

## Dataset and disclosure

The real public [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) contains {r['dataset']['raw_rows']} rows and {r['dataset']['raw_categories']} categories; no synthetic training data was used. Source-provided labels and website-derived examples were not independently verified. The retained dataset has {r['cleaning']['retained_rows']} rows. Source and dataset hash are recorded in [data/provenance.json](data/provenance.json); current CSV SHA-256: `{r['dataset']['csv_sha256']}`.

Public availability does not establish consent for unrelated reuse. Do not log private resumes. No independent licensing or anonymization guarantee is made for source data.

## Methodology

- Preserve the original stratified seed-{meth['seed']} split: {meth['train_rows']} training and {meth['test_rows']} held-out rows in `reports/split.json`. No rows were resplit after masking.
- Apply a **fixed universal dictionary** containing every category name and the earlier audit's variants to every text, without using its label. Lowercase, remove URLs/nonletters, remove NLTK stopwords, noun-lemmatize, and remove canonical lemma forms of the same dictionary. This removes useful domain vocabulary too.
- The saved v2 vectorizer calls `clean_text_universal` during both fitting and API inference. `clean_text_legacy` preserves old behavior; the `clean_text` compatibility entry point keeps frozen baseline/L1 pickles reproducible. Do not substitute the legacy function in new production fits.
- Keep baseline TF-IDF parameters (`max_features={meth['max_features']}`, `ngram_range={tuple(meth['ngram_range'])}`) and all Random Forest hyperparameters unchanged. Fit vocabulary/IDF inside each training fold. See [protocol.json](experiments/production_v2/protocol.json).
- Fixed sigmoid calibration is refitted from scratch. Outer and inner stratified CV each use {r['calibration']['protocol']['outer_cv']} folds; calibrators see out-of-fold base-model probabilities. Final calibration uses training OOF predictions; the final forest fits only the full training partition. No method search was repeated.
- Select abstention thresholds using only training OOF predictions and the original grid/utility rule. Freeze choices before the one final held-out experiment evaluation. Later metric audits use saved predictions; the regression suite also verifies saved-artifact reproducibility, without tuning settings.

## Actual results

Generated from [experiments/production_v2/results.json](experiments/production_v2/results.json). CV standard deviation describes fold variation, not a confidence interval. Every row below uses calibrated predictions before abstention.

{table}

The earlier audit found own-label variants in {labels['own_variant_count']}/{labels['total_raw_rows']} raw resumes (share {labels['own_variant_share']}). Baseline v1 therefore included a known shortcut. L1 removed only each resume's own true category variants, an informative diagnostic that cannot be used on unlabeled inputs. Production v2 removes all variants regardless of category.

{investigation}

[Per-class F1 for all three systems](experiments/production_v2/comparison.md). Historical results and artifacts remain in `experiments/baseline_v1/` and `experiments/leakage_impact/`; current serving metrics and checksums are in root `results.json` and `models/manifest.json`.

## Calibration and abstention

{chr(10).join(calibration)}

Brier is the mean sum across classes of squared probability error. Log loss uses natural logarithms. ECE uses top-label confidence and ten equal-width bins; it depends on binning and is not a correctness guarantee.

Current thresholds: **T1={thresholds['t1']}, T2={thresholds['t2']}** (previously {b['abstain']['t1']} and {b['abstain']['t2']}). Select the maximum training OOF utility: correct answer +1, wrong answer -1, abstention 0; ties favor coverage, then lower thresholds. This is a declared demo cost assumption, not a validated operational cost. Inputs shorter than {thresholds['short_input_words']} whitespace words also trigger uncertainty. Reason priority is short input, low top probability, small top-two margin. Numerical comparisons are strict `<`.

Held-out coverage: **{policy['coverage']}**, answering **{policy['answered']}/{policy['total']}** resumes. Accuracy on answered resumes: **{policy['accuracy_on_answered']}**. This selective accuracy must not be confused with all-row accuracy. Training selection scores are optimistic after threshold tuning.

![Training threshold tradeoff](experiments/production_v2/coverage_vs_accuracy.png)
![Held-out reliability](experiments/production_v2/reliability_before_after.png)

## Run locally

Use Python {r['environment']['python']} and the pinned `requirements.txt`. From the project root:

```bash
python -m venv .venv
# PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another terminal: `cd frontend`, `npm install`, `npm run dev`. Open the displayed Vite URL. Included artifacts serve offline; NLTK downloads are explicit setup steps. The Dockerfile downloads corpora during build, serves on `PORT` (default 7860), and runs as a non-root user. An actual Docker build has not been verified locally.

## API contract and real example

`GET /health`, `POST /predict` with JSON `{{"text": "..."}}`, and `POST /predict/file` with multipart field `file` keep their existing schemas. Create `request.json` locally with your text; never commit personal CVs:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary @request.json
curl -X POST http://127.0.0.1:8000/predict/file -F "file=@cv.pdf"
```

Use `curl.exe` in PowerShell if needed. This response was captured from the promoted artifacts through FastAPI TestClient using an authored accounting demo, not a held-out record. Input content is not logged:

```json
{json.dumps(e,indent=2)}
```

Legacy `predicted_category`, `confidence` and `top_predictions` still describe raw RF probabilities. The calibrated category/confidence/top-three fields form a separate coherent view. Predictions may change with the new model; field names, types, validation and raw-versus-calibrated semantics are unchanged. `is_uncertain` and `uncertainty_reason` remain explicit. The frontend shows possible categories when uncertain and preserves the ethical footer.

`top_terms` uses present TF-IDF features multiplied by global RF impurity importance. These are influential terms, not class-specific or causal explanations. The full vocabulary and a real API example were checked for forbidden label phrases and lemma forms. File responses return a preview of original extracted text to the requesting browser; the model uses the universally masked representation.

Uploads accept PDF, DOCX and TXT, validate signatures and enforce bounded in-memory processing. No uploaded file is permanently stored. Scanned/image-only and encrypted PDFs are unsupported; no OCR is performed. Detailed existing upload bounds and errors are enforced in `src/extract.py` and `api/uploads.py`. Missing artifacts fail readiness; validation errors and unexpected failures return friendly responses without resume content or stack traces.

## Verification and reproduction

Backend: **{verify['backend']['passed']} passed, {verify['backend']['failures']} failed, {verify['backend']['errors']} errors, {verify['backend']['skipped']} skipped**. Frontend: **{verify['frontend']['passed']} passed, {verify['frontend']['failed']} failed, {verify['frontend']['errors']} errors, {verify['frontend']['skipped']} skipped**. Raw test evidence and schema comparison are in `experiments/production_v2/`. Run `python -m pytest -v` and, from `frontend`, `npm test`.

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
"""
    write(ROOT/'README.md',text,run)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--final',action='store_true')
    main(parser.parse_args().final)
