"""One-time final documentation from recorded evidence; never evaluates models."""
import json
import shutil
from pathlib import Path
from src.data import ROOT, sha256
from src.experiment_log import append_row


def main():
    out=ROOT/'reports/production_v2_final'
    out.mkdir(exist_ok=False)
    before=out/'before';before.mkdir()
    for name in ['README.md','data/provenance.json','experiments/experiment_log.md','tests/test_exploration.py']:
        dest=before/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
    r=read('experiments/production_v2/results.json');t=read('reports/tech_dataset_audit/results.json')
    l=read('reports/error_analysis/analysis_results.json')['label_shortcuts']
    v=r['leakage_comparison']['production_v2'];b=r['leakage_comparison']['baseline_v1'];d=r['leakage_comparison']['L1']
    e=read('experiments/exploration/word_svc/results.json');ec=read('experiments/exploration/word_char_svc/results.json')
    a=r['abstain'];c=r['calibration']['test'];m=r['methodology']
    manifest=read('experiments/production_v2/models/manifest.json')
    checks={}
    for name,h in manifest['artifacts'].items():
        actual=sha256(ROOT/'models'/name);archived=sha256(ROOT/'experiments/production_v2/models'/name)
        assert actual==archived==h
        checks[name]={'production_sha256':actual,'archived_sha256':archived,'manifest_sha256':h,'match':True}
    assert sha256(ROOT/'models/manifest.json')==sha256(ROOT/'experiments/production_v2/models/manifest.json')
    (out/'artifact_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    provenance=read('data/provenance.json')
    for source in provenance['additional_sources']:
        if source.get('local_path')=='data/resumes_tech.csv':
            source.update({'used_in_training':False,'status':'rejected_after_audit',
                           'audit_report':'reports/tech_dataset_audit/SUMMARY.md',
                           'retention_reason':'Provenance and reproducibility only; no merging or classifier training'})
    (ROOT/'data/provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    findings=f"""## Methodology and honesty findings

The initial benchmark contained a label-name shortcut: the earlier error analysis found own-label variants in {l['own_variant_count']}/{l['total_raw_rows']} resumes (share {l['own_variant_share']}). Baseline v1's recorded test macro F1 was {b['macro_f1']}. That internal score allowed job-title lookup and was not convincing evidence of generalization. Label presence alone does not measure how much of a prediction is caused by the shortcut.

The L1 diagnostic removed each resume's own category variants during fitting and evaluation, recording test macro F1 {d['macro_f1']}. It measured sensitivity to label wording, but required the true category and therefore could not serve as an inference pipeline. It also removed genuine domain evidence; the difference is not a causal decomposition of performance.

Production_v2 applies the entire fixed category-variant dictionary, including canonical lemma forms, to every input without knowing its label. The same transformation runs during training and inference. TF-IDF plus Random Forest, sigmoid calibration and a training-selected abstention policy remain the official model. Its recorded test macro F1 is {v['macro_f1']}. Promotion addressed the known shortcut; it did not improve the old headline score or prove that every remaining shortcut was eliminated.

The later classical-ML exploration evaluated seven configurations, including a freshly measured control, technical-token preservation, balanced weights, Logistic Regression, word/character SVMs and combined features. Word SVC ranked first on raw training CV at {e['cv_mean']} ± {e['cv_std']}; its calibrated test macro F1 was {e['test']['macro_f1']}. The combined word/character candidate recorded {ec['test']['macro_f1']}. Only the frozen finalists received final test evaluation. These results did not demonstrate a meaningful held-out improvement over production_v2. Training CV did improve over the freshly measured raw RF control, so it would be inaccurate to say all CV differences were noise. Raw ranking CV and calibrated production CV are different measurements; fold standard deviation is not a significance test. No exploration candidate was adopted. See the [complete ranked exploration](experiments/exploration/SUMMARY.md).

Jillani's technical-role dataset was then audited and rejected for training as-is: {t['rows']} rows contained only {t['duplicates']['normalized_unique']} unique texts, with {t['duplicates']['raw_redundant_rows']} redundant exact duplicates (fraction {t['duplicates']['normalized_redundant_share']}). Python Developer has 6 unique texts, DevOps Engineer 7 and Java Developer 13, as recorded in the audit category table. Own-label variants occur in {t['leakage']['variant_count']}/{t['rows']} rows; {t['formatting']['tech_all']['possible_mojibake_documents']} rows carry heuristic encoding-corruption markers. After exact deduplication, no unique text had a same-category cosine neighbor above 0.9 under the declared audit representation: repeated copies, rather than additional detected near-duplicates, dominate this problem. The CSV is retained solely to reproduce the [audit](reports/tech_dataset_audit/SUMMARY.md), not used in training.

All selection used the saved training partition. Held-out results are reporting evidence, not a basis for further tuning. The existing split has nevertheless been inspected across historical experiments and is not an untouched external benchmark.
"""
    # Category-specific support is read from the audit, never maintained manually.
    cats={x['category']:x for x in t['category_counts']}
    findings=findings.replace('Python Developer has 6 unique texts, DevOps Engineer 7 and Java Developer 13',
        f"Python Developer has {cats['Python Developer']['unique_normalized']} unique texts, DevOps Engineer {cats['DevOps Engineer']['unique_normalized']} and Java Developer {cats['Java Developer']['unique_normalized']}")
    (out/'methodology.md').write_text(findings,encoding='utf-8')
    metrics=[('Calibrated CV macro F1',f"{v['cv_mean']} ± {v['cv_std']}"),('Test macro F1',v['macro_f1']),('Test accuracy',v['accuracy']),
        ('Test abstention coverage',a['test']['coverage']),('Test answered accuracy',a['test']['accuracy_on_answered'])]
    table='\n'.join(['| Metric | Recorded value |','|---|---|']+[f'| {k} | {value} |' for k,value in metrics])
    calibration='\n'.join(['| Test metric | Raw | Sigmoid calibrated |','|---|---|---|']+
        [f"| {key} | {c['before'][key]} | {c['after'][key]} |" for key in ['brier_score','log_loss','ece_10_bins']])
    per='\n'.join(['| Category | Test F1 |','|---|---|']+[f"| {name} | {v['per_class'][name]['f1-score']} |" for name in r['classes']])
    metadata=(before/'README.md').read_text(encoding='utf-8').split('---',2)[1]
    text=f"""---{metadata}---

# Resume Screener AI

A locally trained job-category classifier using scikit-learn, FastAPI and React. It demonstrates reproducible training, calibration and uncertainty handling. **This is not a hiring decision tool, a candidate ranking system or a measure of applicant quality.**

**Official model: production_v2.** Experimental candidates are archived and not served.

## Live demo

Public deployment is pending; no live URL is claimed. See [deployment instructions](DEPLOY_HF.md). The API exposes `/health` and interactive documentation at `/docs`.

## Dataset and disclosure

Training uses the public [Sneha Anbhawal Kaggle Resume Dataset]({r['dataset']['source']}): {r['dataset']['raw_rows']} source rows and {r['dataset']['raw_categories']} broad categories, with {r['cleaning']['retained_rows']} retained rows after recorded cleaning. No synthetic training data was used. Labels and scraped source content were not independently validated. Public availability is not evidence of consent for every downstream use.

The seed-{m['seed']} stratified split remains {m['train_rows']} training and {m['test_rows']} held-out rows in [reports/split.json](reports/split.json). [Provenance](data/provenance.json) records sources and hashes. `data/resumes.csv` is the training source. **`data/resumes_tech.csv` is rejected audit material, NOT training data**; it remains in the repository for reproducibility. Its publisher states CC0. Do not upload or commit private resumes.

{findings}
## Final pipeline and results

Lowercase text, remove URLs/nonletters, remove NLTK stopwords, noun-lemmatize, and universally mask label variants and their lemma forms. Fit TF-IDF (`max_features={m['max_features']}`, `ngram_range={tuple(m['ngram_range'])}`) within each training fold. The saved vectorizer calls `clean_text_universal`; the historical legacy cleaner is retained only for archived artifact compatibility.

Random Forest and sigmoid calibration are fixed from the recorded protocol. Nested training CV keeps calibration separate from fitting; vocabulary/IDF are fold-local. The final forest is fitted only on training rows. See [protocol](experiments/production_v2/protocol.json) and [results](experiments/production_v2/results.json). No metrics were recomputed for this wrap-up.

{table}

These test scores describe calibrated predictions on all held-out rows before abstention. CV spread is fold standard deviation, not a confidence interval. The per-class results show substantial variation:

<details><summary>Recorded per-class test F1</summary>

{per}

</details>

## Calibration and uncertainty

{calibration}

Brier is the mean sum of squared class-probability errors; log loss uses natural logarithms. ECE uses top-label confidence with ten equal-width bins and depends on binning. Calibration is not a correctness guarantee.

The policy flags uncertainty when calibrated top probability is below **{a['t1']}**, the top-two margin is below **{a['t2']}**, or input has fewer than **{a['short_input_words']}** whitespace words. Threshold selection used training out-of-fold utility (correct +1, wrong -1, abstain 0), with coverage/lower-threshold tie-breaks. This is a demo cost assumption, not an operationally validated policy. On test, it answers {a['test']['answered']}/{a['test']['total']} rows. Selective accuracy must not be confused with all-row accuracy.

![Reliability](experiments/production_v2/reliability_before_after.png)
![Coverage tradeoff](experiments/production_v2/coverage_vs_accuracy.png)

## Run locally

Use the pinned Python dependencies; the recorded training environment used Python {r['environment']['python']}. From the project root:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe -m nltk.downloader stopwords wordnet omw-1.4
.\\.venv\\Scripts\\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173. The frontend targets http://127.0.0.1:8000 by default; use `VITE_API_BASE_URL` to override it. See [frontend setup](frontend/README.md) for Node requirements. On Unix, use `.venv/bin/python` for the Python commands. Included model artifacts serve without model downloads; NLTK downloads are explicit setup steps. Git LFS is required to retrieve the stored model/data files when cloning (`git lfs pull`).

## API and privacy

`GET /health` reports readiness. `POST /predict` accepts a JSON object with a string `text` field. `POST /predict/file` accepts multipart field `file` for PDF, DOCX or TXT. Create `request.json` locally, then:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary @request.json
curl.exe -X POST http://127.0.0.1:8000/predict/file -F "file=@cv.pdf"
```

An existing response captured from production_v2 using an authored demo (not a held-out resume) is preserved in [api_examples.json](experiments/production_v2/api_examples.json). No new prediction was run for this document.

Legacy `predicted_category`, `confidence`, and `top_predictions` describe raw forest probabilities. `calibrated_predicted_category`, `calibrated_confidence`, and `calibrated_top_predictions` provide the calibrated view. `is_uncertain` and `uncertainty_reason` make uncertainty explicit; the UI shows possible categories when uncertain. No field scores candidate quality.

`top_terms` multiplies present TF-IDF features by global forest feature importance. These are influential terms, not causal or class-specific explanations. Uploads use bounded in-memory processing, signature validation and friendly errors. Files are not permanently stored and resume contents are not logged. File responses include an extracted-text preview for the requesting browser. Scanned/image-only and encrypted PDFs are unsupported; no OCR is performed.

## Verification and evidence

Run `.\\.venv\\Scripts\\python.exe -m pytest -v` and `npm test` inside `frontend`. Final wrap-up reports are in [reports/production_v2_final](reports/production_v2_final). Artifact checksum verification matches [models/manifest.json](models/manifest.json) against the archived production_v2 artifacts.

The evidence trail retains [baseline_v1](experiments/baseline_v1), [L1](experiments/leakage_impact), [production_v2](experiments/production_v2), [exploration](experiments/exploration), and the [technical dataset audit](reports/tech_dataset_audit). The [experiment log](experiments/experiment_log.md) records the closing decision. `src/finalize_documentation.py` generated this final document from saved results; its one-time guard prevents accidental overwrite. Earlier report generators produce historical documents and should not overwrite this final README.

## Deployment

The Docker SDK setup uses `PORT` with a default of 7860 and binds to `0.0.0.0`, running as a non-root user. NLTK resources are installed during the image build and model artifacts are included. See [DEPLOY_HF.md](DEPLOY_HF.md) and [README_HF.md](README_HF.md). An actual Docker build/public deployment has not been verified locally. Restrict the currently permissive CORS configuration to the real frontend origin before production deployment.

## Limitations

- The observed classical TF-IDF experiments remain around the production test macro F1 of {v['macro_f1']}; this is an empirical plateau in the investigated settings, not a proven upper bound. Broad occupational categories cannot finely resolve Java, Python or DevOps roles.
- Universal masking removes useful occupational vocabulary too. Related skill words, templates and source bias may remain. Alphabetic cleaning loses distinctions such as C++ and C#.
- The rejected technical dataset has extensive exact duplication, minimal independent support per role, frequent label mentions and encoding-warning markers. Retaining it for an audit does not make it suitable training data.
- Small, imbalanced, single-source English data limits per-class reliability and calibration. Non-English, short and out-of-domain CVs are unreliable; thresholds are dataset-specific. The repeatedly inspected historical holdout is not an independent external validation set.
- There is no production monitoring, fairness or hiring-outcome validation, authentication, rate limiting or deployment load test. This remains a classification demonstration, not a hiring tool.
- Load only trusted joblib artifacts. Checksums detect mismatches, not malicious replacement of both model and manifest. Restart the API after intentional artifact updates.

## What I'd improve with more time

Obtain a properly sourced, consented and independently labeled technical-role dataset; audit duplication, taxonomy and source separation before defining a new split. Compare pretrained embeddings under training-only selection with explicit download approval, measured resource costs and honest explanation limits; embeddings have not yet been evaluated. Investigate per-class uncertainty thresholds using training-only validation and enough independent examples, followed by external calibration validation. Add privacy-aware monitoring and deployment load tests.
"""
    (ROOT/'README.md').write_text(text,encoding='utf-8')
    row=append_row('FINAL-P2','Finalize production_v2; documentation only, no retraining',r,True,
        'Official final model. Exploration and technical dataset audit investigated but not adopted; see experiments/exploration/SUMMARY.md and reports/tech_dataset_audit/SUMMARY.md. Metrics and timing are historical production_v2 results, not a new run.')
    (out/'closing_row.md').write_text(row+'\n',encoding='utf-8')
    print(row)


if __name__=='__main__':main()
