# Experiment log

Current frozen baseline; historical records follow unchanged.

| ID | Change | CV macro F1 (mean ± std) | Test macro F1 | Test accuracy | Train time (s) | Kept? | Notes |
|---|---|---|---|---|---|---|---|
| B0 | Freeze current Random Forest + sigmoid calibration | 0.710961947058129 ± 0.014977244812944244 | 0.7783811932565733 | 0.8048289738430584 | 452.60083699999814 | Yes | Commit 6cb1912. Metrics include calibration; all test rows, before abstention. Time is nested calibration method-comparison wall time, not a single fit. |
| L1 | Own-label masking diagnostic; RF + fixed sigmoid | 0.5317317225299487 ± 0.009913121069946657 | 0.5580907457690872 | 0.5955734406438632 | 13.244835800000146 | No | Measurement only, not model selection or adoption. Production unchanged. Time includes full-training RF and calibration fit. See leakage_impact/SUMMARY.md. Time is selected base-model fit time. |
| P1 | Production universal masking; fixed RF + sigmoid | 0.4861169105304429 ± 0.014420560374723257 | 0.520205602976937 | 0.5633802816901409 | 361.9881848999994 | Yes | Requested methodological promotion, not a test-score improvement. All-label and canonical lemma masking; thresholds chosen on training OOF. See production_v2/comparison.md. Time is nested CV wall time with fixed sigmoid, not a model-method comparison or a single fit. |
| V3-1 | family_screen logistic_regression {} | 0.5379772586292838 ± 0.01563035063836193 | not evaluated | not evaluated | 48.80190560000119 | Yes | Accepted as search incumbent only; not promoted. Held-out data unused for this trial. See v3/search_log.md. Time is measured five-fold fit/scoring cost, not one full-training fit. |
| V3-2 | family_screen random_forest {} | 0.6089439014893414 ± 0.0154157434862607 | not evaluated | not evaluated | 43.438257499998144 | Yes | Accepted as search incumbent only; not promoted. Held-out data unused for this trial. See v3/search_log.md. Time is measured five-fold fit/scoring cost, not one full-training fit. |
| V3-3 | family_screen linear_svc {} | 0.5876247295748038 ± 0.02490672877849458 | not evaluated | not evaluated | 17.723251799998252 | No | Accepted as search incumbent only; not promoted. Held-out data unused for this trial. See v3/search_log.md. Time is measured five-fold fit/scoring cost, not one full-training fit. |
| V3-4 | randomized_search random_forest {"classifier__max_depth": 30, "classifier__min_samples_leaf": 1, "tfidf__max_df": 0.9, "tfidf__max_features": 10000, "tfidf__min_df": 3} | 0.5974385504651646 ± 0.022209375478586108 | not evaluated | not evaluated | 27.77335023880005 | No | Accepted as search incumbent only; not promoted. Held-out data unused for this trial. See v3/search_log.md. Time is measured five-fold fit/scoring cost, not one full-training fit. |
| V3-5 | randomized_search random_forest {"classifier__max_depth": 30, "classifier__min_samples_leaf": 2, "tfidf__max_df": 0.9, "tfidf__max_features": 20000, "tfidf__min_df": 2} | 0.5990794772803449 ± 0.021857708076941453 | not evaluated | not evaluated | 25.585954427719116 | No | Accepted as search incumbent only; not promoted. Held-out data unused for this trial. See v3/search_log.md. Time is measured five-fold fit/scoring cost, not one full-training fit. |
| V3-6 | randomized_search random_forest {"classifier__max_depth": 30, "classifier__min_samples_leaf": 1, "tfidf__max_df": 0.9, "tfidf__max_features": 10000, "tfidf__min_df": 2} | 0.6151059008733112 ± 0.02859508580488019 | not evaluated | not evaluated | 28.187321424484253 | No | Accepted as search incumbent only; not promoted. Held-out data unused for this trial. See v3/search_log.md. Time is measured five-fold fit/scoring cost, not one full-training fit. |
| V3-7 | randomized_search random_forest {"classifier__max_depth": 30, "classifier__min_samples_leaf": 2, "tfidf__max_df": 0.95, "tfidf__max_features": 5000, "tfidf__min_df": 3} | 0.6124279816555676 ± 0.019472818367348797 | not evaluated | not evaluated | 27.1693332195282 | No | Accepted as search incumbent only; not promoted. Held-out data unused for this trial. See v3/search_log.md. Time is measured five-fold fit/scoring cost, not one full-training fit. |

## Historical records (before baseline freeze)

# Experiment log

The requested baseline freeze was not completed because this is not a Git repository. No baseline model row is fabricated.

## Error analysis

### Error analysis 20260928T125553085962Z

| Note | Finding |
|---|---|
| Scope | Descriptive audit of current saved artifacts; no model fitted, selected or changed. |
| Weakest classes | BPO, AUTOMOBILE, ARTS |
| Label shortcuts | Own-label variants in 2199/2484; share 0.8852657004830918. |
| Near duplicates | 4 test rows above 0.9 cosine; accuracy 1.0 versus 0.742393509127789 for the rest. |
| Report | [Generated summary](../reports/error_analysis/SUMMARY.md); all values come from its generated result files. This is a notes group, not a model-comparison row. |
| Selection guardrail | Future selection uses training-only CV; keep the existing held-out split unchanged. |

## Calibration and abstention

<!-- calibration-run:20260928T1302135841413Z -->

| ID | Change | CV macro F1 (mean ± std) | Test macro F1 | Test accuracy | Train time (s) | Kept? | Notes |
|---|---|---|---|---|---|---|---|
| C1 | Sigmoid calibration; base RF unchanged | 0.710961947058129 ± 0.014977244812944244 | 0.7783811932565733 | 0.8048289738430584 | 452.60083699999814 | Yes | Time is nested method-comparison wall time, not a single model fit. Selection used training OOF log loss/Brier; thresholds 0.3, 0.15. [Generated report](../experiments/calibration_v1/20260928T1302135841413Z/REPORT.md). |
