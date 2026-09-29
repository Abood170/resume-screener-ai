# Experiment log

Current frozen baseline; historical records follow unchanged.

| ID | Change | CV macro F1 (mean ± std) | Test macro F1 | Test accuracy | Train time (s) | Kept? | Notes |
|---|---|---|---|---|---|---|---|
| B0 | Freeze current Random Forest + sigmoid calibration | 0.710961947058129 ± 0.014977244812944244 | 0.7783811932565733 | 0.8048289738430584 | 452.60083699999814 | Yes | Commit 6cb1912. Metrics include calibration; all test rows, before abstention. Time is nested calibration method-comparison wall time, not a single fit. |
| L1 | Own-label masking diagnostic; RF + fixed sigmoid | 0.5317317225299487 ± 0.009913121069946657 | 0.5580907457690872 | 0.5955734406438632 | 13.244835800000146 | No | Measurement only, not model selection or adoption. Production unchanged. Time includes full-training RF and calibration fit. See leakage_impact/SUMMARY.md. Time is selected base-model fit time. |
| P1 | Production universal masking; fixed RF + sigmoid | 0.4861169105304429 ± 0.014420560374723257 | 0.520205602976937 | 0.5633802816901409 | 361.9881848999994 | Yes | Requested methodological promotion, not a test-score improvement. All-label and canonical lemma masking; thresholds chosen on training OOF. See production_v2/comparison.md. Time is nested CV wall time with fixed sigmoid, not a model-method comparison or a single fit. |

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
