# Experiment log

Current frozen baseline; historical records follow unchanged.

| ID | Change | CV macro F1 (mean ? std) | Test macro F1 | Test accuracy | Train time (s) | Kept? | Notes |
|---|---|---|---|---|---|---|---|

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
