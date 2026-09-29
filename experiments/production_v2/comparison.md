# Production v2 comparison

Generated from results.json and investigation_review.json. All-row calibrated predictions, before abstention; no numeric rounding.

| System | CV macro F1 (mean ± std) | Test macro F1 | Test accuracy |
|---|---|---|---|
| baseline_v1 | 0.710961947058129 ± 0.014977244812944244 | 0.7783811932565733 | 0.8048289738430584 |
| L1 | 0.5317317225299487 ± 0.009913121069946657 | 0.5580907457690872 | 0.5955734406438632 |
| production_v2 | 0.4861169105304429 ± 0.014420560374723257 | 0.520205602976937 | 0.5633802816901409 |

| Category | Baseline v1 F1 | L1 F1 | Production v2 F1 |
|---|---|---|---|
| ACCOUNTANT | 0.8301886792452831 | 0.7727272727272727 | 0.631578947368421 |
| ADVOCATE | 0.8235294117647058 | 0.41379310344827586 | 0.37735849056603776 |
| AGRICULTURE | 0.8181818181818182 | 0.3 | 0.43478260869565216 |
| APPAREL | 0.6896551724137931 | 0.36363636363636365 | 0.3333333333333333 |
| ARTS | 0.6153846153846154 | 0.5142857142857142 | 0.35294117647058826 |
| AUTOMOBILE | 0.5 | 0.4444444444444444 | 0.25 |
| AVIATION | 0.8372093023255814 | 0.7391304347826086 | 0.7727272727272727 |
| BANKING | 0.7272727272727273 | 0.5531914893617021 | 0.6341463414634146 |
| BPO | 0.36363636363636365 | 0.0 | 0.0 |
| BUSINESS-DEVELOPMENT | 0.9583333333333334 | 0.49122807017543857 | 0.5517241379310345 |
| CHEF | 0.7727272727272727 | 0.7441860465116279 | 0.7727272727272727 |
| CONSTRUCTION | 0.9545454545454546 | 0.6666666666666666 | 0.6956521739130435 |
| CONSULTANT | 0.7391304347826086 | 0.09523809523809523 | 0.05405405405405406 |
| DESIGNER | 0.9302325581395349 | 0.7027027027027027 | 0.5789473684210527 |
| DIGITAL-MEDIA | 0.7878787878787878 | 0.5945945945945946 | 0.6486486486486487 |
| ENGINEERING | 0.875 | 0.6382978723404256 | 0.6363636363636364 |
| FINANCE | 0.8 | 0.8301886792452831 | 0.5333333333333333 |
| FITNESS | 0.88 | 0.7083333333333334 | 0.6666666666666666 |
| HEALTHCARE | 0.6976744186046512 | 0.46808510638297873 | 0.41509433962264153 |
| HR | 0.875 | 0.8260869565217391 | 0.8163265306122449 |
| INFORMATION-TECHNOLOGY | 0.9019607843137255 | 0.7241379310344828 | 0.7142857142857143 |
| PUBLIC-RELATIONS | 0.7555555555555555 | 0.5957446808510638 | 0.5833333333333334 |
| SALES | 0.6909090909090909 | 0.5098039215686274 | 0.44 |
| TEACHER | 0.8571428571428571 | 0.6976744186046512 | 0.5909090909090909 |

## Investigation before promotion

Production v2 minus L1: test macro F1 -0.03788514279215027; training CV macro F1 -0.04561481199950579. The drop is present in training CV too; it is not solely a held-out anomaly.

Universal masking removes an average of 15.551307847082494 whitespace-delimited words per test resume, versus 7.334004024144869 for L1. Including canonical lemma masking, it removes 19.80482897384306 cleaned tokens on average, versus 8.50503018108652 for L1. On training resumes the corresponding raw-word averages are 15.145665322580646 versus 7.120463709677419. Raw-word counts and cleaned-token counts use different tokenization and should not be subtracted from one another.

This broader removal includes useful domain evidence as well as titles. L1 conditions its removal on the true category; v2 does not. The additional canonical lemma pass also removes singular forms such as sale. These differences are consistent with a lower score, but do not establish a causal decomposition. Neither a runtime masking bug nor altered CV folds is indicated by the integrity audit. There are 0 empty test texts after masking, 0 exact masked cross-split overlaps, and 0 forbidden vocabulary features.

The hypothesis that v2 might fall between baseline and L1 was not a guaranteed bound and was not borne out. No preprocessing, model parameter, calibration method or threshold was selected using this held-out result. Promotion implements the requested label-independent methodology, not a claim of predictive improvement. This is materially lower than the old internal benchmark, but more defensible against the known label-word shortcut. It is not proof that all remaining predictions are genuine content understanding or that the model is ready for hiring decisions.
