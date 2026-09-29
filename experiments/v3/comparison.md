# v3 candidate comparison

Generated from results.json, position_results.json and archived production results; unrounded values.

| System | CV macro F1 (mean ± std) | Test macro F1 | Test accuracy |
|---|---|---|---|
| baseline_v1 | 0.710961947058129 ± 0.014977244812944244 | 0.7783811932565733 | 0.8048289738430584 |
| L1 | 0.5317317225299487 ± 0.009913121069946657 | 0.5580907457690872 | 0.5955734406438632 |
| production_v2 | 0.4861169105304429 ± 0.014420560374723257 | 0.520205602976937 | 0.5633802816901409 |
| v3 | 0.6556926481524633 ± 0.01777500474604841 | 0.6803971791286144 | 0.7203219315895373 |

| Category | Baseline v1 F1 | L1 F1 | Production v2 F1 | v3 F1 | v3 minus v2 |
|---|---|---|---|---|---|
| ACCOUNTANT | 0.8301886792452831 | 0.7727272727272727 | 0.631578947368421 | 0.7272727272727273 | 0.09569377990430628 |
| ADVOCATE | 0.8235294117647058 | 0.41379310344827586 | 0.37735849056603776 | 0.6938775510204082 | 0.3165190604543704 |
| AGRICULTURE | 0.8181818181818182 | 0.3 | 0.43478260869565216 | 0.7272727272727273 | 0.29249011857707513 |
| APPAREL | 0.6896551724137931 | 0.36363636363636365 | 0.3333333333333333 | 0.6 | 0.26666666666666666 |
| ARTS | 0.6153846153846154 | 0.5142857142857142 | 0.35294117647058826 | 0.41025641025641024 | 0.057315233785821984 |
| AUTOMOBILE | 0.5 | 0.4444444444444444 | 0.25 | 0.4444444444444444 | 0.19444444444444442 |
| AVIATION | 0.8372093023255814 | 0.7391304347826086 | 0.7727272727272727 | 0.8 | 0.027272727272727337 |
| BANKING | 0.7272727272727273 | 0.5531914893617021 | 0.6341463414634146 | 0.6829268292682927 | 0.04878048780487809 |
| BPO | 0.36363636363636365 | 0.0 | 0.0 | 0.0 | 0.0 |
| BUSINESS-DEVELOPMENT | 0.9583333333333334 | 0.49122807017543857 | 0.5517241379310345 | 0.7857142857142857 | 0.23399014778325122 |
| CHEF | 0.7727272727272727 | 0.7441860465116279 | 0.7727272727272727 | 0.8444444444444444 | 0.07171717171717173 |
| CONSTRUCTION | 0.9545454545454546 | 0.6666666666666666 | 0.6956521739130435 | 0.8095238095238095 | 0.11387163561076608 |
| **CONSULTANT** | 0.7391304347826086 | 0.09523809523809523 | 0.05405405405405406 | 0.5777777777777777 | 0.5237237237237237 |
| **DESIGNER** | 0.9302325581395349 | 0.7027027027027027 | 0.5789473684210527 | 0.8292682926829268 | 0.25032092426187413 |
| **DIGITAL-MEDIA** | 0.7878787878787878 | 0.5945945945945946 | 0.6486486486486487 | 0.7058823529411765 | 0.05723370429252783 |
| **ENGINEERING** | 0.875 | 0.6382978723404256 | 0.6363636363636364 | 0.782608695652174 | 0.1462450592885376 |
| FINANCE | 0.8 | 0.8301886792452831 | 0.5333333333333333 | 0.6666666666666666 | 0.1333333333333333 |
| FITNESS | 0.88 | 0.7083333333333334 | 0.6666666666666666 | 0.8333333333333334 | 0.16666666666666674 |
| HEALTHCARE | 0.6976744186046512 | 0.46808510638297873 | 0.41509433962264153 | 0.6808510638297872 | 0.2657567242071457 |
| HR | 0.875 | 0.8260869565217391 | 0.8163265306122449 | 0.8936170212765957 | 0.07729049066435079 |
| **INFORMATION-TECHNOLOGY** | 0.9019607843137255 | 0.7241379310344828 | 0.7142857142857143 | 0.8214285714285714 | 0.1071428571428571 |
| PUBLIC-RELATIONS | 0.7555555555555555 | 0.5957446808510638 | 0.5833333333333334 | 0.723404255319149 | 0.1400709219858156 |
| SALES | 0.6909090909090909 | 0.5098039215686274 | 0.44 | 0.6071428571428571 | 0.1671428571428571 |
| TEACHER | 0.8571428571428571 | 0.6976744186046512 | 0.5909090909090909 | 0.6818181818181818 | 0.09090909090909083 |

v3 minus production v2: test macro F1 0.16019157615167745; calibrated CV macro F1 0.16957573762202038.

**Not promoted.** Production files remain unchanged pending the user's review. Model settings were selected from training CV using the predeclared standard-deviation gate; test scores did not choose settings or trigger promotion. CV improvement gate passed: True. The candidate is not proven leakage-free: capping body counts limits repetition, but even one explicit label mention remains predictive. Higher scores cannot be attributed exclusively to better content understanding or technical-token preservation without further training-only ablations.

The training-position audit found 2159/14206 matches within the first twenty cleaned tokens (fraction 0.15197803744896524); 12047/14206 were later (fraction 0.8480219625510348). 1332/1750 positive documents had an early first mention. K=13 is the ceiling of the training first-match end upper quartile. All-category prefix masking is inference-safe; body label-containing features are capped at 2 raw counts before sublinear TF-IDF. The cap is per unigram/bigram feature, not a collective cap across different contextual bigrams. Technical shapes such as C++, C#, .NET, Node.js, CI/CD and alphanumeric versions survive via placeholder-and-restore; generic hyphenated words can also survive.

Residual own-label presence after prefix masking: 1739/1984 training resumes (fraction 0.8765120967741935). Frequency capping does not remove these occurrences. This is concrete evidence against claiming that v3 eliminates the known lexical shortcut.

The first three screens compare balanced LR, RF and LinearSVC, then eight sampled combinations are scored for each of the two highest-mean families. Every score is five-fold training CV. Settings and K are selected using this training partition; subsequent nested calibration CV is conditional on those choices, not a fully nested search estimate. Fold standard deviation is a heuristic gate, not a significance test. The repeatedly inspected holdout is not a pristine external benchmark.

Selected trial: V3-2; family: random_forest. Fixed sigmoid calibration uses nested training-only OOF responses, and the final calibrator uses full-training OOF responses. No new external data or dependencies were added. If LinearSVC is chosen, legacy raw confidence in the isolated candidate API is explicitly a softmax-margin proxy, not a native probability. This would require an API semantic compatibility decision before production promotion; calibrated probabilities are fitted directly to decision margins, not this proxy.


| Metric | Raw candidate | Sigmoid candidate |
|---|---|---|
| macro_f1 | 0.6327871835984821 | 0.6803971791286144 |
| accuracy | 0.6901408450704225 | 0.7203219315895373 |
| brier_score | 0.5721394366197183 | 0.398177476644112 |
| log_loss | 1.4014854652267514 | 0.9951252925222299 |
| ece_10_bins | 0.3333400402414487 | 0.10643136970178817 |

Training-only thresholds: T1=0.4, T2=0.1. Held-out coverage 0.7806841046277666 (388/497), accuracy among answered 0.8221649484536082. Short inputs below fifty whitespace words remain uncertain. The original utility, grid and tie-breaks were reused.

## Fixed Java/backend demo

The actual /predict call returned CONSULTANT with confidence 0.26396082782574387; uncertainty reason: low_confidence. IT rank within calibrated top three: None. The intended IT headline was not achieved. The input was fixed before results and was not rewritten to get a favorable prediction.

Preserved technical spellings do not guarantee inclusion in the learned vocabulary after document-frequency and feature-budget filtering. Fixed demo terms absent from that vocabulary: spring boot, node.js, ci/cd, docker, kafka, junit, oauth, jwt, postgresql, redis. This is observed feature coverage, not a complete causal explanation of the prediction. No model or sample adjustments were made in response.

```json
{
  "predicted_category": "SALES",
  "confidence": 0.115,
  "calibrated_predicted_category": "CONSULTANT",
  "calibrated_confidence": 0.26396082782574387,
  "calibrated_top_predictions": [
    {
      "category": "CONSULTANT",
      "probability": 0.26396082782574387
    },
    {
      "category": "SALES",
      "probability": 0.15210478749713724
    },
    {
      "category": "ARTS",
      "probability": 0.0800658479839846
    }
  ],
  "is_uncertain": true,
  "uncertainty_reason": "low_confidence",
  "top_predictions": [
    {
      "category": "SALES",
      "probability": 0.115
    },
    {
      "category": "CONSULTANT",
      "probability": 0.105
    },
    {
      "category": "INFORMATION-TECHNOLOGY",
      "probability": 0.1
    }
  ],
  "top_terms": [
    "service",
    "server",
    "wrote",
    "building",
    "sql",
    "database",
    "using",
    "maintained"
  ],
  "text_stats": {
    "word_count": 85,
    "short_input": false
  },
  "source": "text",
  "text_preview": null
}
```
