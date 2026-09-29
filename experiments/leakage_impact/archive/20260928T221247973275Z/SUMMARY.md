# Label-shortcut impact diagnostic

All numbers below come from [results.json](results.json).

| System | CV macro F1 (mean ± std) | Test macro F1 | Test accuracy |
|---|---|---|---|
| Frozen baseline v1 (RF + sigmoid) | 0.710961947058129 ± 0.014977244812944244 | 0.7783811932565733 | 0.8048289738430584 |
| Own-label masked train and test (RF + sigmoid) | 0.5317317225299487 ± 0.009913121069946657 | 0.5580907457690872 | 0.5955734406438632 |

Masked minus baseline: test macro F1 -0.22029044748748605; accuracy -0.2092555331991952.

Of 400 baseline-correct test predictions, 120 became incorrect after masked retraining (fraction 0.3); 16 previously incorrect predictions became correct.
Own-label variants were available in 449/497 test resumes (fraction 0.903420523138833) versus 2199/2484 raw resumes (fraction 0.8852657004830918). The test subset and raw corpus have different denominators; the saved split excludes cleaned empty/duplicate rows and samples within categories.

| Category (largest drop first) | Support | Baseline F1 | Masked F1 | Delta |
|---|---|---|---|---|
| CONSULTANT | 23.0 | 0.7391304347826086 | 0.09523809523809523 | -0.6438923395445134 |
| AGRICULTURE | 13.0 | 0.8181818181818182 | 0.3 | -0.5181818181818183 |
| BUSINESS-DEVELOPMENT | 24.0 | 0.9583333333333334 | 0.49122807017543857 | -0.4671052631578948 |
| ADVOCATE | 24.0 | 0.8235294117647058 | 0.41379310344827586 | -0.40973630831643 |
| BPO | 4.0 | 0.36363636363636365 | 0.0 | -0.36363636363636365 |
| APPAREL | 19.0 | 0.6896551724137931 | 0.36363636363636365 | -0.3260188087774295 |
| CONSTRUCTION | 22.0 | 0.9545454545454546 | 0.6666666666666666 | -0.28787878787878796 |
| ENGINEERING | 24.0 | 0.875 | 0.6382978723404256 | -0.23670212765957444 |
| HEALTHCARE | 23.0 | 0.6976744186046512 | 0.46808510638297873 | -0.22958931222167245 |
| DESIGNER | 21.0 | 0.9302325581395349 | 0.7027027027027027 | -0.22752985543683213 |
| DIGITAL-MEDIA | 19.0 | 0.7878787878787878 | 0.5945945945945946 | -0.19328419328419322 |
| SALES | 23.0 | 0.6909090909090909 | 0.5098039215686274 | -0.1811051693404635 |
| INFORMATION-TECHNOLOGY | 24.0 | 0.9019607843137255 | 0.7241379310344828 | -0.17782285327924274 |
| BANKING | 23.0 | 0.7272727272727273 | 0.5531914893617021 | -0.17408123791102514 |
| FITNESS | 24.0 | 0.88 | 0.7083333333333334 | -0.17166666666666663 |
| PUBLIC-RELATIONS | 22.0 | 0.7555555555555555 | 0.5957446808510638 | -0.15981087470449173 |
| TEACHER | 20.0 | 0.8571428571428571 | 0.6976744186046512 | -0.15946843853820591 |
| ARTS | 21.0 | 0.6153846153846154 | 0.5142857142857142 | -0.10109890109890118 |
| AVIATION | 23.0 | 0.8372093023255814 | 0.7391304347826086 | -0.09807886754297279 |
| ACCOUNTANT | 24.0 | 0.8301886792452831 | 0.7727272727272727 | -0.05746140651801035 |
| AUTOMOBILE | 7.0 | 0.5 | 0.4444444444444444 | -0.05555555555555558 |
| HR | 22.0 | 0.875 | 0.8260869565217391 | -0.048913043478260865 |
| CHEF | 24.0 | 0.7727272727272727 | 0.7441860465116279 | -0.02854122621564481 |
| FINANCE | 24.0 | 0.8 | 0.8301886792452831 | 0.030188679245283012 |

## Interpretation and limits

Removing the recorded own-label variants changes test macro F1 from 0.7783811932565733 to 0.5580907457690872. The remaining accuracy is 0.5955734406438632. This measures sensitivity to the listed vocabulary, not a causal partition into genuine content versus lookup.
A binary 'mostly genuine' versus 'mostly label lookup' conclusion is not identified by this experiment alone. The category-specific deletion uses ground truth, can create label-dependent missing-word patterns, and removes useful domain content as well as titles (e.g. financial, design, accounting). Other label variants, templates and near-duplicates remain. Consequently this is neither a clean content-only benchmark nor an honest performance ceiling.
The three largest F1 decreases are CONSULTANT (-0.6438923395445134), AGRICULTURE (-0.5181818181818183), BUSINESS-DEVELOPMENT (-0.4671052631578948). These are the most sensitive classes in this diagnostic, not proof that their original predictions only read labels.
Supplemental evaluation-only masking of the frozen model gives macro F1 0.3891695620364666 and accuracy 0.4024144869215292; 200/400 original correct answers become wrong (fraction 0.5). This includes distribution shift and must not be confused with the retrained comparison above.
The same frozen TF-IDF/RF hyperparameters, seed, split, outer CV and nested calibration folds were used; sigmoid was fixed from baseline, not selected anew. TF-IDF was fitted independently inside every training fold. All-row metrics are before abstention, with no threshold tuning. Baseline scores were reproduced from frozen artifacts. No production artifacts, preprocessing, API, or existing tests were changed; no permanent fix is recommended or selected here.
