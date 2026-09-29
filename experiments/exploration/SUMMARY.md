# Exploration summary

All values are freshly computed in each candidate results.json. Ranking uses raw five-fold training CV macro F1 only; final test metrics use sigmoid calibration. Only the frozen top two were evaluated on test. No production files were changed.

| Rank | Candidate | CV macro F1 (mean ± std) | Calibrated test macro F1 | Test accuracy | Honest observation |
|---|---|---|---|---|---|
| 1 | word_svc | 0.5187029137588132 ± 0.02127154152059278 | 0.5178233530804878 | 0.5593561368209256 | CV mean minus fresh control: 0.05722462958385888; improves the point estimate. Fold std is not a significance test. |
| 2 | word_char_svc | 0.5124825586834483 ± 0.023887983042986304 | 0.5191586134385697 | 0.5633802816901409 | CV mean minus fresh control: 0.051004274508493985; improves the point estimate. Fold std is not a significance test. Combined-view delta versus better single SVC view: -0.006220355075364892. |
| 3 | char_svc | 0.49565674512444124 ± 0.021827142943292903 | Not evaluated | Not evaluated | CV mean minus fresh control: 0.034178460949486955; improves the point estimate. Fold std is not a significance test. |
| 4 | word_lr | 0.4936969400812468 ± 0.024615376416498184 | Not evaluated | Not evaluated | CV mean minus fresh control: 0.032218655906292526; improves the point estimate. Fold std is not a significance test. |
| 5 | balanced_rf | 0.47324222748900946 ± 0.011746546677539016 | Not evaluated | Not evaluated | CV mean minus fresh control: 0.01176394331405517; improves the point estimate. Fold std is not a significance test. Balancing delta versus otherwise identical E2: 0.004432551067943269. |
| 6 | technical_rf | 0.4688096764210662 ± 0.026459834704900573 | Not evaluated | Not evaluated | CV mean minus fresh control: 0.007331392246111901; improves the point estimate. Fold std is not a significance test. Technical-token preservation alone helped. |
| 7 | control_rf | 0.4614782841749543 ± 0.02383798966098138 | Not evaluated | Not evaluated | Fresh control. |

## Reasoning recorded before fitting

- **Fresh production-style RF control:** Recompute the existing universal-mask raw RF five-fold score on exactly the saved training rows; anchor all comparisons in fresh measurements. [Rationale](control_rf/rationale.json)
- **Universal mask plus technical tokens, RF:** Keep RF and TF-IDF settings fixed; change only technical-preserving cleaning/tokenization. Expect useful punctuation-bearing skills to survive, but sparse training coverage may prevent improvement. [Rationale](technical_rf/rationale.json)
- **Technical RF with class balancing:** Change only class_weight from E2 to balanced. Expect better minority recall; majority precision may decline. [Rationale](balanced_rf/rationale.json)
- **Technical word/bigram balanced Logistic Regression:** Sparse linear decision boundaries and broader vocabulary may generalize better than trees after label removal. Fixed C=1, sublinear TF, min_df=2, max_df=.95, max_features=20000. [Rationale](word_lr/rationale.json)
- **Technical word/bigram balanced LinearSVC:** Use the same features as E4 with a margin-based classifier, fixed C=1. May help overlapping classes; native probabilities are unavailable. [Rationale](word_svc/rationale.json)
- **Masked character n-grams with balanced LinearSVC:** Character 3-5 grams within word boundaries may capture spelling/morphology and technical forms despite sparse exact tokens. Input stays universally masked; character fragments can still encode source/template bias. [Rationale](char_svc/rationale.json)
- **Combined masked word and character features:** Complement explicit skill phrases with character robustness using equal-weight concatenated L2-normalized feature blocks. Extra complexity is worthwhile only if CV improves over both single views. [Rationale](word_char_svc/rationale.json)

## Finalist calibration and abstention

| Candidate | Calibrated CV F1 mean ± std | Brier | Log loss | ECE | T1 / T2 | Test coverage | Accuracy answered |
|---|---|---|---|---|---|---|---|
| word_svc | 0.5213417576233275 ± 0.01659900364619594 | 0.5620131939650492 | 1.4438181128698713 | 0.10154195049289037 | 0.4 / 0.05 | 0.6237424547283702 | 0.7290322580645161 |
| word_char_svc | 0.5144394483063955 ± 0.020389398001687757 | 0.5755263226351806 | 1.491123066605034 | 0.0826720220875122 | 0.4 / 0.05 | 0.6257545271629779 | 0.7106109324758842 |

## Recommendation and limits

Training-CV preference: **word_svc**. Its mean minus the fresh control is 0.05722462958385888. This preference was fixed before test reporting. No promotion is performed; wait for the user's choice.
Practical recommendation frozen before test access: **Consider candidate after user review**; preferred candidate **word_svc**. Highest training CV mean. CV gain 0.05722462958385888; declared variability gate 0.02383798966098138. This is a heuristic, not a significance test.
Universal masking remains in every candidate, including the input to character features. This avoids deliberately restoring body label words, but cannot remove every correlated skill, template or source cue. Character fragments may encode residual source bias.
The masks are fixed from existing policy; all feature vocabularies and IDF are fitted within each training fold. Nested calibration uses five outer and three inner training folds. Calibrated CV is conditional on training-selected candidate settings and is not a fully nested search estimate. Fold standard deviation is not a confidence interval.
The same test split has been inspected historically, so it is not a pristine external benchmark. No external data or pretrained weights were downloaded. No extra calibration methods, per-class thresholds or ensembling were tried in this bounded run; this is not an exhaustive search.
LinearSVC has no native probabilities. These artifacts are offline candidates, not API drop-ins: any later integration must preserve the existing probability-field semantics and explanation honesty. Production API code, privacy handling and ethical framing remain unchanged.
Training mask audit: 0 residual dictionary documents and 0 empty documents among 1984 training rows. This verifies the explicit dictionary only, not all possible proxies.

## Verification

Backend: 126 passed, 0 failed, 0 errors, 0 skipped. Frontend: 46 passed, 0 failed, 0 errors, 0 skipped. Counts come from the saved full-suite reports. Production models, API code, privacy handling, and previous experiments are unchanged. Nothing was promoted.
