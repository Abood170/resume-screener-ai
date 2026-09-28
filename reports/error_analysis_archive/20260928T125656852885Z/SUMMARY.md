# Baseline error analysis

Analysis only: the saved Random Forest, preprocessing, API and split are unchanged. The prior freeze stopped because this is not a Git repository; no `baseline-v1` tag or frozen archive exists. This audit identifies the current artifacts by hashes in [analysis_results.json](analysis_results.json).

## Scope and reproduced baseline

The existing split contains 1984 training and 497 test resumes. Its confusion matrix reproduces the recorded baseline exactly. Test accuracy: 0.744466800804829; macro F1: 0.6803518093463513. Training CV macro F1: 0.6577281212720567 ± 0.01833295124954121. Test minus CV: 0.0226236880742946. This is a descriptive gap, not evidence of significance or improvement; CV models also use smaller fitting partitions. No model was fitted or selected. Source: [cv_vs_test_gap.csv](cv_vs_test_gap.csv).

These held-out diagnostics are final descriptive reporting. Recommendations below are hypotheses to investigate on training-only out-of-fold predictions, not settings chosen from test performance. Repeatedly consulting this holdout would compromise its independence for future improvement claims.

## Weakest classes

| Category | Train support | Test support | Precision | Recall | F1 |
|---|---|---|---|---|---|
| BPO | 18 | 4 | 0.0 | 0.0 | 0.0 |
| AUTOMOBILE | 29 | 7 | 1.0 | 0.14285714285714285 | 0.25 |
| ARTS | 82 | 21 | 0.625 | 0.23809523809523808 | 0.3448275862068966 |

Training support and class F1 have Pearson correlation 0.8436961196509216 and Spearman correlation 0.5036140765301462. Small classes are among the weakest, but ARTS shows that support alone is insufficient. These are descriptive associations across categories, not causal effects. Sources: [per_class_report.csv](per_class_report.csv), [support plot](class_support_vs_f1.png).

## Most common confusions

| True -> predicted | Count | True-class support | Fraction of true class |
|---|---|---|---|
| ARTS -> TEACHER | 9 | 21 | 0.42857142857142855 |
| FINANCE -> ACCOUNTANT | 8 | 24 | 0.3333333333333333 |
| APPAREL -> SALES | 3 | 19 | 0.15789473684210525 |

Third place is tied at 3: APPAREL -> SALES; CONSULTANT -> ACCOUNTANT; CONSULTANT -> PUBLIC-RELATIONS. Ties are ordered alphabetically, not selectively. The requested top-pair file has a fixed cutoff; additional tied pairs at its boundary can be omitted. Sources: [top_confusions.csv](top_confusions.csv), [normalized matrix](confusion_matrix_normalized.png). Plot annotations are truncated to two decimals; [matrix CSV](confusion_matrix_normalized.csv) preserves the underlying values.

## Leakage and shortcuts

- **Label wording:** own-label variants appear in 2199/2484 raw resumes (share 0.8852657004830918); 20/24 categories have shares at least one half. In the first eighty words, matches occur in 1753/2484 (share 0.7057165861513688). This benchmark partly permits job-title lookup. These frequencies do not establish how much the model relies on it, and a label mention can describe education or past work rather than the target occupation.
- **Literal-match caveat:** [label_shortcuts.csv](label_shortcuts.csv) reports every category separately for raw, training and test data. Exact case-insensitive substring matching can count `arts` inside `parts`, or miss space-separated versions of hyphenated labels. The separate boundary-aware variant measure normalizes punctuation/whitespace and uses the explicit [variant dictionary](label_variants.json). Variants can still be broad; these are lexical indicators, not human annotations of job titles.
- **Near duplicates:** using the saved training-fitted TF-IDF space, 4/497 test resumes have nearest-training cosine similarity strictly above 0.9. Accuracy is 1.0 (4/4) versus 0.742393509127789 (366/493) for the remainder. All 4 high-similarity matches share the training neighbor's category. The higher accuracy is descriptive and the group is small; similarity in a reduced vocabulary is not proof of a duplicate or a causal estimate of score inflation. No rows were removed. Source: [near_duplicate_summary.csv](near_duplicate_summary.csv), [nearest neighbors](nearest_train_similarity.csv).
- **Exact overlap:** 0 test resumes share an exact preprocessing-normalized text hash with training. This does not exclude template, author or source overlap. The overall macro F1 did not cross the requested suspicious-gain threshold, and no improvement is claimed.

## Structured review of misclassified examples

Whole documents were read in memory by fixed lexical/structure checks; only sanitized counts and dataset row IDs are retained. This is a structured evidence review, not a human semantic annotation or causal attribution. Up to five errors per class were sampled without replacement using seed 42. BPO has fewer than five test examples, so all available errors were inspected. Sources and indicator definitions: [misclassification_review.json](misclassification_review.json).

- **BPO**: inspected 4 of 4 errors (4 test resumes); word counts 645–1278. Own-label variants occur in 2/4; predicted-label variants in 4/4. Domain indicators: customer operations 4/4, sales business 4/4, technology 4/4, finance accounting 4/4, healthcare 1/4. Predictions: CONSULTANT: 1, FINANCE: 1, HEALTHCARE: 1, SALES: 1. Under-50-word inputs: 0; fewer than two recognized section markers: 0.
- **AUTOMOBILE**: inspected 5 of 6 errors (7 test resumes); word counts 278–1001. Own-label variants occur in 2/5; predicted-label variants in 5/5. Domain indicators: customer operations 4/5, sales business 2/5, technology 4/5, education 2/5, finance accounting 3/5, engineering automotive 3/5, healthcare 1/5. Predictions: ACCOUNTANT: 1, ARTS: 1, AVIATION: 1, BANKING: 1, ENGINEERING: 1. Under-50-word inputs: 0; fewer than two recognized section markers: 0.
- **ARTS**: inspected 5 of 16 errors (21 test resumes); word counts 244–941. Own-label variants occur in 3/5; predicted-label variants in 4/5. Domain indicators: customer operations 3/5, sales business 3/5, technology 3/5, creative design 2/5, education 1/5, finance accounting 1/5, engineering automotive 2/5, healthcare 1/5. Predictions: HEALTHCARE: 1, INFORMATION-TECHNOLOGY: 1, PUBLIC-RELATIONS: 1, SALES: 1, TEACHER: 1. Under-50-word inputs: 0; fewer than two recognized section markers: 0.

Multiple domain signals and predicted-label wording support overlapping vocabulary as a plausible source of confusion. The inspected errors provide no evidence that very short text is the explanation. Section markers do not show obvious missing structure, but cannot rule out unusual formatting or distinguish real employment from education/history mentions. No resume excerpts or personal data are published.

## Next investigations — training data only

1. **Compare unweighted and class-weighted baselines within training CV**, keeping the existing test split untouched. The low-support BPO/AUTOMOBILE failures motivate the comparison; ARTS prevents assuming imbalance explains every weakness. Inspect per-class out-of-fold recall as well as macro F1. [Evidence](per_class_report.csv)
2. **Run a predeclared title/label-masking diagnostic inside training CV.** Compare with unchanged text to measure dependence on explicit title wording. Do not adopt masking from this held-out analysis or claim it must improve accuracy. [Evidence](label_shortcuts.csv)
3. **Audit training-only near-duplicate/template clusters and use grouped training CV as a sensitivity check.** Keep the held-out split fixed and report the different CV protocol separately. Do not remove difficult or similar test rows. [Evidence](near_duplicate_summary.csv)
4. **Review training labels and an annotation rubric for overlapping occupations**, especially ARTS/TEACHER and FINANCE/ACCOUNTANT, before changing label definitions. Inspect analogous training examples rather than relabeling test errors. [Evidence](top_confusions.csv)
5. **Stratify training out-of-fold errors by label presence, domain indicators and length before changing text cleaning.** The sampled errors contain substantial text and mixed-domain wording, so a short-input or formatting fix is not supported by this review. Any feature/preprocessing experiment must earn adoption through training CV. [Evidence](misclassification_review.json)

No fixes, new dependencies or external downloads were made. Any later data acquisition still requires approval.

## Verification and reproducibility

Backend: 63 tests, 63 passed, 0 failed, 0 errors, 0 skipped. Frontend: 26 tests, 26 passed, 0 failed, 0 skipped; runner success: True. See [verification.json](verification.json), [pytest.xml](pytest.xml) and [frontend_tests.json](frontend_tests.json).

Run `python -m src.error_analysis` to regenerate the analysis. Existing report directories are copied to timestamped `reports/error_analysis_archive/` before replacement, and an existing experiment log is archived before notes are appended. `python -m src.error_analysis --render-only` regenerates this summary from saved results. Original `results.json`, artifacts and split are never written. All numeric findings above are rendered from generated result files.
