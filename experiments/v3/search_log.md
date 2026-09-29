# Full v3 search log

No held-out scores were computed for individual search trials. Acceptance means retained as the incumbent during search, not production promotion. Final selected trial: V3-2.

| ID | Stage / family | Parameters | CV macro F1 (mean ± std) | Accepted by gate? |
|---|---|---|---|---|
| V3-1 | family_screen / logistic_regression | {} | 0.5379772586292838 ± 0.01563035063836193 | True |
| V3-2 | family_screen / random_forest | {} | 0.6089439014893414 ± 0.0154157434862607 | True |
| V3-3 | family_screen / linear_svc | {} | 0.5876247295748038 ± 0.02490672877849458 | False |
| V3-4 | randomized_search / random_forest | {"classifier__max_depth": 30, "classifier__min_samples_leaf": 1, "tfidf__max_df": 0.9, "tfidf__max_features": 10000, "tfidf__min_df": 3} | 0.5974385504651646 ± 0.022209375478586108 | False |
| V3-5 | randomized_search / random_forest | {"classifier__max_depth": 30, "classifier__min_samples_leaf": 2, "tfidf__max_df": 0.9, "tfidf__max_features": 20000, "tfidf__min_df": 2} | 0.5990794772803449 ± 0.021857708076941453 | False |
| V3-6 | randomized_search / random_forest | {"classifier__max_depth": 30, "classifier__min_samples_leaf": 1, "tfidf__max_df": 0.9, "tfidf__max_features": 10000, "tfidf__min_df": 2} | 0.6151059008733112 ± 0.02859508580488019 | False |
| V3-7 | randomized_search / random_forest | {"classifier__max_depth": 30, "classifier__min_samples_leaf": 2, "tfidf__max_df": 0.95, "tfidf__max_features": 5000, "tfidf__min_df": 3} | 0.6124279816555676 ± 0.019472818367348797 | False |
| V3-8 | randomized_search / random_forest | {"classifier__max_depth": 30, "classifier__min_samples_leaf": 1, "tfidf__max_df": 0.9, "tfidf__max_features": 5000, "tfidf__min_df": 2} | 0.5969134138930842 ± 0.02828358867040845 | False |
| V3-9 | randomized_search / random_forest | {"classifier__max_depth": 30, "classifier__min_samples_leaf": 2, "tfidf__max_df": 0.9, "tfidf__max_features": 5000, "tfidf__min_df": 3} | 0.607025561589161 ± 0.016555827546976293 | False |
| V3-10 | randomized_search / random_forest | {"classifier__max_depth": null, "classifier__min_samples_leaf": 2, "tfidf__max_df": 0.9, "tfidf__max_features": 5000, "tfidf__min_df": 2} | 0.6076410229575248 ± 0.020333002413534994 | False |
| V3-11 | randomized_search / random_forest | {"classifier__max_depth": null, "classifier__min_samples_leaf": 2, "tfidf__max_df": 0.95, "tfidf__max_features": 5000, "tfidf__min_df": 3} | 0.6039615159538217 ± 0.020218387077669323 | False |
| V3-12 | randomized_search / linear_svc | {"classifier__C": 3.0, "tfidf__max_df": 0.95, "tfidf__max_features": 20000, "tfidf__min_df": 3} | 0.5834928550297416 ± 0.02271494433116875 | False |
| V3-13 | randomized_search / linear_svc | {"classifier__C": 1.0, "tfidf__max_df": 0.9, "tfidf__max_features": 5000, "tfidf__min_df": 3} | 0.585901824166778 ± 0.02748242300394449 | False |
| V3-14 | randomized_search / linear_svc | {"classifier__C": 3.0, "tfidf__max_df": 0.9, "tfidf__max_features": 10000, "tfidf__min_df": 2} | 0.5890533467903338 ± 0.023575326091077202 | False |
| V3-15 | randomized_search / linear_svc | {"classifier__C": 3.0, "tfidf__max_df": 0.95, "tfidf__max_features": 5000, "tfidf__min_df": 2} | 0.5881539725228419 ± 0.023772597069991012 | False |
| V3-16 | randomized_search / linear_svc | {"classifier__C": 1.0, "tfidf__max_df": 0.9, "tfidf__max_features": 20000, "tfidf__min_df": 2} | 0.5825735945155708 ± 0.02343373431702715 | False |
| V3-17 | randomized_search / linear_svc | {"classifier__C": 3.0, "tfidf__max_df": 0.95, "tfidf__max_features": 5000, "tfidf__min_df": 3} | 0.586808081432828 ± 0.0245988082485922 | False |
| V3-18 | randomized_search / linear_svc | {"classifier__C": 1.0, "tfidf__max_df": 0.95, "tfidf__max_features": 10000, "tfidf__min_df": 3} | 0.5867904561044599 ± 0.022659524207840294 | False |
| V3-19 | randomized_search / linear_svc | {"classifier__C": 1.0, "tfidf__max_df": 0.9, "tfidf__max_features": 5000, "tfidf__min_df": 2} | 0.5869928188022513 ± 0.02457614745143372 | False |
