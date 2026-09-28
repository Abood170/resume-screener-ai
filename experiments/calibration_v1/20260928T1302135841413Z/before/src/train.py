"""Run leakage-safe CV, select on training folds, then evaluate once on test."""

import json
import platform
import time
import importlib.metadata
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from src.data import ROOT, sha256
from src.preprocess import clean_text
from src.evaluate import evaluate, write_results

SEED = 42


def main() -> None:
    data = pd.read_csv(ROOT / "data/resumes.csv")
    data["clean"] = data.Resume.map(clean_text)
    empty = data.clean.eq("")
    conflicts = data.groupby("clean").Category.nunique()
    ambiguous = set(conflicts[conflicts > 1].index)
    keep = data.loc[~empty & ~data.clean.isin(ambiguous)].copy()
    before = len(keep)
    keep = keep.drop_duplicates("clean").copy()
    train, test = train_test_split(
        keep.index.to_numpy(), test_size=0.2, random_state=SEED, stratify=keep.Category
    )
    x_train, y_train = keep.loc[train, "Resume"], keep.loc[train, "Category"]
    x_test, y_test = keep.loc[test, "Resume"], keep.loc[test, "Category"]
    classes = sorted(keep.Category.unique().tolist())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    estimators = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "multinomial_nb": MultinomialNB(alpha=1.0),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=SEED, n_jobs=2
        ),
    }
    fitted, comparison = {}, {}
    for name, estimator in estimators.items():
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        preprocessor=clean_text,
                        lowercase=False,
                        max_features=5000,
                        ngram_range=(1, 2),
                    ),
                ),
                ("classifier", estimator),
            ]
        )
        start = time.perf_counter()
        scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
            error_score="raise",
        )
        cv_seconds = time.perf_counter() - start
        start = time.perf_counter()
        pipeline.fit(x_train, y_train)
        fitted[name] = pipeline
        comparison[name] = {
            "cv_macro_f1_folds": scores.tolist(),
            "cv_macro_f1_mean": float(scores.mean()),
            "cv_macro_f1_std": float(scores.std()),
            "cv_seconds": cv_seconds,
            "fit_seconds": time.perf_counter() - start,
        }
        print(name, json.dumps(comparison[name]), flush=True)
    # Selection is frozen BEFORE any held-out metrics are computed.
    winner = max(comparison, key=lambda n: comparison[n]["cv_macro_f1_mean"])
    figure_dir = ROOT / "reports/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, pipeline in fitted.items():
        comparison[name].update(
            evaluate(pipeline, x_test, y_test, classes, figure_dir, name)
        )
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(fitted[winner]["classifier"], model_dir / "model.joblib")
    joblib.dump(fitted[winner]["tfidf"], model_dir / "vectorizer.joblib")
    manifest = {
        "selected_model": winner,
        "classes": classes,
        "dataset_sha256": sha256(ROOT / "data/resumes.csv"),
        "artifacts": {p.name: sha256(p) for p in model_dir.glob("*.joblib")},
    }
    (model_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (ROOT / "reports/split.json").write_text(
        json.dumps(
            {"train_row_indices": train.tolist(), "test_row_indices": test.tolist()},
            indent=2,
        ),
        encoding="utf-8",
    )
    results = {
        "dataset": json.loads((ROOT / "data/provenance.json").read_text()),
        "cleaning": {
            "empty_rows_removed": int(empty.sum()),
            "conflicting_text_groups_removed": len(ambiguous),
            "conflicting_rows_removed": int(data.clean.isin(ambiguous).sum()),
            "duplicate_rows_removed": before - len(keep),
            "retained_rows": len(keep),
        },
        "eda": json.loads((ROOT / "reports/eda.json").read_text()),
        "methodology": {
            "seed": SEED,
            "test_fraction": 0.2,
            "cv_folds": 5,
            "max_features": 5000,
            "ngram_range": [1, 2],
            "train_rows": len(train),
            "test_rows": len(test),
            "selection_metric": "training CV macro F1",
            "refit_on_test": False,
            "train_category_counts": y_train.value_counts().sort_index().to_dict(),
            "test_category_counts": y_test.value_counts().sort_index().to_dict(),
        },
        "classes": classes,
        "models": comparison,
        "selected_model": winner,
        "artifacts": manifest,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {
                p: importlib.metadata.version(p)
                for p in ["scikit-learn", "numpy", "pandas", "nltk", "joblib"]
            },
        },
    }
    write_results(results, ROOT / "results.json")
    print("Selected:", winner, "train:", len(train), "test:", len(test), flush=True)


if __name__ == "__main__":
    main()
