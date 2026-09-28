import json
import shutil
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from src.data import ROOT
from src.predict import Predictor, InvalidResumeError
from src.preprocess import clean_text


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


def test_prediction_matches_artifacts(predictor):
    text = "Software engineer developing Python applications and SQL databases"
    result = predictor.predict(text)
    probabilities = predictor.model.predict_proba(
        predictor.vectorizer.transform([text])
    )[0]
    assert (
        result["predicted_category"]
        == predictor.model.classes_[np.argmax(probabilities)]
    )
    assert result["confidence"] == float(max(probabilities))
    assert 0 <= result["confidence"] <= 1
    assert result == predictor.predict(text)


@pytest.mark.parametrize(
    "text",
    ["", "   ", "!!!", "the and", "zzqxzzqxzzqx", "a" * 50001],
    ids=["empty", "whitespace", "punctuation", "stopwords", "unknown", "oversized"],
)
def test_invalid_text(predictor, text):
    with pytest.raises(InvalidResumeError):
        predictor.predict(text)


def test_non_string(predictor):
    with pytest.raises(TypeError):
        predictor.predict(123)


def test_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        Predictor(tmp_path)


def test_corrupt_artifacts(tmp_path):
    for name in ("manifest.json", "model.joblib", "vectorizer.joblib"):
        shutil.copy(ROOT / "models" / name, tmp_path / name)
    with (tmp_path / "model.joblib").open("ab") as f:
        f.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        Predictor(tmp_path)


def test_saved_model_reproduces_holdout(predictor):
    data = pd.read_csv(ROOT / "data/resumes.csv")
    split = json.loads((ROOT / "reports/split.json").read_text())
    results = json.loads((ROOT / "results.json").read_text())
    train = data.loc[split["train_row_indices"]]
    test = data.loc[split["test_row_indices"]]
    assert set(train.Resume.map(clean_text)).isdisjoint(test.Resume.map(clean_text))
    predictions = predictor.model.predict(predictor.vectorizer.transform(test.Resume))
    score = f1_score(test.Category, predictions, average="macro")
    assert score == results["models"][results["selected_model"]]["macro_f1"]
