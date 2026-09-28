"""Calibration contract and probability math, without held-out tuning."""

import json
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.calibration import probability_metrics, reliability_bins, uncertainty
from src.data import ROOT
from src.predict import Predictor


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


def test_calibrator_artifact_loads_and_normalizes(predictor):
    raw = np.full((3, len(predictor.model.classes_)), 1 / len(predictor.model.classes_))
    raw[1] = 0
    raw[1, 0] = 1
    raw[2] = 0
    raw[2, :2] = 0.5
    probabilities = predictor.calibrator.predict_proba(raw)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "probabilities,words,t1,t2,expected",
    [
        ([0.9, 0.07, 0.03], 100, 0.5, 0.1, None),
        ([0.5, 0.5, 0], 100, 0.4, 0.1, "small_margin"),
        ([0.4, 0.35, 0.25], 100, 0.5, 0.1, "low_confidence"),
        ([0.52, 0.47, 0.01], 100, 0.5, 0.1, "small_margin"),
        ([0.99, 0.01, 0], 49, 0.5, 0.1, "short_input"),
        ([0.4, 0.35, 0.25], 5, 0.5, 0.1, "short_input"),
        ([0.75, 0.25, 0], 50, 0.75, 0.5, None),
    ],
    ids=[
        "clear",
        "tie",
        "low-top",
        "narrow-margin",
        "short",
        "short-priority",
        "strict-boundaries",
    ],
)
def test_uncertainty_policy(probabilities, words, t1, t2, expected):
    assert uncertainty(probabilities, words, t1, t2) == expected


@pytest.mark.parametrize("values", [[0.5, 0.8], [float("nan"), 0.5], [-0.1, 1.1]])
def test_uncertainty_rejects_invalid_probabilities(values):
    with pytest.raises(ValueError):
        uncertainty(values, 100, 0.5, 0.1)


def test_metric_definitions_on_known_probabilities():
    values = np.full((3, 3), 1 / 3)
    metrics = probability_metrics(["a", "b", "c"], values, ["a", "b", "c"])
    assert metrics["brier_score"] == pytest.approx(2 / 3)
    assert metrics["log_loss"] == pytest.approx(np.log(3))
    assert metrics["ece_10_bins"] == pytest.approx(0)
    assert metrics["accuracy"] == pytest.approx(1 / 3)


def test_reliability_bins_include_probability_one():
    bins = reliability_bins(["a", "b"], np.eye(2), ["a", "b"])
    assert len(bins) == 10
    assert bins[-1]["count"] == 2
    assert bins[-1]["accuracy"] == bins[-1]["mean_confidence"] == 1
    assert sum(row["count"] for row in bins) == 2


def test_missing_calibrator_fails_closed(tmp_path):
    for name in ("manifest.json", "model.joblib", "vectorizer.joblib"):
        shutil.copy2(ROOT / "models" / name, tmp_path / name)
    with pytest.raises(FileNotFoundError):
        Predictor(tmp_path)


def test_corrupted_calibrator_fails_checksum(tmp_path):
    for name in (
        "manifest.json",
        "model.joblib",
        "vectorizer.joblib",
        "calibrator.joblib",
    ):
        shutil.copy2(ROOT / "models" / name, tmp_path / name)
    with (tmp_path / "calibrator.joblib").open("ab") as file:
        file.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        Predictor(tmp_path)


def test_calibrated_response_matches_artifact(predictor):
    result = predictor.predict("accounting audit financial reporting " * 20)
    raw = predictor.model.predict_proba(
        predictor.vectorizer.transform(["accounting audit financial reporting " * 20])
    )
    calibrated = predictor.calibrator.predict_proba(raw)[0]
    ranked = np.argsort(-calibrated, kind="stable")[:3]
    assert result["calibrated_confidence"] == float(calibrated[ranked[0]])
    assert (
        result["calibrated_predicted_category"] == predictor.model.classes_[ranked[0]]
    )
    assert result["calibrated_top_predictions"] == [
        {
            "category": str(predictor.model.classes_[index]),
            "probability": float(calibrated[index]),
        }
        for index in ranked
    ]
    assert result["uncertainty_reason"] == uncertainty(
        calibrated, 80, predictor.t1, predictor.t2
    )
    assert result["is_uncertain"] == (result["uncertainty_reason"] is not None)


def test_api_short_input_and_file_consistency():
    with TestClient(create_app()) as client:
        text = "skills teamwork customer service"
        response = client.post("/predict", json={"text": text})
        assert response.status_code == 200
        body = response.json()
        assert body["is_uncertain"] is True
        assert body["uncertainty_reason"] == "short_input"
        assert (
            len(body["top_predictions"]) == len(body["calibrated_top_predictions"]) == 3
        )
        file = client.post("/predict/file", files={"file": ("cv.txt", text)}).json()
        assert file == {**body, "source": "file", "text_preview": text}
        schema = client.get("/openapi.json").json()["components"]["schemas"][
            "PredictionResponse"
        ]
        assert {"calibrated_confidence", "is_uncertain", "uncertainty_reason"} <= set(
            schema["required"]
        )


def test_calibration_selection_excludes_held_out_rows():
    manifest = json.loads((ROOT / "models/manifest.json").read_text())
    run = ROOT / manifest["calibration"]["run_directory"]
    split = json.loads((ROOT / "reports/split.json").read_text())
    train, test = set(split["train_row_indices"]), set(split["test_row_indices"])
    validation = []
    for fold in json.loads((run / "folds.json").read_text()):
        fitting, held = (
            set(fold["fit_row_indices"]),
            set(fold["validation_row_indices"]),
        )
        assert fitting.isdisjoint(held)
        assert fitting | held == train
        assert (fitting | held).isdisjoint(test)
        validation.extend(held)
    assert len(validation) == len(set(validation)) == len(train)
