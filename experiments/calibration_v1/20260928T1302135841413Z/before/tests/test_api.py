import pytest
from fastapi.testclient import TestClient
from api.main import create_app
from src.predict import Predictor


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as client:
        yield client


def test_health(client):
    assert client.get("/health").status_code == 200
    assert client.get("/health").json()["status"] == "ok"


def test_predict(client):
    text = "Accountant managing audits, financial statements and tax reporting"
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    assert response.json() == Predictor().predict(text)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"text": ""},
        {"text": "  "},
        {"text": 123},
        {"text": None},
        {"text": []},
        {"text": "!!!"},
        {"text": "zzqxzzqxzzqx"},
        {"text": "x" * 50001},
        {"text": "python", "unexpected": True},
    ],
)
def test_validation(client, body):
    assert client.post("/predict", json=body).status_code == 422


def test_unavailable(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health").status_code == 503
        assert (
            client.post("/predict", json={"text": "Python engineer"}).status_code == 503
        )


def test_internal_error(client, monkeypatch):
    def fail(_):
        raise RuntimeError("private internal information")

    monkeypatch.setattr(client.app.state.predictor, "predict", fail)
    response = client.post("/predict", json={"text": "Python engineer"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}
