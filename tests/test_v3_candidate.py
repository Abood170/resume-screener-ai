"""Candidate contract, calibration isolation and production-integrity regressions."""
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from api.main import create_app
from src.data import ROOT,sha256
from src.v3_api import CandidatePredictor,candidate_app
from src.v3_experiment import response


@pytest.fixture(scope='module')
def candidate(): return CandidatePredictor()


def test_candidate_probabilities_match_artifacts(candidate):
    text='Java Spring Boot SQL database REST API Docker CI/CD development testing'
    p=candidate.calibrator.predict_proba(response(candidate.pipeline,[text]))[0]
    assert np.isfinite(p).all() and np.all(p>=0)
    assert p.sum()==pytest.approx(1)
    body=candidate.predict(text)
    assert body['calibrated_confidence']==float(p.max())
    assert body['calibrated_predicted_category']==candidate.classes[p.argmax()]


def test_candidate_real_routes_and_schema():
    assert candidate_app().openapi()==create_app().openapi()
    with TestClient(candidate_app()) as client:
        assert client.get('/health').status_code==200
        text='Java Spring Boot SQL database REST API Docker CI/CD development testing'
        body=client.post('/predict',json={'text':text})
        assert body.status_code==200
        assert body.json()['uncertainty_reason']=='short_input'
        uploaded=client.post('/predict/file',files={'file':('cv.txt',text)})
        assert uploaded.status_code==200
        assert uploaded.json()=={**body.json(),'source':'file','text_preview':text}
        assert client.post('/predict',json={'text':'!!!'}).status_code==422


def test_nested_calibration_never_uses_test_rows():
    run=ROOT/'experiments/v3'
    split=json.loads((ROOT/'reports/split.json').read_text())
    training,testing=set(split['train_row_indices']),set(split['test_row_indices'])
    validation=[]
    for outer in json.loads((run/'folds.json').read_text()):
        fit,val=set(outer['fit']),set(outer['validation'])
        assert not fit&val and fit|val==training and not (fit|val)&testing
        validation+=outer['validation']
        seen=[]
        for inner in outer['inner']:
            a,b=set(inner['fit']),set(inner['validation'])
            assert not a&b and a|b==fit
            seen+=inner['validation']
        assert len(seen)==len(set(seen))==len(fit)
    assert len(validation)==len(set(validation))==len(training)


def test_production_and_previous_experiments_unchanged():
    checks=json.loads((ROOT/'experiments/v3/protection.json').read_text())
    assert all(sha256(ROOT/name)==digest for name,digest in checks.items())


def test_search_gate_matches_predeclared_rule():
    search=json.loads((ROOT/'experiments/v3/search_results.json').read_text())
    accepted=[]
    for row in search['trials']:
        assert row['kept_in_search']==(row['mean']>row['incumbent_mean_before']+row['incumbent_std_before'])
        if row['kept_in_search']: accepted.append(row['id'])
    if accepted: assert search['chosen']['id']==accepted[-1]
