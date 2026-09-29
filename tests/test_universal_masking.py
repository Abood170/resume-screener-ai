"""Inference-safe masking, baseline compatibility, and deployed feature checks."""
import json
import joblib
import numpy as np
import pytest

from src.data import ROOT
from src.label_masking import VARIANTS, PHRASES, universal_mask, strip_phrases
from src.preprocess import clean_text, clean_text_legacy, clean_text_universal, masked_feature_phrases


def test_dictionary_matches_prior_audit():
    prior=json.loads((ROOT/'reports/error_analysis/label_variants.json').read_text())
    assert VARIANTS==prior
    assert len(VARIANTS)==24


@pytest.mark.parametrize('text',[
    'ACCOUNTANT and HR with financial and human resources work',
    'Business-development; INFORMATION_TECHNOLOGY; digital/media',
    'Artists, art, arts, teaching, teachers, chefs and sales',
    'public relations healthcare health-care agriculture automotive',
])
def test_removes_all_labels_without_receiving_a_category(text):
    cleaned=clean_text_universal(text)
    assert universal_mask(universal_mask(text))==universal_mask(text)
    assert strip_phrases(cleaned,masked_feature_phrases())==cleaned


def test_preserves_boundaries_and_nonlabel_content():
    assert universal_mask('parts earth cart SQL Python audit')=='parts earth cart SQL Python audit'
    assert clean_text_universal('parts earth cart SQL Python audit')=='part earth cart sql python audit'


def test_lemma_forms_cannot_reintroduce_label_features():
    assert clean_text_universal('sales sale artists arts teacher teaching')==''
    assert clean_text_universal('digital and media')==''


@pytest.mark.parametrize('value',[None,12,[]])
def test_invalid_type(value):
    with pytest.raises(TypeError): clean_text_universal(value)


def test_empty_and_label_only_inputs():
    assert clean_text_universal('')==''
    assert clean_text_universal(' !!! ')==''
    assert clean_text_universal(' '.join(PHRASES))==''


def test_legacy_entry_point_preserved():
    text='ACCOUNTANT financial accounting sales SQL'
    assert clean_text(text)==clean_text_legacy(text)
    assert clean_text(text)=='accountant financial accounting sale sql'
    assert clean_text_universal(text)=='sql'


def test_frozen_vectorizer_uses_legacy_preprocessing():
    old=joblib.load(ROOT/'experiments/baseline_v1/models/vectorizer.joblib')
    assert old.preprocessor is clean_text
    assert 'accounting' in old.preprocessor('accounting')


def test_production_vectorizer_embeds_universal_mask():
    from src.predict import Predictor,InvalidResumeError
    predictor=Predictor()
    assert predictor.vectorizer.preprocessor is clean_text_universal
    vocabulary=predictor.vectorizer.get_feature_names_out()
    assert all(strip_phrases(str(v),masked_feature_phrases()).strip()==v for v in vocabulary)
    with pytest.raises(InvalidResumeError): predictor.predict('accountant financial accounting')
    result=predictor.predict('Accountant financial accounting audit payroll ledger reconciliation tax compliance')
    assert result['top_terms']
    assert all(strip_phrases(term,masked_feature_phrases()).strip()==term for term in result['top_terms'])
    # The same feature input must result regardless of inserted category words.
    np.testing.assert_array_equal(predictor.vectorizer.transform(['SQL database']).toarray(),
        predictor.vectorizer.transform(['ACCOUNTANT SQL financial database']).toarray())
