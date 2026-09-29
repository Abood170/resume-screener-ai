"""Synthetic checks of experimental features without touching production."""
from collections import Counter
import numpy as np
import pytest
from sklearn.base import clone
from src.v3_features import clean_technical,SmartMasker,CappedTfidfVectorizer


def test_preserves_technical_tokens():
    text='C++ C# .NET Node.js CI/CD Java17 HTTP/2 spring-boot SQL'
    assert clean_technical(text).split()==['c++','c#','.net','node.js','ci/cd','java17','http/2','spring-boot','sql']


def test_urls_removed_and_placeholders_cannot_collide():
    value=clean_technical('zztechnicalplaceholderazz C++ https://example.com/foo Node.js')
    assert value=='zztechnicalplaceholderazz c++ node.js'


def test_prefix_is_label_independent_and_body_preserved():
    text='Accounting audit payroll financial Python sales'
    assert SmartMasker(2)(text)=='audit payroll financial python sale'
    assert SmartMasker(100)(text)=='audit payroll python'


def test_prefix_masks_complete_cross_boundary_phrase():
    assert SmartMasker(2)('skills business development databases')=='skill database'


def test_count_cap_does_not_delete_body_terms_or_cap_unrelated_terms():
    vectorizer=CappedTfidfVectorizer(k=0,cap=2,min_df=1,max_df=1.0)
    features=Counter(vectorizer.build_analyzer()('sales sales sales sales audit audit audit audit'))
    assert features['sale']==2
    assert features['sale sale']==2
    assert features['audit']==4
    assert SmartMasker(0)('sales sales sales sales')=='sale sale sale sale'


def test_tfidf_uses_capped_counts_before_sublinear_tf():
    v=CappedTfidfVectorizer(k=0,cap=2,min_df=1,max_df=1.0)
    v.set_params(max_features=5000)
    row=v.fit_transform(['sales sales sales sales audit audit audit audit']).toarray()[0]
    # One-document IDF is one and common normalization cancels in the ratio.
    ratio=row[v.vocabulary_['sale']]/row[v.vocabulary_['audit']]
    assert ratio==pytest.approx((1+np.log(2))/(1+np.log(4)))


def test_clone_and_search_parameter_update():
    v=clone(CappedTfidfVectorizer(k=0))
    v.set_params(k=2)
    assert list(v.build_analyzer()('sales audit'))==['audit']


@pytest.mark.parametrize('text',['',' !!! ','the and'])
def test_empty(text): assert clean_technical(text)==''


def test_calibration_adapter_has_complete_classifier_interface():
    from src.v3_experiment import calibrate
    p=np.array([[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8]]*5)
    mapping=calibrate(p,np.array(['a','b','c']*5),['a','b','c'])
    np.testing.assert_allclose(mapping.predict_proba(p).sum(axis=1),1)
    assert mapping.predict(p).tolist()==['a','b','c']*5
