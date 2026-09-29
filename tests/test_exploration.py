"""Synthetic feature checks and saved-evidence guardrails for exploration."""
import json
import numpy as np
from src.data import ROOT,sha256
from src.exploration_features import masked_technical
from src.label_masking import PHRASES,strip_phrases
from src.preprocess import masked_feature_phrases


def test_universal_mask_preserves_tech_without_retaining_category_words():
    text='Accountant financial sales C++ C# .NET Node.js CI/CD SQL'
    assert masked_technical(text)=='c++ c# .net node.js ci/cd sql'
    assert masked_technical(' '.join(PHRASES))==''


def test_mask_is_idempotent_and_removes_lemma_forms():
    text=masked_technical('arts sale digital and media SQL database')
    assert text=='sql database'
    assert masked_technical(text)==text
    assert strip_phrases(text,masked_feature_phrases())==text


def test_character_and_combined_branches_cannot_bypass_mask():
    from src.exploration import estimator
    char=estimator('char_svc')['features']
    combined=estimator('word_char_svc')['features']
    for vectorizer in [char,*[v for _,v in combined.transformer_list]]:
        assert vectorizer.build_preprocessor()('ACCOUNTANT sales C++ SQL')=='c++ sql'


def test_only_frozen_finalists_have_test_results():
    run=ROOT/'experiments/exploration'
    selection=json.loads((run/'selection.json').read_text())
    evaluated=[]
    for name in selection['ranking']:
        folder=run/name
        if (folder/'test_results.json').exists():
            evaluated.append(name)
            assert json.loads((folder/'test_results.json').read_text())['evaluations']==1
            assert json.loads((folder/'test_started.json').read_text())['selection_sha256']==sha256(run/'selection.json')
    assert set(evaluated)==set(selection['finalists'])
    assert len(evaluated)<=2


def test_nested_calibration_has_disjoint_rows():
    run=ROOT/'experiments/exploration'
    split=json.loads((ROOT/'reports/split.json').read_text())
    train,test=set(split['train_row_indices']),set(split['test_row_indices'])
    for name in json.loads((run/'selection.json').read_text())['finalists']:
        folds=json.loads((run/name/'calibration.json').read_text())['folds'];seen=[]
        for fold in folds:
            a,b=set(fold['fit']),set(fold['validation'])
            assert not a&b and a|b==train and not (a|b)&test
            seen+=fold['validation'];inner_seen=[]
            for inner in fold['inner']:
                c,d=set(inner['fit']),set(inner['validation'])
                assert not c&d and c|d==a
                inner_seen+=inner['validation']
            assert len(inner_seen)==len(set(inner_seen))==len(a)
        assert len(seen)==len(set(seen))==len(train)


def test_all_protected_files_unchanged():
    checks=json.loads((ROOT/'experiments/exploration/protected.json').read_text())
    assert all(sha256(ROOT/name)==value for name,value in checks.items())
