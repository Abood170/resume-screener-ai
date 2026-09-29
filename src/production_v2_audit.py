"""Validate saved v2 evidence and quantify masking, without held-out inference."""
import json
import joblib
import numpy as np
import pandas as pd
from src.data import ROOT, sha256
from src.calibration import probability_metrics, selective_metrics
from src.leakage_impact import mask_own_label
from src.preprocess import clean_text_legacy, clean_text_universal
from src.production_v2 import save


def main():
    run=ROOT/'experiments/production_v2'
    r=json.loads((run/'results.json').read_text())
    p=json.loads((run/'protocol.json').read_text())
    selection=json.loads((run/'selection.json').read_text())
    manifest=json.loads((run/'models/manifest.json').read_text())
    assert sha256(ROOT/'src/preprocess.py')==p['preprocess_sha256']
    assert sha256(ROOT/'src/label_masking.py')==p['masking_sha256']
    assert sha256(ROOT/'reports/split.json')==p['split_hash']
    assert sha256(ROOT/'data/resumes.csv')==manifest['dataset_sha256']
    assert sha256(run/'protocol.json')==selection['protocol_sha256']
    assert sha256(run/'selection.json')==r['calibration']['test']['selection_sha256_before_test']
    for name,digest in manifest['artifacts'].items(): assert sha256(run/'models'/name)==digest
    split=json.loads((ROOT/'reports/split.json').read_text())
    train,test=set(split['train_row_indices']),set(split['test_row_indices'])
    folds=json.loads((run/'folds.json').read_text())
    baseline=json.loads((ROOT/'experiments/baseline_v1/results.json').read_text())
    original=json.loads((ROOT/baseline['calibration']['run_directory']/'folds.json').read_text())
    for new,old in zip(folds,original):
        a,b=set(new['fit_row_indices']),set(new['validation_row_indices'])
        assert a.isdisjoint(b) and a|b==train and (a|b).isdisjoint(test)
        assert new['fit_row_indices']==old['fit_row_indices']
        assert new['validation_row_indices']==old['validation_row_indices']
        seen=[]
        for inner in new['inner_folds']:
            c,d=set(inner['fit_row_indices']),set(inner['validation_row_indices'])
            assert c.isdisjoint(d) and c|d==a
            seen+=inner['validation_row_indices']
        assert len(seen)==len(set(seen))==len(a)
    oof=np.load(run/'training_oof.npz',allow_pickle=False)
    data=pd.read_csv(ROOT/'data/resumes.csv',keep_default_na=False)
    counts=data.loc[split['train_row_indices'],'Resume'].map(lambda t:len(t.split())).to_numpy()
    grid=[selective_metrics(oof['labels'],oof['calibrated'],r['classes'],counts,t1,t2)
          for t1 in p['threshold_t1_grid'] for t2 in p['threshold_t2_grid']]
    grid.append(selective_metrics(oof['labels'],oof['calibrated'],r['classes'],counts,0,0))
    assert min(grid,key=lambda x:(-x['utility'],-x['coverage'],x['t1'],x['t2']))==selection['training_policy']
    cached=np.load(run/'test_probabilities.npz',allow_pickle=False)
    assert cached['row_indices'].tolist()==split['test_row_indices']
    for phase,key in [('before','raw'),('after','calibrated')]:
        for k,v in probability_metrics(cached['labels'],cached[key],r['classes']).items():
            assert v==r['calibration']['test'][phase][k]
    model=joblib.load(run/'models/model.joblib')
    assert model.get_params()==joblib.load(ROOT/'experiments/baseline_v1/models/model.joblib').get_params()
    vectorizer=joblib.load(run/'models/vectorizer.joblib')
    assert vectorizer.preprocessor is clean_text_universal
    assert model.classes_.tolist()==r['classes']
    removal={}
    for name,ids in [('train',split['train_row_indices']),('test',split['test_row_indices'])]:
        diffs=[]
        for row in data.loc[ids].itertuples():
            raw=len(row.Resume.split())
            before=len(clean_text_legacy(row.Resume).split())
            l1=mask_own_label(row.Resume,row.Category)
            diffs.append((raw-len(l1.split()),before-len(clean_text_legacy(l1).split())))
        removal[name]={'l1_mean_raw_words_removed':float(np.mean([x[0] for x in diffs])),
                       'l1_mean_clean_tokens_removed':float(np.mean([x[1] for x in diffs])),
                       'universal':r['masking_audit'][name]['overall']}
    save(run/'investigation_review.json',{'integrity_passed':True,'cached_metrics_reproduced':True,
        'outer_folds_match_baseline':True,'nested_folds_disjoint':True,'threshold_selection_reproduced':True,
        'additional_heldout_inference':False,'removal_comparison':removal,
        'investigation':r['investigation'],
        'conclusion':'The lower score also occurs in training CV, consistent with a broader removal of useful domain evidence. L1 is label-conditioned and lacks the canonical lemma pass, so it is not a fair lower bound. No settings changed after inspecting test results. Proceed with the requested methodological promotion; remaining shortcuts and external validity are unresolved.'})
    print(json.dumps({'checks':'passed','removal_comparison':removal},indent=2))


if __name__=='__main__': main()
