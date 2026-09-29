"""Reproduce the universal-mask production experiment without implicit promotion.

Training and threshold choices use only the saved training partition. A fresh
run directory is required. Serving artifacts are changed only by explicit publish.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.calibration import fit_calibrator, probability_metrics, reliability_bins, selective_metrics
from src.data import ROOT, sha256
from src.label_masking import VARIANTS, PHRASES, universal_mask
from src.preprocess import clean_text_legacy, clean_text_universal, masked_feature_phrases


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def archive(path: Path, run: Path) -> None:
    if path.exists():
        target = run / "revisions" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def metrics(y, probs, classes):
    pred = np.array(classes)[probs.argmax(1)]
    report = classification_report(y, pred, labels=classes, output_dict=True, zero_division=0)
    return {**probability_metrics(y, probs, classes), "per_class": report,
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist()}


def masking_audit(frame):
    rows = []
    for item in frame.itertuples():
        raw_words = len(item.Resume.split())
        raw_removed = raw_words - len(universal_mask(item.Resume).split())
        before = len(clean_text_legacy(item.Resume).split())
        after = len(clean_text_universal(item.Resume).split())
        rows.append({"category": item.Category, "raw_words": raw_words,
                     "raw_words_removed": raw_removed, "clean_tokens_before": before,
                     "clean_tokens_removed": before-after, "empty_after": int(after == 0)})
    df = pd.DataFrame(rows)
    def aggregate(part):
        return {"rows": len(part), "mean_raw_words_removed": float(part.raw_words_removed.mean()),
                "total_raw_words_removed": int(part.raw_words_removed.sum()),
                "raw_word_fraction_removed": float(part.raw_words_removed.sum()/part.raw_words.sum()),
                "mean_clean_tokens_removed": float(part.clean_tokens_removed.mean()),
                "clean_token_fraction_removed": float(part.clean_tokens_removed.sum()/part.clean_tokens_before.sum()),
                "empty_after": int(part.empty_after.sum())}
    return {"overall": aggregate(df), "per_class": {c: aggregate(part) for c,part in df.groupby("category")}}


def train(run: Path) -> None:
    if (run / "protocol.json").exists():
        raise ValueError("Use a new run directory; existing experiment evidence is immutable")
    run.mkdir(parents=True, exist_ok=True)
    bdir = ROOT / "experiments/baseline_v1"
    baseline = json.loads((bdir / "results.json").read_text())
    checksum = json.loads((bdir / "checksums.json").read_text())["sha256"]
    for name,digest in checksum.items():
        assert sha256(bdir/name) == digest
    assert sha256(ROOT/'reports/split.json') == checksum['reports/split.json']
    assert sha256(ROOT/'data/resumes.csv') == baseline['dataset']['csv_sha256']
    assert VARIANTS == json.loads((ROOT/'reports/error_analysis/label_variants.json').read_text())
    split = json.loads((bdir/'reports/split.json').read_text())
    assert not set(split['train_row_indices']) & set(split['test_row_indices'])
    vectorizer = clone(joblib.load(bdir/'models/vectorizer.joblib')).set_params(preprocessor=clean_text_universal)
    model = clone(joblib.load(bdir/'models/model.joblib'))
    estimator = Pipeline([('tfidf', vectorizer), ('classifier', model)])
    protocol = {**baseline['calibration']['protocol'], 'method': 'sigmoid',
        'selection': 'RF hyperparameters and sigmoid fixed from baseline; no model or calibration method selection.',
        'preprocessing': 'Universal fixed phrase mask, legacy cleaning, canonical lemma mask; no labels required.',
        'variants': VARIANTS, 'raw_phrases': PHRASES, 'canonical_phrases': masked_feature_phrases(),
        'preprocess_sha256': sha256(ROOT/'src/preprocess.py'), 'masking_sha256': sha256(ROOT/'src/label_masking.py'),
        'baseline_checksums': checksum, 'rf_params': model.get_params(),
        'vectorizer_params': {k: str(v) for k,v in vectorizer.get_params().items()},
        'hypothesis': 'Performance may lie between baseline and L1; not a guaranteed ordering. Universal removal is broader.',
        'promotion': 'Requested methodological change, not selected using held-out performance.'}
    save(run/'protocol.json',protocol)
    data = pd.read_csv(ROOT/'data/resumes.csv',keep_default_na=False)
    tr = data.loc[split['train_row_indices']]
    x,y = tr.Resume.to_numpy(),tr.Category.to_numpy()
    classes = baseline['classes']
    raw_oof=np.zeros((len(y),len(classes))); cal_oof=np.zeros_like(raw_oof)
    folds=[]; scores=[]; cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    start=time.perf_counter()
    for fold,(fit,val) in enumerate(cv.split(x,y)):
        inner=np.zeros((len(fit),len(classes)))
        inner_folds=[]
        for number,(a,b) in enumerate(cv.split(x[fit],y[fit])):
            fitted=clone(estimator).fit(x[fit[a]],y[fit[a]])
            assert fitted.classes_.tolist()==classes
            inner[b]=fitted.predict_proba(x[fit[b]])
            inner_folds.append({'fit_row_indices':tr.index[fit[a]].tolist(),'validation_row_indices':tr.index[fit[b]].tolist()})
            print(f'Outer {fold+1}/5 inner {number+1}/5 complete',flush=True)
        fitted=clone(estimator).fit(x[fit],y[fit])
        raw_oof[val]=fitted.predict_proba(x[val])
        cal_oof[val]=fit_calibrator(inner,y[fit],classes,'sigmoid').predict_proba(raw_oof[val])
        scores.append({'fold':fold,'raw':probability_metrics(y[val],raw_oof[val],classes),
                       'sigmoid':probability_metrics(y[val],cal_oof[val],classes)})
        folds.append({'fold':fold,'fit_row_indices':tr.index[fit].tolist(),
                      'validation_row_indices':tr.index[val].tolist(),'inner_folds':inner_folds})
        np.savez_compressed(run/f'outer_fold_{fold}.npz',row_indices=tr.index[val],raw=raw_oof[val],calibrated=cal_oof[val])
    cv_seconds=time.perf_counter()-start
    save(run/'folds.json',folds)
    cv_stats={method:{'mean':float(np.mean([s[method]['macro_f1'] for s in scores])),
                     'std':float(np.std([s[method]['macro_f1'] for s in scores]))} for method in ['raw','sigmoid']}
    training={'fold_metrics':scores,'cv_fold_macro_f1':cv_stats,'elapsed_seconds':cv_seconds,
              'pooled_nested_oof':{'raw':probability_metrics(y,raw_oof,classes),'sigmoid':probability_metrics(y,cal_oof,classes)}}
    word_counts=np.array([len(t.split()) for t in x])
    grid=[selective_metrics(y,cal_oof,classes,word_counts,t1,t2)
          for t1 in protocol['threshold_t1_grid'] for t2 in protocol['threshold_t2_grid']]
    grid.append(selective_metrics(y,cal_oof,classes,word_counts,0,0))
    chosen=min(grid,key=lambda r:(-r['utility'],-r['coverage'],r['t1'],r['t2']))
    selection={'method':'sigmoid','t1':chosen['t1'],'t2':chosen['t2'],'short_input_words':50,
               'training_policy':chosen,'protocol_sha256':sha256(run/'protocol.json'),
               'selection_data':'Training nested OOF only; fixed historical utility/grid/tie-breaks'}
    save(run/'selection.json',selection)
    pd.DataFrame(grid).to_csv(run/'threshold_grid.csv',index=False)
    np.savez_compressed(run/'training_oof.npz',row_indices=tr.index,labels=y.astype(str),raw=raw_oof,calibrated=cal_oof)
    start=time.perf_counter(); fitted=clone(estimator).fit(x,y)
    final_cal=fit_calibrator(raw_oof,y,classes,'sigmoid'); fit_seconds=time.perf_counter()-start
    (run/'models').mkdir()
    joblib.dump(fitted['tfidf'],run/'models/vectorizer.joblib')
    joblib.dump(fitted['classifier'],run/'models/model.joblib')
    joblib.dump(final_cal,run/'models/calibrator.joblib')
    training['fit_seconds']=fit_seconds
    save(run/'training_results.json',training)
    print('Training and thresholds frozen. Starting final held-out evaluation.',flush=True)
    te=data.loc[split['test_row_indices']]
    raw=fitted.predict_proba(te.Resume); calibrated=final_cal.predict_proba(raw)
    before=metrics(te.Category,raw,classes); after=metrics(te.Category,calibrated,classes)
    policy=selective_metrics(te.Category,calibrated,classes,te.Resume.map(lambda t:len(t.split())),chosen['t1'],chosen['t2'])
    test={'before':before,'after':after,'policy':policy,'selection_sha256_before_test':sha256(run/'selection.json'),
          'held_out_evaluations':1,'delta_after_minus_before':{k:after[k]-before[k] for k in ['macro_f1','accuracy','brier_score','log_loss','ece_10_bins']}}
    save(run/'test_results.json',test)
    np.savez_compressed(run/'test_probabilities.npz',row_indices=te.index,labels=te.Category.to_numpy().astype(str),raw=raw,calibrated=calibrated)
    audit={'train':masking_audit(tr),'test':masking_audit(te)}
    train_clean=[clean_text_universal(t) for t in tr.Resume]
    test_clean=[clean_text_universal(t) for t in te.Resume]
    audit['masked_exact_cross_split_overlap']=sum(t in set(train_clean) for t in test_clean)
    names=fitted['tfidf'].get_feature_names_out()
    from src.label_masking import strip_phrases
    bad=[str(t) for t in names if strip_phrases(str(t),masked_feature_phrases()).strip()!=str(t)]
    audit['forbidden_features']=bad
    assert not bad
    save(run/'masking_audit.json',audit)
    relative=run.relative_to(ROOT).as_posix()
    manifest={'selected_model':'random_forest','classes':classes,'dataset_sha256':baseline['dataset']['csv_sha256'],
              'artifacts':{p.name:sha256(p) for p in (run/'models').glob('*.joblib')},
              'preprocessing':{'name':'universal_label_mask_v2','function':'src.preprocess.clean_text_universal',
                               'source_sha256':protocol['preprocess_sha256'],'masking_sha256':protocol['masking_sha256']},
              'calibration':{'artifact':'calibrator.joblib','method':'sigmoid','run_directory':relative,'protocol_sha256':selection['protocol_sha256']},
              'abstain':{k:selection[k] for k in ['t1','t2','short_input_words']}}
    save(run/'models/manifest.json',manifest)
    raw_entry={**before,'cv_macro_f1_mean':cv_stats['raw']['mean'],'cv_macro_f1_std':cv_stats['raw']['std'],
               'cv_macro_f1_folds':[s['raw']['macro_f1'] for s in scores], 'fit_seconds':fit_seconds,'cv_seconds':cv_seconds}
    result={k:baseline[k] for k in ['dataset','cleaning','eda','methodology','classes','environment']}
    result.update({'selected_model':'random_forest','models':{'random_forest':raw_entry},'artifacts':manifest,
                   'production_version':'production_v2','preprocessing':manifest['preprocessing'],
                   'calibration':{'method':'sigmoid','run_directory':relative,'artifact':'calibrator.joblib','protocol':protocol,
                       'training_cv':training,'cv_fold_macro_f1':cv_stats,'test':test},
                   'abstain':{'t1':chosen['t1'],'t2':chosen['t2'],'short_input_words':50,'training_oof':chosen,'test':policy},
                   'masking_audit':audit,'leakage_comparison':{},'environment':{**baseline['environment'],'python':platform.python_version()}})
    l1=json.loads((ROOT/'experiments/leakage_impact/results.json').read_text())
    for key,report,cvv in [('baseline_v1',baseline['calibration']['test']['after'],baseline['calibration']['cv_fold_macro_f1']['sigmoid']),
                          ('L1',l1['masked'],{'mean':l1['masked']['cv_mean'],'std':l1['masked']['cv_std']}),
                          ('production_v2',after,cv_stats['sigmoid'])]:
        per=report.get('per_class') or baseline['calibration']['audit']['class_report_after']
        result['leakage_comparison'][key]={'cv_mean':cvv['mean'],'cv_std':cvv['std'],'macro_f1':report['macro_f1'],
                                         'accuracy':report['accuracy'],'per_class':per}
    result['investigation']={'below_l1':after['macro_f1']<l1['masked']['macro_f1'],
        'test_macro_f1_delta_vs_l1':after['macro_f1']-l1['masked']['macro_f1'],
        'cv_macro_f1_delta_vs_l1':cv_stats['sigmoid']['mean']-l1['masked']['cv_mean'],
        'test_macro_f1_delta_vs_baseline':after['macro_f1']-baseline['calibration']['test']['after']['macro_f1'],
        'note':'Label-independent removal avoids L1 ground-truth conditioning; more domain vocabulary is removed. Quantified removal is in masking_audit. Deltas describe sensitivity, not causal proof.'}
    result['methodology']={**result['methodology'],'selection_metric':'RF and sigmoid fixed from baseline; no model selection. Thresholds: training OOF utility only.',
                           'saved_split_sha256':sha256(ROOT/'reports/split.json')}
    save(run/'results.json',result)
    fig,ax=plt.subplots(figsize=(9,6))
    for t2 in protocol['threshold_t2_grid']:
        points=sorted([r for r in grid if r['t2']==t2 and r['accuracy_on_answered'] is not None],key=lambda r:r['coverage'])
        ax.plot([r['coverage'] for r in points],[r['accuracy_on_answered'] for r in points],'o-',label=f'margin {t2}')
    ax.scatter(chosen['coverage'],chosen['accuracy_on_answered'],marker='*',s=180,color='black',label='Chosen on training OOF')
    ax.set(xlabel='Training OOF coverage',ylabel='Accuracy among answered',title='Universal masking: training-only threshold tradeoff'); ax.legend(); fig.tight_layout()
    fig.savefig(run/'coverage_vs_accuracy.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6))
    for name,values in [('Raw RF',raw),('Sigmoid',calibrated)]:
        bins=[b for b in reliability_bins(te.Category,values,classes) if b['count']]
        ax.plot([b['mean_confidence'] for b in bins],[b['accuracy'] for b in bins],'o-',label=name)
    ax.plot([0,1],[0,1],'--',color='gray'); ax.set(xlabel='Mean top confidence',ylabel='Bin accuracy',title='Production v2 held-out reliability'); ax.legend(); fig.tight_layout()
    fig.savefig(run/'reliability_before_after.png',dpi=160); plt.close(fig)
    print('Final metrics saved; review investigation before explicit publication.',flush=True)


def publish(run: Path) -> None:
    if (run/'published.json').exists(): raise ValueError('Already published')
    r=json.loads((run/'results.json').read_text()); s=json.loads((run/'selection.json').read_text())
    assert r['calibration']['test']['selection_sha256_before_test']==sha256(run/'selection.json')
    assert s['protocol_sha256']==sha256(run/'protocol.json')
    assert (run/'investigation_review.json').exists(), 'Review diagnostic evidence before promotion'
    assert sha256(ROOT/'reports/split.json')==r['methodology']['saved_split_sha256']
    for source in (run/'models').iterdir():
        target=ROOT/'models'/source.name; archive(target,run); shutil.copy2(source,target)
    archive(ROOT/'results.json',run); shutil.copy2(run/'results.json',ROOT/'results.json')
    for name in ['coverage_vs_accuracy.png','reliability_before_after.png']:
        target=ROOT/'reports/figures'/name; archive(target,run); shutil.copy2(run/name,target)
    save(run/'published.json',{'manifest_sha256':sha256(ROOT/'models/manifest.json'),'results_sha256':sha256(ROOT/'results.json')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'experiments/production_v2')
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    (publish if args.publish else train)(args.run_dir.resolve())
