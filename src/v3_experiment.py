"""Training-only v3 search, fixed-choice nested calibration, one test report.

No promotion. Never writes serving artifacts or earlier experiment directories.
CV after search is conditional on training-selected settings, not an independent
nested-search performance estimate. Prefix K is a training-derived fixed rule.
"""
import json
import time
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.base import BaseEstimator,ClassifierMixin,clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold,RandomizedSearchCV,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning

from src.calibration import probability_metrics,selective_metrics,reliability_bins
from src.data import ROOT,sha256
from src.production_v2 import save,metrics
from src.v3_features import CappedTfidfVectorizer
from src.v3_calibration import ResponseAdapter


def response(estimator,text):
    return estimator.decision_function(text) if hasattr(estimator,'decision_function') else estimator.predict_proba(text)


def raw_probabilities(estimator,text):
    """SVC has no raw probabilities: softmax margins are a disclosed score proxy."""
    return estimator.predict_proba(text) if hasattr(estimator,'predict_proba') else softmax(estimator.decision_function(text),axis=1)


def calibrate(values,y,classes):
    adapter=ResponseAdapter(tuple(classes)).fit(values)
    return CalibratedClassifierCV(FrozenEstimator(adapter),method='sigmoid',ensemble=False).fit(values,y)


def main(resume=False):
    run=ROOT/'experiments/v3'
    if (run/'results.json').exists(): raise ValueError('Held-out result already exists; do not evaluate again')
    if (run/'protocol.json').exists() and not resume: raise ValueError('Existing run; use --resume only for completed search before test evaluation')
    if resume:
        assert (run/'search_results.json').exists()
        assert not (run/'selection.json').exists(), 'Frozen evaluation stage requires separate recovery, not retraining'
        protocol=json.loads((run/'protocol.json').read_text())
        assert protocol['feature_source_sha256']==sha256(ROOT/'src/v3_features.py')
    pos=json.loads((run/'position_results.json').read_text())
    production=json.loads((ROOT/'experiments/production_v2/results.json').read_text())
    split=json.loads((ROOT/'reports/split.json').read_text())
    assert sha256(ROOT/'reports/split.json')==pos['split_sha256']
    data=pd.read_csv(ROOT/'data/resumes.csv',keep_default_na=False)
    train=data.loc[split['train_row_indices']]; x=train.Resume.to_numpy(); y=train.Category.to_numpy()
    cv=StratifiedKFold(5,shuffle=True,random_state=42)
    families={'logistic_regression':LogisticRegression(class_weight='balanced',C=1,max_iter=3000,random_state=42),
              'random_forest':RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=2),
              'linear_svc':LinearSVC(class_weight='balanced',C=1,dual='auto',max_iter=10000,random_state=42)}
    spaces={name:{'tfidf__min_df':[2,3],'tfidf__max_df':[0.9,0.95],'tfidf__max_features':[5000,10000,20000],
                  **({'classifier__max_depth':[None,30],'classifier__min_samples_leaf':[1,2]} if name=='random_forest'
                     else {'classifier__C':[0.3,1.0,3.0]})} for name in families}
    if not resume: save(run/'protocol.json',{'seed':42,'k':pos['k'],'cap':2,'cv':5,'sublinear_tf':True,
        'families':{k:str(v) for k,v in families.items()},'search_spaces':spaces,'search_iterations_per_shortlisted_family':8,
        'shortlist':'Highest mean 5-fold macro F1 at defaults; two families; deterministic alphabetical tie-break',
        'keep_rule':'In recorded trial order accept only if mean exceeds current incumbent mean plus incumbent fold std; incumbent starts at production_v2 raw RF CV. Final is last accepted candidate; if none accepted, best v3 is diagnostic only.',
        'calibration':'Fixed sigmoid; nested outer/inner five-fold training-only calibration after settings selected. CV conditional on selected settings, not full nested-search validation.',
        'threshold_rule':production['calibration']['protocol']['threshold_selection'],
        'test_policy':'One frozen-candidate evaluation; test never selects or promotes. Production remains unchanged pending user review.',
        'residual_risk':'Capped body label presence remains a shortcut. Technical shapes include some nontechnical hyphenated words.',
        'data_sha256':sha256(ROOT/'data/resumes.csv'),'split_sha256':sha256(ROOT/'reports/split.json'),
        'feature_source_sha256':sha256(ROOT/'src/v3_features.py')})
    warnings.filterwarnings('error',category=ConvergenceWarning)
    rows=[]; pipelines={}
    for name,model in families.items():
        pipe=Pipeline([('tfidf',CappedTfidfVectorizer(k=pos['k'])),('classifier',model)])
        pipelines[name]=pipe
        if resume: continue
        start=time.perf_counter(); scores=cross_val_score(pipe,x,y,cv=cv,scoring='f1_macro',error_score='raise',n_jobs=1)
        rows.append({'id':f'V3-{len(rows)+1}','stage':'family_screen','family':name,'params':{},'fold_scores':scores.tolist(),
                     'mean':float(scores.mean()),'std':float(scores.std()),'seconds':time.perf_counter()-start})
        save(run/f"trial_{len(rows)}.json",rows[-1]); print(json.dumps(rows[-1]),flush=True)
    if resume:
        prior=json.loads((run/'search_results.json').read_text())
        rows=prior['trials'];shortlist=prior['shortlisted_families']
    else: shortlist=[r['family'] for r in sorted(rows,key=lambda r:(-r['mean'],r['family']))[:2]]
    for name in ([] if resume else shortlist):
        search=RandomizedSearchCV(pipelines[name],spaces[name],n_iter=8,random_state=42,scoring='f1_macro',
                                  cv=cv,n_jobs=1,refit=False,error_score='raise',return_train_score=False,verbose=0)
        start=time.perf_counter(); search.fit(x,y)
        elapsed=time.perf_counter()-start
        results=search.cv_results_
        for i,params in enumerate(results['params']):
            row={'id':f'V3-{len(rows)+1}','stage':'randomized_search','family':name,'params':params,
                 'fold_scores':[float(results[f'split{fold}_test_score'][i]) for fold in range(5)],
                 'mean':float(results['mean_test_score'][i]),'std':float(results['std_test_score'][i]),
                 'seconds':float(5*(results['mean_fit_time'][i]+results['mean_score_time'][i]))}
            rows.append(row); save(run/f'trial_{len(rows)}.json',row); print(json.dumps(row),flush=True)
        save(run/f'{name}_search.json',{'space':spaces[name],'elapsed_seconds':elapsed,'trials':8})
    incumbent=production['models']['random_forest']; mean=incumbent['cv_macro_f1_mean']; std=incumbent['cv_macro_f1_std']; chosen=None
    for row in rows:
        row['incumbent_mean_before']=mean; row['incumbent_std_before']=std
        row['kept_in_search']=row['mean']>mean+std
        if row['kept_in_search']: chosen=row; mean=row['mean']; std=row['std']
    gate_passed=chosen is not None
    if chosen is None: chosen=max(rows,key=lambda r:r['mean'])
    if not resume: save(run/'search_results.json',{'trials':rows,'shortlisted_families':shortlist,'chosen':chosen,'cv_gate_passed':gate_passed})
    else: assert chosen==prior['chosen'] and gate_passed==prior['cv_gate_passed']
    fitted_template=clone(pipelines[chosen['family']]).set_params(**chosen['params'])
    classes=production['classes']; score_oof=np.zeros((len(y),len(classes))); calibrated_oof=np.zeros_like(score_oof)
    fold_metrics=[]; folds=[]; start=time.perf_counter()
    for number,(fit,val) in enumerate(cv.split(x,y)):
        inner=np.zeros((len(fit),len(classes))); inner_rows=[]
        for j,(a,b) in enumerate(cv.split(x[fit],y[fit])):
            fitted=clone(fitted_template).fit(x[fit[a]],y[fit[a]])
            assert fitted.classes_.tolist()==classes
            inner[b]=response(fitted,x[fit[b]])
            inner_rows.append({'fit':train.index[fit[a]].tolist(),'validation':train.index[fit[b]].tolist()})
        fitted=clone(fitted_template).fit(x[fit],y[fit]); score_oof[val]=response(fitted,x[val])
        calibrated_oof[val]=calibrate(inner,y[fit],classes).predict_proba(score_oof[val])
        fold_metrics.append(probability_metrics(y[val],calibrated_oof[val],classes))
        folds.append({'fit':train.index[fit].tolist(),'validation':train.index[val].tolist(),'inner':inner_rows})
        print(f'Nested sigmoid calibration outer fold {number+1}/5 complete',flush=True)
    calibration_seconds=time.perf_counter()-start
    save(run/'folds.json',folds)
    np.savez_compressed(run/'training_oof.npz',row_indices=train.index,labels=y.astype(str),scores=score_oof,calibrated=calibrated_oof)
    rule=production['calibration']['protocol']; counts=np.array([len(t.split()) for t in x])
    grid=[selective_metrics(y,calibrated_oof,classes,counts,t1,t2) for t1 in rule['threshold_t1_grid'] for t2 in rule['threshold_t2_grid']]
    grid.append(selective_metrics(y,calibrated_oof,classes,counts,0,0))
    policy=min(grid,key=lambda r:(-r['utility'],-r['coverage'],r['t1'],r['t2']))
    final_cal=calibrate(score_oof,y,classes)
    start=time.perf_counter(); final=clone(fitted_template).fit(x,y); fit_seconds=time.perf_counter()-start
    selection={'family':chosen['family'],'params':chosen['params'],'k':pos['k'],'cap':2,'trial_id':chosen['id'],
               'cv_gate_passed':gate_passed,'policy':policy,'protocol_sha256':sha256(run/'protocol.json')}
    save(run/'selection.json',selection)
    pd.DataFrame(grid).to_csv(run/'threshold_grid.csv',index=False)
    (run/'models').mkdir()
    joblib.dump(final,run/'models/pipeline.joblib');joblib.dump(final_cal,run/'models/calibrator.joblib')
    save(run/'models/manifest.json',{'classes':classes,'family':chosen['family'],'policy':policy,
         'artifacts':{p.name:sha256(p) for p in (run/'models').glob('*.joblib')}})
    # First and only candidate held-out evaluation, after all settings frozen.
    test=data.loc[split['test_row_indices']]
    raw=raw_probabilities(final,test.Resume); cal=final_cal.predict_proba(response(final,test.Resume))
    before=metrics(test.Category,raw,classes); after=metrics(test.Category,cal,classes)
    np.savez_compressed(run/'test_probabilities.npz',row_indices=test.index,labels=test.Category.to_numpy().astype(str),raw=raw,calibrated=cal)
    test_policy=selective_metrics(test.Category,cal,classes,test.Resume.map(lambda t:len(t.split())),policy['t1'],policy['t2'])
    cv_values=[f['macro_f1'] for f in fold_metrics]
    result={'selected_family':chosen['family'],'selected_trial':chosen['id'],'search':json.loads((run/'search_results.json').read_text()),
        'cv_macro_f1_mean':float(np.mean(cv_values)),'cv_macro_f1_std':float(np.std(cv_values)),
        'calibration_cv_folds':fold_metrics,'training_calibration':probability_metrics(y,calibrated_oof,classes),
        'test':{'before':before,'after':after,'policy':test_policy},'thresholds':policy,
        'selection_sha256_before_test':sha256(run/'selection.json'),'held_out_evaluations':1,
        'cv_gate_passed':gate_passed,'production_unchanged':True,'promoted':False,
        'promotion_reason':'Candidate only; awaiting user review. Test results are reporting only, not a selection rule.',
        'calibration_cv_caveat':'Conditional on settings and K selected using this training partition; not fully nested hyperparameter search.',
        'legacy_svc_caveat':'If selected family is LinearSVC, raw API confidence is a disclosed softmax-margin proxy, not a native probability; compatibility decision required before promotion.',
        'timing':{'calibration_cv_seconds':calibration_seconds,'final_fit_seconds':fit_seconds}}
    save(run/'results.json',result)
    # Plot only recorded training grid and held-out probabilities; no selection.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,6))
    for t2 in rule['threshold_t2_grid']:
        rs=sorted([g for g in grid if g['t2']==t2 and g['accuracy_on_answered'] is not None],key=lambda g:g['coverage'])
        ax.plot([g['coverage'] for g in rs],[g['accuracy_on_answered'] for g in rs],'o-',label=f'margin {t2}')
    ax.scatter(policy['coverage'],policy['accuracy_on_answered'],marker='*',s=180,c='black')
    ax.set(xlabel='Training OOF coverage',ylabel='Accuracy on answered',title='v3 training-only abstention policy');ax.legend();fig.tight_layout();fig.savefig(run/'coverage_vs_accuracy.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6))
    for name,p in [('raw',raw),('sigmoid',cal)]:
        bins=[b for b in reliability_bins(test.Category,p,classes) if b['count']]
        ax.plot([b['mean_confidence'] for b in bins],[b['accuracy'] for b in bins],'o-',label=name)
    ax.plot([0,1],[0,1],'--',c='gray');ax.legend();ax.set(xlabel='Confidence',ylabel='Bin accuracy',title='v3 held-out reliability');fig.tight_layout();fig.savefig(run/'reliability.png',dpi=150);plt.close(fig)
    protection=json.loads((run/'protection.json').read_text())
    assert all(sha256(ROOT/name)==digest for name,digest in protection.items())
    print(json.dumps({'selected':chosen['id'],'test':{k:v for k,v in after.items() if k not in ['per_class','confusion_matrix']},'policy':policy}),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true')
    main(parser.parse_args().resume)
