"""Bounded, resumable training-only exploration; final tests are a separate phase.

Usage: python -m src.exploration --phase search|finalize|report
Every idea has a saved rationale before fitting. Only the frozen top two can
access held-out rows; a start marker prevents accidental repeated evaluation.
No serving file, API code, or previous experiment is modified.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,classification_report
from sklearn.exceptions import ConvergenceWarning

from src.data import ROOT,sha256
from src.production_v2 import save,archive
from src.experiment_log import append_row
from src.calibration import probability_metrics,selective_metrics
from src.v3_experiment import calibrate,response
from src.exploration_features import masked_technical

RUN=ROOT/'experiments/exploration'
IDEAS=[
 ('E1','control_rf','Fresh production-style RF control',
  'Recompute the existing universal-mask raw RF five-fold score on exactly the saved training rows; anchor all comparisons in fresh measurements.'),
 ('E2','technical_rf','Universal mask plus technical tokens, RF',
  'Keep RF and TF-IDF settings fixed; change only technical-preserving cleaning/tokenization. Expect useful punctuation-bearing skills to survive, but sparse training coverage may prevent improvement.'),
 ('E3','balanced_rf','Technical RF with class balancing',
  'Change only class_weight from E2 to balanced. Expect better minority recall; majority precision may decline.'),
 ('E4','word_lr','Technical word/bigram balanced Logistic Regression',
  'Sparse linear decision boundaries and broader vocabulary may generalize better than trees after label removal. Fixed C=1, sublinear TF, min_df=2, max_df=.95, max_features=20000.'),
 ('E5','word_svc','Technical word/bigram balanced LinearSVC',
  'Use the same features as E4 with a margin-based classifier, fixed C=1. May help overlapping classes; native probabilities are unavailable.'),
 ('E6','char_svc','Masked character n-grams with balanced LinearSVC',
  'Character 3-5 grams within word boundaries may capture spelling/morphology and technical forms despite sparse exact tokens. Input stays universally masked; character fragments can still encode source/template bias.'),
 ('E7','word_char_svc','Combined masked word and character features',
  'Complement explicit skill phrases with character robustness using equal-weight concatenated L2-normalized feature blocks. Extra complexity is worthwhile only if CV improves over both single views.'),
]


def read_rows(indices):
    """Only selected rows leave the loader; never inspect held-out text in search."""
    wanted=set(indices); rows={}
    with (ROOT/'data/resumes.csv').open(encoding='utf-8',newline='') as stream:
        for i,row in enumerate(csv.DictReader(stream)):
            if i in wanted: rows[i]=(row['Resume'],row['Category'])
    return np.array([rows[i][0] for i in indices]),np.array([rows[i][1] for i in indices])


def estimator(name):
    rf=clone(joblib.load(ROOT/'models/model.joblib'))
    old=clone(joblib.load(ROOT/'models/vectorizer.joblib'))
    technical=clone(old).set_params(preprocessor=masked_technical,tokenizer=str.split,token_pattern=None)
    word=TfidfVectorizer(preprocessor=masked_technical,tokenizer=str.split,token_pattern=None,
        lowercase=False,ngram_range=(1,2),sublinear_tf=True,min_df=2,max_df=.95,max_features=20000)
    char=TfidfVectorizer(preprocessor=masked_technical,lowercase=False,analyzer='char_wb',
        ngram_range=(3,5),sublinear_tf=True,min_df=3,max_df=.95,max_features=20000)
    svc=LinearSVC(C=1,class_weight='balanced',dual='auto',max_iter=10000,random_state=42)
    choices={'control_rf':(old,rf),'technical_rf':(technical,rf),
      'balanced_rf':(technical,clone(rf).set_params(class_weight='balanced')),
      'word_lr':(word,LogisticRegression(C=1,class_weight='balanced',max_iter=3000,random_state=42)),
      'word_svc':(word,svc),'char_svc':(char,svc),
      'word_char_svc':(FeatureUnion([('word',word),('char',char)]),svc)}
    features,model=choices[name]
    return Pipeline([('features',features),('classifier',model)])


def initialize():
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists(): return
    split=json.loads((ROOT/'reports/split.json').read_text())
    save(RUN/'protocol.json',{'seed':42,'cv_folds':5,'finalists':2,'ideas':IDEAS,
      'selection':'Rank mean raw training CV macro F1 descending; alphabetical tie-break. Freeze top two BEFORE any test access. Calibrated CV will not rerank finalists.',
      'calibration':'Fixed sigmoid, outer five-fold/inner three-fold training-only calibration; no calibration-method search. Inner three folds bound compute cost, fixed before scores.',
      'thresholds':{'t1':[.3,.4,.5,.6,.7,.8],'t2':[.05,.1,.15,.2]},
      'threshold_rule':'Maximize training OOF utility +1 correct, -1 wrong, 0 abstain. Ties higher coverage, lower T1, lower T2. Include (0,0). Short-input limit remains 50 whitespace words.',
      'suspicious_score':.75,'split_sha256':sha256(ROOT/'reports/split.json'),
      'data_sha256':sha256(ROOT/'data/resumes.csv'),'feature_source_sha256':sha256(ROOT/'src/exploration_features.py'),
      'holdout_policy':'Only frozen finalists, one final evaluation each. Others receive null test metrics. No automatic promotion.'})
    protected={}
    for folder in ['models','api','experiments/baseline_v1','experiments/leakage_impact','experiments/production_v2','experiments/v3']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts: protected[p.relative_to(ROOT).as_posix()]=sha256(p)
    for name in ['results.json','README.md','reports/split.json','src/preprocess.py','src/predict.py']:
        protected[name]=sha256(ROOT/name)
    save(RUN/'protected.json',protected)
    assert not set(split['train_row_indices'])&set(split['test_row_indices'])


def assert_protected():
    protected=json.loads((RUN/'protected.json').read_text())
    assert all(sha256(ROOT/name)==digest for name,digest in protected.items())
    protocol=json.loads((RUN/'protocol.json').read_text())
    assert sha256(ROOT/'data/resumes.csv')==protocol['data_sha256']
    assert sha256(ROOT/'reports/split.json')==protocol['split_sha256']
    assert sha256(ROOT/'src/exploration_features.py')==protocol['feature_source_sha256']


def log_cv(r):
    if f"| EXP-{r['id']} |" in (ROOT/'experiments/experiment_log.md').read_text(encoding='utf-8'):return
    obj={'selected_model':r['name'],'models':{r['name']:{
       'cv_macro_f1_mean':r['cv_mean'],'cv_macro_f1_std':r['cv_std'],
       'macro_f1':'not evaluated','accuracy':'not evaluated','fit_seconds':r['cv_seconds'],
       'timing_description':'Time is fresh five-fold fitting and scoring wall time.'}}}
    append_row('EXP-'+r['id'],r['title'],obj,False,'Exploration only; no production promotion. '+r['observation'])


def search():
    initialize();assert_protected()
    if (RUN/'selection.json').exists():
        print('Frozen selection already exists; no search repeated.');return
    split=json.loads((ROOT/'reports/split.json').read_text());indices=split['train_row_indices']
    x,y=read_rows(indices);cv=StratifiedKFold(5,shuffle=True,random_state=42)
    warnings.filterwarnings('error',category=ConvergenceWarning)
    results=[]
    for id,name,title,reason in IDEAS:
        folder=RUN/name;folder.mkdir(exist_ok=True)
        if (folder/'results.json').exists():
            r=json.loads((folder/'results.json').read_text());results.append(r);log_cv(r);continue
        template=estimator(name)
        if not (folder/'rationale.json').exists():
            save(folder/'rationale.json',{'id':id,'name':name,'title':title,'expectation_before_fitting':reason,
                 'parameters':{k:str(v) for k,v in template.get_params().items()},'test_access':False})
        folds=[];start=time.perf_counter();out=np.empty_like(y)
        for number,(fit,val) in enumerate(cv.split(x,y)):
            model=clone(template).fit(x[fit],y[fit]);pred=model.predict(x[val]);out[val]=pred
            folds.append(float(f1_score(y[val],pred,average='macro')))
            print(f'{id} {name}: training fold {number+1}/5 complete',flush=True)
        mean=float(np.mean(folds));std=float(np.std(folds))
        control=results[0]['cv_mean'] if results else mean
        note=('Fresh control.' if not results else
            f"CV mean minus fresh control: {mean-control}; {'improves' if mean>control else 'does not improve'} the point estimate. Fold std is not a significance test.")
        if name=='technical_rf':
            note+=' Technical-token preservation alone '+('helped.' if mean>control else 'did not help here.')
        if name=='balanced_rf':
            delta=mean-results[1]['cv_mean'];note+=f' Balancing delta versus otherwise identical E2: {delta}.'
        if name=='word_char_svc':
            best=max(r['cv_mean'] for r in results if r['name'] in ['word_svc','char_svc'])
            note+=f' Combined-view delta versus better single SVC view: {mean-best}.'
        r={'id':id,'name':name,'title':title,'cv_mean':mean,'cv_std':std,'cv_folds':folds,
           'cv_seconds':time.perf_counter()-start,'observation':note,'test':None,'test_evaluations':0,
           'cv_per_class':classification_report(y,out,output_dict=True,zero_division=0)}
        save(folder/'results.json',r);np.savez_compressed(folder/'oof.npz',row_indices=indices,labels=y,predictions=out)
        results.append(r);log_cv(r)
        print(json.dumps({k:r[k] for k in ['id','cv_mean','cv_std','observation']}),flush=True)
    ranked=sorted(results,key=lambda r:(-r['cv_mean'],r['name']))
    save(RUN/'selection.json',{'ranking':[r['name'] for r in ranked],
       'finalists':[r['name'] for r in ranked[:2]],'selected_on':'Training CV only',
       'protocol_sha256':sha256(RUN/'protocol.json')})
    assert_protected();print('Finalists frozen using training CV only.',flush=True)


def finish_candidate(name,x,y,indices,split,protocol):
    folder=RUN/name
    if (folder/'test_started.json').exists():
        if (folder/'test_results.json').exists():return
        raise RuntimeError('Test evaluation already started; recover saved outputs, never rerun automatically')
    template=estimator(name);classes=sorted(np.unique(y).tolist())
    cv=StratifiedKFold(5,shuffle=True,random_state=42);innercv=StratifiedKFold(3,shuffle=True,random_state=42)
    scores=np.zeros((len(y),len(classes)));calibrated=np.zeros_like(scores);fold_metrics=[];folds=[]
    for number,(fit,val) in enumerate(cv.split(x,y)):
        cache=folder/f'calibration_fold_{number}.npz'
        inner_rows=[]
        if cache.exists():
            saved=np.load(cache,allow_pickle=False);assert saved['validation'].tolist()==np.asarray(indices)[val].tolist()
            scores[val]=saved['scores'];calibrated[val]=saved['calibrated']
            inner_rows=json.loads((folder/f'calibration_fold_{number}.json').read_text())['inner']
        else:
            raw=np.zeros((len(fit),len(classes)))
            for a,b in innercv.split(x[fit],y[fit]):
                fitted=clone(template).fit(x[fit[a]],y[fit[a]])
                assert fitted.classes_.tolist()==classes
                raw[b]=response(fitted,x[fit[b]])
                inner_rows.append({'fit':np.asarray(indices)[fit[a]].tolist(),'validation':np.asarray(indices)[fit[b]].tolist()})
            fitted=clone(template).fit(x[fit],y[fit]);scores[val]=response(fitted,x[val])
            calibrated[val]=calibrate(raw,y[fit],classes).predict_proba(scores[val])
            np.savez_compressed(cache,validation=np.asarray(indices)[val],scores=scores[val],calibrated=calibrated[val])
            save(folder/f'calibration_fold_{number}.json',{'inner':inner_rows})
        folds.append({'fit':np.asarray(indices)[fit].tolist(),'validation':np.asarray(indices)[val].tolist(),'inner':inner_rows})
        fold_metrics.append(probability_metrics(y[val],calibrated[val],classes))
        print(f'{name}: nested calibration outer fold {number+1}/5 complete',flush=True)
    counts=np.array([len(t.split()) for t in x])
    grid=[selective_metrics(y,calibrated,classes,counts,t1,t2) for t1 in protocol['thresholds']['t1'] for t2 in protocol['thresholds']['t2']]
    grid.append(selective_metrics(y,calibrated,classes,counts,0,0))
    policy=min(grid,key=lambda r:(-r['utility'],-r['coverage'],r['t1'],r['t2']))
    fit=clone(template).fit(x,y);mapping=calibrate(scores,y,classes)
    models=folder/'models';models.mkdir(exist_ok=True)
    for file,obj in [('pipeline.joblib',fit),('calibrator.joblib',mapping)]:
        archive(models/file,RUN);joblib.dump(obj,models/file)
    if not (models/'manifest.json').exists():
        save(models/'manifest.json',{'candidate':name,'classes':classes,'calibration':'sigmoid',
             'policy':policy,'selection_sha256':sha256(RUN/'selection.json'),
             'artifacts':{p.name:sha256(p) for p in models.glob('*.joblib')},
             'deployment_status':'Offline candidate; not integrated into or promoted to production API'})
    if not (folder/'calibration.json').exists():
        save(folder/'calibration.json',{'fold_metrics':fold_metrics,'folds':folds,
             'cv_mean':float(np.mean([r['macro_f1'] for r in fold_metrics])),
             'cv_std':float(np.std([r['macro_f1'] for r in fold_metrics])),
             'threshold_grid':grid,'selected_policy':policy,'method':'sigmoid'})
    # Hard boundary: all settings and candidates frozen before reading test rows.
    assert_protected()
    save(folder/'test_started.json',{'selection_sha256':sha256(RUN/'selection.json'),
         'calibration_sha256':sha256(folder/'calibration.json'),'model_sha256':sha256(models/'pipeline.joblib')})
    xt,yt=read_rows(split['test_row_indices'])
    p=mapping.predict_proba(response(fit,xt));prediction=np.asarray(classes)[p.argmax(1)]
    np.savez_compressed(folder/'test_probabilities.npz',row_indices=split['test_row_indices'],labels=yt,probabilities=p)
    measured={**probability_metrics(yt,p,classes),'per_class':classification_report(yt,prediction,output_dict=True,zero_division=0),
        'policy':selective_metrics(yt,p,classes,np.array([len(t.split()) for t in xt]),policy['t1'],policy['t2']),
        'evaluations':1}
    save(folder/'test_results.json',measured)
    if measured['macro_f1']>.75:
        # Investigate FIRST; never print a high score as a success or continue.
        train_clean=[masked_technical(t) for t in x];test_clean=[masked_technical(t) for t in xt]
        from src.label_masking import strip_phrases
        from src.preprocess import masked_feature_phrases
        from sklearn.metrics.pairwise import cosine_similarity
        v=TfidfVectorizer().fit(train_clean)
        nearest=cosine_similarity(v.transform(test_clean),v.transform(train_clean)).max(axis=1)
        leakage={'trigger':'>0.75 test macro F1','not_a_success_claim':True,
            'label_residual_documents':sum(strip_phrases(t,masked_feature_phrases())!=t for t in test_clean),
            'exact_train_test_overlap':sum(t in set(train_clean) for t in test_clean),
            'nearest_training_cosine_above_point9':int((nearest>.9).sum()),
            'interpretation':'Dictionary absence does not exclude author/template/source shortcuts; independent external validation remains necessary.'}
        save(folder/'leakage_investigation.json',leakage)
        raise RuntimeError('Suspicious-score guard triggered; leakage investigation saved before any success report.')
    r=json.loads((folder/'results.json').read_text());archive(folder/'results.json',RUN)
    r.update({'test':measured,'test_evaluations':1,'calibrated_cv':{'mean':float(np.mean([m['macro_f1'] for m in fold_metrics])),
               'std':float(np.std([m['macro_f1'] for m in fold_metrics]))},'promoted':False})
    (folder/'results.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    obj={'selected_model':name,'models':{name:{'cv_macro_f1_mean':r['cv_mean'],'cv_macro_f1_std':r['cv_std'],
       'macro_f1':measured['macro_f1'],'accuracy':measured['accuracy'],'fit_seconds':'not separately timed',
       'timing_description':'Original ranking CV and final calibrated test metrics; no model selection on test.'}}}
    append_row('EXP-'+r['id']+'-final','Frozen finalist '+name,obj,False,'Candidate only; waiting for user decision. Raw ranking CV is not calibrated CV.')
    print(f'{name}: single held-out evaluation saved.',flush=True)


def finalize():
    assert_protected();selection=json.loads((RUN/'selection.json').read_text())
    assert (RUN/'recommendation.json').exists(), 'Freeze the CV-only recommendation first'
    protocol=json.loads((RUN/'protocol.json').read_text());assert sha256(RUN/'protocol.json')==selection['protocol_sha256']
    split=json.loads((ROOT/'reports/split.json').read_text());x,y=read_rows(split['train_row_indices'])
    for name in selection['finalists']:finish_candidate(name,x,y,split['train_row_indices'],split,protocol)
    assert_protected()


def report():
    selection=json.loads((RUN/'selection.json').read_text());rows=[]
    for name in selection['ranking']:
        rows.append(json.loads((RUN/name/'results.json').read_text()))
    lines=['# Exploration summary','',
      'All values are freshly computed in each candidate results.json. Ranking uses raw five-fold training CV macro F1 only; final test metrics use sigmoid calibration. Only the frozen top two were evaluated on test. No production files were changed.',
      '', '| Rank | Candidate | CV macro F1 (mean ± std) | Calibrated test macro F1 | Test accuracy | Honest observation |', '|---|---|---|---|---|---|']
    for rank,r in enumerate(rows,1):
        test=r.get('test');f1=test['macro_f1'] if test else 'Not evaluated';accuracy=test['accuracy'] if test else 'Not evaluated'
        lines.append(f"| {rank} | {r['name']} | {r['cv_mean']} ± {r['cv_std']} | {f1} | {accuracy} | {r['observation']} |")
    lines+=['','## Reasoning recorded before fitting','']
    for _,name,title,reason in IDEAS: lines.append(f'- **{title}:** {reason} [Rationale]({name}/rationale.json)')
    lines+=['','## Finalist calibration and abstention','',
            '| Candidate | Calibrated CV F1 mean ± std | Brier | Log loss | ECE | T1 / T2 | Test coverage | Accuracy answered |',
            '|---|---|---|---|---|---|---|---|']
    for r in rows:
        if not r.get('test'):continue
        c=r['calibrated_cv'];t=r['test'];p=t['policy']
        lines.append(f"| {r['name']} | {c['mean']} ± {c['std']} | {t['brier_score']} | {t['log_loss']} | {t['ece_10_bins']} | {p['t1']} / {p['t2']} | {p['coverage']} | {p['accuracy_on_answered']} |")
    best=rows[0];control=next(r for r in rows if r['name']=='control_rf')
    recommendation=json.loads((RUN/'recommendation.json').read_text())
    lines+=['','## Recommendation and limits','',
      f"Training-CV preference: **{best['name']}**. Its mean minus the fresh control is {best['cv_mean']-control['cv_mean']}. This preference was fixed before test reporting. No promotion is performed; wait for the user's choice.",
      f"Practical recommendation frozen before test access: **{recommendation['recommendation']}**; preferred candidate **{recommendation['preferred_candidate']}**. {recommendation['reason']} CV gain {recommendation['cv_gain_over_control']}; declared variability gate {recommendation['variability_gate']}. This is a heuristic, not a significance test.",
      'Universal masking remains in every candidate, including the input to character features. This avoids deliberately restoring body label words, but cannot remove every correlated skill, template or source cue. Character fragments may encode residual source bias.',
      'The masks are fixed from existing policy; all feature vocabularies and IDF are fitted within each training fold. Nested calibration uses five outer and three inner training folds. Calibrated CV is conditional on training-selected candidate settings and is not a fully nested search estimate. Fold standard deviation is not a confidence interval.',
      'The same test split has been inspected historically, so it is not a pristine external benchmark. No external data or pretrained weights were downloaded. No extra calibration methods, per-class thresholds or ensembling were tried in this bounded run; this is not an exhaustive search.',
      'LinearSVC has no native probabilities. These artifacts are offline candidates, not API drop-ins: any later integration must preserve the existing probability-field semantics and explanation honesty. Production API code, privacy handling and ethical framing remain unchanged.']
    audit=json.loads((RUN/'training_mask_audit.json').read_text())
    lines.append(f"Training mask audit: {audit['residual_dictionary_documents']} residual dictionary documents and {audit['empty_documents']} empty documents among {audit['rows']} training rows. This verifies the explicit dictionary only, not all possible proxies.")
    if (RUN/'backend_tests.xml').exists() and (RUN/'frontend_tests.json').exists():
        import xml.etree.ElementTree as ET
        totals={k:0 for k in ['tests','failures','errors','skipped']}
        for suite in ET.parse(RUN/'backend_tests.xml').getroot().iter('testsuite'):
            for k in totals:totals[k]+=int(suite.attrib[k])
        totals['passed']=totals['tests']-totals['failures']-totals['errors']-totals['skipped']
        frontend=json.loads((RUN/'frontend_tests.json').read_text())
        verification={'backend':totals,'frontend':{'passed':frontend['numPassedTests'],
            'failed':frontend['numFailedTests'],'errors':frontend.get('numRuntimeErrorTestSuites',0),
            'skipped':frontend['numPendingTests']},'production_and_archives_unchanged':True}
        archive(RUN/'verification.json',RUN)
        (RUN/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
        f=verification['frontend']
        lines+=['','## Verification','',f"Backend: {totals['passed']} passed, {totals['failures']} failed, {totals['errors']} errors, {totals['skipped']} skipped. Frontend: {f['passed']} passed, {f['failed']} failed, {f['errors']} errors, {f['skipped']} skipped. Counts come from the saved full-suite reports. Production models, API code, privacy handling, and previous experiments are unchanged. Nothing was promoted."]
    path=RUN/'SUMMARY.md';archive(path,RUN);path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    assert_protected();print('\n'.join(lines[:12]))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['search','finalize','report'],required=True)
    args=parser.parse_args();{'search':search,'finalize':finalize,'report':report}[args.phase]()
