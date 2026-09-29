"""Generate candidate-only v3 evidence and append all search attempts to the log."""
import json
import xml.etree.ElementTree as ET
from src.data import ROOT,sha256
from src.production_v2 import save,archive
from src.experiment_log import append_row


def write(path,text,run):
    archive(path,run);path.write_text(text,encoding='utf-8')


def main():
    run=ROOT/'experiments/v3'
    r=json.loads((run/'results.json').read_text());pos=json.loads((run/'position_results.json').read_text())
    production=json.loads((ROOT/'experiments/production_v2/results.json').read_text())
    residual=json.loads((run/'residual_shortcuts.json').read_text())
    systems=dict(production['leakage_comparison'])
    systems['v3']={'cv_mean':r['cv_macro_f1_mean'],'cv_std':r['cv_macro_f1_std'],
                   **r['test']['after']}
    comparison=['| System | CV macro F1 (mean ± std) | Test macro F1 | Test accuracy |','|---|---|---|---|']
    for name,m in systems.items():
        comparison.append(f"| {name} | {m['cv_mean']} ± {m['cv_std']} | {m['macro_f1']} | {m['accuracy']} |")
    per=['| Category | Baseline v1 F1 | L1 F1 | Production v2 F1 | v3 F1 | v3 minus v2 |','|---|---|---|---|---|---|']
    for category in production['classes']:
        values=[s['per_class'][category]['f1-score'] for s in systems.values()]
        label='**'+category+'**' if category in ['INFORMATION-TECHNOLOGY','ENGINEERING','CONSULTANT','DESIGNER','DIGITAL-MEDIA'] else category
        per.append('| '+label+' | '+' | '.join(map(str,values))+' | '+str(values[-1]-values[-2])+' |')
    delta=r['test']['after']['macro_f1']-production['calibration']['test']['after']['macro_f1']
    cdelta=r['cv_macro_f1_mean']-production['calibration']['cv_fold_macro_f1']['sigmoid']['mean']
    conclusion=f"""v3 minus production v2: test macro F1 {delta}; calibrated CV macro F1 {cdelta}.

**Not promoted.** Production files remain unchanged pending the user's review. Model settings were selected from training CV using the predeclared standard-deviation gate; test scores did not choose settings or trigger promotion. CV improvement gate passed: {r['cv_gate_passed']}. The candidate is not proven leakage-free: capping body counts limits repetition, but even one explicit label mention remains predictive. Higher scores cannot be attributed exclusively to better content understanding or technical-token preservation without further training-only ablations.

The training-position audit found {pos['occurrences_first_20']}/{pos['own_label_occurrences']} matches within the first twenty cleaned tokens (fraction {pos['first_20_occurrence_fraction']}); {pos['occurrences_later']}/{pos['own_label_occurrences']} were later (fraction {pos['later_occurrence_fraction']}). {pos['positive_docs_with_first_match_in_20']}/{pos['positive_documents']} positive documents had an early first mention. K={pos['k']} is the ceiling of the training first-match end upper quartile. All-category prefix masking is inference-safe; body label-containing features are capped at {pos['cap']} raw counts before sublinear TF-IDF. The cap is per unigram/bigram feature, not a collective cap across different contextual bigrams. Technical shapes such as C++, C#, .NET, Node.js, CI/CD and alphanumeric versions survive via placeholder-and-restore; generic hyphenated words can also survive.

Residual own-label presence after prefix masking: {residual['own_label_present_after_prefix_mask']}/{residual['train_rows']} training resumes (fraction {residual['fraction']}). Frequency capping does not remove these occurrences. This is concrete evidence against claiming that v3 eliminates the known lexical shortcut.

The first three screens compare balanced LR, RF and LinearSVC, then eight sampled combinations are scored for each of the two highest-mean families. Every score is five-fold training CV. Settings and K are selected using this training partition; subsequent nested calibration CV is conditional on those choices, not a fully nested search estimate. Fold standard deviation is a heuristic gate, not a significance test. The repeatedly inspected holdout is not a pristine external benchmark.

Selected trial: {r['selected_trial']}; family: {r['selected_family']}. Fixed sigmoid calibration uses nested training-only OOF responses, and the final calibrator uses full-training OOF responses. No new external data or dependencies were added. If LinearSVC is chosen, legacy raw confidence in the isolated candidate API is explicitly a softmax-margin proxy, not a native probability. This would require an API semantic compatibility decision before production promotion; calibrated probabilities are fitted directly to decision margins, not this proxy.
"""
    policy=r['test']['policy']; threshold=r['thresholds']
    calibration=['| Metric | Raw candidate | Sigmoid candidate |','|---|---|---|']
    for key in ['macro_f1','accuracy','brier_score','log_loss','ece_10_bins']:
        calibration.append(f"| {key} | {r['test']['before'][key]} | {r['test']['after'][key]} |")
    body='# v3 candidate comparison\n\nGenerated from results.json, position_results.json and archived production results; unrounded values.\n\n'+'\n'.join(comparison)+'\n\n'+'\n'.join(per)+'\n\n'+conclusion+'\n\n'+'\n'.join(calibration)
    body+=f"\n\nTraining-only thresholds: T1={threshold['t1']}, T2={threshold['t2']}. Held-out coverage {policy['coverage']} ({policy['answered']}/{policy['total']}), accuracy among answered {policy['accuracy_on_answered']}. Short inputs below fifty whitespace words remain uncertain. The original utility, grid and tie-breaks were reused.\n"
    if (run/'api_example.json').exists():
        example=json.loads((run/'api_example.json').read_text())
        diagnostic=json.loads((run/'demo_diagnostics.json').read_text())
        missing=[term for term,present in diagnostic['fixed_demo_technical_features_in_selected_vocabulary'].items() if not present]
        body+='\n## Fixed Java/backend demo\n\n'+f"The actual /predict call returned {example['response']['calibrated_predicted_category']} with confidence {example['response']['calibrated_confidence']}; uncertainty reason: {example['response']['uncertainty_reason']}. IT rank within calibrated top three: {example['target_rank']}. The intended IT headline was not achieved. The input was fixed before results and was not rewritten to get a favorable prediction.\n\n"
        body+='Preserved technical spellings do not guarantee inclusion in the learned vocabulary after document-frequency and feature-budget filtering. Fixed demo terms absent from that vocabulary: '+', '.join(missing)+'. This is observed feature coverage, not a complete causal explanation of the prediction. No model or sample adjustments were made in response.\n\n```json\n'+json.dumps(example['response'],indent=2)+'\n```\n'
    write(run/'comparison.md',body,run)
    rows=r['search']['trials']; log=['| ID | Stage / family | Parameters | CV macro F1 (mean ± std) | Accepted by gate? |','|---|---|---|---|---|']
    for row in rows:
        log.append(f"| {row['id']} | {row['stage']} / {row['family']} | {json.dumps(row['params'],sort_keys=True)} | {row['mean']} ± {row['std']} | {row['kept_in_search']} |")
    write(run/'search_log.md','# Full v3 search log\n\nNo held-out scores were computed for individual search trials. Acceptance means retained as the incumbent during search, not production promotion. Final selected trial: '+r['selected_trial']+'.\n\n'+'\n'.join(log)+'\n',run)
    existing=(ROOT/'experiments/experiment_log.md').read_text(encoding='utf-8')
    for row in rows:
        if f"| {row['id']} |" in existing: continue
        metric={'selected_model':row['family'],'models':{row['family']:{
            'cv_macro_f1_mean':row['mean'],'cv_macro_f1_std':row['std'],
            'macro_f1':'not evaluated','accuracy':'not evaluated','fit_seconds':row['seconds'],
            'timing_description':'Time is measured five-fold fit/scoring cost, not one full-training fit.'}}}
        append_row(row['id'],row['stage']+' '+row['family']+' '+json.dumps(row['params'],sort_keys=True),metric,
                   row['kept_in_search'],'Accepted as search incumbent only; not promoted. Held-out data unused for this trial. See v3/search_log.md.')
    final_id=f'V3-{len(rows)+1}'
    if f'| {final_id} |' not in existing:
        metric={'selected_model':r['selected_family'],'models':{r['selected_family']:{
            'cv_macro_f1_mean':r['cv_macro_f1_mean'],'cv_macro_f1_std':r['cv_macro_f1_std'],
            'macro_f1':r['test']['after']['macro_f1'],'accuracy':r['test']['after']['accuracy'],
            'fit_seconds':r['timing']['final_fit_seconds']}}}
        append_row(final_id,'Frozen v3 candidate with sigmoid calibration',metric,False,
                   'Candidate only; pending user review. CV conditional on training-selected settings; residual body-label shortcuts. See v3/comparison.md.')
    protection=json.loads((run/'protection.json').read_text())
    assert all(sha256(ROOT/name)==digest for name,digest in protection.items())
    if (run/'backend_tests.xml').exists() and (run/'frontend_tests.json').exists():
        counts={k:0 for k in ['tests','failures','errors','skipped']}
        for suite in ET.parse(run/'backend_tests.xml').getroot().iter('testsuite'):
            for key in counts: counts[key]+=int(suite.attrib[key])
        counts['passed']=counts['tests']-counts['failures']-counts['errors']-counts['skipped']
        f=json.loads((run/'frontend_tests.json').read_text())
        save(run/'verification.json',{'backend':counts,'frontend':{'passed':f['numPassedTests'],
            'failed':f['numFailedTests'],'errors':f.get('numRuntimeErrorTestSuites',0),'skipped':f['numPendingTests']},
            'production_and_prior_experiments_unchanged':True})
    print('\n'.join(comparison))


if __name__=='__main__':main()
