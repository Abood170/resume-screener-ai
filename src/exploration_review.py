"""Freeze a practical recommendation using CV only, before held-out evaluation."""
import json
from src.exploration import RUN,assert_protected
from src.production_v2 import save
from src.data import sha256


def main():
    assert not list(RUN.glob('*/test_started.json')), 'Recommendation must precede test access'
    selection=json.loads((RUN/'selection.json').read_text())
    rows={name:json.loads((RUN/name/'results.json').read_text()) for name in selection['ranking']}
    best=rows[selection['ranking'][0]];control=rows['control_rf'];preferred=best
    reason='Highest training CV mean.'
    if best['name']=='word_char_svc':
        single=max([rows['word_svc'],rows['char_svc']],key=lambda r:r['cv_mean'])
        if best['cv_mean']-single['cv_mean']<=max(best['cv_std'],single['cv_std']):
            preferred=single
            reason='Combined features did not clear the declared variability/complexity gate; prefer the better single view.'
    gain=preferred['cv_mean']-control['cv_mean']
    passed=gain>max(preferred['cv_std'],control['cv_std'])
    save(RUN/'recommendation.json',{'preferred_candidate':preferred['name'],
         'recommendation':'Consider candidate after user review' if passed else 'Retain production; no convincing CV gain by the declared heuristic',
         'reason':reason,'cv_gain_over_control':gain,'variability_gate':max(preferred['cv_std'],control['cv_std']),
         'gate_passed':passed,'selection_sha256':sha256(RUN/'selection.json'),
         'rule_sha256':sha256(RUN/'recommendation_rule.json'),'test_access':False,'auto_promote':False})
    assert_protected()
    print('CV-only recommendation frozen before held-out evaluation.')


if __name__=='__main__':main()
