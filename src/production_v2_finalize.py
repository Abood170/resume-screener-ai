"""Attach verification evidence and publish generated docs; never train/evaluate."""
import json
import shutil
import xml.etree.ElementTree as ET
from api.main import app
from src.data import ROOT, sha256
from src.experiment_log import append_row
from src.label_masking import strip_phrases
from src.preprocess import masked_feature_phrases
from src.production_v2 import archive, save
from src.production_v2_report import main as render


def main():
    run=ROOT/'experiments/production_v2'
    before=json.loads((run/'openapi_before.json').read_text())
    assert before==app.openapi()
    protected=json.loads((run/'protected_before.json').read_text())
    assert all(sha256(ROOT/name)==digest for name,digest in protected.items())
    manifest=json.loads((ROOT/'models/manifest.json').read_text())
    for name,digest in manifest['artifacts'].items():
        assert sha256(ROOT/'models'/name)==sha256(run/'models'/name)==digest
    example=json.loads((run/'api_examples.json').read_text())
    for case in example.values():
        assert case['status_code']==200
        assert all(strip_phrases(term,masked_feature_phrases()).strip()==term for term in case['response']['top_terms'])
    suites=ET.parse(run/'backend_tests.xml').getroot().iter('testsuite')
    totals={k:0 for k in ['tests','failures','errors','skipped']}
    for suite in suites:
        for k in totals: totals[k]+=int(suite.attrib[k])
    totals['passed']=totals['tests']-totals['failures']-totals['errors']-totals['skipped']
    front=json.loads((run/'frontend_tests.json').read_text())
    verification={'backend':totals,'frontend':{'tests':front['numTotalTests'],
        'passed':front['numPassedTests'],'failed':front['numFailedTests'],
        'errors':front.get('numRuntimeErrorTestSuites',0),'skipped':front['numPendingTests']},
        'api_schema_unchanged':True,'prior_tests_and_baseline_archives_unchanged':True,
        'artifact_checksums_match':True,'example_influential_terms_exclude_mask_dictionary':True}
    assert totals['failures']==totals['errors']==0 and front['success']
    save(run/'verification.json',verification)
    r=json.loads((run/'results.json').read_text())
    r['verification']=verification
    r['api_examples']=example
    r['calibration']['training_cv']['timing_description']='Time is nested CV wall time with fixed sigmoid, not a model-method comparison or a single fit.'
    archive(run/'results.json',run)
    (run/'results.json').write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf-8')
    archive(ROOT/'results.json',run)
    shutil.copy2(run/'results.json',ROOT/'results.json')
    append_row('P1','Production universal masking; fixed RF + sigmoid',r,True,
        'Requested methodological promotion, not a test-score improvement. All-label and canonical lemma masking; thresholds chosen on training OOF. See production_v2/comparison.md.')
    render(final=True)
    save(run/'finalized.json',{'results_sha256':sha256(ROOT/'results.json'),
        'manifest_sha256':sha256(ROOT/'models/manifest.json'),'readme_sha256':sha256(ROOT/'README.md')})
    print(json.dumps(verification,indent=2))


if __name__=='__main__': main()
