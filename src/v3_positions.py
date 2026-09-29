"""Training-only label-position measurement, before candidate fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import ROOT,sha256
from src.production_v2 import save
from src.v3_features import clean_technical,variant_tokens,occurrences


def main():
    out=ROOT/'experiments/v3'; out.mkdir(exist_ok=True)
    if (out/'position_results.json').exists(): raise ValueError('Measurement already exists')
    split=json.loads((ROOT/'reports/split.json').read_text())
    data=pd.read_csv(ROOT/'data/resumes.csv',keep_default_na=False).loc[split['train_row_indices']]
    first_ends=[]; matches=early=docs_early=0; rows=[]
    for row in data.itertuples():
        spans=occurrences(clean_technical(row.Resume).split(),variant_tokens()[row.Category])
        if spans:
            first_ends.append(spans[0][1]); docs_early+=spans[0][0]<20
        front=sum(start<20 for start,end in spans)
        matches+=len(spans); early+=front
        rows.append({'category':row.Category,'occurrences':len(spans),'early':front,'positive':bool(spans)})
    # Predeclared descriptive rule: ceil upper-quartile first-match END position.
    # This targets the first mention in most positive training resumes, not all
    # repetitions. K is not optimized against classification performance.
    k=int(np.ceil(np.quantile(first_ends,0.75)))
    result={'scope':'training partition only','split_sha256':sha256(ROOT/'reports/split.json'),
        'train_rows':len(data),'positive_documents':len(first_ends),
        'own_label_occurrences':matches,'occurrences_first_20':early,'occurrences_later':matches-early,
        'first_20_occurrence_fraction':early/matches,'later_occurrence_fraction':(matches-early)/matches,
        'positive_docs_with_first_match_in_20':int(docs_early),'positive_docs_first_20_fraction':docs_early/len(first_ends),
        'first_match_end_quantiles':{str(q):float(np.quantile(first_ends,q)) for q in [0.25,0.5,0.75,0.9]},
        'k':k,'k_rule':'ceil 75th percentile of first own-label match end among positive TRAIN documents; fixed before candidate scoring',
        'cap':2,'cap_rule':'Fixed two raw counts per label-containing feature; not tuned on test',
        'tokenization':'Technical-preserving cleaned tokens; canonical category/variant forms; start positions below 20 are early. Not identical to the raw-text audit denominator.'}
    save(out/'position_results.json',result)
    save(out/'protection.json',{str(p.relative_to(ROOT)):sha256(p)
        for folder in ['models','experiments/baseline_v1','experiments/leakage_impact','experiments/production_v2']
        for p in (ROOT/folder).rglob('*') if p.is_file()})
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
