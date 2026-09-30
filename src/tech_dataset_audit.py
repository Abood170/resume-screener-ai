"""Audit the explicitly approved Jillani dataset; no model fitting or split changes.

Run with python -m src.tech_dataset_audit. Downloads only if the separate CSV
does not exist. Reports contain aggregate statistics and row IDs, never text.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data import ROOT, sha256
from src.label_masking import VARIANTS

OUT = ROOT / 'reports/tech_dataset_audit'
SOURCE = 'https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset'
URL = 'https://www.kaggle.com/api/v1/datasets/download/jillanisofttech/updated-resume-dataset'


def norm(text: str) -> str:
    return re.sub(r'[\W_]+', ' ', text.casefold()).strip()


def stats(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {k: float(v) for k, v in zip(
        ['min', 'p25', 'median', 'p75', 'p90', 'p95', 'p99', 'max'],
        np.quantile(a, [0, .25, .5, .75, .9, .95, .99, 1]))} | {'mean': float(a.mean()), 'count': len(a)}


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = ROOT / 'reports/tech_dataset_audit_archive' / stamp
    protected = [ROOT / 'data/resumes.csv', ROOT / 'reports/split.json']
    for folder in ['models', 'experiments']:
        protected += [p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    if OUT.exists():
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copytree(OUT, archive / 'previous_report')
    OUT.mkdir(parents=True, exist_ok=True)
    target = ROOT / 'data/resumes_tech.csv'
    if not target.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read(20_000_001)
        if len(payload) > 20_000_000:
            raise ValueError('Archive exceeds expected bounded download size')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            members = [n for n in z.namelist() if n.endswith('UpdatedResumeDataSet.csv')]
            if len(members) != 1 or z.getinfo(members[0]).file_size > 20_000_000:
                raise ValueError('Unexpected archive contents')
            csv_bytes = z.read(members[0])
        df = pd.read_csv(io.BytesIO(csv_bytes))
        assert {'Category', 'Resume'} <= set(df.columns)
        provenance_path = ROOT / 'data/provenance.json'
        provenance = json.loads(provenance_path.read_text())
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(provenance_path, archive / 'provenance.json')
        target.write_bytes(csv_bytes)
        entry = {'source': SOURCE, 'download_url': URL, 'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                 'license_stated_by_publisher': 'CC0: Public Domain', 'archive_sha256': hashlib.sha256(payload).hexdigest(),
                 'csv_sha256': sha256(target), 'archive_member': members[0], 'local_path': 'data/resumes_tech.csv',
                 'raw_rows': len(df), 'raw_categories': int(df.Category.nunique()), 'transformation': 'None; original CSV bytes preserved'}
        provenance.setdefault('additional_sources', []).append(entry)
        provenance_path.write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    df = pd.read_csv(target)
    assert df[['Resume', 'Category']].notna().all().all()
    texts = df.Resume.astype(str)
    normalized = texts.map(norm)
    # Reuse earlier literal/normalized-boundary method; extend the frozen dictionary
    # for the new taxonomy. No variants are mined from resume contents.
    variants = {c: VARIANTS.get(c.upper(), [norm(c)]) for c in sorted(df.Category.unique())}
    variants.update({'Data Science': ['data science', 'data scientist', 'data scientists'],
        'Web Designing': ['web designing', 'web design', 'web designer'],
        'DevOps Engineer': ['devops engineer', 'devops engineering'],
        'DotNet Developer': ['dotnet developer', 'dot net developer', 'net developer'],
        'Health and fitness': ['health and fitness', 'health fitness'],
        'Mechanical Engineer': ['mechanical engineer', 'mechanical engineers', 'mechanical engineering'],
        'Civil Engineer': ['civil engineer', 'civil engineers', 'civil engineering'],
        'Electrical Engineering': ['electrical engineering', 'electrical engineer']})
    technical = {'Data Science','Web Designing','Java Developer','SAP Developer','Automation Testing',
                 'Python Developer','DevOps Engineer','Network Security Engineer','Database','Hadoop',
                 'ETL Developer','DotNet Developer','Blockchain','Testing'}
    records=[]
    for c,g in df.groupby('Category', sort=True):
        t=g.Resume.astype(str); n=t.map(norm)
        literal=sum(c.casefold() in s.casefold() for s in t)
        boundary=sum(any(' '+v+' ' in ' '+s+' ' for v in variants[c]) for s in n)
        records.append({'category': c, 'count': len(g), 'technical_developer_scope': c in technical,
                        'unique_normalized': n.nunique(), 'redundant_normalized': int(n.duplicated().sum()),
                        'literal_count': literal, 'literal_share': literal/len(g),
                        'variant_count': boundary, 'variant_share': boundary/len(g)})
    counts=pd.DataFrame(records); counts.to_csv(OUT/'category_audit.csv', index=False)
    # One corpus-wide vocabulary; all within-category pairs, excluding diagonal.
    # Descriptive analysis only: this is not a classifier or performance estimate.
    vectorizer=TfidfVectorizer(lowercase=True, strip_accents='unicode', ngram_range=(1,2), sublinear_tf=True)
    x=vectorizer.fit_transform(texts)
    nearest=[]; distributions=[]; all_pairs=[]; unique_pairs=[]
    for c,g in df.groupby('Category',sort=True):
        ids=g.index.to_numpy(); sim=cosine_similarity(x[ids]); tri=sim[np.triu_indices(len(ids),1)]
        all_pairs.extend(tri.tolist())
        np.fill_diagonal(sim,-1); maxima=sim.max(axis=1)
        unique_ids=g.loc[~normalized.loc[ids].duplicated()].index.to_numpy()
        unique_sim=cosine_similarity(x[unique_ids]); up=unique_sim[np.triu_indices(len(unique_ids),1)]
        unique_pairs.extend(up.tolist()); np.fill_diagonal(unique_sim,-1)
        distributions.append({'category':c, 'all_pairs':stats(tri), 'nearest_neighbor':stats(maxima),
            'pairs_gt_0_9':int((tri>.9).sum()), 'resumes_with_neighbor_gt_0_9':int((maxima>.9).sum()),
            'unique_normalized_pairs':stats(up), 'unique_resumes_with_neighbor_gt_0_9':int((unique_sim.max(axis=1)>.9).sum())})
        nearest.extend({'row_id':int(i),'category':c,'nearest_row_id':int(ids[j]),'cosine':float(v)}
                       for i,j,v in zip(ids,sim.argmax(axis=1),maxima))
    pd.DataFrame(nearest).to_csv(OUT/'nearest_neighbors.csv',index=False)
    primary=pd.read_csv(ROOT/'data/resumes.csv')
    split=json.loads((ROOT/'reports/split.json').read_text())
    # Existing held-out rows are NOT used for dataset comparisons or selection.
    train=primary.iloc[split['train_row_indices']]
    def formatting(t):
        return {'words':stats(t.map(lambda s:len(s.split()))),'characters':stats(t.map(len)),
                'newline_documents':int(t.str.contains(r'[\r\n]',regex=True).sum()),
                'html_tag_documents':int(t.str.contains(r'<[A-Za-z][^>]*>',regex=True).sum()),
                'non_ascii_documents':int(t.map(lambda s:any(ord(ch)>127 for ch in s)).sum()),
                'replacement_character_documents':int(t.str.contains('\ufffd',regex=False).sum()),
                'possible_mojibake_documents':int(t.str.contains('â|Ã|Â',regex=True).sum()),
                'under_50_words':int(t.map(lambda s:len(s.split())<50).sum())}
    group_sizes=normalized.value_counts()
    conflicts=df.assign(normalized=normalized).groupby('normalized').Category.nunique()
    result={'source':SOURCE,'csv_sha256':sha256(target),'rows':len(df),'categories':len(counts),
        'technical_scope_definition':'Software/data/network/web/testing roles; excludes business analysis, PMO, electrical/mechanical/civil engineering.',
        'technical_categories':int(counts.technical_developer_scope.sum()),
        'technical_rows':int(counts.loc[counts.technical_developer_scope,'count'].sum()),
        'duplicates':{'raw_redundant_rows':int(texts.duplicated().sum()),'normalized_unique':int(normalized.nunique()),
            'normalized_redundant_rows':int(normalized.duplicated().sum()),'normalized_redundant_share':float(normalized.duplicated().mean()),
            'rows_in_duplicate_groups':int(group_sizes[group_sizes>1].sum()),'conflicting_label_groups':int((conflicts>1).sum())},
        'similarity':{'method':'Whole-corpus TF-IDF word unigrams+bigrams, sublinear TF, unicode accent stripping; within-category cosine; self excluded. Unique-pair analysis removes normalized exact duplicates, retains same fitted IDF.',
            'all_within_category_pairs':stats(all_pairs),'unique_within_category_pairs':stats(unique_pairs),
            'nearest_within_category':stats([r['cosine'] for r in nearest]),
            'rows_neighbor_gt_0_9':sum(r['cosine']>.9 for r in nearest),
            'unique_rows_neighbor_gt_0_9':sum(r['unique_resumes_with_neighbor_gt_0_9'] for r in distributions),
            'by_category':distributions},
        'leakage':{'literal_count':int(counts.literal_count.sum()),'literal_share':float(counts.literal_count.sum()/len(df)),
            'variant_count':int(counts.variant_count.sum()),'variant_share':float(counts.variant_count.sum()/len(df)),
            'method':'Case-insensitive own-category substring plus punctuation/whitespace-normalized boundary-aware variants, as in the prior audit; variants saved separately. Literal substring can give false positives.'},
        'formatting':{'tech_all':formatting(texts),'primary_training_only':formatting(train.Resume.astype(str))},
        'overlap':{'primary_training_categories':sorted(train.Category.unique()),
            'exact_casefold_label_overlap':sorted(set(df.Category.str.casefold())&set(train.Category.str.casefold())),
            'normalized_text_matches_primary_training':int(normalized.isin(set(train.Resume.map(norm))).sum()),
            'heldout_not_inspected':True},'category_counts':records}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (OUT/'label_variants.json').write_text(json.dumps(variants,indent=2)+'\n',encoding='utf-8')
    assert all(sha256(ROOT/p)==h for p,h in hashes.items())
    (OUT/'protected_checksums.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['rows','categories','technical_categories','technical_rows','duplicates','leakage','overlap']}))


if __name__=='__main__':
    main()
