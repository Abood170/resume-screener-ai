"""Generate audit prose exclusively from saved aggregate results."""
import json
from src.tech_dataset_audit import OUT


def main():
    r=json.loads((OUT/'results.json').read_text()); d=r['duplicates']; s=r['similarity']; f=r['formatting']; l=r['leakage']
    lines=['# Technical resume dataset audit','',
        '**Verdict: do not merge or replace production with this dataset as-is.** The nominal size is dominated by repeated resumes. It is useful for a small, explicitly limited technical-role pilot after a separately approved deduplication and evaluation design; it does not establish a solution to the current model ceiling.',
        '',f"Source: {r['source']}. Publisher-stated license: CC0. Original bytes preserved in data/resumes_tech.csv; provenance, retrieval time and archive/CSV hashes are appended in data/provenance.json. CSV SHA-256: `{r['csv_sha256']}`.",
        '',f"Measured {r['rows']} rows across {r['categories']} categories. Under the explicit software/data/network/web/testing definition, {r['technical_categories']} categories contain {r['technical_rows']} rows. Business analysis/PMO and other engineering disciplines are excluded from that count.",
        '', '## Categories and actual independent text support','',
        '| Category | Rows | Unique normalized texts | Technical scope | Literal own-label share | Boundary variant share |',
        '|---|---|---|---|---|---|']
    for c in r['category_counts']:
        lines.append(f"| {c['category']} | {c['count']} | {c['unique_normalized']} | {c['technical_developer_scope']} | {c['literal_share']} | {c['variant_share']} |")
    lines+=['','## Duplicate and template audit','',
        f"Raw exact-text redundant rows: {d['raw_redundant_rows']}. Normalized redundant rows: {d['normalized_redundant_rows']}; fraction {d['normalized_redundant_share']}. Only {d['normalized_unique']} unique texts remain. {d['rows_in_duplicate_groups']} rows belong to repeated-text groups. Conflicting-label duplicate groups: {d['conflicting_label_groups']}.",
        'Redundant means rows beyond the first in each group; group membership counts every member. Normalization casefolds and collapses punctuation/whitespace. Neither representation deletes any rows from the saved CSV.',
        '',s['method'],'',
        '| Population | Pair count | Median cosine | P90 | P99 | Maximum |', '|---|---|---|---|---|---|']
    for key in ['all_within_category_pairs','unique_within_category_pairs']:
        a=s[key];lines.append(f"| {key} | {a['count']} | {a['median']} | {a['p90']} | {a['p99']} | {a['max']} |")
    lines += ['',f"{s['rows_neighbor_gt_0_9']} original rows have a same-category neighbor above 0.9 cosine. After exact normalized deduplication, {s['unique_rows_neighbor_gt_0_9']} unique rows have such a neighbor. Thus this measure identifies repeated copies, not additional high-similarity templates among the distinct texts. Small floating-point excursions above 1 are numerical artifacts, not meaningful similarities above 1.",
        'Low pairwise similarity cannot exclude shared structural templates, semantic paraphrases, or a common source. This audit uses one declared lexical representation, not a comprehensive template detector. Splitting repeated rows independently would make validation unreliable; any future split must group duplicate families first. No split or model was created.',
        '', '## Label shortcuts','',
        f"Literal own-category substring: {l['literal_count']}/{r['rows']} (fraction {l['literal_share']}). Boundary-aware variants: {l['variant_count']}/{r['rows']} (fraction {l['variant_share']}).",
        l['method'],
        'The earlier dictionary is reused for matching categories and extended with explicit technical phrases in label_variants.json. These are lexical presence indicators, not proof of causal model reliance or ground-truth annotation quality. Broad terms and substring fragments can create false positives; unlisted aliases can create false negatives. No text or personal data is included in reports.',
        '', '## Length and formatting','',
        '| Measure | New dataset, all rows | Existing dataset, training only |','|---|---|---|']
    for field in ['words','characters']:
        for stat in ['min','median','mean','max']:
            lines.append(f"| {field}: {stat} | {f['tech_all'][field][stat]} | {f['primary_training_only'][field][stat]} |")
    for field in ['newline_documents','html_tag_documents','non_ascii_documents','replacement_character_documents','possible_mojibake_documents','under_50_words']:
        lines.append(f"| {field} | {f['tech_all'][field]} | {f['primary_training_only'][field]} |")
    lines += ['', 'Mojibake is a heuristic flag for the characters â/Ã/Â, not a verified encoding-error count. Non-ASCII characters are not intrinsically errors. Raw-row summaries weight duplicated texts repeatedly. Shorter texts, pervasive line breaks and encoding artifacts indicate a source/style shift; no automatic repair was applied.',
        '', '## Label overlap and conflicts','',
        f"Case-insensitive identical labels: {', '.join(r['overlap']['exact_casefold_label_overlap'])}. Normalized exact-text matches against primary training rows: {r['overlap']['normalized_text_matches_primary_training']}. The primary held-out partition was not inspected; this is not a full cross-dataset duplicate clearance.",
        'There is no Information Technology label in the new category list. Its technical roles overlap conceptually with existing INFORMATION-TECHNOLOGY. Web Designing overlaps DESIGNER and potentially DIGITAL-MEDIA; Health and fitness overlaps FITNESS/HEALTHCARE; civil/mechanical/electrical roles overlap ENGINEERING and potentially CONSTRUCTION/AUTOMOBILE. These are taxonomy conflicts requiring a policy, not verified one-to-one label equivalences. Testing and Automation Testing also overlap internally; Database, Hadoop and ETL describe intersecting technologies rather than disjoint occupations.',
        '', '## Decision required: no implementation performed','',
        '- **A — Replace broad categories:** a clearer technical scope, but discards existing coverage and leaves very few independent examples per role. A deduplicated pilot would require a newly approved dataset/split protocol.',
        '- **B — Add categories:** retains breadth but mixes broad sectors, roles and technologies. Resolve identical labels and broad-versus-specific conflicts before any merging; source-specific formatting could become a shortcut.',
        '- **C — Separate second stage:** preserves the broad model and isolates the technical taxonomy, but first-stage routing errors propagate and the second stage still has very limited unique support. Routing scope and uncertainty behavior require an explicit decision.',
        '', 'Recommendation: do not train on the repeated raw rows. If proceeding, approve a small deduplicated pilot and its taxonomy first; do not describe it as a validated general-purpose technical classifier. All production artifacts, prior experiments and saved split were checksum-verified unchanged. No held-out predictions, classifier training, merging, or promotion occurred.']
    (OUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': main()
