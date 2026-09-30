# Technical resume dataset audit

**Verdict: do not merge or replace production with this dataset as-is.** The nominal size is dominated by repeated resumes. It is useful for a small, explicitly limited technical-role pilot after a separately approved deduplication and evaluation design; it does not establish a solution to the current model ceiling.

Source: https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset. Publisher-stated license: CC0. Original bytes preserved in data/resumes_tech.csv; provenance, retrieval time and archive/CSV hashes are appended in data/provenance.json. CSV SHA-256: `c076ae68623eab41403dd6a863aef089b38e4c7e1294d17f9593b67ad2f181fa`.

Measured 962 rows across 25 categories. Under the explicit software/data/network/web/testing definition, 14 categories contain 600 rows. Business analysis/PMO and other engineering disciplines are excluded from that count.

## Categories and actual independent text support

| Category | Rows | Unique normalized texts | Technical scope | Literal own-label share | Boundary variant share |
|---|---|---|---|---|---|
| Advocate | 20 | 10 | False | 1.0 | 1.0 |
| Arts | 36 | 6 | False | 1.0 | 1.0 |
| Automation Testing | 26 | 7 | True | 0.8461538461538461 | 0.8461538461538461 |
| Blockchain | 40 | 5 | True | 1.0 | 1.0 |
| Business Analyst | 28 | 6 | False | 1.0 | 1.0 |
| Civil Engineer | 24 | 6 | False | 1.0 | 1.0 |
| Data Science | 40 | 10 | True | 0.9 | 1.0 |
| Database | 33 | 11 | True | 1.0 | 1.0 |
| DevOps Engineer | 55 | 7 | True | 0.9272727272727272 | 0.9272727272727272 |
| DotNet Developer | 28 | 7 | True | 0.0 | 0.8571428571428571 |
| ETL Developer | 40 | 5 | True | 1.0 | 1.0 |
| Electrical Engineering | 30 | 5 | False | 0.6 | 1.0 |
| HR | 44 | 10 | False | 1.0 | 1.0 |
| Hadoop | 42 | 7 | True | 1.0 | 1.0 |
| Health and fitness | 30 | 6 | False | 0.3333333333333333 | 0.5 |
| Java Developer | 84 | 13 | True | 0.9285714285714286 | 0.9285714285714286 |
| Mechanical Engineer | 40 | 5 | False | 1.0 | 1.0 |
| Network Security Engineer | 25 | 5 | True | 0.4 | 0.4 |
| Operations Manager | 40 | 4 | False | 1.0 | 1.0 |
| PMO | 30 | 3 | False | 1.0 | 1.0 |
| Python Developer | 48 | 6 | True | 1.0 | 1.0 |
| SAP Developer | 24 | 6 | True | 0.0 | 0.0 |
| Sales | 40 | 5 | False | 1.0 | 1.0 |
| Testing | 70 | 7 | True | 0.8571428571428571 | 0.8571428571428571 |
| Web Designing | 45 | 4 | True | 0.4 | 1.0 |

## Duplicate and template audit

Raw exact-text redundant rows: 796. Normalized redundant rows: 796; fraction 0.8274428274428275. Only 166 unique texts remain. 958 rows belong to repeated-text groups. Conflicting-label duplicate groups: 0.
Redundant means rows beyond the first in each group; group membership counts every member. Normalization casefolds and collapses punctuation/whitespace. Neither representation deletes any rows from the saved CSV.

Whole-corpus TF-IDF word unigrams+bigrams, sublinear TF, unicode accent stripping; within-category cosine; self excluded. Unique-pair analysis removes normalized exact duplicates, retains same fitted IDF.

| Population | Pair count | Median cosine | P90 | P99 | Maximum |
|---|---|---|---|---|---|
| all_within_category_pairs | 20577 | 0.06621882045190768 | 0.999999999999999 | 1.0000000000000182 | 1.0000000000000262 |
| unique_within_category_pairs | 538 | 0.059729386638137386 | 0.11177929442260717 | 0.1793723367830321 | 0.23935663236002885 |

958 original rows have a same-category neighbor above 0.9 cosine. After exact normalized deduplication, 0 unique rows have such a neighbor. Thus this measure identifies repeated copies, not additional high-similarity templates among the distinct texts. Small floating-point excursions above 1 are numerical artifacts, not meaningful similarities above 1.
Low pairwise similarity cannot exclude shared structural templates, semantic paraphrases, or a common source. This audit uses one declared lexical representation, not a comprehensive template detector. Splitting repeated rows independently would make validation unreliable; any future split must group duplicate families first. No split or model was created.

## Label shortcuts

Literal own-category substring: 808/962 (fraction 0.83991683991684). Boundary-aware variants: 880/962 (fraction 0.9147609147609148).
Case-insensitive own-category substring plus punctuation/whitespace-normalized boundary-aware variants, as in the prior audit; variants saved separately. Literal substring can give false positives.
The earlier dictionary is reused for matching categories and extended with explicit technical phrases in label_variants.json. These are lexical presence indicators, not proof of causal model reliance or ground-truth annotation quality. Broad terms and substring fragments can create false positives; unlisted aliases can create false negatives. No text or personal data is included in reports.

## Length and formatting

| Measure | New dataset, all rows | Existing dataset, training only |
|---|---|---|
| words: min | 19.0 | 113.0 |
| words: median | 329.0 | 758.0 |
| words: mean | 450.49792099792097 | 809.219254032258 |
| words: max | 2209.0 | 3994.0 |
| characters: min | 142.0 | 881.0 |
| characters: median | 2355.0 | 5895.5 |
| characters: mean | 3160.364864864865 | 6274.157258064516 |
| characters: max | 14816.0 | 30055.0 |
| newline_documents | 962 | 934 |
| html_tag_documents | 4 | 0 |
| non_ascii_documents | 737 | 1546 |
| replacement_character_documents | 0 | 5 |
| possible_mojibake_documents | 734 | 3 |
| under_50_words | 30 | 0 |

Mojibake is a heuristic flag for the characters â/Ã/Â, not a verified encoding-error count. Non-ASCII characters are not intrinsically errors. Raw-row summaries weight duplicated texts repeatedly. Shorter texts, pervasive line breaks and encoding artifacts indicate a source/style shift; no automatic repair was applied.

## Label overlap and conflicts

Case-insensitive identical labels: advocate, arts, hr, sales. Normalized exact-text matches against primary training rows: 0. The primary held-out partition was not inspected; this is not a full cross-dataset duplicate clearance.
There is no Information Technology label in the new category list. Its technical roles overlap conceptually with existing INFORMATION-TECHNOLOGY. Web Designing overlaps DESIGNER and potentially DIGITAL-MEDIA; Health and fitness overlaps FITNESS/HEALTHCARE; civil/mechanical/electrical roles overlap ENGINEERING and potentially CONSTRUCTION/AUTOMOBILE. These are taxonomy conflicts requiring a policy, not verified one-to-one label equivalences. Testing and Automation Testing also overlap internally; Database, Hadoop and ETL describe intersecting technologies rather than disjoint occupations.

## Decision required: no implementation performed

- **A — Replace broad categories:** a clearer technical scope, but discards existing coverage and leaves very few independent examples per role. A deduplicated pilot would require a newly approved dataset/split protocol.
- **B — Add categories:** retains breadth but mixes broad sectors, roles and technologies. Resolve identical labels and broad-versus-specific conflicts before any merging; source-specific formatting could become a shortcut.
- **C — Separate second stage:** preserves the broad model and isolates the technical taxonomy, but first-stage routing errors propagate and the second stage still has very limited unique support. Routing scope and uncertainty behavior require an explicit decision.

Recommendation: do not train on the repeated raw rows. If proceeding, approve a small deduplicated pilot and its taxonomy first; do not describe it as a validated general-purpose technical classifier. All production artifacts, prior experiments and saved split were checksum-verified unchanged. No held-out predictions, classifier training, merging, or promotion occurred.
