# Business Entity Resolution: baseline pipeline

Blocking -> pairwise LightGBM -> threshold tuned for macro F0.5. No external lookups, no country-specific
logic (country is only used as a blocking partition and a same-country flag), no pretrained models.

## Layout
```
src/normalize.py   accent stripping / transliteration, abbreviations, legal-suffix split, postcode extraction
src/blocking.py    3-channel candidate generation (name char-ngram TF-IDF, name+address word TF-IDF, postcode)
src/features.py    ~45 pair features (similarities, exact flags, competition ranks/gaps)
src/metrics.py     macro F0.5 (singletons count)
src/pipeline.py    end-to-end CLI
src/synthetic.py   tiny fake US/India/France data, smoke test only
```

## Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# from code/business_entity_resolution/ ; DATA has train/ and test/ subfolders
python3 src/pipeline.py --data-dir ../../dataset --out-dir ../../output --work-dir work

# validate (script from the organisers), from student_resource/
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
Useful flags: `--cache` (reuse pair features between runs), `--k-name/--k-word/--k-pc` (candidates per channel),
`--folds`, `--thr` (override tuned threshold), `--by-country auto|yes|no`.

The run prints: ground-truth stats, blocking recall on train, out-of-fold macro F0.5 (grouped by Source 1
entity), per-country OOF scores, a leave-one-country-out check (proxy for unseen France), and the predicted
match rate per test country (watch for France deviating a lot from US/India).

## Method
1. **Normalise**: NFKC, `anyascii` transliteration (offline table, strips accents, handles non-Latin scripts),
   casefold, punctuation removal, abbreviation expansion (corp/inc/ltd/pvt, st/rd/ave, French rue/av/bd ...),
   legal suffix split from the core name, postcode = last 5/6-digit token, house number.
2. **Block** per country group (auto-enabled when >=99% of true train pairs are same-country): union of
   top-K name char n-gram TF-IDF, top-K name+address word TF-IDF, and same-postcode candidates. The union is
   exactly what the model scores and is written to `candidate_pairs.tsv`.
3. **Features** per pair: char/word TF-IDF cosines (name, address, combined), Jaro-Winkler, Levenshtein,
   token sort/set/partial ratios, Jaccard, exact core/sorted/initials/first/last-token flags, postcode and
   house-number equality, missingness, lengths, and rank/gap of the pair's score among the S1 entity's
   candidates and among the S1 entities competing for the same candidate.
4. **Model**: LightGBM on labelled pairs (hard negatives from blocking). 4-fold out-of-fold predictions grouped by
   S1 entity; the probability threshold and the "each S2/S3 record goes to its best S1 entity only" rule are
   chosen to maximise macro F0.5 (empty predictions for singletons earn 1.0).
5. **Output**: final model retrained on all training pairs; matches = pairs above threshold.

## Licences
LightGBM (MIT), scikit-learn (BSD), rapidfuzz (MIT), anyascii (ISC), pandas/numpy/scipy (BSD).
No neural model is used, so the "MIT/Apache 2.0, up to 8B parameters" model constraint is trivially met.
