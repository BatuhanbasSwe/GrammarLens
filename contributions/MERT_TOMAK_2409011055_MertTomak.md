# Mert Tomak · Data Cleaning and Preprocessing

Student number: 2409011055

## 1. Work done and responsibilities

My role in GrammarLens is data cleaning and preprocessing. Everyone else needs my output before training any model, so my work is the first link of the project chain.

- **Repository setup.** I accepted the collaborator invitation and committed the starter package (`scripts/download_data.py`, `scripts/build_dataset.py`, `requirements.txt`, `.gitignore`, `.gitattributes`) from my own GitHub account.
- **Raw data download.** I ran `scripts/download_data.py`. It downloaded the three UD English-EWT files (train 12,544, dev 2,001, test 2,077 sentences) with the license, and 14 BLiMP files (14,000 minimal pairs). It also wrote `data/raw/checksums.txt`, 18 files in total.
- **Dataset build.** I ran `scripts/build_dataset.py`. The cleaning kept 10,134 of 16,622 sentences, and the error injection plus balancing produced 20,300 labeled examples (2,287 per class in train, 311 in dev, 302 in test). I committed `data/raw/` and `data/processed/`.
- **Course dataset table.** I registered our dataset in the class dataset table, with the description of our self-built, script-generated data and BLiMP as external test.
- **Data documentation.** I wrote `docs/preprocessing.md`: sources and licenses, every cleaning rule with its count, how each error type is generated, balancing and splitting, label check results and limitations.
- **Label check.** I sampled 20 test examples per class (140 in total) and checked by hand whether each label is right. A second full review of all 140 examples marked 3 labels as wrong. Results are in `results/label_audit.csv`.
- **Error analysis.** I analyzed the mistakes of DeBERTa-v3 (seed 42, trained by Batuhan) and wrote `results/error_analysis.md`.

- **Review feedback.** After Batuhan rebuilt the dataset from scratch and confirmed identical content, I applied his notes: CSV files are written with LF line endings, the audit file uses English column names and values, and the documents no longer mention a seed that was still running.

The second label review and the error analysis were done with AI assistance, and the judgments are mine and subjective (one annotator, no second opinion). This is stated in both documents.

## 2. Related code files

| File | What it does |
| --- | --- |
| `scripts/download_data.py` | Starter script (from the team's starter package). Downloads EWT, its license and 14 BLiMP files into `data/raw/` and writes `checksums.txt`. I ran it unchanged. |
| `scripts/build_dataset.py` | Starter script (same package). Cleans EWT, injects errors, balances the classes and writes the CSV files. I documented how it works and made one change: the CSV files are now written with `lineterminator="\n"`, so they are byte-identical on every operating system (suggested by Batuhan, who rebuilt the data from scratch and got identical content). The cleaning rules and the error injection are unchanged. |
| `requirements.txt`, `.gitignore`, `.gitattributes` | Starter configuration files that I committed. `.gitignore` keeps the virtual environment, `__pycache__` and model weights out of the repository. |
| `data/raw/` | Raw EWT (`UD_English-EWT/`), BLiMP (`blimp/`) and `checksums.txt`. |
| `data/processed/` | `train.csv`, `dev.csv`, `test.csv` and `stats.json`. |
| `docs/preprocessing.md` | Data preparation document. |
| `results/label_audit.csv` | 140 hand-checked test examples with columns `label_correct` (`yes`/`no`) and `note`. |
| `results/error_analysis.md` | Error analysis of DeBERTa-v3 seed 42 with 6 findings and 20 example sentences. |

## 3. How the code works

`scripts/build_dataset.py` has four parts:

1. **Reading.** `read_conllu` reads the `.conllu` files sentence by sentence. For every word it keeps the form, lemma, universal and PTB part of speech, morphological features, head and dependency relation. Multiword tokens (such as "don't") are stored as ranges. From the whole treebank the script also builds an inflection lexicon: for each (lemma, PTB tag) pair it remembers the most frequent word form. `inflect` uses it to find, for example, the plural of a noun.
2. **Cleaning.** `main` applies six rules in order and counts what each one drops: annotated typo, length outside 5-40 words, link or e-mail, mostly uppercase, no finite verb, duplicate (also across splits).
3. **Error injection.** For every clean sentence `main` keeps the original as `CORRECT` and calls six injector functions (`make_sva`, `make_verb_form`, `make_det`, `make_noun_num`, `make_prep`, `make_word_order`). Each one searches the gold annotation for a suitable spot, picks one at random (`seed=42`) and changes exactly one word or one pair of words; if no spot exists it returns nothing. `render` turns the changed word list back into text and respects `SpaceAfter=No` and multiword tokens. If the result equals the original, the example is dropped.
4. **Balancing and output.** Inside each split every class is downsampled to the size of the smallest class (`NOUN_NUM`). Each split is shuffled, gets ids like `test-00042`, and is written to CSV together with `stats.json`.

`scripts/download_data.py` downloads the pinned EWT and BLiMP files, writes the checksums and prints the sentence counts.

## 4. Algorithms, methods and libraries

- **Methods:** CoNLL-U parsing, rule-based cleaning, rule-based error injection based on gold Universal Dependencies annotation, an inflection lexicon built from the treebank, class balancing by random downsampling, manual label audit, and confusion-matrix based error analysis.
- **Libraries:** `pandas` (tables, sampling, `crosstab`), and the Python standard library (`re`, `random`, `collections`, `json`, `pathlib`).
- **Data:** UD English-EWT (CC BY-SA 4.0) as main data and BLiMP (CC BY 4.0) as external test.

## 5. Running and testing

From the repository root:

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/build_dataset.py
```

**Checks I did.** The output has to match the expected numbers of the task file: 16,622 raw sentences, 10,134 kept, 20,300 examples, and 2,287 / 311 / 302 examples per class in train / dev / test. All numbers matched. I also checked that `data/processed/` has the four expected files and that `git status` did not list the virtual environment.

**Label check.** For each class I sampled 20 test examples with `random_state=42` and compared the text with the error-free `source` sentence.

**Error analysis.** I joined the prediction file `results/preds/deberta-v3_seed42.csv` with `data/processed/test.csv` on `id` and printed the confusion matrix with:

```bash
python -c 'import pandas as pd; d=pd.read_csv("results/preds/deberta-v3_seed42.csv"); o=["CORRECT","SVA","VERB_FORM","DET","NOUN_NUM","PREP","WORD_ORDER"]; print(pd.crosstab(d.label,d.pred).reindex(index=o,columns=o,fill_value=0))'
```

## 6. Experiments and results

| Metric | Value |
| --- | --- |
| Raw EWT sentences | 16,622 |
| Dropped: annotated typo / length / link / uppercase / no finite verb / duplicate | 1,347 / 3,438 / 209 / 130 / 1,127 / 237 |
| Clean sentences | 10,134 |
| Final examples (7 classes × 2,900) | 20,300 |
| Per class: train / dev / test | 2,287 / 311 / 302 |
| BLiMP minimal pairs downloaded | 14,000 |
| Label check: wrong labels in 140 examples | 3 (2.1%): `SVA` 1/20, `PREP` 2/20, all other classes 0/20 |
| DeBERTa-v3 seed 42: wrong test predictions | 101 of 2,114 |
| Error analysis: reviewed errors | 51 of 101 (`PREP→CORRECT` 24, `DET→CORRECT` 13, `CORRECT→DET` 14) |

**Main findings of the error analysis** (details in `results/error_analysis.md`):

- 74 of the 101 errors involve `CORRECT`, and the four confusions `PREP→CORRECT`, `CORRECT→DET`, `DET→CORRECT` and `CORRECT→PREP` make up 58 of them (57%).
- Of the 24 `PREP→CORRECT` sentences, 13 are still grammatical after the preposition swap and only 2 look like real model misses.
- Of the 13 `DET→CORRECT` sentences, 8 are still acceptable after the article change. Of the 14 `CORRECT→DET` sentences, 10 contain an article problem in the original text.
- So part of the measured model errors is label noise, but the reviewed sentences are exactly those the model got wrong, so the share of noise in a whole class cannot be estimated from them.

I report no model scores here; the model results belong to the people who trained the models.

## 7. Problems, gaps and improvement suggestions

**Problems and gaps**

- The injected `PREP` and `DET` errors sometimes leave a natural sentence, so these two classes contain label noise. `CORRECT` is not perfectly clean either, because rule 1 only removes sentences that the treebank itself marks as typos.
- The label check was done by one annotator, the 20 examples per class are a small sample, and borderline cases (listed in `results/label_audit.csv`) can be judged differently.
- The error analysis covers one seed (42) and 51 of the 101 errors. The other 50 errors and the other seeds were not reviewed.
- Each sentence has at most one error, and the errors are produced by a script, not written by real learners.
- I did not change the cleaning rules or the error injection (the only change to `scripts/build_dataset.py` is the line terminator of the CSV files), and I did not relabel or filter any data.

**Improvement suggestions (not done)**

- Filter out injected sentences that still look acceptable, for example with a language model score or a grammar checker, and drop original sentences with article errors.
- Add a second annotator and a larger audit sample, so that the label noise rate can be estimated per class.
- Add sentences with several errors, and test on real learner errors.
