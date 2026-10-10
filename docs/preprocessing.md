# Data Preparation (GrammarLens-EWT)

This document describes how the GrammarLens dataset was built: sources, cleaning rules, error injection, balancing and splitting, and a manual check of label quality. All numbers come from the output of `scripts/build_dataset.py` and from `data/processed/stats.json`.

## 1. Data sources, versions and licenses

| Source | Use | Pinned version | License |
| --- | --- | --- | --- |
| [UD English-EWT](https://github.com/UniversalDependencies/UD_English-EWT) | Main data: 16,622 real English sentences with word-level grammatical annotation | commit `4a4d77f` | CC BY-SA 4.0 |
| [BLiMP](https://github.com/alexwarstadt/blimp) | External test: 14 files, 14,000 grammatical/ungrammatical sentence pairs | commit `3e56b06` | CC BY 4.0 |

- EWT files: `en_ewt-ud-train.conllu` (12,544 sentences), `en_ewt-ud-dev.conllu` (2,001), `en_ewt-ud-test.conllu` (2,077) and `LICENSE.txt`.
- Only the 14 BLiMP files that match our classes were used: 8 for `NOUN_NUM` and 6 for `SVA`.
- Raw files are downloaded by `scripts/download_data.py` into `data/raw/`. That the files are unchanged is verified with `data/raw/checksums.txt`.
- The dataset derived from EWT is shared under CC BY-SA 4.0.

## 2. Cleaning rules

`scripts/build_dataset.py` reads the sentences in order (train, dev, test) and applies the rules below in order. A sentence that fails a rule is dropped and is not checked against later rules.

| # | Rule | How it is applied | Why | Dropped |
| --- | --- | --- | --- | --- |
| 1 | Annotated typo or error | Any word carries `Typo=Yes`, `CorrectForm` or `CorrectSpaceAfter` | "Correct" sentences must really be correct | 1,347 |
| 2 | Length | Fewer than 5 or more than 40 words | Short ones are not sentences, long ones look like several sentences | 3,438 |
| 3 | Link or e-mail | Text contains `http://`, `https://`, `www.`, `@` or `.com` | Noise | 209 |
| 4 | Capital letters | More than 50% of the letters are uppercase | Noise | 130 |
| 5 | No finite verb | No word has `VerbForm=Fin` | Not a full sentence | 1,127 |
| 6 | Duplicate sentence | The lowercased text was already seen (including earlier splits) | Prevents leakage between training and test | 237 |
| | **Total dropped** | | | **6,488** |
| | **Remaining** | 16,622 − 6,488 | | **10,134** |

Note: the duplicate check also works across splits. If a sentence was seen in train, it is dropped when it appears again in dev or test.

## 3. Error injection

For every clean sentence we keep the original (`CORRECT`). In addition, for each error type, if the sentence has a suitable spot, we create a version with **exactly one error**. Errors are generated from the gold annotation of EWT (part of speech, lemma, dependency head). If no suitable spot exists, that error type is not generated for that sentence. Random choices use `seed=42`, so the script produces the same data on every run.

Apart from the changed word (and fixing the capitalization of the first word), the sentence stays as it is. If the generated text equals the original, the example is discarded.

| Class | Error type | How it is generated | Example |
| --- | --- | --- | --- |
| `CORRECT` | No error | The sentence itself | I am a preferred provider with most insurance companies. |
| `SVA` | Subject-verb disagreement | The number of a verb that has a visible subject is flipped (runs↔run, is↔are, was↔were) | He **are** concerned about the allocation … |
| `VERB_FORM` | Wrong verb form | A verb after a modal or "to", or after have/be, is put in a wrong form (can go → can goes, has gone → has went, is going → is go) | I want to **went** all over … |
| `DET` | Article error | With 50% probability "a/an" is deleted, with 50% probability a↔an is swapped | I recently took **an** rescue puppy … |
| `NOUN_NUM` | Singular/plural mismatch | A noun after a singular determiner (this, a, every …) is made plural, or a noun after a plural determiner (these, many, two …) is made singular. Subjects are excluded, otherwise the error would also be an SVA error | We are planning **this events** for Thursday … |
| `PREP` | Wrong preposition | A preposition (in, on, at, to, for, of, with, from, by, about) is replaced by a random different one from the list | … being given directly **with** John … |
| `WORD_ORDER` | Two words swapped | A determiner or adjective is swapped with the noun right after it | Later they kept the same name for **company the**. |

In word-order errors the words stay the same, so the bag of words of a `WORD_ORDER` sentence is identical to that of its correct version. Models that only count words (Naive Bayes) therefore cannot separate this class from `CORRECT`.

## 4. Balancing and splitting

- **Split:** EWT's own train/dev/test split is kept. No new random split is made. All versions of a sentence (correct and erroneous) stay in the same split, so the same sentence never appears in both train and test. The `sent_id` column makes this traceable.
- **Balancing:** Inside each split, all 7 classes are downsampled to the size of the smallest class in that split. The extra examples are dropped randomly with `seed=42`. The smallest class is `NOUN_NUM` in every split.
- **Shuffling:** Each split is shuffled and given an `id` (`train-00000`, `dev-00000`, `test-00000`, …).

### Class counts before balancing

| Class | train | dev | test |
| --- | --- | --- | --- |
| `CORRECT` | 7,960 | 1,086 | 1,088 |
| `SVA` | 5,021 | 705 | 674 |
| `VERB_FORM` | 4,056 | 487 | 514 |
| `DET` | 2,395 | 312 | 303 |
| `NOUN_NUM` | 2,287 | 311 | 302 |
| `PREP` | 5,191 | 629 | 638 |
| `WORD_ORDER` | 5,580 | 713 | 705 |

### Class counts after balancing

| Class | train | dev | test |
| --- | --- | --- | --- |
| Every class (7 classes) | 2,287 | 311 | 302 |
| Split total | 16,009 | 2,177 | 2,114 |

Total: **20,300 examples** (7 classes × 2,900).

### File format

Columns of `data/processed/train.csv`, `dev.csv` and `test.csv`: `id`, `sent_id`, `label`, `text`, `source`. `source` is the error-free version of the same sentence; it is for analysis only and is not a model input. The counts and settings are also stored in `data/processed/stats.json`.

## 5. Label check

From the test set, 20 examples per class (140 in total) were sampled at random (`random_state=42`) and checked by hand. For each example one question was asked: is the sentence really as its label says? (For `CORRECT`: none of the 6 error types is present. For the other classes: an error of that type is present.) The error-free `source` sentence was used for comparison. The results are in `results/label_audit.csv` (column `label_dogru_mu`).

| Class | Checked | Wrong labels | Rate |
| --- | --- | --- | --- |
| `CORRECT` | 20 | 0 | 0% |
| `SVA` | 20 | 0 | 0% |
| `VERB_FORM` | 20 | 0 | 0% |
| `DET` | 20 | 0 | 0% |
| `NOUN_NUM` | 20 | 0 | 0% |
| `PREP` | 20 | 0 | 0% |
| `WORD_ORDER` | 20 | 0 | 0% |
| **Total** | **140** | **0** | **0%** |

**Limits of this result:**

- The check was done by one person; there is no second annotator. The result reflects one person's judgment.
- 20 examples per class is a small sample. "0 wrong labels" does not mean the true rate is zero. It only means no wrong label was found in these 140 examples.
- In some `PREP` examples the sentence still sounds natural after the preposition change, so they are borderline. In this audit they were accepted as matching their label: "The staff **at** Allentown are friendly…", "the team **for** barton car wash…", "…a revised draft **by** the CDWR risk memo" and "…a friend out **at** Chicago…". One `SVA` example ("Google-**hate**-privacy argument") is also borderline, because it can be read as a compound modifier. A stricter judge could count these as wrong labels.
- Since prepositions are replaced randomly, borderline cases are most expected in the `PREP` class.

**Follow-up: evidence from the model error analysis.** After the DeBERTa-v3 predictions (seed 42) became available, we reviewed the 24 test sentences labeled `PREP` that the model predicted as `CORRECT` (details in `results/error_analysis.md`). In our reading, 13 of them are still grammatical after the preposition swap and 3 more are probably acceptable; only 2 look like real model misses. A similar pattern appears for `DET` (8 of the 13 `DET→CORRECT` sentences are still acceptable) and for `CORRECT` (10 of the 14 `CORRECT→DET` sentences contain an article problem in the original text). So the 0-of-20 result above is too optimistic for `PREP` and `DET`, and `CORRECT` is not perfectly clean either, because cleaning rule 1 only removes sentences that the treebank itself marks as typos. These sentences were selected because the model got them wrong, so they cannot be used to estimate the noise rate of a whole class; they are a lower bound on the noise the model exposed. The judgments come from one annotator with AI assistance and have no second opinion. We did not relabel or filter the data: all models were trained and tested on the dataset exactly as described in this document.

## 6. Limitations

- The errors are produced by a script, not written by real learners. Real errors are more diverse.
- Every sentence contains at most one error. Sentences with several errors are not in the dataset.
- `NOUN_NUM` errors are not generated on subjects.
- Erroneous sentences are corrupted versions of real sentences, while `CORRECT` is the original text.
- The `PREP` and `DET` classes contain label noise (an injected edit sometimes leaves a natural sentence), and `CORRECT` contains some original sentences with article errors (Section 5 and `results/error_analysis.md`).

## 7. Reproducing the data

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/build_dataset.py
```

Output: `data/raw/` (raw data and `checksums.txt`) and `data/processed/` (`train.csv`, `dev.csv`, `test.csv`, `stats.json`). Expected result: 10,134 clean sentences and 20,300 examples.
