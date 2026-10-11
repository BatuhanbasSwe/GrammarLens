# GrammarLens

**English grammatical error-type classification on a self-built dataset.** Given a sentence,
predict which of six error types it contains — or that it is correct. The dataset is built from
real web English (UD English-EWT) with scripted single-error injection, and the model ladder runs
from Naive Bayes to DeBERTa-v3.

| Model | Macro-F1 (test) | 95% CI | Accuracy | BLiMP pair acc. |
|---|---:|---:|---:|---:|
| Naive Bayes | 0.298 | 0.279–0.318 | 0.310 | 0.444 |
| CNN | 0.547 ± 0.009 | 0.530–0.563 | 0.550 | 0.483 |
| DistilBERT | 0.890 ± 0.002 | 0.878–0.902 | 0.892 | 0.943 |
| **DeBERTa-v3** | **0.952 ± 0.001** | 0.944–0.960 | 0.952 | **0.983** |

Test split: 2,114 sentences (7 classes × 302). Neural models: mean over seeds 13/42/2026;
CI = bootstrap 95% of the seed mean. Full table: `results/benchmark.csv`.

## The task

7-way classification. One class is clean text, six are error types, each produced by one scripted
edit of a real sentence:

| Class | Error | Example edit |
|---|---|---|
| `CORRECT` | none | the original sentence |
| `SVA` | subject–verb disagreement | runs ↔ run, is ↔ are |
| `VERB_FORM` | wrong verb form | can go → can goes, has gone → has went |
| `DET` | article error | a/an deleted or swapped |
| `NOUN_NUM` | singular/plural mismatch | this event → this events |
| `PREP` | wrong preposition | swapped within a 10-item list |
| `WORD_ORDER` | two adjacent words swapped | the company → company the |

A `WORD_ORDER` sentence has the same bag of words as its correct version — a pure word-counting
model cannot see it. That is the point of the ladder: each rung isolates a capability.

## Dataset: GrammarLens-EWT

Full documentation with every number and rule: **[docs/preprocessing.md](docs/preprocessing.md)**.

1. **Source** — [UD English-EWT](https://github.com/UniversalDependencies/UD_English-EWT)
   (commit `4a4d77f`), 16,622 real web sentences with gold POS/lemma/dependency annotation.
2. **Cleaning** — 6 ordered rules drop 6,488 sentences (annotated typos 1,347 · length 3,438 ·
   links/e-mail 209 · all-caps 130 · no finite verb 1,127 · duplicates across splits 237);
   10,134 remain. Duplicates are dropped *across* splits, so no sentence appears in both train
   and test.
3. **Error injection** — errors are generated from the gold annotation: exactly one edit per
   sentence, six types; if the sentence has no suitable spot, that type is skipped. Deterministic
   (`seed=42`).
4. **Balancing** — EWT's own train/dev/test split is kept; inside each split all 7 classes are
   downsampled to the smallest class. Final: **20,300 examples** (7 × 2,287 / 311 / 302).

Label quality was audited by hand: 140 sampled test examples → 3 wrong labels (2.1%); a
model-driven review found more noise concentrated in `PREP`/`DET` (details in
`results/error_analysis.md` and preprocessing.md §5).

## Models

Every model sees identical splits and is scored identically (`src/metrics.py`).

| Model | Key facts |
|---|---|
| **Naive Bayes** (`src/train_nb.py`) | lowercase unigram counts + MultinomialNB (α=1.0); deterministic; the floor. |
| **CNN** (`src/train_cnn.py`) | TextCNN trained from scratch: 100-d embeddings, Conv1d windows 3/4/5 × 100 filters, max-pool, dropout 0.5; Adam 1e-3, batch 64, ≤30 epochs, best epoch by dev macro-F1. Sees only 3–5-word windows. |
| **DistilBERT** (`src/train_transformer.py`) | `distilbert-base-uncased` fine-tuned; lr 3e-5, batch 32, 3 epochs, bf16 autocast with fp32 weights. |
| **DeBERTa-v3** (`src/train_transformer.py`) | `deberta-v3-base` fine-tuned; lr 2e-5, rest as DistilBERT. Our strongest model. |

No hyperparameter search was done; all selection used the dev split only. Model choice per epoch:
weights of the best dev macro-F1 epoch.

## Repository structure

```
data/
  raw/UD_English-EWT/    pinned, unmodified EWT treebank (3 .conllu + license)
  raw/blimp/             14 pinned BLiMP files used as external test
  raw/checksums.txt      integrity check for all raw files
  processed/             the built dataset: train/dev/test.csv + stats.json
                         (columns: id, sent_id, label, text, source)
docs/
  preprocessing.md       full dataset-construction documentation
scripts/
  download_data.py       fetches the pinned raw data and verifies checksums
  build_dataset.py       cleaning → error injection → balancing → processed CSVs
src/
  metrics.py             shared metrics (macro-F1, per-class, confusion) + result writers
  train_nb.py            trains/evaluates the Naive Bayes baseline
  train_cnn.py           trains/evaluates the TextCNN baseline
  train_transformer.py   fine-tunes DistilBERT or DeBERTa-v3
  evaluate_blimp.py      BLiMP minimal-pair evaluation (pair + type accuracy)
  build_benchmark.py     validates all runs, writes results/benchmark.csv + figures
  analyze_long_distance.py  CNN SVA error rate vs. subject–verb distance
results/
  *_seed*.json           per-run test metrics, BLiMP scores, hardware, train time
  benchmark.csv          official aggregated table
  preds/                 id,label,pred for every final run (2,114 rows each)
  label_audit.csv        manual label-quality audit of 140 sampled examples
  error_analysis.md      DeBERTa-v3 error analysis + label-noise findings
  figures/               benchmark charts (SVG)
contributions/           per-member contribution reports
presentation/            slides + speaker script for the project talk
```

## Reproducing

```bash
pip install -r requirements.txt        # pandas, scikit-learn, torch>=2.0, transformers>=5.0, sentencepiece, protobuf

python scripts/download_data.py        # raw data + checksum verification
python scripts/build_dataset.py        # → data/processed/ (expected: 10,134 clean, 20,300 examples)

python src/train_nb.py --final                        # Naive Bayes (seed 42)
python src/train_cnn.py --final --seed 42             # CNN (also 13, 2026)
python src/train_transformer.py --model distilbert  --final --seed 42
python src/train_transformer.py --model deberta-v3  --final --seed 42   # also 13, 2026

python src/build_benchmark.py         # validates every run, writes benchmark.csv + figures
python src/analyze_long_distance.py   # prints the CNN distance analysis (writes nothing)
```

Without `--final`, the training scripts touch only train/dev and write nothing. Neural training
is deterministic per seed (fixed seeds, deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG`).

## Evaluation

- **Main metric:** macro-F1 on the test split; accuracy, macro precision/recall, per-class F1 and
  the confusion matrix come from the same run. Single-run CIs and the benchmark CI are percentile
  bootstrap (1,000 resamples, 95%). `build_benchmark.py` recomputes every stored metric from the
  stored predictions before writing the table — nothing is hand-entered.
- **External test (BLiMP):** 14,000 hand-built minimal pairs (6,000 subject–verb, 8,000
  determiner–noun agreement) from the pinned BLiMP subset. *Pair accuracy*: the good sentence gets
  the higher `CORRECT` probability (chance 0.50). *Type accuracy*: the bad sentence is classified
  with the expected error type (chance ≈ 0.14). Naive Bayes and the CNN stay at chance; the
  transformers transfer (0.94 / 0.98) — the models did not just fit our injection style.

## Error analysis (why 0.952 is a lower bound)

DeBERTa-v3 (seed 42) makes 101 test errors; 74 involve `CORRECT`. We re-read the three largest
confusions (`results/error_analysis.md`):

- `PREP → CORRECT` (24): 13 swapped sentences are still grammatical, 3 more probably fine —
  only 2 are real model misses.
- `DET → CORRECT` (13): 8 remain acceptable after the article edit.
- `CORRECT → DET` (14): in 10 the *original* web text already contains the article error
  (e.g. "a ebook") — the model is right, the label is wrong.

So a visible share of the residual "errors" is label noise, not model failure: the reported
macro-F1 is somewhat pessimistic. The sample is model-selected, so this is a lower bound on the
noise, not a rate. Suggested fix (not done here): filter injected sentences that remain
acceptable, drop originals with article errors.

## Licenses

See **[LICENSE](LICENSE)** for the full text.

- **Code and documentation** (`src/`, `scripts/`, `docs/`): MIT License.
- **`data/raw/UD_English-EWT/`** and the derived dataset in **`data/processed/`**:
  CC BY-SA 4.0 (ShareAlike carries over to the derived dataset).
- **`data/raw/blimp/`**: CC BY 4.0.
- Raw files are unmodified, pinned by commit and verified via `data/raw/checksums.txt`.

## Team

Built for an NLP course project by Cemal Batuhan Baş, Görkem Veysel Göksu, Mert Tomak and
Zeynep Çelik. Per-member details: [`contributions/`](contributions/).
