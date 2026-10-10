# Zeynep Çelik · CNN model

Student number: 2309111135 · GitHub: `zeynepcelikx`

## 1. Work done and responsibilities

My part of GrammarLens is the CNN model: a Kim (2014) style text CNN that looks at the sentence through 3-, 4- and 5-word windows. It sees the order of nearby words, but it has not learned English beforehand, so it sits between the bag-of-words Naive Bayes baseline and the pre-trained Transformers.

What I did:

- **Training script.** Wrote `src/train_cnn.py`, a CNN trained from scratch (no pre-trained word vectors) on `data/processed/train.csv`. The vocabulary is built from the train split only.
- **Dev-based tuning.** All choices were made on the dev split. The only setting I changed from the starting configuration was the epoch limit (see below).
- **Shared evaluation.** Connected the script to Görkem's `src/metrics.py` and `src/evaluate_blimp.py`, so the CNN results are saved in the same format as every other model.
- **Reproducibility.** Made GPU training deterministic and checked it: running seed 13 twice gave identical dev and test numbers (only the training time differed).
- **Final runs.** Trained with seeds 13, 42 and 2026 and measured each on the test split and on BLiMP. Results and predictions are in `results/`.
- **Long-distance agreement analysis.** Measured how the CNN's errors on `SVA` sentences depend on the distance between subject and verb (`src/analyze_long_distance.py`).
- **Summary of CNN vs Naive Bayes.** Per-class comparison written for the README and the presentation (Section 6).
- **Dependencies.** Added `torch>=2.0` to `requirements.txt`.

### Settings and the change made on the dev split

| Setting | Value |
| --- | --- |
| Vocabulary | train split only, words seen fewer than 2 times become `<unk>` (10,529 entries) |
| Tokenization | lowercase, words and punctuation marks split with a regular expression |
| Sentence length | padded or cut to 64 tokens |
| Word vectors | 100 dimensions, learned from scratch |
| Convolutions | window sizes 3, 4, 5 with 100 filters each, max-pooling over the sentence |
| Dropout | 0.5 |
| Optimizer | Adam, learning rate 0.001, batch size 64, cross-entropy loss |
| Epochs | up to 30; the epoch with the best dev macro-F1 is kept |

Batch size 64 and the 64-token length are my own choices; the task description did not fix them.

The only change from the starting configuration was the **maximum number of epochs, 10 to 30**. With 10 epochs, seed 13 reached a best dev macro-F1 of 0.4926, and the best epoch was the last one (10) with the score still rising. With 30 epochs, the best dev macro-F1 was 0.5474 (best epoch 27) and the curve flattened from about epoch 20. This was decided only from dev scores. These two dev runs were made before the deterministic setting was added, so their exact numbers differ slightly from later runs. The dev logs are on my Google Drive and are not in the repository.

### Test-set discipline

Training and tuning runs without `--final` never read the test split. The test split was used only in the final runs (`--final`). One disclosure: the seed-13 final run was done twice. The first run was before the deterministic setting was added (test macro-F1 0.5376); I repeated it afterwards to make it reproducible (0.5363). I did not change any setting (epochs, learning rate, ...) based on a test score. The numbers reported below are from the deterministic runs.

## 2. Related code files

| File | What it does |
| --- | --- |
| `src/train_cnn.py` | Builds the vocabulary, trains the CNN, picks the best epoch on dev and, with `--final`, measures test and BLiMP and saves the results |
| `src/analyze_long_distance.py` | Measures the CNN's `SVA` error rate against the number of words between subject and verb, using the EWT dependency trees |
| `results/cnn_seed13.json`, `cnn_seed42.json`, `cnn_seed2026.json` | Test metrics, BLiMP results, training time and hardware for each seed |
| `results/preds/cnn_seed13.csv`, `cnn_seed42.csv`, `cnn_seed2026.csv` | Test predictions (`id`, `label`, `pred`) for each seed |
| `requirements.txt` | I added the `torch` line |

`src/train_cnn.py` relies on Görkem's `src/metrics.py` (result files) and `src/evaluate_blimp.py` (BLiMP). I did not modify them.

## 3. How the code works

**`src/train_cnn.py`**

- `tokenize` lowercases a sentence and splits it into words and punctuation marks.
- `build_vocab` counts the words of the train sentences and keeps those seen at least twice. Id 0 is padding and id 1 is the unknown word.
- `encode` turns each sentence into a fixed-length list of vocabulary ids, padded with zeros.
- `TextCNN` is the model. An embedding layer gives every word a 100-number vector. Three `Conv1d` layers with window sizes 3, 4 and 5 (100 filters each) slide over the sentence, and each filter acts as a detector for one short word pattern. Max-pooling keeps the strongest match of every filter, so the sentence becomes a vector of 300 numbers. After dropout (0.5), a linear layer maps this vector to the 7 classes.
- `train` runs the epochs: shuffled mini-batches, loss and Adam update, then macro-F1 on dev. It stores the weights of the best dev epoch and loads them at the end.
- `predict_logits` and `predict_proba` score sentences in batches; `predict_proba` returns softmax probabilities in the class order `CORRECT, SVA, VERB_FORM, DET, NOUN_NUM, PREP, WORD_ORDER`. It is the function handed to `evaluate_blimp`.
- `final_test` (only with `--final`) predicts the test split, calls `evaluate_blimp`, and writes the predictions and the result JSON through `save_predictions` and `save_result_json`.
- `set_seed` fixes the Python, NumPy and PyTorch random generators and turns on deterministic GPU settings.

**`src/analyze_long_distance.py`**

For every `SVA` test sentence it compares the error-injected text with its original (`source`) to find the verb that was changed. It finds that word in the EWT `.conllu` tree of the same sentence and looks up its subject (`nsubj`/`csubj`, ignoring `:outer` links; for copulas and auxiliaries the subject of the main predicate is used; if there are several subjects, the one closest to the verb is taken). The distance is the number of non-punctuation words between subject and verb. It then prints the CNN's error rate per distance for each seed and the sentences with the longest distances that the CNN got wrong.

## 4. Algorithms, methods and libraries

- **Text CNN (Kim, 2014):** word embeddings, parallel 1-D convolutions with different window sizes, max-over-time pooling, dropout, softmax classifier.
- **Training:** Adam optimizer, cross-entropy loss, model selection by the best epoch on the dev split.
- **Deterministic training:** fixed seeds, `cudnn.deterministic`, no `cudnn.benchmark`, `torch.use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG`.
- **Evaluation:** macro-F1 as the main metric, with accuracy, macro precision, macro recall, per-class F1 and the confusion matrix from `src/metrics.py`; BLiMP pair and type accuracy from `src/evaluate_blimp.py`.
- **Libraries:** PyTorch, pandas, NumPy, scikit-learn (`f1_score`), and the standard library (`re`, `difflib`, `csv`).
- **Hardware:** Google Colab, NVIDIA Tesla T4.

## 5. Running and testing

Training runs on Google Colab with a T4 GPU (PyTorch, pandas and scikit-learn are already installed there). Locally, install the requirements first (`pip install -r requirements.txt`).

```
git clone https://github.com/BatuhanbasSwe/GrammarLens.git
cd GrammarLens

python src/train_cnn.py --seed 13
python src/train_cnn.py --seed 13 --final
python src/train_cnn.py --seed 42 --final
python src/train_cnn.py --seed 2026 --final
```

- Without `--final` the script trains and prints the dev macro-F1 of every epoch; it does not touch the test split and writes no files.
- With `--final` it also measures the test split and BLiMP and writes `results/cnn_seed<seed>.json` and `results/preds/cnn_seed<seed>.csv`.
- Other options: `--epochs`, `--lr`, `--batch_size`, `--emb_dim`, `--num_filters`, `--dropout`, `--min_freq`, `--max_len`.
- A run takes about one minute on a T4.
- On Colab, `cd /content` before deleting and re-cloning the repository, otherwise the shell is left in a removed folder.

The long-distance analysis needs only the standard library. It reads `data/processed/test.csv`, `results/preds/cnn_seed*.csv` and the EWT test trees `data/raw/UD_English-EWT/en_ewt-ud-test.conllu` (all in the repository), prints the result to the screen and writes no files. Run it from the repository root:

```
python src/analyze_long_distance.py
```

Testing: running the same seed twice produces identical logs, apart from the `train_time_sec` line. I checked this with `diff` on two runs of seed 13.

## 6. Experiments and results

All numbers below come from `results/cnn_seed*.json`, `results/nb_seed42.json` and `src/analyze_long_distance.py`. Test split: 7 classes × 302 = 2,114 sentences.

### CNN on the test split (3 seeds)

| Metric | Seed 13 | Seed 42 | Seed 2026 | Mean ± std |
| --- | --- | --- | --- | --- |
| Macro-F1 | 0.5363 | 0.5468 | 0.5585 | 0.5472 ± 0.0111 |
| Accuracy | 0.5421 | 0.5468 | 0.5596 | 0.5495 ± 0.0091 |
| Macro precision | 0.5389 | 0.5540 | 0.5681 | 0.5537 ± 0.0146 |
| Macro recall | 0.5421 | 0.5468 | 0.5596 | 0.5495 ± 0.0091 |
| Best dev macro-F1 | 0.5459 | 0.5489 | 0.5576 | |
| Best epoch | 30 | 30 | 30 | |
| Training time (s) | 51.1 | 51.6 | 50.9 | |

The standard deviation is the sample standard deviation over the 3 seeds. Hardware: GPU (Tesla T4). Bootstrap confidence intervals are Görkem's part and are not computed here.

### Per-class F1 on the test split: CNN vs Naive Bayes

| Class | Naive Bayes | CNN (mean of 3 seeds) | Difference |
| --- | --- | --- | --- |
| CORRECT | 0.127 | 0.339 | +0.212 |
| SVA | 0.345 | 0.618 | +0.273 |
| VERB_FORM | 0.349 | 0.599 | +0.251 |
| DET | 0.399 | 0.533 | +0.133 |
| NOUN_NUM | 0.422 | 0.662 | +0.240 |
| PREP | 0.250 | 0.419 | +0.169 |
| WORD_ORDER | 0.197 | 0.660 | +0.463 |
| **Macro-F1** | **0.298** | **0.547** | **+0.249** |

Naive Bayes is a single run (`results/nb_seed42.json`); it gives the same result every time. Its accuracy is 0.3098.

### BLiMP (external test, 14,000 pairs: 6,000 `SVA`, 8,000 `NOUN_NUM`)

| Metric | CNN (mean ± std) | Naive Bayes |
| --- | --- | --- |
| Pair accuracy, overall | 0.4835 ± 0.0187 | 0.4442 |
| Pair accuracy, `SVA` | 0.4596 ± 0.0047 | 0.4740 |
| Pair accuracy, `NOUN_NUM` | 0.5014 ± 0.0307 | 0.4219 |
| Type accuracy, overall | 0.2482 ± 0.0226 | 0.2681 |
| Type accuracy, `SVA` | 0.1636 ± 0.0152 | 0.1625 |
| Type accuracy, `NOUN_NUM` | 0.3116 ± 0.0281 | 0.3474 |

Pair accuracy asks whether the model gives the correct sentence a higher `CORRECT` probability than the wrong one (chance is 0.50). Type accuracy asks whether the predicted class of the wrong sentence is the class mapped to that BLiMP file (random guessing among 7 classes would give about 0.14). The CNN is close to chance level on pair accuracy and does not transfer to BLiMP in a useful way; it is not clearly better than Naive Bayes there.

### Long-distance agreement analysis (`SVA` sentences)

`SVA` error rate = share of `SVA` test sentences that the CNN did not classify as `SVA`. The distance is the number of words between the subject and the verb. The CNN's widest window is 5 words, so subject and verb fit in one window only if at most 3 words lie between them. 298 of the 302 `SVA` test sentences could be measured (in 4 the subject was not found in the tree).

| Words between subject and verb | Sentences | Wrong, seed 13 | Rate, seed 13 | Rate, seed 42 | Rate, seed 2026 |
| --- | --- | --- | --- | --- | --- |
| 0 | 227 | 75 | 0.330 | 0.308 | 0.300 |
| 1 | 27 | 14 | 0.519 | 0.444 | 0.407 |
| 2 | 16 | 10 | 0.625 | 0.750 | 0.562 |
| 3 | 15 | 9 | 0.600 | 0.667 | 0.600 |
| 4 or more | 13 | 9 | 0.692 | 0.615 | 0.769 |

Fitting in the window (0 to 3 words between): error rate 0.340 to 0.379 depending on the seed; not fitting (4 or more words): 0.615 to 0.769. 65 of the 302 `SVA` sentences were misclassified by all three seeds.

Examples that the CNN got wrong (seed 13), with subject and verb far apart:

- `...local Iraqi Sunni fundamentalists opposed to the US presence in Iraq has begun joining...` (7 words between; predicted `VERB_FORM`)
- `...the recent article by Daniel Okrent of The New York Times ... demonstrate ...` (8 words between; predicted `WORD_ORDER`)
- `...the political will to end the crisis expressed a few short weeks ago seem to have ebbed.` (10 words between; predicted `PREP`)

Reading of the result: the error rate grows with the distance, and it is already about 1.5 to 2 times higher at 1 to 3 words than for adjacent subject and verb, so the difficulty does not start only at the window limit. The group beyond the window has only 13 sentences, so a sharp break at the window size cannot be claimed. Sentences with a long subject-verb distance are also longer and more complex in general, so this is a correlation.

### What the CNN does with `CORRECT` sentences

`CORRECT` has the lowest F1 of all classes (0.339 on average). From the confusion matrices (test split, rows are true classes):

| Seed | True `CORRECT` found (recall) | Sentences predicted `CORRECT` that are really correct (precision) | Error sentences predicted `CORRECT` |
| --- | --- | --- | --- |
| 13 | 96 of 302 = 0.318 | 0.340 | 10.3% |
| 42 | 112 of 302 = 0.371 | 0.325 | 12.9% |
| 2026 | 110 of 302 = 0.364 | 0.324 | 12.6% |

The sentences without errors that the CNN does not recognise are spread over all six error classes. Summed over the 3 seeds, 35.1% are predicted `CORRECT`, 15.2% `PREP`, 12.7% `NOUN_NUM`, 12.1% `SVA`, 9.5% `VERB_FORM`, 7.7% `DET` and 7.6% `WORD_ORDER`. So the problem has two sides: the CNN often sees an error in a correct sentence, and only about a third of the sentences it calls `CORRECT` really are correct. Why this happens was not tested; it would need further experiments (for example looking at which words trigger the error classes).

### Summary: CNN vs Naive Bayes

The CNN is better than Naive Bayes on every error type: macro-F1 rises from 0.298 to 0.547 on average over 3 seeds (std about 0.011). The biggest jump is in `WORD_ORDER` (F1 0.197 to 0.660): Naive Bayes counts single words and cannot see their order, while the CNN reads windows of 3 to 5 neighbouring words. `SVA` (0.345 to 0.618), `VERB_FORM` (0.349 to 0.599) and `NOUN_NUM` (0.422 to 0.662) also improve clearly.

The CNN still struggles most with `CORRECT` (F1 0.339) and `PREP` (F1 0.419). On `SVA`, the error rate grows with the distance between subject and verb, from about 33% when they are adjacent to about 62-77% when more than 3 words separate them (only 13 sentences). It also stays near chance level on the external BLiMP test (pair accuracy 0.46 to 0.50).

Example for "what the CNN cannot see": `...fundamentalists opposed to the US presence in Iraq has begun joining...` (7 words between subject and verb). The CNN predicted `VERB_FORM`.

## 7. Problems, gaps and suggestions

- **BLiMP stays near chance.** Pair accuracy is 0.46 to 0.50 for every seed. I did not test why; it may be the difference between the BLiMP template sentences and our real web sentences, but this is not measured.
- **The best epoch was the last one (30) in all three final runs.** Dev macro-F1 was still rising very slowly. I did not raise the limit again after seeing the test results, since that would be tuning on the test split. A longer schedule could be tested on dev in the future.
- **Low `CORRECT` F1 (0.339).** Measured: the CNN finds only 32-37% of the correct sentences and sends the rest to the error classes, and only about a third of its `CORRECT` predictions are right (table in Section 6). The reason behind this pattern was not tested.
- **`PREP` label noise.** Mert's data document notes that after a preposition is replaced the sentence sometimes still sounds natural, so part of the low `PREP` score may come from noisy labels. I did not measure how much.
- **Long-distance analysis is small and correlational.** Only 13 sentences lie beyond the window, and the distance is measured on the injected verb. A few injected errors are borderline in colloquial English, which may count as a model mistake.
- **Naive Bayes comparison.** Naive Bayes is one deterministic run, the CNN is the mean of 3 seeds.
- **Training time.** `train_time_sec` includes the vocabulary building and the dev evaluation after every epoch, not only the weight updates.
- **Model weights are not saved.** Only metrics and predictions are stored, so a result is reproduced by re-training (about one minute on a T4). Dev-run logs are on my Google Drive, not in the repository.
- **Local testing.** `src/train_cnn.py` was run on Colab; it was not run on a local machine.
- **Suggestions.** Try pre-trained word vectors, a character-level branch, or dilated convolutions to see longer distances; compare against the Transformer models with the same per-class and long-distance analysis.
