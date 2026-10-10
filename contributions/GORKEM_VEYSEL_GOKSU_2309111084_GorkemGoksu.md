# Görkem Veysel Göksu · Shared Evaluation, Naive Bayes, and Benchmarking

- **Student number:** 2309111084
- **GitHub:** [GorkemGoksu](https://github.com/GorkemGoksu)

## 1. Work done and responsibilities

I was responsible for the shared evaluation infrastructure used by every model, the project-specific BLiMP evaluation, the Naive Bayes baseline, and the final benchmark aggregation.

My completed work includes:

- Implementing the shared seven-class metric calculation with a fixed class order.
- Writing validated and atomic result JSON and `id,label,pred` prediction CSV outputs.
- Implementing a reproducible 95% percentile-bootstrap confidence interval for Macro-F1.
- Implementing model-independent evaluation over the 14 selected BLiMP paradigms.
- Training and evaluating the unigram `CountVectorizer` + `MultinomialNB` baseline without data leakage.
- Producing the final benchmark table after all required model and seed outputs became available.
- Producing the two required benchmark figures: test Macro-F1 with confidence intervals and BLiMP pair accuracy.
- Measuring why a bag-of-words model cannot represent almost every `WORD_ORDER` corruption in the dataset.

## 2. Related code files

| File | Purpose |
| --- | --- |
| `src/metrics.py` | Defines the canonical class order, computes classification metrics, validates outputs, writes prediction/result files, and provides bootstrap confidence intervals. |
| `src/evaluate_blimp.py` | Validates the pinned 14-file BLiMP subset and computes pair and type accuracy from model probabilities. |
| `src/train_nb.py` | Trains the unigram Multinomial Naive Bayes baseline and performs the final test/BLiMP evaluation only when explicitly requested. |
| `src/build_benchmark.py` | Validates every required model/seed JSON and prediction CSV, aggregates the final metrics, computes confidence intervals, and writes the benchmark artifacts. |
| `results/nb_seed42.json` | Measured Naive Bayes test and BLiMP results. |
| `results/preds/nb_seed42.csv` | Gold labels and Naive Bayes predictions for all 2,114 test examples. |
| `results/benchmark.csv` | Final four-model benchmark table generated from the validated frozen results. |
| `results/figures/macro_f1.svg` | Mean test Macro-F1 with 95% bootstrap confidence intervals. |
| `results/figures/blimp_pair_accuracy.svg` | Mean BLiMP pair accuracy for the four models. |

## 3. How the code works

### Shared metrics

`src/metrics.py` uses the following canonical class order for every model:

`CORRECT, SVA, VERB_FORM, DET, NOUN_NUM, PREP, WORD_ORDER`

`compute_classification_metrics()` validates the true and predicted labels and then calculates accuracy, macro precision, macro recall, Macro-F1, per-class F1, and a 7×7 confusion matrix. All seven labels remain in the calculation even when a model never predicts one of them. Rows in the confusion matrix are gold labels and columns are predicted labels.

`save_predictions()` writes exactly the `id,label,pred` schema. `save_result_json()` recalculates the classification metrics, validates the BLiMP object, and writes only the required result fields. Both functions reject missing, duplicated, or unknown values and use atomic replacement so that interrupted writes do not leave partial final files.

The bootstrap helper samples test examples with replacement 1,000 times, recalculates seven-class Macro-F1 for every sample, and reports the 2.5th and 97.5th percentiles as the 95% confidence interval. A fixed random state makes the interval reproducible.

### BLiMP evaluation

`src/evaluate_blimp.py` maps eight determiner-noun agreement paradigms to `NOUN_NUM` and six subject-verb/distractor agreement paradigms to `SVA`. Each file must contain exactly 1,000 unique minimal pairs, giving 14,000 pairs in total.

The evaluator accepts a `predict_proba` function and the model's probability-column order. It safely reorders the columns into the shared class order before computing the two project metrics:

- **Pair accuracy:** a pair is correct only when the grammatical sentence has a strictly higher `CORRECT` probability than the ungrammatical sentence. A tie is counted as incorrect.
- **Type accuracy:** the predicted class of the ungrammatical sentence must match its mapped error type, either `SVA` or `NOUN_NUM`.

This is the project-defined adaptation of BLiMP for a seven-class error classifier. It is not the original full-sentence language-model scoring protocol.

### Naive Bayes baseline

`src/train_nb.py` validates the processed CSV schema and labels. `CountVectorizer` is fitted only on `train.csv` and produces unigram word-count features. Development, test, and BLiMP sentences are transformed with the already fitted vocabulary, so evaluation data cannot affect training.

The classifier is `MultinomialNB` with the default `alpha=1.0` Laplace smoothing. The model is deterministic, so the project records one final run under seed 42. A normal command evaluates only the development split. Test and BLiMP data are read only with the explicit `--final` flag.

### Final benchmark

`src/build_benchmark.py` requires seed 42 for Naive Bayes and seeds 13, 42, and 2026 for CNN, DistilBERT, and DeBERTa-v3. It performs the following checks before writing any final artifact:

1. Every required result JSON and prediction CSV exists.
2. JSON keys and class order exactly match the shared contract.
3. Every prediction file has exactly the `id,label,pred` header.
4. Prediction IDs and gold labels match `data/processed/test.csv` after ID-based alignment.
5. Accuracy, precision, recall, F1 values, and confusion matrices recomputed from predictions match the corresponding JSON.
6. BLiMP sections and pair counts match the pinned 14-file definition.

The script calculates model means, population standard deviations across the required seeds, and a 95% bootstrap confidence interval for mean Macro-F1. In a multi-seed model, every bootstrap repetition applies the same sampled test indices to all seeds before averaging their Macro-F1 values.

## 4. Algorithms, methods and libraries

| Tool or method | Use in the project |
| --- | --- |
| Python | Validation, training, evaluation, and output generation |
| pandas | Reading the processed train, development, and test data |
| NumPy | Probability arrays, bootstrap sampling, and numerical operations |
| scikit-learn `CountVectorizer` | Converting sentences to unigram word-count vectors |
| scikit-learn `MultinomialNB` | Naive Bayes baseline classifier |
| scikit-learn metrics | Accuracy, precision, recall, F1, and confusion matrix |
| Percentile bootstrap | Quantifying test-sample uncertainty for Macro-F1 |
| JSONL, CSV, and JSON | BLiMP inputs, predictions, and the shared result contract |
| SVG | Dependency-free and reproducible benchmark figures |

## 5. Running and testing

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the Naive Bayes development evaluation without accessing the test set:

```bash
python src/train_nb.py
```

Run the single final Naive Bayes evaluation:

```bash
python src/train_nb.py --final
```

This produces:

```text
results/nb_seed42.json
results/preds/nb_seed42.csv
```

After all required model outputs are frozen, build the benchmark and figures:

```bash
python src/build_benchmark.py
```

This validates all inputs before producing:

```text
results/benchmark.csv
results/figures/macro_f1.svg
results/figures/blimp_pair_accuracy.svg
```

### Validation performed

- Verified the fixed seven-class order and the 7×7 confusion-matrix orientation.
- Verified exact JSON key order and prediction CSV headers.
- Recomputed the classification metrics from every final prediction CSV and matched them to the JSON values.
- Verified 2,114 aligned test predictions for every required run.
- Verified one NB run and three runs each for CNN, DistilBERT, and DeBERTa-v3.
- Verified all 14 BLiMP files, 1,000 pairs per file, and 14,000 pairs in total.
- Verified probability-column reordering for different model class orders.
- Verified that tied `CORRECT` probabilities count as an incorrect BLiMP pair.
- Verified reproducible bootstrap output with the same random state.
- Verified that incomplete benchmark inputs cause the script to stop without writing partial final artifacts.
- Verified the final benchmark model order and numeric ranges.
- Parsed both generated SVG files as valid XML and confirmed that each contains all four model bars.

## 6. Experiments and results

### Four-model benchmark

All values below are taken from the generated final benchmark. Standard deviation is the population standard deviation over the required seeds. The confidence interval is the reproducible 95% test-example bootstrap interval for mean Macro-F1.

| Model | Method | Runs | Macro-F1 mean | Macro-F1 std. | 95% CI | Accuracy | Macro precision | Macro recall | BLiMP pair acc. | BLiMP type acc. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive Bayes | Baseline | 1 | 0.2983 | 0.0000 | [0.2785, 0.3177] | 0.3098 | 0.3009 | 0.3098 | 0.4442 | 0.2681 |
| CNN | Baseline | 3 | 0.5472 | 0.0090 | [0.5304, 0.5629] | 0.5495 | 0.5537 | 0.5495 | 0.4835 | 0.2482 |
| DistilBERT | Baseline | 3 | 0.8904 | 0.0023 | [0.8780, 0.9019] | 0.8915 | 0.8901 | 0.8915 | 0.9427 | 0.9040 |
| DeBERTa-v3 | New method | 3 | 0.9518 | 0.0011 | [0.9436, 0.9596] | 0.9519 | 0.9520 | 0.9519 | 0.9827 | 0.9604 |

DeBERTa-v3 has the best mean test Macro-F1 and BLiMP scores. It improves mean Macro-F1 by 0.0614 absolute over DistilBERT and by 0.4046 over CNN. Its seed-to-seed standard deviation is also the smallest among the three multi-seed models. DistilBERT is the strongest baseline. CNN improves substantially over Naive Bayes on the test set, but its mean BLiMP pair accuracy remains below 0.5.

### Naive Bayes details

| Metric | Value |
| --- | ---: |
| Test accuracy | 0.3098 |
| Test macro precision | 0.3009 |
| Test macro recall | 0.3098 |
| Test Macro-F1 | 0.2983 |
| Macro-F1 95% bootstrap CI | [0.2785, 0.3177] |
| BLiMP pair accuracy, overall | 0.4442 |
| BLiMP pair accuracy, SVA | 0.4740 |
| BLiMP pair accuracy, NOUN_NUM | 0.4219 |
| BLiMP type accuracy, overall | 0.2681 |
| BLiMP type accuracy, SVA | 0.1625 |
| BLiMP type accuracy, NOUN_NUM | 0.3474 |
| Training time | 0.307 seconds |
| Hardware | CPU, Intel64 Family 6 Model 186 |

Naive Bayes per-class F1 values are:

| Class | F1 |
| --- | ---: |
| `CORRECT` | 0.1270 |
| `SVA` | 0.3450 |
| `VERB_FORM` | 0.3486 |
| `DET` | 0.3994 |
| `NOUN_NUM` | 0.4219 |
| `PREP` | 0.2495 |
| `WORD_ORDER` | 0.1968 |

The development Macro-F1 was 0.2935 and the final test Macro-F1 was 0.2983. The benchmark uses the final test value.

### Bag-of-words limitation

Across the train, development, and test splits, 2,899 of the 2,900 `WORD_ORDER` examples have exactly the same unigram counts as their source sentences under the fitted vectorizer analysis: **2,899 / 2,900 = 99.97%**.

The single exception is `train-09432`. The source contains `modern m16`, while the corrupted text contains `M modern16`, which merges two tokens. I did not edit this row because dataset construction belongs to another team member.

Because Naive Bayes receives word counts but not word positions, it cannot distinguish almost all source/corrupted `WORD_ORDER` pairs from its input representation. This explains the low `WORD_ORDER` F1 of 0.1968 and motivates sequence-aware models.

## 7. Problems, gaps and improvement suggestions

- The unigram Naive Bayes baseline cannot encode word order and is especially weak for `WORD_ORDER` and `CORRECT`.
- An overall BLiMP pair accuracy below 0.5 shows that Naive Bayes does not consistently assign a higher `CORRECT` probability to the grammatical sentence.
- CNN performs better on the test set than Naive Bayes, but its BLiMP pair and type results show that this performance does not translate into robust minimal-pair grammatical sensitivity.
- Future controlled experiments could compare bigrams, character n-grams, or TF-IDF against the unchanged unigram-count baseline. These should be reported as separate experiments rather than replacing the required baseline.
- The benchmark intentionally refuses partial result sets. If a future required run is missing, the absence should be reported rather than filled with copied or invented values.
