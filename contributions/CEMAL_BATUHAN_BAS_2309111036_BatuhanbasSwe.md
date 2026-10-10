# Cemal Batuhan Baş · Transformer Models and Repository Owner

- **Student number:** 2309111036
- **GitHub:** [BatuhanbasSwe](https://github.com/BatuhanbasSwe)

## 1. Work done and responsibilities

I was responsible for fine-tuning two pre-trained Transformer models on our dataset and for owning the GitHub repository. DistilBERT is a baseline; DeBERTa-v3 is the project's new method.

What I did:

- Created the public GitHub repository and added the team members as collaborators.
- Wrote a single script that trains both models with the same code (`src/train_transformer.py`). Only the Hugging Face checkpoint name and the default learning rate differ between the models, so any difference between them comes from the model, not from the code.
- As an early check, trained DistilBERT with one seed on the dev split only. Its dev macro-F1 was 0.8882, clearly above the best classical result of the planning prototype (0.49).
- Diagnosed and fixed, at its root cause, a numerical stability problem in DeBERTa-v3 training: every weight became NaN after the first update (details in Sections 3 and 7).
- Trained both models three times each with seeds 13, 42 and 2026, and saved the test and BLiMP results in the shared format with Görkem's evaluation code.
- Put the DeBERTa-v3 predictions in `results/preds/` for Mert's error analysis.
- Checked that all my result files pass Görkem's benchmark validation without errors.

## 2. Related code files

| File | Purpose |
| --- | --- |
| `src/train_transformer.py` | Fine-tunes DistilBERT or DeBERTa-v3 on the train split, picks the epoch by dev macro-F1; only with `--final` it measures test and BLiMP and saves the results. |
| `results/distilbert_seed{13,42,2026}.json` | Measured test and BLiMP results, training time and hardware of DistilBERT for the three seeds. |
| `results/deberta-v3_seed{13,42,2026}.json` | Measured test and BLiMP results, training time and hardware of DeBERTa-v3 for the three seeds. |
| `results/preds/distilbert_seed*.csv`, `results/preds/deberta-v3_seed*.csv` | `id,label,pred` rows for the 2,114 test examples. |

The script uses Görkem's `src/metrics.py` (`save_predictions`, `save_result_json`, `CLASS_ORDER`) and `src/evaluate_blimp.py` (`evaluate_blimp`) without changing them.

## 3. How the code works

### Overall flow

```text
train.csv ──tokenize──> training loop (3 epochs) ──every epoch──> dev macro-F1
                                                     │
                           weights of the best epoch (kept in memory)
                                                     │
                                  with --final       ▼
                     test predictions + BLiMP ──> save_predictions / save_result_json
```

### Model definitions

The `MODELS` dictionary maps the model name used in the result files to a Hugging Face checkpoint and a default learning rate:

| `--model` | Checkpoint | Default LR |
| --- | --- | --- |
| `distilbert` | `distilbert/distilbert-base-uncased` | 3e-5 |
| `deberta-v3` | `microsoft/deberta-v3-base` | 2e-5 |

`AutoModelForSequenceClassification` adds a new classification layer with 7 outputs on top of the pre-trained encoder. During fine-tuning this new layer and the whole encoder are trained together.

### Reproducibility (`set_seed`)

The Python, NumPy and PyTorch random generators are seeded, cuDNN is put into deterministic mode and `torch.use_deterministic_algorithms` is enabled. The shuffling order of the training data is also fixed by a seeded `torch.Generator`. As a result, two runs with the same seed give the same dev scores (verified in Section 5).

### Tokenization and dynamic padding (`encode`, `make_collate`)

Sentences are split into subword tokens by each model's own tokenizer and truncated at 64 tokens. `encode` also reports how many sentences are longer than 64 tokens; this number is **0** for train, dev and test, so no sentence was truncated.

Sentences are not padded to 64 in advance. `make_collate` pads each batch only to the length of its longest sentence (dynamic padding). The result does not change, but computation drops considerably.

### Training loop (`train`)

- **Optimizer:** AdamW with weight decay 0.01. Bias and LayerNorm weights get no weight decay.
- **Learning-rate schedule:** linear warmup from 0 to the target LR over the first 10% of the steps, then linear decay to 0. This keeps large early updates from damaging the pre-trained weights.
- **Gradient clipping:** the gradient norm is clipped at 1.0.
- **Mixed precision:** the forward pass runs in bf16 inside `torch.autocast` (supported by the RTX 4070); the loss and softmax are computed in 32-bit with `float()`.
- **Numerical stability check:** if the loss is not finite (NaN/inf) at any step, training stops immediately with a `FloatingPointError`, so a broken model is never trained silently.
- **Epoch selection:** after each epoch, macro-F1 is computed on the dev split. The weights of the best-scoring epoch are copied to CPU memory and loaded back into the model at the end. Weights are never written to disk.

### fp32 master weights

The model is loaded with `from_pretrained(..., dtype=torch.float32)`. Transformers 5 loads a checkpoint in the data type it was saved in, and `microsoft/deberta-v3-base` is stored in fp16. With fp16 weights, AdamW's `eps = 1e-8` is below the smallest value fp16 can represent and rounds to 0; the squared gradient also rounds to 0, and the update becomes 0/0 = NaN. This line keeps the weights and the optimizer state in fp32; bf16 is used only in the forward pass. DistilBERT is already stored in fp32, so nothing changes for it, and both models are still trained with the same code.

### Final evaluation (`final_test`)

The test split is read only when `--final` is given, and only after training has finished. Test predictions are saved with `save_predictions` and all metrics with `save_result_json`. For BLiMP, `evaluate_blimp` receives a function that takes a list of sentences and returns the softmax probabilities of the 7 classes; the `CORRECT` probability comes from this softmax output. Probabilities are computed in 64-bit so that every row sums to exactly 1.

## 4. Algorithms, methods and libraries

| Tool or method | Purpose |
| --- | --- |
| Transformer architecture | Network design that relates every token to every other token in the sentence through self-attention |
| DistilBERT (`distilbert-base-uncased`) | Baseline model: a 6-layer, lowercased model obtained from BERT by knowledge distillation; 67.0M parameters |
| DeBERTa-v3 (`deberta-v3-base`) | New method: a 12-layer model that computes word content and position with separate vectors (disentangled attention) and is pre-trained ELECTRA-style; 184.4M parameters (98.4M of them are the embeddings of its 128,100-token vocabulary) |
| Fine-tuning | Retraining a pre-trained model end to end on our 7-class task |
| AdamW, linear warmup/decay, gradient clipping | Standard optimization settings for stable fine-tuning |
| bf16 autocast + fp32 master weights | Mixed precision for speed, 32-bit weights for stability |
| PyTorch 2.14.1 (CUDA) | Model training |
| Hugging Face Transformers 5.19.0 | Models, tokenizers and the LR scheduler |
| scikit-learn 1.9.1, pandas 2.3.3 | Dev macro-F1 and reading the CSV files |
| `sentencepiece` | Required by the DeBERTa-v3 tokenizer |

## 5. Running and testing

Installation (the CUDA build of PyTorch should be installed first):

```bash
python -m pip install -r requirements.txt
```

Development run that does not read the test split:

```bash
python src/train_transformer.py --model distilbert --seed 42
python src/train_transformer.py --model deberta-v3 --seed 42
```

Final runs (3 seeds per model):

```bash
python src/train_transformer.py --model distilbert --seed 13 --final
python src/train_transformer.py --model distilbert --seed 42 --final
python src/train_transformer.py --model distilbert --seed 2026 --final
python src/train_transformer.py --model deberta-v3 --seed 13 --final
python src/train_transformer.py --model deberta-v3 --seed 42 --final
python src/train_transformer.py --model deberta-v3 --seed 2026 --final
```

The other options (`--epochs`, `--lr`, `--batch_size`, `--max_len`, `--warmup_ratio`, `--weight_decay`, `--max_grad_norm`, `--precision {bf16,fp32}`) were used with their default values.

Checks performed:

- Both tokenizers load and split a sample sentence as expected.
- No sentence in train, dev or test is longer than 64 tokens (0 / 0 / 0).
- The DeBERTa-v3 NaN problem was reproduced step by step with the same seed and batch order; after the fix, the loss, the gradients and the weights stayed finite for 4 steps.
- **Reproducibility:** the dev-only run and the final run of seed 42 gave the same dev macro-F1 at every epoch for both models (DistilBERT 0.7856 / 0.8808 / 0.8882, DeBERTa-v3 0.9387 / 0.9501 / 0.9516).
- Every prediction CSV has 2,114 rows.
- Görkem's `build_benchmark()` was run on all result files with its outputs redirected to a temporary folder outside the repository. It recomputes the metrics of every JSON from the prediction CSV and compares them; it finished without errors.

## 6. Experiments and results

### Settings

The starting settings from the task plan were used; since the dev results were good, no setting was changed.

| Setting | Value |
| --- | --- |
| Max tokens | 64 |
| Batch size | 32 |
| Epochs | 3 (best epoch selected by dev macro-F1) |
| Learning rate | DistilBERT 3e-5, DeBERTa-v3 2e-5 |
| Warmup | 10% of the steps, then linear decay |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Precision | bf16 autocast, fp32 weights |
| Seeds | 13, 42, 2026 |
| Hardware | NVIDIA GeForce RTX 4070 Laptop GPU (8 GB) |

### Early check (dev only, seed 42)

| Model | Epoch 1 | Epoch 2 | Epoch 3 |
| --- | ---: | ---: | ---: |
| DistilBERT | 0.7856 | 0.8808 | **0.8882** |
| DeBERTa-v3 | 0.9387 | 0.9501 | **0.9516** |

### Final results per seed

| Model | Seed | Best epoch | Dev macro-F1 | Test macro-F1 | Accuracy | Macro P | Macro R | BLiMP pair | BLiMP type | Training time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DistilBERT | 13 | 3 | 0.8891 | 0.8871 | 0.8884 | 0.8866 | 0.8884 | 0.9354 | 0.8940 | 172 s |
| DistilBERT | 42 | 3 | 0.8882 | 0.8916 | 0.8926 | 0.8918 | 0.8926 | 0.9495 | 0.9104 | 181 s |
| DistilBERT | 2026 | 3 | 0.8930 | 0.8924 | 0.8936 | 0.8918 | 0.8936 | 0.9433 | 0.9075 | 149 s |
| DeBERTa-v3 | 13 | 2 | 0.9523 | 0.9530 | 0.9532 | 0.9529 | 0.9532 | 0.9827 | 0.9608 | 447 s |
| DeBERTa-v3 | 42 | 3 | 0.9516 | 0.9522 | 0.9522 | 0.9526 | 0.9522 | 0.9834 | 0.9615 | 668 s |
| DeBERTa-v3 | 2026 | 2 | 0.9511 | 0.9503 | 0.9503 | 0.9504 | 0.9503 | 0.9820 | 0.9588 | 501 s |

### Summary over three seeds

Mean and standard deviation are computed the same way as in Görkem's benchmark code (population standard deviation). The confidence intervals are 1,000-resample bootstrap intervals of the seed mean, computed with `build_benchmark.py`. The official table is `results/benchmark.csv`.

| Metric | DistilBERT | DeBERTa-v3 |
| --- | ---: | ---: |
| Test macro-F1 (mean ± std) | 0.8904 ± 0.0023 | **0.9518 ± 0.0011** |
| Macro-F1 95% confidence interval | [0.8780, 0.9019] | [0.9436, 0.9596] |
| Accuracy | 0.8915 | 0.9519 |
| Macro precision | 0.8901 | 0.9520 |
| Macro recall | 0.8915 | 0.9519 |
| BLiMP pair accuracy: overall / SVA / NOUN_NUM | 0.943 / 0.898 / 0.976 | 0.983 / 0.977 / 0.987 |
| BLiMP type accuracy: overall / SVA / NOUN_NUM | 0.904 / 0.839 / 0.952 | 0.960 / 0.955 / 0.964 |

The confidence intervals of the two models do not overlap.

### Per-class F1 (mean ± std over three seeds)

| Class | DistilBERT | DeBERTa-v3 | Difference |
| --- | ---: | ---: | ---: |
| `CORRECT` | 0.720 ± 0.003 | 0.873 ± 0.005 | +0.153 |
| `SVA` | 0.943 ± 0.006 | 0.978 ± 0.002 | +0.035 |
| `VERB_FORM` | 0.975 ± 0.001 | 0.986 ± 0.001 | +0.011 |
| `DET` | 0.878 ± 0.005 | 0.931 ± 0.001 | +0.053 |
| `NOUN_NUM` | 0.988 ± 0.002 | 0.993 ± 0.002 | +0.005 |
| `PREP` | 0.772 ± 0.007 | 0.922 ± 0.003 | +0.150 |
| `WORD_ORDER` | 0.958 ± 0.001 | 0.980 ± 0.001 | +0.022 |

### Interpretation

- Both pre-trained models beat the baselines by a large margin: macro-F1 is 0.298 for Naive Bayes and 0.547 ± 0.009 for the CNN (from my teammates' result files).
- DeBERTa-v3 improves macro-F1 by 0.062 over DistilBERT. The largest gains are in `CORRECT` and `PREP`, the classes that require looking at the meaning of the sentence rather than the form of a single word.
- On BLiMP, the pair accuracy of Naive Bayes and the CNN is below chance (0.5); DistilBERT reaches 0.943 and DeBERTa-v3 0.983. The models can separate subject-verb and determiner-noun agreement errors in minimal pairs that were not produced by our own script.
- For two DeBERTa-v3 seeds the best epoch was 2; epoch 3 lowered the dev score, so epoch 2 was selected automatically.
- The pair DeBERTa-v3 (seed 42) confuses most is `PREP → CORRECT` (24 of its 101 errors). In Mert's error analysis (`results/error_analysis.md`), 13 of these 24 sentences are still grammatical after the preposition swap, so many of these "errors" are label noise rather than model mistakes.

## 7. Problems, limitations and suggestions

- **DeBERTa-v3 NaN problem:** in the first attempt, training stopped with a NaN loss at the second step. A step-by-step diagnostic script showed that at the first step the loss and the gradients were finite, yet 202 weight tensors became NaN after the update even though the learning rate was 0. Getting the same result with bf16 and with fp32 autocast showed that the problem was in the weights, not in the computation; the weights turned out to be loaded in fp16. A one-line change that loads the weights in fp32 fixed it.
- **Environment:** scikit-learn was not installed on the training machine; it was installed from `requirements.txt`. The CUDA build of PyTorch must be installed separately.
- **Training times:** on the same hardware, DeBERTa-v3 training times ranged from 447 to 668 seconds. This may depend on the current state of the laptop GPU, but the cause was not measured; the times should only be used for rough comparison. The CNN was trained on a different GPU (Tesla T4), so times cannot be compared directly across models.
- **No hyperparameter search:** the starting settings gave good dev results, so no LR, epoch or batch-size search was done. A small LR search on the dev split might improve the results slightly.
- **Weakest classes are `CORRECT` and `PREP`:** part of the errors in these classes comes from label noise (see Mert's error analysis); this can only be separated by reviewing wrong predictions by hand.
- **Data limitations:** the errors are generated by a script and every sentence contains a single error, so performance on real learner text with several errors was not measured. The BLiMP external test covers only the `SVA` and `NOUN_NUM` classes.
- **Suggestions:** a larger model such as `deberta-v3-large`, an error analysis of the `SVA` sentences DeBERTa-v3 gets wrong by subject-verb distance, and a calibration analysis that measures how reliable the model's probabilities are.
