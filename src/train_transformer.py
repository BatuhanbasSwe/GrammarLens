"""Fine-tune a pretrained Transformer (DistilBERT or DeBERTa-v3) on GrammarLens.

Both models are trained by this single script; only ``--model`` changes, so any
difference between them comes from the pretrained model, not from the code.

Development run (train + dev only, does not read the test split):

    python src/train_transformer.py --model distilbert --seed 42

Final run (also evaluates test + BLiMP and writes the required result files):

    python src/train_transformer.py --model deberta-v3 --seed 42 --final

The epoch with the highest dev macro-F1 is kept. Model weights are only held in
memory and are never written to disk.
"""

import argparse
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from src.evaluate_blimp import evaluate_blimp
    from src.metrics import CLASS_ORDER, save_predictions, save_result_json
except ModuleNotFoundError as error:
    # ``python src/train_transformer.py`` puts src/ (not the repo root) on sys.path.
    if error.name not in {"src", "src.evaluate_blimp", "src.metrics"}:
        raise
    from evaluate_blimp import evaluate_blimp
    from metrics import CLASS_ORDER, save_predictions, save_result_json

LABELS = list(CLASS_ORDER)
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
ROOT = Path(__file__).resolve().parent.parent

# Result-file model name -> Hugging Face checkpoint and default learning rate.
MODELS = {
    "distilbert": {"checkpoint": "distilbert/distilbert-base-uncased", "lr": 3e-5},
    "deberta-v3": {"checkpoint": "microsoft/deberta-v3-base", "lr": 2e-5},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def encode(tokenizer, texts, max_len):
    """Tokenize without padding; batches are padded to their own longest sentence."""
    full = tokenizer(texts)["input_ids"]
    n_truncated = sum(len(ids) > max_len for ids in full)
    enc = tokenizer(texts, truncation=True, max_length=max_len)
    features = [{key: enc[key][i] for key in enc.keys()} for i in range(len(texts))]
    return features, n_truncated


def load_split(name, tokenizer, max_len):
    df = pd.read_csv(ROOT / "data" / "processed" / f"{name}.csv")
    features, n_truncated = encode(tokenizer, df["text"].tolist(), max_len)
    labels = [LABEL_TO_ID[x] for x in df["label"]]
    print(f"{name}: {len(df)} examples, {n_truncated} truncated to {max_len} tokens")
    return df, features, labels


def make_collate(tokenizer):
    def collate(batch):
        features = [f for f, _ in batch]
        labels = torch.tensor([y for _, y in batch])
        return tokenizer.pad(features, return_tensors="pt"), labels

    return collate


def autocast(device, precision):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=precision == "bf16" and device.type == "cuda",
    )


def predict_logits(model, features, collate, device, precision, batch_size=128):
    model.eval()
    loader = DataLoader(
        [(f, 0) for f in features], batch_size=batch_size, collate_fn=collate
    )
    out = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with autocast(device, precision):
                logits = model(**inputs).logits
            out.append(logits.float().cpu())
    return torch.cat(out)


def predict_proba(model, tokenizer, sentences, collate, device, args):
    features, _ = encode(tokenizer, list(sentences), args.max_len)
    logits = predict_logits(model, features, collate, device, args.precision)
    return torch.softmax(logits.double(), dim=1).numpy()


def train(args, tokenizer, collate, device):
    _, train_features, train_labels = load_split("train", tokenizer, args.max_len)
    _, dev_features, dev_labels = load_split("dev", tokenizer, args.max_len)

    loader = DataLoader(
        list(zip(train_features, train_labels)),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Keep fp32 master weights: transformers loads a checkpoint in its stored dtype,
    # and deberta-v3-base is stored in fp16, where AdamW's eps (1e-8) rounds to 0
    # and the first update turns every weight into NaN. bf16 is only used in autocast.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODELS[args.model]["checkpoint"],
        dtype=torch.float32,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL_TO_ID,
    ).to(device)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * total_steps), total_steps
    )
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        for step, (inputs, labels) in enumerate(loader, start=1):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            with autocast(device, precision=args.precision):
                logits = model(**inputs).logits
            loss = loss_fn(logits.float(), labels)
            # Numerical stability check: stop at once instead of training on NaN/inf.
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss {loss.item()} at epoch {epoch}, step {step}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(labels)

        dev_pred = predict_logits(
            model, dev_features, collate, device, args.precision
        ).argmax(dim=1)
        dev_f1 = f1_score(dev_labels, dev_pred.numpy(), average="macro")
        print(
            f"epoch {epoch}  train_loss {total_loss / len(train_labels):.4f}  "
            f"dev_macro_f1 {dev_f1:.4f}  ({time.time() - epoch_start:.0f}s)"
        )
        if dev_f1 > best_f1:
            best_f1, best_epoch = dev_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"best epoch: {best_epoch}  dev_macro_f1 {best_f1:.4f}")
    return model


def hardware_description(device):
    if device.type == "cuda":
        return f"GPU ({torch.cuda.get_device_name(0)})"
    return "CPU"


def final_test(model, tokenizer, collate, args, device, train_time_sec):
    test_df, test_features, _ = load_split("test", tokenizer, args.max_len)
    pred = predict_logits(model, test_features, collate, device, args.precision)
    pred_labels = [LABELS[i] for i in pred.argmax(dim=1).numpy()]
    true_labels = test_df["label"].tolist()

    blimp = evaluate_blimp(
        lambda sentences: predict_proba(model, tokenizer, sentences, collate, device, args),
        probability_class_order=LABELS,
        blimp_dir=ROOT / "data" / "raw" / "blimp",
    )

    save_predictions(
        ids=test_df["id"].tolist(),
        y_true=true_labels,
        y_pred=pred_labels,
        model=args.model,
        seed=args.seed,
        results_dir=ROOT / "results",
    )
    result_path = save_result_json(
        model=args.model,
        seed=args.seed,
        y_true=true_labels,
        y_pred=pred_labels,
        blimp=blimp,
        train_time_sec=train_time_sec,
        hardware=hardware_description(device),
        results_dir=ROOT / "results",
    )
    test_f1 = f1_score(true_labels, pred_labels, average="macro")
    print(f"test_macro_f1 {test_f1:.4f}")
    print(f"blimp_pair_accuracy {blimp['pair_accuracy']['overall']:.4f}")
    print(f"blimp_type_accuracy {blimp['type_accuracy']['overall']:.4f}")
    print(f"saved {result_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=None, help="default depends on --model")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--final", action="store_true")
    args = p.parse_args()
    if args.lr is None:
        args.lr = MODELS[args.model]["lr"]
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"model: {args.model} ({MODELS[args.model]['checkpoint']})  device: {device}  "
        f"seed: {args.seed}  lr: {args.lr}  precision: {args.precision}"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model]["checkpoint"])
    collate = make_collate(tokenizer)

    start = time.time()
    model = train(args, tokenizer, collate, device)
    train_time_sec = time.time() - start
    print(f"train_time_sec {train_time_sec:.1f}")

    if args.final:
        final_test(model, tokenizer, collate, args, device, train_time_sec)


if __name__ == "__main__":
    main()
