import argparse
import copy
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate_blimp import evaluate_blimp
from metrics import save_predictions, save_result_json

LABELS = ["CORRECT", "SVA", "VERB_FORM", "DET", "NOUN_NUM", "PREP", "WORD_ORDER"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
PAD_ID = 0
UNK_ID = 1
TOKEN_RE = re.compile(r"\w+|[^\w\s]")
ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts, min_freq):
    counts = Counter(tok for text in texts for tok in tokenize(text))
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    for tok, count in counts.most_common():
        if count >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode(texts, vocab, max_len):
    ids = np.full((len(texts), max_len), PAD_ID, dtype=np.int64)
    for i, text in enumerate(texts):
        toks = [vocab.get(tok, UNK_ID) for tok in tokenize(text)][:max_len]
        ids[i, : len(toks)] = toks
    return torch.from_numpy(ids)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_filters, kernel_sizes, dropout, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, num_filters, k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)
        pooled = [torch.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


def predict_logits(model, ids, device, batch_size=256):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size].to(device)
            out.append(model(batch).cpu())
    return torch.cat(out)


def predict_proba(model, vocab, sentences, device, max_len=64):
    ids = encode(sentences, vocab, max_len)
    return torch.softmax(predict_logits(model, ids, device), dim=1).numpy()


def load_split(name, vocab, max_len):
    df = pd.read_csv(ROOT / "data" / "processed" / f"{name}.csv")
    ids = encode(df["text"].tolist(), vocab, max_len)
    labels = torch.tensor([LABEL_TO_ID[x] for x in df["label"]])
    return df, ids, labels


def train(args, device):
    train_df = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    vocab = build_vocab(train_df["text"].tolist(), args.min_freq)
    print(f"vocab size: {len(vocab)}")

    _, train_ids, train_labels = load_split("train", vocab, args.max_len)
    _, dev_ids, dev_labels = load_split("dev", vocab, args.max_len)

    loader = DataLoader(
        TensorDataset(train_ids, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    model = TextCNN(
        len(vocab), args.emb_dim, args.num_filters, [3, 4, 5], args.dropout, len(LABELS)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        dev_pred = predict_logits(model, dev_ids, device).argmax(dim=1).numpy()
        dev_f1 = f1_score(dev_labels.numpy(), dev_pred, average="macro")
        print(
            f"epoch {epoch:2d}  train_loss {total_loss / len(train_ids):.4f}  "
            f"dev_macro_f1 {dev_f1:.4f}"
        )
        if dev_f1 > best_f1:
            best_f1, best_epoch = dev_f1, epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(f"best epoch: {best_epoch}  dev_macro_f1 {best_f1:.4f}")
    return model, vocab, best_epoch, best_f1


def hardware_description(device):
    if device.type == "cuda":
        return f"GPU ({torch.cuda.get_device_name(0)})"
    return "CPU"


def final_test(model, vocab, args, device, train_time_sec):
    test_df, test_ids, _ = load_split("test", vocab, args.max_len)
    pred = predict_logits(model, test_ids, device).argmax(dim=1).numpy()
    pred_labels = [LABELS[i] for i in pred]
    true_labels = test_df["label"].tolist()

    blimp = evaluate_blimp(
        lambda sentences: predict_proba(model, vocab, sentences, device, args.max_len),
        probability_class_order=LABELS,
        blimp_dir=ROOT / "data" / "raw" / "blimp",
    )

    save_predictions(
        ids=test_df["id"].tolist(),
        y_true=true_labels,
        y_pred=pred_labels,
        model="cnn",
        seed=args.seed,
        results_dir=ROOT / "results",
    )
    result_path = save_result_json(
        model="cnn",
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--emb_dim", type=int, default=100)
    p.add_argument("--num_filters", type=int, default=100)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--min_freq", type=int, default=2)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--final", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  seed: {args.seed}")

    start = time.time()
    model, vocab, best_epoch, best_f1 = train(args, device)
    train_time_sec = time.time() - start
    print(f"train_time_sec {train_time_sec:.1f}")

    if args.final:
        final_test(model, vocab, args, device, train_time_sec)


if __name__ == "__main__":
    main()
