"""Long-distance agreement analysis for the CNN on the SVA test sentences.

For every SVA sentence in the test split, finds the verb that was changed by the
error injection and its subject (from the EWT dependency trees), counts the words
between them, and prints the CNN's error rate per distance for each seed.

Reads: data/processed/test.csv, results/preds/cnn_seed*.csv and
data/raw/UD_English-EWT/en_ewt-ud-test.conllu. Writes nothing; the result is
printed to the screen.

Run from the repository root:  python src/analyze_long_distance.py
"""

import argparse
import csv
import difflib
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOKEN_RE = re.compile(r"\w+|[^\w\s]")
WINDOW_FIT_MAX_BETWEEN = 3


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def load_conllu(path):
    sentences = {}
    current = None
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            line = line.rstrip("\n")
            if line.startswith("# sent_id"):
                current = line.split("=", 1)[1].strip()
                sentences[current] = []
            elif line and not line.startswith("#"):
                fields = line.split("\t")
                if fields[0].isdigit():
                    sentences[current].append(
                        {
                            "id": int(fields[0]),
                            "form": fields[1],
                            "upos": fields[3],
                            "head": int(fields[6]),
                            "rel": fields[7],
                        }
                    )
    return sentences


def changed_word(source, text):
    src, new = tokenize(source), tokenize(text)
    ops = [op for op in difflib.SequenceMatcher(None, src, new).get_opcodes() if op[0] != "equal"]
    if len(ops) != 1 or ops[0][2] - ops[0][1] != 1 or ops[0][4] - ops[0][3] != 1:
        return None
    index = ops[0][1]
    occurrence = sum(1 for tok in src[:index] if tok == src[index])
    return src[index], occurrence


def words_between(words, verb):
    by_id = {w["id"]: w for w in words}
    target = verb
    if verb["rel"] in ("cop", "aux", "aux:pass") and verb["head"] in by_id:
        target = by_id[verb["head"]]
    subjects = [
        w
        for w in words
        if w["head"] == target["id"]
        and w["rel"].split(":")[0] in ("nsubj", "csubj")
        and not w["rel"].endswith(":outer")
    ]
    if not subjects:
        return None, None
    subject = min(subjects, key=lambda w: abs(w["id"] - verb["id"]))
    low, high = sorted((subject["id"], verb["id"]))
    count = sum(1 for w in words if low < w["id"] < high and w["upos"] != "PUNCT")
    return subject, count


def measure(seeds):
    conllu = load_conllu(ROOT / "data" / "raw" / "UD_English-EWT" / "en_ewt-ud-test.conllu")
    with open(ROOT / "data" / "processed" / "test.csv", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["label"] == "SVA"]
    predictions = {}
    for seed in seeds:
        with open(ROOT / "results" / "preds" / f"cnn_seed{seed}.csv", encoding="utf-8") as stream:
            predictions[seed] = {r["id"]: r["pred"] for r in csv.DictReader(stream)}

    records = []
    for row in rows:
        change = changed_word(row["source"], row["text"])
        words = conllu.get(row["sent_id"])
        if change is None or not words:
            continue
        form, occurrence = change
        candidates = [w for w in words if w["form"].lower() == form]
        if len(candidates) <= occurrence:
            continue
        verb = candidates[occurrence]
        subject, between = words_between(words, verb)
        if subject is None:
            continue
        records.append(
            {
                "text": row["text"],
                "subject": subject["form"],
                "verb": verb["form"],
                "between": between,
                "pred": {s: predictions[s][row["id"]] for s in seeds},
            }
        )
    return len(rows), records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 2026])
    parser.add_argument("--examples-seed", type=int, default=13)
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()

    total, records = measure(args.seeds)
    print(f"SVA test sentences: {total}, measured: {len(records)}")

    for seed in args.seeds:
        print(f"\nseed {seed}")
        groups = defaultdict(lambda: [0, 0])
        for rec in records:
            key = min(rec["between"], 4)
            groups[key][0] += 1
            groups[key][1] += rec["pred"][seed] != "SVA"
        for key in sorted(groups):
            n, wrong = groups[key]
            label = f"{key}+" if key == 4 else str(key)
            print(f"  words between {label:>2}: n={n:3d} wrong={wrong:3d} rate={wrong / n:.3f}")
        inside = [r for r in records if r["between"] <= WINDOW_FIT_MAX_BETWEEN]
        outside = [r for r in records if r["between"] > WINDOW_FIT_MAX_BETWEEN]
        for name, group in (("inside window (0-3)", inside), ("outside window (4+)", outside)):
            wrong = sum(r["pred"][seed] != "SVA" for r in group)
            print(f"  {name}: n={len(group)} wrong={wrong} rate={wrong / len(group):.3f}")

    wrong_all = [r for r in records if all(r["pred"][s] != "SVA" for s in args.seeds)]
    print(f"\nwrong in every seed: {len(wrong_all)}")
    shown = [r for r in records if r["pred"][args.examples_seed] != "SVA"]
    shown.sort(key=lambda r: -r["between"])
    print(f"\nlongest-distance errors, seed {args.examples_seed}:")
    for rec in shown[: args.examples]:
        print(
            f"  between={rec['between']:2d} subject={rec['subject']} verb={rec['verb']} "
            f"pred={rec['pred'][args.examples_seed]} | {rec['text'][:110]}"
        )


if __name__ == "__main__":
    main()
