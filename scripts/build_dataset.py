"""Build the GrammarLens dataset from the raw UD English-EWT treebank.

Usage (run from the repository root, after scripts/download_data.py):
    python scripts/build_dataset.py

Pipeline
  1. Read the three .conllu files (train / dev / test, the official EWT document-level split).
  2. Clean: drop sentences that the treebank itself marks as containing typos/errors,
     sentences shorter than 5 or longer than 40 tokens, sentences with URLs/e-mails,
     mostly-uppercase sentences, sentences without a finite verb, and duplicates
     (a sentence already seen in an earlier split is dropped -> no train/test leakage).
  3. For every clean sentence, keep the original (label CORRECT) and inject exactly ONE error
     of each applicable type, using the gold POS / morphology / dependency annotation:
        SVA         subject-verb agreement    "He are concerned ..."
        VERB_FORM   wrong verb form           "I want to went ..."
        DET         article missing / a-an    "took an rescue puppy"
        NOUN_NUM    determiner-noun number    "this events"
        PREP        wrong preposition         "given directly with John"
        WORD_ORDER  swapped neighbours        "for company the"
  4. Balance the 7 classes inside each split (downsample to the smallest class).
Outputs: data/processed/{train,dev,test}.csv and data/processed/stats.json
"""
import collections
import json
import pathlib
import random
import re

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "UD_English-EWT"
OUT = ROOT / "data" / "processed"
SEED = 42
random.seed(SEED)

LABELS = ["CORRECT", "SVA", "VERB_FORM", "DET", "NOUN_NUM", "PREP", "WORD_ORDER"]


# ----------------------------------------------------------------------------- reading
def read_conllu(path):
    """Return a list of sentences. Each sentence = dict(meta=..., w=[word dicts], r={multiword ranges})."""
    sents, words, ranges, meta = [], [], {}, {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:                                   # blank line = end of sentence
            if words:
                sents.append(dict(meta=meta, w=words, r=ranges))
            words, ranges, meta = [], {}, {}
            continue
        if line.startswith("#"):                       # comments: sent_id, text
            if " = " in line:
                k, v = line[2:].split(" = ", 1)
                meta[k] = v
            continue
        c = line.split("\t")
        if "." in c[0]:                                # empty nodes (enhanced UD) -> skip
            continue
        misc = dict(x.split("=", 1) for x in c[9].split("|") if "=" in x)
        if "-" in c[0]:                                # multiword token, e.g. 19-20 don't
            a, b = map(int, c[0].split("-"))
            ranges[a] = (b, c[1], misc)
            continue
        feats = dict(x.split("=", 1) for x in c[5].split("|") if "=" in x)
        words.append(dict(id=int(c[0]), form=c[1], lemma=c[2].lower(), upos=c[3], xpos=c[4],
                          feats=feats, head=int(c[6]) if c[6] != "_" else 0, rel=c[7], misc=misc))
    return sents


splits = {s: read_conllu(RAW / f"en_ewt-ud-{s}.conllu") for s in ["train", "dev", "test"]}

# inflection lexicon built from the treebank itself: (lemma, PTB tag) -> most frequent word form
lex = collections.defaultdict(collections.Counter)
for sentences in splits.values():
    for s in sentences:
        for w in s["w"]:
            if w["form"].isalpha():
                lex[(w["lemma"], w["xpos"])][w["form"].lower()] += 1


def inflect(lemma, xpos):
    c = lex.get((lemma, xpos))
    return c.most_common(1)[0][0] if c else None


# ----------------------------------------------------------------------------- helpers
def render(words, ranges):
    """Turn a word list back into a sentence string, respecting SpaceAfter=No and multiword tokens."""
    out, n = [], len(words)
    pos = {w["id"]: k for k, w in enumerate(words)}
    k = 0
    while k < n:
        w = words[k]
        if w["id"] in ranges and not w.get("_touched"):
            b, form, misc = ranges[w["id"]]
            out.append(form + ("" if misc.get("SpaceAfter") == "No" else " "))
            k = pos.get(b, k) + 1
            continue
        out.append(w["form"] + ("" if w["misc"].get("SpaceAfter") == "No" else " "))
        k += 1
    return "".join(out).strip()


def in_range(s, wid):
    return any(a <= wid <= b for a, (b, _, _) in s["r"].items())


def match_case(new, old):
    return new[:1].upper() + new[1:] if old[:1].isupper() and old.lower() != "i" else new


def children(s, wid):
    return [w for w in s["w"] if w["head"] == wid]


def replace_word(s, w, new_form):
    w2 = dict(w, form=match_case(new_form, w["form"]))
    return [w2 if x["id"] == w["id"] else x for x in s["w"]]


# ----------------------------------------------------------------------------- error injectors
# each function returns a new word list with ONE error, or None if the sentence has no suitable spot
def make_sva(s):
    cands = []
    for w in s["w"]:
        if w["upos"] not in ("VERB", "AUX") or not w["form"].isalpha() or in_range(s, w["id"]):
            continue
        clause = w["id"] if w["rel"] not in ("cop", "aux", "aux:pass") else w["head"]
        if not any(c["rel"].startswith("nsubj") or c["rel"] == "expl" for c in children(s, clause)):
            continue                                   # verb must have a visible subject
        if w["xpos"] in ("VBZ", "VBP"):                # runs <-> run, is <-> are
            new = inflect(w["lemma"], "VBP" if w["xpos"] == "VBZ" else "VBZ")
            if w["lemma"] == "be":
                new = "are" if w["xpos"] == "VBZ" else "is"
        elif w["lemma"] == "be" and w["form"].lower() in ("was", "were"):
            new = "were" if w["form"].lower() == "was" else "was"
        else:
            continue
        if new and new != w["form"].lower():
            cands.append((w, new))
    if not cands:
        return None
    w, new = random.choice(cands)
    return replace_word(s, w, new)


def make_verb_form(s):
    cands = []
    for w in s["w"]:
        if w["upos"] != "VERB" or not w["form"].isalpha() or in_range(s, w["id"]):
            continue
        ch = children(s, w["id"])
        aux = [c["lemma"] for c in ch if c["rel"] in ("aux", "aux:pass")]
        has_modal = any(c["xpos"] == "MD" for c in ch if c["rel"] == "aux")
        has_to = any(c["lemma"] == "to" and c["rel"] == "mark" for c in ch)
        options = []
        if w["xpos"] == "VB" and (has_modal or has_to):          # can go -> can goes / to went
            options = ["VBZ", "VBD"]
        elif w["xpos"] == "VBN" and ("have" in aux or "be" in aux):  # has gone -> has go / has went
            options = ["VB", "VBD"]
        elif w["xpos"] == "VBG" and "be" in aux:                 # is going -> is go
            options = ["VB"]
        for tag in options:
            new = inflect(w["lemma"], tag)
            if new and new != w["form"].lower():
                cands.append((w, new))
    if not cands:
        return None
    w, new = random.choice(cands)
    return replace_word(s, w, new)


def capitalize_first(words):
    words = [dict(x) for x in words]
    if words and words[0]["form"][:1].islower():
        words[0]["form"] = words[0]["form"][:1].upper() + words[0]["form"][1:]
    return words


def make_det(s):
    byid = {w["id"]: w for w in s["w"]}
    cands = [w for w in s["w"] if w["lemma"] == "a" and w["rel"] == "det" and w["form"].lower() in ("a", "an")
             and byid.get(w["head"], {}).get("xpos") == "NN" and not in_range(s, w["id"])]
    if not cands:
        return None
    w = random.choice(cands)
    if random.random() < 0.5:                                    # delete the article: "a car" -> "car"
        words = [x for x in s["w"] if x["id"] != w["id"]]
        return capitalize_first(words) if w is s["w"][0] else words
    return replace_word(s, w, "an" if w["form"].lower() == "a" else "a")   # a <-> an


SG_DET = {"a", "an", "this", "that", "every", "each", "another"}
PL_DET = {"these", "those", "many", "several", "few", "both", "two", "three", "four", "five", "various", "numerous"}


def make_noun_num(s):
    cands = []
    for w in s["w"]:
        if w["upos"] != "NOUN" or not w["form"].isalpha() or w["rel"].startswith("nsubj") or in_range(s, w["id"]):
            continue                                   # subjects excluded so the error is not also an SVA error
        dets = {c["form"].lower() for c in children(s, w["id"]) if c["rel"] in ("det", "nummod")}
        if w["xpos"] == "NN" and dets & SG_DET:        # this event -> this events
            new = inflect(w["lemma"], "NNS")
        elif w["xpos"] == "NNS" and dets & PL_DET:     # these events -> these event
            new = inflect(w["lemma"], "NN")
        else:
            continue
        if new and new != w["form"].lower():
            cands.append((w, new))
    if not cands:
        return None
    w, new = random.choice(cands)
    return replace_word(s, w, new)


PREPS = ["in", "on", "at", "to", "for", "of", "with", "from", "by", "about"]


def make_prep(s):
    byid = {w["id"]: w for w in s["w"]}
    cands = [w for w in s["w"] if w["upos"] == "ADP" and w["rel"] == "case" and w["form"].lower() in PREPS
             and byid.get(w["head"], {}).get("upos") in ("NOUN", "PROPN", "PRON") and not in_range(s, w["id"])]
    if not cands:
        return None
    w = random.choice(cands)
    return replace_word(s, w, random.choice([p for p in PREPS if p != w["form"].lower()]))


def make_word_order(s):
    W = s["w"]
    cands = []
    for k in range(len(W) - 1):
        a, b = W[k], W[k + 1]                          # determiner/adjective directly before its noun
        if b["upos"] == "NOUN" and a["head"] == b["id"] and (a["rel"] == "det" or (a["rel"] == "amod" and a["upos"] == "ADJ")) \
           and a["form"].isalpha() and b["form"].isalpha() and not in_range(s, a["id"]) and not in_range(s, b["id"]):
            cands.append(k)
    if not cands:
        return None
    k = random.choice(cands)
    W = [dict(x) for x in W]
    a, b = W[k], W[k + 1]
    a["form"], b["form"] = b["form"], a["form"]       # "the car" -> "car the"
    if k == 0:
        a["form"] = a["form"][:1].upper() + a["form"][1:]
        if b["upos"] != "PROPN" and b["form"] != "I":
            b["form"] = b["form"][:1].lower() + b["form"][1:]
    return W


INJECTORS = {"SVA": make_sva, "VERB_FORM": make_verb_form, "DET": make_det,
             "NOUN_NUM": make_noun_num, "PREP": make_prep, "WORD_ORDER": make_word_order}


# ----------------------------------------------------------------------------- cleaning + generation
def main():
    stats = collections.Counter()
    rows, seen = [], set()
    for split, sentences in splits.items():
        for s in sentences:
            stats["raw"] += 1
            text = s["meta"].get("text", "")
            n = len(s["w"])
            if any(w["feats"].get("Typo") == "Yes" or "CorrectForm" in w["misc"] or "CorrectSpaceAfter" in w["misc"]
                   for w in s["w"]):
                stats["drop_annotated_error"] += 1; continue
            if not (5 <= n <= 40):
                stats["drop_length"] += 1; continue
            if re.search(r"https?://|www\.|@|\.com\b", text):
                stats["drop_url_email"] += 1; continue
            letters = [ch for ch in text if ch.isalpha()]
            if letters and sum(ch.isupper() for ch in letters) / len(letters) > 0.5:
                stats["drop_caps"] += 1; continue
            if not any(w["feats"].get("VerbForm") == "Fin" for w in s["w"]):
                stats["drop_no_finite_verb"] += 1; continue
            key = text.lower().strip()
            if key in seen:
                stats["drop_duplicate"] += 1; continue
            seen.add(key)
            stats["kept"] += 1

            sid = s["meta"]["sent_id"]
            base = render(s["w"], s["r"])
            rows.append((split, sid, "CORRECT", base, base))
            for label, inject in INJECTORS.items():
                out = inject(s)
                if out is None:
                    continue
                orig = {w["id"]: w["form"] for w in s["w"]}
                out = [dict(w, _touched=(w["form"] != orig[w["id"]])) for w in out]
                ids = {w["id"] for w in out}
                keep_ranges = {a: v for a, v in s["r"].items()
                               if all(i in ids for i in range(a, v[0] + 1))
                               and all(not w["_touched"] for w in out if a <= w["id"] <= v[0])}
                t = render(out, keep_ranges)
                if t != base:
                    rows.append((split, sid, label, t, base))

    df = pd.DataFrame(rows, columns=["split", "sent_id", "label", "text", "source"])
    available = df.groupby(["split", "label"]).size().unstack(0).to_dict()

    parts = []                                        # balance: every class gets the size of the smallest one
    for split, g in df.groupby("split"):
        cap = g.label.value_counts().min()
        for label, x in g.groupby("label"):
            parts.append(x.sample(cap, random_state=SEED))
    df = pd.concat(parts).reset_index(drop=True)

    OUT.mkdir(parents=True, exist_ok=True)
    for split in ["train", "dev", "test"]:
        part = df[df.split == split].drop(columns="split").sample(frac=1, random_state=SEED).reset_index(drop=True)
        part.insert(0, "id", [f"{split}-{i:05d}" for i in range(len(part))])
        part.to_csv(OUT / f"{split}.csv", index=False)

    final = df.groupby(["split", "label"]).size().unstack(0).fillna(0).astype(int).to_dict()
    (OUT / "stats.json").write_text(json.dumps({"seed": SEED, "cleaning": dict(stats),
                                                "available_before_balancing": available,
                                                "final_per_class": final}, indent=2), encoding="utf-8")
    print("Cleaning:", dict(stats))
    print(df.groupby(["split", "label"]).size().unstack(0).loc[LABELS])
    print("Total examples:", len(df), "->", OUT)


if __name__ == "__main__":
    main()
