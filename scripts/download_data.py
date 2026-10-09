"""Download the raw data for GrammarLens into data/raw/.

Usage (run from the repository root):
    python scripts/download_data.py

What it downloads (pinned to exact commits, so everyone gets identical files):
  1. UD English-EWT treebank (CC BY-SA 4.0) -> data/raw/UD_English-EWT/
  2. The 14 BLiMP phenomena that match our classes (CC BY 4.0) -> data/raw/blimp/
It also writes data/raw/checksums.txt (SHA-256 of every file) so the raw data can be verified.
Only the Python standard library is used.
"""
import hashlib
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

EWT_COMMIT = "4a4d77f599ea53cc405f85d0cec4b2f14f81d42b"     # UD_English-EWT, 2026-05-06
BLIMP_COMMIT = "3e56b06fcabca9b30822fc66435fca6b1aa40bb1"   # blimp, 2022-12-13

EWT_URL = f"https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/{EWT_COMMIT}/"
BLIMP_URL = f"https://raw.githubusercontent.com/alexwarstadt/blimp/{BLIMP_COMMIT}/data/"

EWT_FILES = ["en_ewt-ud-train.conllu", "en_ewt-ud-dev.conllu", "en_ewt-ud-test.conllu", "LICENSE.txt"]

# BLiMP phenomena used as an external test set (mapped to our labels in the evaluation script)
BLIMP_FILES = [
    # determiner-noun number agreement -> NOUN_NUM
    "determiner_noun_agreement_1", "determiner_noun_agreement_2",
    "determiner_noun_agreement_irregular_1", "determiner_noun_agreement_irregular_2",
    "determiner_noun_agreement_with_adjective_1", "determiner_noun_agreement_with_adj_2",
    "determiner_noun_agreement_with_adj_irregular_1", "determiner_noun_agreement_with_adj_irregular_2",
    # subject-verb agreement -> SVA
    "regular_plural_subject_verb_agreement_1", "regular_plural_subject_verb_agreement_2",
    "irregular_plural_subject_verb_agreement_1", "irregular_plural_subject_verb_agreement_2",
    "distractor_agreement_relational_noun", "distractor_agreement_relative_clause",
]


def download(url: str, dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already there: {dest.relative_to(ROOT)}")
        return
    print(f"  downloading  : {dest.relative_to(ROOT)}")
    with urllib.request.urlopen(url, timeout=60) as response:
        dest.write_bytes(response.read())


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    print("1/2 UD English-EWT")
    for name in EWT_FILES:
        download(EWT_URL + name, RAW / "UD_English-EWT" / name)

    print("2/2 BLiMP (14 phenomena)")
    for name in BLIMP_FILES:
        download(BLIMP_URL + name + ".jsonl", RAW / "blimp" / (name + ".jsonl"))

    # quick sanity check: number of sentences per EWT split
    for split in ["train", "dev", "test"]:
        path = RAW / "UD_English-EWT" / f"en_ewt-ud-{split}.conllu"
        n = sum(1 for line in path.open(encoding="utf-8") if line.startswith("# sent_id"))
        print(f"  EWT {split:5s}: {n} sentences")
    n_pairs = sum(sum(1 for _ in (RAW / "blimp" / (f + ".jsonl")).open(encoding="utf-8")) for f in BLIMP_FILES)
    print(f"  BLiMP      : {n_pairs} minimal pairs")

    files = sorted(p for p in RAW.rglob("*") if p.is_file() and p.name != "checksums.txt")
    lines = [f"{sha256(p)}  {p.relative_to(RAW).as_posix()}" for p in files]
    (RAW / "checksums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Done. {len(files)} files in data/raw/, checksums written to data/raw/checksums.txt")


if __name__ == "__main__":
    main()
