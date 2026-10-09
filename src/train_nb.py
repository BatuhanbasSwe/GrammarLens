"""Train and evaluate the GrammarLens multinomial Naive Bayes baseline.

Development run (does not read the test split or write result files):

    python src/train_nb.py

Final run (evaluates test + BLiMP and writes the required artifacts):

    python src/train_nb.py --final

The vectorizer and classifier are fitted only on ``data/processed/train.csv``.
The development, test, and BLiMP texts are transformed with that frozen
vocabulary, which prevents evaluation data from leaking into the model.
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

try:
    from src.evaluate_blimp import evaluate_blimp
    from src.metrics import (
        CLASS_ORDER,
        compute_classification_metrics,
        save_predictions,
        save_result_json,
    )
except ModuleNotFoundError as error:
    # ``python src/train_nb.py`` places src/ rather than the repository root
    # on sys.path. Only fall back for that specific package-resolution case.
    if error.name not in {"src", "src.evaluate_blimp", "src.metrics"}:
        raise
    from evaluate_blimp import evaluate_blimp
    from metrics import (
        CLASS_ORDER,
        compute_classification_metrics,
        save_predictions,
        save_result_json,
    )


ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
BLIMP_DIR = ROOT / "data" / "raw" / "blimp"
RESULTS_DIR = ROOT / "results"
REQUIRED_COLUMNS = ("id", "label", "text")
DEFAULT_SEED = 42


def load_split(path: str | Path) -> pd.DataFrame:
    """Load one processed split and validate the fields used by the model."""

    split_path = Path(path)
    if not split_path.is_file():
        raise FileNotFoundError(f"Dataset split not found: {split_path}")

    frame = pd.read_csv(split_path)
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{split_path} is missing required columns: {missing_columns}"
        )
    if frame.empty:
        raise ValueError(f"{split_path} must contain at least one row")
    if frame[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError(f"{split_path} contains missing id, label, or text values")

    frame = frame.copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = frame[column].astype(str)
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"{split_path} contains empty {column} values")

    unknown_labels = sorted(set(frame["label"]) - set(CLASS_ORDER))
    if unknown_labels:
        raise ValueError(f"{split_path} contains unknown labels: {unknown_labels}")
    if frame["id"].duplicated().any():
        raise ValueError(f"{split_path} contains duplicate ids")
    return frame


def train_nb(
    train_frame: pd.DataFrame,
    *,
    alpha: float = 1.0,
) -> tuple[CountVectorizer, MultinomialNB]:
    """Fit unigram word counts and a multinomial NB model on train only."""

    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a finite number greater than zero")

    observed_labels = set(train_frame["label"])
    missing_labels = [label for label in CLASS_ORDER if label not in observed_labels]
    if missing_labels:
        raise ValueError(f"Training split is missing classes: {missing_labels}")

    vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 1))
    train_features = vectorizer.fit_transform(train_frame["text"].tolist())
    model = MultinomialNB(alpha=float(alpha))
    model.fit(train_features, train_frame["label"].tolist())
    return vectorizer, model


def predict_proba(
    model: MultinomialNB,
    vectorizer: CountVectorizer,
    sentences: Sequence[str],
) -> np.ndarray:
    """Return class probabilities in ``model.classes_`` order."""

    sentence_list = list(sentences)
    if any(not isinstance(sentence, str) for sentence in sentence_list):
        raise TypeError("sentences must contain only strings")
    features = vectorizer.transform(sentence_list)
    return np.asarray(model.predict_proba(features), dtype=float)


def _predict_labels(
    model: MultinomialNB,
    vectorizer: CountVectorizer,
    sentences: Sequence[str],
) -> list[str]:
    features = vectorizer.transform(list(sentences))
    return [str(label) for label in model.predict(features)]


def _hardware_description() -> str:
    processor = (
        platform.processor().strip()
        or os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
        or platform.machine().strip()
        or "unknown processor"
    )
    return f"CPU ({processor})"


def run(
    *,
    seed: int = DEFAULT_SEED,
    alpha: float = 1.0,
    final: bool = False,
    blimp_batch_size: int = 256,
) -> dict[str, Any]:
    """Train NB, report dev metrics, and optionally create final artifacts."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    train_frame = load_split(PROCESSED_DATA_DIR / "train.csv")
    dev_frame = load_split(PROCESSED_DATA_DIR / "dev.csv")

    start = time.perf_counter()
    vectorizer, model = train_nb(train_frame, alpha=alpha)
    train_time_sec = time.perf_counter() - start

    dev_predictions = _predict_labels(
        model, vectorizer, dev_frame["text"].tolist()
    )
    dev_metrics = compute_classification_metrics(
        dev_frame["label"].tolist(), dev_predictions
    )
    outcome: dict[str, Any] = {
        "model": "nb",
        "seed": seed,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "train_time_sec": train_time_sec,
        "dev_metrics": dev_metrics,
    }

    if not final:
        return outcome

    test_frame = load_split(PROCESSED_DATA_DIR / "test.csv")
    test_predictions = _predict_labels(
        model, vectorizer, test_frame["text"].tolist()
    )
    blimp = evaluate_blimp(
        lambda sentences: predict_proba(model, vectorizer, sentences),
        probability_class_order=[str(label) for label in model.classes_],
        blimp_dir=BLIMP_DIR,
        batch_size=blimp_batch_size,
    )

    predictions_path = save_predictions(
        ids=test_frame["id"].tolist(),
        y_true=test_frame["label"].tolist(),
        y_pred=test_predictions,
        model="nb",
        seed=seed,
        results_dir=RESULTS_DIR,
    )
    result_path = save_result_json(
        model="nb",
        seed=seed,
        y_true=test_frame["label"].tolist(),
        y_pred=test_predictions,
        blimp=blimp,
        train_time_sec=train_time_sec,
        hardware=_hardware_description(),
        results_dir=RESULTS_DIR,
    )
    outcome.update(
        {
            "test_metrics": compute_classification_metrics(
                test_frame["label"].tolist(), test_predictions
            ),
            "blimp": blimp,
            "predictions_path": predictions_path,
            "result_path": result_path,
        }
    )
    return outcome


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the GrammarLens multinomial Naive Bayes baseline."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Result-file seed label; NB itself is deterministic (default: 42).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="MultinomialNB additive smoothing value (default: 1.0).",
    )
    parser.add_argument(
        "--blimp-batch-size",
        type=int,
        default=256,
        help="Number of BLiMP pairs scored per batch (default: 256).",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Evaluate test and BLiMP, then write final result artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outcome = run(
        seed=args.seed,
        alpha=args.alpha,
        final=args.final,
        blimp_batch_size=args.blimp_batch_size,
    )
    dev_metrics = outcome["dev_metrics"]
    print(
        f"model=nb seed={outcome['seed']} "
        f"vocabulary_size={outcome['vocabulary_size']} "
        f"train_time_sec={outcome['train_time_sec']:.3f}"
    )
    print(
        f"dev_accuracy={dev_metrics['accuracy']:.4f} "
        f"dev_macro_f1={dev_metrics['macro_f1']:.4f}"
    )

    if args.final:
        test_metrics = outcome["test_metrics"]
        blimp = outcome["blimp"]
        print(
            f"test_accuracy={test_metrics['accuracy']:.4f} "
            f"test_macro_f1={test_metrics['macro_f1']:.4f}"
        )
        print(
            f"blimp_pair_accuracy={blimp['pair_accuracy']['overall']:.4f} "
            f"blimp_type_accuracy={blimp['type_accuracy']['overall']:.4f}"
        )
        print(f"predictions={outcome['predictions_path']}")
        print(f"result={outcome['result_path']}")
    else:
        print("development mode: test/BLiMP were not read and no results were written")


if __name__ == "__main__":
    main()
