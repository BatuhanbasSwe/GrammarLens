"""Shared evaluation and result-writing utilities for GrammarLens.

All models must use the class order in ``CLASS_ORDER``. Metric values are stored
as proportions in the [0, 1] interval. Confusion-matrix rows are true labels and
columns are predicted labels, both in ``CLASS_ORDER``.

Typical final-evaluation usage::

    from src.metrics import save_predictions, save_result_json

    save_predictions(
        ids=test_ids,
        y_true=test_labels,
        y_pred=test_predictions,
        model="cnn",
        seed=42,
    )
    save_result_json(
        model="cnn",
        seed=42,
        y_true=test_labels,
        y_pred=test_predictions,
        blimp=blimp_results,
        train_time_sec=train_time_sec,
        hardware=hardware,
    )

Only measured values should be passed to these functions. The bootstrap helper
does not modify result JSON files; its output is intended for benchmark.csv.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


CLASS_ORDER: tuple[str, ...] = (
    "CORRECT",
    "SVA",
    "VERB_FORM",
    "DET",
    "NOUN_NUM",
    "PREP",
    "WORD_ORDER",
)
MODEL_NAMES: tuple[str, ...] = ("nb", "cnn", "distilbert", "deberta-v3")

_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

_BLIMP_CLASS_ORDER: tuple[str, ...] = ("SVA", "NOUN_NUM")
_BLIMP_GROUP_ORDER: tuple[str, ...] = ("overall", *_BLIMP_CLASS_ORDER)
_EXPECTED_BLIMP_PAIR_COUNTS: dict[str, int] = {
    "overall": 14_000,
    "SVA": 6_000,
    "NOUN_NUM": 8_000,
}

__all__ = [
    "CLASS_ORDER",
    "MODEL_NAMES",
    "bootstrap_macro_f1_ci",
    "compute_classification_metrics",
    "save_predictions",
    "save_result_json",
]


def _as_list(values: Sequence[str], *, name: str) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings, not one string")
    try:
        items = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable sequence") from exc
    if not all(isinstance(item, str) for item in items):
        raise TypeError(f"every value in {name} must be a string")
    return items


def _validate_labels(values: Sequence[str], *, name: str) -> list[str]:
    labels = _as_list(values, name=name)
    unknown = sorted(set(labels) - set(CLASS_ORDER))
    if unknown:
        raise ValueError(f"{name} contains unknown labels: {unknown}")
    return labels


def _validate_prediction_vectors(
    y_true: Sequence[str], y_pred: Sequence[str]
) -> tuple[list[str], list[str]]:
    true_labels = _validate_labels(y_true, name="y_true")
    predicted_labels = _validate_labels(y_pred, name="y_pred")
    if not true_labels:
        raise ValueError("y_true and y_pred must not be empty")
    if len(true_labels) != len(predicted_labels):
        raise ValueError(
            "y_true and y_pred must have the same length "
            f"({len(true_labels)} != {len(predicted_labels)})"
        )
    return true_labels, predicted_labels


def _validate_model_and_seed(model: str, seed: int) -> tuple[str, int]:
    if model not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}, got {model!r}")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return model, int(seed)


def _finite_float(value: Real, *, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return number


def _validate_blimp_results(blimp: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(blimp, Mapping):
        raise TypeError("blimp must be a mapping")

    expected_sections = {"pair_accuracy", "type_accuracy", "n_pairs"}
    if set(blimp) != expected_sections:
        raise ValueError(
            "blimp must contain exactly pair_accuracy, type_accuracy, and n_pairs"
        )

    validated: dict[str, Any] = {}
    for metric_name in ("pair_accuracy", "type_accuracy"):
        values = blimp[metric_name]
        if not isinstance(values, Mapping) or set(values) != set(_BLIMP_GROUP_ORDER):
            raise ValueError(
                f"blimp.{metric_name} must contain exactly {_BLIMP_GROUP_ORDER}"
            )
        validated[metric_name] = {
            group: _finite_float(
                values[group],
                name=f"blimp.{metric_name}.{group}",
                minimum=0.0,
                maximum=1.0,
            )
            for group in _BLIMP_GROUP_ORDER
        }

    pair_counts = blimp["n_pairs"]
    if not isinstance(pair_counts, Mapping) or set(pair_counts) != set(
        _BLIMP_GROUP_ORDER
    ):
        raise ValueError(f"blimp.n_pairs must contain exactly {_BLIMP_GROUP_ORDER}")

    normalized_counts: dict[str, int] = {}
    for group in _BLIMP_GROUP_ORDER:
        count = pair_counts[group]
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError(f"blimp.n_pairs.{group} must be an integer")
        normalized_counts[group] = int(count)

    if normalized_counts != _EXPECTED_BLIMP_PAIR_COUNTS:
        raise ValueError(
            "blimp.n_pairs must match the pinned 14-file subset: "
            f"{_EXPECTED_BLIMP_PAIR_COUNTS}"
        )
    validated["n_pairs"] = normalized_counts
    return validated


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8", newline="\n")
    temporary_path.replace(path)


def compute_classification_metrics(
    y_true: Sequence[str], y_pred: Sequence[str]
) -> dict[str, Any]:
    """Compute all required classification metrics in the canonical class order."""

    true_labels, predicted_labels = _validate_prediction_vectors(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=CLASS_ORDER,
        average=None,
        zero_division=0,
    )
    matrix = sklearn_confusion_matrix(
        true_labels,
        predicted_labels,
        labels=CLASS_ORDER,
    )

    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class_f1": {
            label: float(score) for label, score in zip(CLASS_ORDER, f1, strict=True)
        },
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def save_predictions(
    *,
    ids: Sequence[str],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    model: str,
    seed: int,
    results_dir: str | Path = _DEFAULT_RESULTS_DIR,
) -> Path:
    """Write ``id,label,pred`` rows to the required predictions CSV path."""

    model, seed = _validate_model_and_seed(model, seed)
    true_labels, predicted_labels = _validate_prediction_vectors(y_true, y_pred)
    identifiers = _as_list(ids, name="ids")
    if len(identifiers) != len(true_labels):
        raise ValueError(
            "ids, y_true, and y_pred must have the same length "
            f"({len(identifiers)} != {len(true_labels)})"
        )
    if any(not identifier.strip() for identifier in identifiers):
        raise ValueError("ids must not contain empty strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("ids must be unique")

    output_path = Path(results_dir) / "preds" / f"{model}_seed{seed}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("id", "label", "pred"))
        writer.writerows(zip(identifiers, true_labels, predicted_labels, strict=True))
    temporary_path.replace(output_path)
    return output_path


def save_result_json(
    *,
    model: str,
    seed: int,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    blimp: Mapping[str, Any],
    train_time_sec: Real,
    hardware: str,
    results_dir: str | Path = _DEFAULT_RESULTS_DIR,
) -> Path:
    """Compute metrics and write one required ``<model>_seed<seed>.json`` file."""

    model, seed = _validate_model_and_seed(model, seed)
    metrics = compute_classification_metrics(y_true, y_pred)
    validated_blimp = _validate_blimp_results(blimp)
    train_time = _finite_float(
        train_time_sec,
        name="train_time_sec",
        minimum=0.0,
        maximum=float("inf"),
    )
    if not isinstance(hardware, str) or not hardware.strip():
        raise ValueError("hardware must be a non-empty string")

    payload = {
        "model": model,
        "seed": seed,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "per_class_f1": metrics["per_class_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "blimp": validated_blimp,
        "train_time_sec": train_time,
        "hardware": hardware.strip(),
    }

    output_path = Path(results_dir) / f"{model}_seed{seed}.json"
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    _atomic_write_text(output_path, serialized + "\n")
    return output_path


def bootstrap_macro_f1_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    n_resamples: int = 1_000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Return a percentile bootstrap confidence interval for macro-F1."""

    true_labels, predicted_labels = _validate_prediction_vectors(y_true, y_pred)
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, Integral):
        raise TypeError("n_resamples must be an integer")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be greater than zero")
    confidence = _finite_float(
        confidence,
        name="confidence",
        minimum=0.0,
        maximum=1.0,
    )
    if confidence in (0.0, 1.0):
        raise ValueError("confidence must be strictly between 0 and 1")
    if isinstance(random_state, bool) or not isinstance(random_state, Integral):
        raise TypeError("random_state must be an integer")

    true_array = np.asarray(true_labels, dtype=object)
    predicted_array = np.asarray(predicted_labels, dtype=object)
    rng = np.random.default_rng(int(random_state))
    scores = np.empty(int(n_resamples), dtype=float)

    for index in range(int(n_resamples)):
        sampled_indices = rng.integers(0, len(true_array), size=len(true_array))
        scores[index] = f1_score(
            true_array[sampled_indices],
            predicted_array[sampled_indices],
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        )

    alpha = 1.0 - confidence
    lower, upper = np.quantile(scores, (alpha / 2.0, 1.0 - alpha / 2.0))
    return {"lower": float(lower), "upper": float(upper)}
