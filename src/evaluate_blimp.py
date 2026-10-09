"""Evaluate GrammarLens classifiers on the pinned 14-file BLiMP subset.

This project uses a classifier-specific adaptation of BLiMP:

* Pair accuracy: ``P(CORRECT | sentence_good) > P(CORRECT | sentence_bad)``.
* Type accuracy: the predicted class for ``sentence_bad`` is the class mapped
  to that BLiMP paradigm (``SVA`` or ``NOUN_NUM``).

The callback must accept a sequence of sentences and return an ``(N, 7)``
array of class probabilities. Its column order is supplied separately so this
module can safely reorder outputs into ``CLASS_ORDER``. For example::

    blimp = evaluate_blimp(
        lambda sentences: predict_proba(model, sentences),
        probability_class_order=model.classes_,
    )

Ties in ``CORRECT`` probability are failures because pair accuracy requires the
good sentence to receive a strictly higher probability.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from src.metrics import CLASS_ORDER
except ModuleNotFoundError as exc:
    if exc.name != "src":
        raise
    from metrics import CLASS_ORDER


_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BLIMP_DIR = _ROOT / "data" / "raw" / "blimp"
_EXPECTED_PAIRS_PER_UID = 1_000
_RESULT_GROUP_ORDER: tuple[str, ...] = ("overall", "SVA", "NOUN_NUM")

BLIMP_UID_TO_LABEL: dict[str, str] = {
    "determiner_noun_agreement_1": "NOUN_NUM",
    "determiner_noun_agreement_2": "NOUN_NUM",
    "determiner_noun_agreement_irregular_1": "NOUN_NUM",
    "determiner_noun_agreement_irregular_2": "NOUN_NUM",
    "determiner_noun_agreement_with_adjective_1": "NOUN_NUM",
    "determiner_noun_agreement_with_adj_2": "NOUN_NUM",
    "determiner_noun_agreement_with_adj_irregular_1": "NOUN_NUM",
    "determiner_noun_agreement_with_adj_irregular_2": "NOUN_NUM",
    "regular_plural_subject_verb_agreement_1": "SVA",
    "regular_plural_subject_verb_agreement_2": "SVA",
    "irregular_plural_subject_verb_agreement_1": "SVA",
    "irregular_plural_subject_verb_agreement_2": "SVA",
    "distractor_agreement_relational_noun": "SVA",
    "distractor_agreement_relative_clause": "SVA",
}

__all__ = [
    "BLIMP_UID_TO_LABEL",
    "BlimpPair",
    "evaluate_blimp",
    "load_blimp_pairs",
]


@dataclass(frozen=True, slots=True)
class BlimpPair:
    """One grammatical/ungrammatical BLiMP minimal pair."""

    uid: str
    pair_id: str
    target_label: str
    sentence_good: str
    sentence_bad: str


def _required_non_empty_string(
    row: dict[str, Any], key: str, *, path: Path, line_number: int
) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{path}:{line_number}: {key} must be a non-empty string"
        )
    return value


def _normalize_pair_id(
    row: dict[str, Any], *, path: Path, line_number: int
) -> str:
    value = row.get("pairID")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(
            f"{path}:{line_number}: pairID must be a string or integer"
        )
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{path}:{line_number}: pairID must not be empty")
    return normalized


def _load_uid_file(path: Path, uid: str, target_label: str) -> list[BlimpPair]:
    if not path.is_file():
        raise FileNotFoundError(f"required BLiMP file is missing: {path}")

    pairs: list[BlimpPair] = []
    seen_pair_ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL line")
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSON value must be an object")

            row_uid = _required_non_empty_string(
                row, "UID", path=path, line_number=line_number
            )
            if row_uid != uid:
                raise ValueError(
                    f"{path}:{line_number}: UID {row_uid!r} does not match {uid!r}"
                )
            pair_id = _normalize_pair_id(row, path=path, line_number=line_number)
            if pair_id in seen_pair_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate pairID {pair_id!r}"
                )
            seen_pair_ids.add(pair_id)

            sentence_good = _required_non_empty_string(
                row, "sentence_good", path=path, line_number=line_number
            )
            sentence_bad = _required_non_empty_string(
                row, "sentence_bad", path=path, line_number=line_number
            )
            if sentence_good == sentence_bad:
                raise ValueError(
                    f"{path}:{line_number}: sentence_good and sentence_bad are identical"
                )

            pairs.append(
                BlimpPair(
                    uid=uid,
                    pair_id=pair_id,
                    target_label=target_label,
                    sentence_good=sentence_good,
                    sentence_bad=sentence_bad,
                )
            )

    if len(pairs) != _EXPECTED_PAIRS_PER_UID:
        raise ValueError(
            f"{path}: expected {_EXPECTED_PAIRS_PER_UID} pairs, found {len(pairs)}"
        )
    return pairs


def load_blimp_pairs(
    blimp_dir: str | Path = _DEFAULT_BLIMP_DIR,
) -> list[BlimpPair]:
    """Load and validate the exact 14 BLiMP paradigms used by GrammarLens."""

    directory = Path(blimp_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"BLiMP directory does not exist: {directory}")

    pairs: list[BlimpPair] = []
    for uid, target_label in BLIMP_UID_TO_LABEL.items():
        pairs.extend(
            _load_uid_file(directory / f"{uid}.jsonl", uid, target_label)
        )
    return pairs


def _validate_probability_class_order(
    probability_class_order: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(probability_class_order, (str, bytes)):
        raise TypeError("probability_class_order must be a sequence of labels")
    try:
        order = tuple(probability_class_order)
    except TypeError as exc:
        raise TypeError("probability_class_order must be iterable") from exc
    if not all(isinstance(label, str) for label in order):
        raise TypeError("every probability_class_order value must be a string")
    if len(order) != len(CLASS_ORDER) or set(order) != set(CLASS_ORDER):
        raise ValueError(
            "probability_class_order must contain every GrammarLens label exactly once"
        )
    return order


def _validated_probabilities(
    raw_probabilities: Any,
    *,
    expected_rows: int,
    batch_start: int,
) -> np.ndarray:
    try:
        probabilities = np.asarray(raw_probabilities, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"predict_proba returned non-numeric values for batch starting {batch_start}"
        ) from exc

    expected_shape = (expected_rows, len(CLASS_ORDER))
    if probabilities.shape != expected_shape:
        raise ValueError(
            "predict_proba returned shape "
            f"{probabilities.shape}, expected {expected_shape} "
            f"for batch starting {batch_start}"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            f"predict_proba returned non-finite values for batch starting {batch_start}"
        )
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError(
            f"predict_proba returned values outside [0, 1] for batch starting {batch_start}"
        )
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(
            "predict_proba rows must sum to 1 for batch starting "
            f"{batch_start}; received range [{row_sums.min()}, {row_sums.max()}]"
        )
    return probabilities


def evaluate_blimp(
    predict_proba: Callable[[Sequence[str]], Any],
    *,
    probability_class_order: Sequence[str],
    blimp_dir: str | Path = _DEFAULT_BLIMP_DIR,
    batch_size: int = 256,
) -> dict[str, dict[str, float | int]]:
    """Return pair/type accuracy overall and for ``SVA``/``NOUN_NUM``."""

    if not callable(predict_proba):
        raise TypeError("predict_proba must be callable")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    source_order = _validate_probability_class_order(probability_class_order)
    canonical_column_indices = [source_order.index(label) for label in CLASS_ORDER]
    correct_index = CLASS_ORDER.index("CORRECT")
    pairs = load_blimp_pairs(blimp_dir)

    counts = {group: 0 for group in _RESULT_GROUP_ORDER}
    pair_correct = {group: 0 for group in _RESULT_GROUP_ORDER}
    type_correct = {group: 0 for group in _RESULT_GROUP_ORDER}

    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start : batch_start + batch_size]
        sentences = [pair.sentence_good for pair in batch]
        sentences.extend(pair.sentence_bad for pair in batch)
        raw_probabilities = predict_proba(sentences)
        probabilities = _validated_probabilities(
            raw_probabilities,
            expected_rows=len(sentences),
            batch_start=batch_start,
        )
        probabilities = probabilities[:, canonical_column_indices]

        split_index = len(batch)
        good_probabilities = probabilities[:split_index]
        bad_probabilities = probabilities[split_index:]
        pair_outcomes = (
            good_probabilities[:, correct_index]
            > bad_probabilities[:, correct_index]
        )
        bad_predictions = np.asarray(CLASS_ORDER, dtype=object)[
            np.argmax(bad_probabilities, axis=1)
        ]

        for pair, is_pair_correct, predicted_label in zip(
            batch, pair_outcomes, bad_predictions, strict=True
        ):
            is_type_correct = predicted_label == pair.target_label
            for group in ("overall", pair.target_label):
                counts[group] += 1
                pair_correct[group] += int(is_pair_correct)
                type_correct[group] += int(is_type_correct)

    return {
        "pair_accuracy": {
            group: pair_correct[group] / counts[group]
            for group in _RESULT_GROUP_ORDER
        },
        "type_accuracy": {
            group: type_correct[group] / counts[group]
            for group in _RESULT_GROUP_ORDER
        },
        "n_pairs": {group: counts[group] for group in _RESULT_GROUP_ORDER},
    }
