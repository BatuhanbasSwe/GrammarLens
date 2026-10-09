"""Validate all final runs and build the GrammarLens benchmark artifacts.

Run this only after the Sunday result freeze, when every required JSON and
prediction CSV is present:

    python src/build_benchmark.py

The command is intentionally all-or-nothing. Missing or inconsistent inputs
raise an error before ``results/benchmark.csv`` or either figure is written.
Every reported scalar comes from a validated result JSON, except confidence
intervals, which are recomputed from the corresponding prediction CSV files.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import f1_score

try:
    from src.metrics import CLASS_ORDER, compute_classification_metrics
except ModuleNotFoundError as error:
    if error.name not in {"src", "src.metrics"}:
        raise
    from metrics import CLASS_ORDER, compute_classification_metrics


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
BENCHMARK_PATH = RESULTS_DIR / "benchmark.csv"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_ORDER = ("nb", "cnn", "distilbert", "deberta-v3")
MODEL_SEEDS: Mapping[str, tuple[int, ...]] = {
    "nb": (42,),
    "cnn": (13, 42, 2026),
    "distilbert": (13, 42, 2026),
    "deberta-v3": (13, 42, 2026),
}
METHOD_TYPES: Mapping[str, str] = {
    "nb": "baseline",
    "cnn": "baseline",
    "distilbert": "baseline",
    "deberta-v3": "new_method",
}
DISPLAY_NAMES: Mapping[str, str] = {
    "nb": "Naive Bayes",
    "cnn": "CNN",
    "distilbert": "DistilBERT",
    "deberta-v3": "DeBERTa-v3",
}
EXPECTED_RESULT_KEYS = (
    "model",
    "seed",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "per_class_f1",
    "confusion_matrix",
    "blimp",
    "train_time_sec",
    "hardware",
)
BENCHMARK_COLUMNS = (
    "model",
    "method_type",
    "n_runs",
    "seeds",
    "macro_f1_mean",
    "macro_f1_std",
    "macro_f1_ci_lower",
    "macro_f1_ci_upper",
    "accuracy_mean",
    "macro_precision_mean",
    "macro_recall_mean",
    "blimp_pair_accuracy_mean",
    "blimp_type_accuracy_mean",
)
SCALAR_METRIC_KEYS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)
BLIMP_GROUPS = ("overall", "SVA", "NOUN_NUM")
EXPECTED_BLIMP_COUNTS = {"overall": 14_000, "SVA": 6_000, "NOUN_NUM": 8_000}
FLOAT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class RunRecord:
    """One validated result JSON and its matching test predictions."""

    model: str
    seed: int
    result: Mapping[str, Any]
    ids: tuple[str, ...]
    labels: tuple[str, ...]
    predictions: tuple[str, ...]


class MissingBenchmarkInputsError(FileNotFoundError):
    """Raised when the complete frozen result set is not yet available."""


def _finite_number(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return numeric


def _require_complete_inputs(results_dir: Path) -> None:
    missing: list[Path] = []
    for model in MODEL_ORDER:
        for seed in MODEL_SEEDS[model]:
            json_path = results_dir / f"{model}_seed{seed}.json"
            predictions_path = results_dir / "preds" / f"{model}_seed{seed}.csv"
            for path in (json_path, predictions_path):
                if not path.is_file():
                    missing.append(path)
    if missing:
        relative_paths = [
            str(path.relative_to(results_dir.parent)).replace("\\", "/")
            for path in missing
        ]
        formatted = "\n  - ".join(relative_paths)
        raise MissingBenchmarkInputsError(
            "Benchmark inputs are incomplete; no outputs were written.\n"
            f"Missing files:\n  - {formatted}"
        )


def _load_test_reference(test_path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not test_path.is_file():
        raise FileNotFoundError(f"Test split not found: {test_path}")
    with test_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"id", "label"}.issubset(reader.fieldnames):
            raise ValueError(f"{test_path} must contain id and label columns")
        rows = list(reader)
    ids = tuple(row["id"] for row in rows)
    labels = tuple(row["label"] for row in rows)
    if not ids or any(not identifier for identifier in ids):
        raise ValueError(f"{test_path} contains no rows or an empty id")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{test_path} contains duplicate ids")
    unknown = sorted(set(labels) - set(CLASS_ORDER))
    if unknown:
        raise ValueError(f"{test_path} contains unknown labels: {unknown}")
    return ids, labels


def _validate_result_payload(payload: Any, *, model: str, seed: int) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"{model}_seed{seed}.json must contain a JSON object")
    if tuple(payload) != EXPECTED_RESULT_KEYS:
        raise ValueError(
            f"{model}_seed{seed}.json has wrong keys/order: {tuple(payload)}"
        )
    if payload["model"] != model or payload["seed"] != seed:
        raise ValueError(
            f"{model}_seed{seed}.json model/seed do not match its filename"
        )

    for key in SCALAR_METRIC_KEYS:
        _finite_number(payload[key], name=f"{model}.{key}", minimum=0.0, maximum=1.0)

    per_class = payload["per_class_f1"]
    if not isinstance(per_class, dict) or tuple(per_class) != CLASS_ORDER:
        raise ValueError(f"{model}.per_class_f1 must use the canonical class order")
    for label in CLASS_ORDER:
        _finite_number(
            per_class[label],
            name=f"{model}.per_class_f1.{label}",
            minimum=0.0,
            maximum=1.0,
        )

    matrix = payload["confusion_matrix"]
    if not isinstance(matrix, list) or len(matrix) != len(CLASS_ORDER):
        raise ValueError(f"{model}.confusion_matrix must be 7x7")
    for row in matrix:
        if not isinstance(row, list) or len(row) != len(CLASS_ORDER):
            raise ValueError(f"{model}.confusion_matrix must be 7x7")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in row):
            raise ValueError(f"{model}.confusion_matrix must contain non-negative integers")

    blimp = payload["blimp"]
    if not isinstance(blimp, dict) or tuple(blimp) != (
        "pair_accuracy",
        "type_accuracy",
        "n_pairs",
    ):
        raise ValueError(f"{model}.blimp has the wrong structure")
    for metric_name in ("pair_accuracy", "type_accuracy"):
        metric = blimp[metric_name]
        if not isinstance(metric, dict) or tuple(metric) != BLIMP_GROUPS:
            raise ValueError(f"{model}.blimp.{metric_name} has wrong groups/order")
        for group in BLIMP_GROUPS:
            _finite_number(
                metric[group],
                name=f"{model}.blimp.{metric_name}.{group}",
                minimum=0.0,
                maximum=1.0,
            )
    if blimp["n_pairs"] != EXPECTED_BLIMP_COUNTS:
        raise ValueError(f"{model}.blimp.n_pairs has unexpected counts")

    _finite_number(payload["train_time_sec"], name=f"{model}.train_time_sec", minimum=0.0)
    if not isinstance(payload["hardware"], str) or not payload["hardware"].strip():
        raise ValueError(f"{model}.hardware must be a non-empty string")


def _load_predictions(
    path: Path,
    *,
    expected_ids: Sequence[str],
    expected_labels: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "label", "pred"]:
            raise ValueError(f"{path} must have exactly the header id,label,pred")
        rows = list(reader)

    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        identifier = row["id"]
        if identifier in by_id:
            raise ValueError(f"{path} contains duplicate id: {identifier}")
        by_id[identifier] = row
    expected_id_set = set(expected_ids)
    missing_ids = expected_id_set - set(by_id)
    extra_ids = set(by_id) - expected_id_set
    if missing_ids or extra_ids:
        raise ValueError(
            f"{path} ids do not match data/processed/test.csv "
            f"(missing={len(missing_ids)}, extra={len(extra_ids)})"
        )

    ordered_rows = [by_id[identifier] for identifier in expected_ids]
    ids = tuple(row["id"] for row in ordered_rows)
    labels = tuple(row["label"] for row in ordered_rows)
    predictions = tuple(row["pred"] for row in ordered_rows)
    if labels != tuple(expected_labels):
        raise ValueError(f"{path} labels do not match data/processed/test.csv")
    unknown = sorted((set(labels) | set(predictions)) - set(CLASS_ORDER))
    if unknown:
        raise ValueError(f"{path} contains unknown labels: {unknown}")
    return ids, labels, predictions


def _assert_metrics_match(
    payload: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    *,
    source_name: str,
) -> None:
    for key in SCALAR_METRIC_KEYS:
        if not math.isclose(
            float(payload[key]),
            float(recomputed[key]),
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ):
            raise ValueError(f"{source_name}: {key} does not match predictions")
    for label in CLASS_ORDER:
        if not math.isclose(
            float(payload["per_class_f1"][label]),
            float(recomputed["per_class_f1"][label]),
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ):
            raise ValueError(
                f"{source_name}: per_class_f1.{label} does not match predictions"
            )
    if payload["confusion_matrix"] != recomputed["confusion_matrix"]:
        raise ValueError(f"{source_name}: confusion_matrix does not match predictions")


def load_run(
    model: str,
    seed: int,
    *,
    results_dir: str | Path = RESULTS_DIR,
    expected_ids: Sequence[str],
    expected_labels: Sequence[str],
) -> RunRecord:
    """Load and cross-check one result JSON plus prediction CSV."""

    results_path = Path(results_dir)
    json_path = results_path / f"{model}_seed{seed}.json"
    predictions_path = results_path / "preds" / f"{model}_seed{seed}.csv"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    _validate_result_payload(payload, model=model, seed=seed)
    ids, labels, predictions = _load_predictions(
        predictions_path,
        expected_ids=expected_ids,
        expected_labels=expected_labels,
    )
    recomputed = compute_classification_metrics(labels, predictions)
    _assert_metrics_match(payload, recomputed, source_name=json_path.name)
    return RunRecord(model, seed, payload, ids, labels, predictions)


def bootstrap_mean_macro_f1_ci(
    runs: Sequence[RunRecord],
    *,
    n_resamples: int = 1_000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Bootstrap the mean Macro-F1 across aligned seed predictions."""

    if not runs:
        raise ValueError("runs must contain at least one result")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    labels = np.asarray(runs[0].labels, dtype=object)
    if any(run.labels != runs[0].labels or run.ids != runs[0].ids for run in runs[1:]):
        raise ValueError("All runs must use the same ordered test examples")
    predictions = [np.asarray(run.predictions, dtype=object) for run in runs]
    rng = np.random.default_rng(random_state)
    scores = np.empty(n_resamples, dtype=float)

    for index in range(n_resamples):
        sampled = rng.integers(0, len(labels), size=len(labels))
        seed_scores = [
            f1_score(
                labels[sampled],
                prediction[sampled],
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            )
            for prediction in predictions
        ]
        scores[index] = float(np.mean(seed_scores))

    alpha = 1.0 - confidence
    lower, upper = np.quantile(scores, (alpha / 2.0, 1.0 - alpha / 2.0))
    return float(lower), float(upper)


def _mean(runs: Sequence[RunRecord], dotted_key: str) -> float:
    values: list[float] = []
    for run in runs:
        current: Any = run.result
        for key in dotted_key.split("."):
            current = current[key]
        values.append(float(current))
    return statistics.fmean(values)


def _summarize_model(
    model: str,
    runs: Sequence[RunRecord],
    *,
    n_resamples: int,
    random_state: int,
) -> dict[str, Any]:
    macro_f1_values = [float(run.result["macro_f1"]) for run in runs]
    ci_lower, ci_upper = bootstrap_mean_macro_f1_ci(
        runs,
        n_resamples=n_resamples,
        confidence=0.95,
        random_state=random_state,
    )
    return {
        "model": model,
        "method_type": METHOD_TYPES[model],
        "n_runs": len(runs),
        "seeds": ";".join(str(run.seed) for run in runs),
        "macro_f1_mean": statistics.fmean(macro_f1_values),
        "macro_f1_std": statistics.pstdev(macro_f1_values),
        "macro_f1_ci_lower": ci_lower,
        "macro_f1_ci_upper": ci_upper,
        "accuracy_mean": _mean(runs, "accuracy"),
        "macro_precision_mean": _mean(runs, "macro_precision"),
        "macro_recall_mean": _mean(runs, "macro_recall"),
        "blimp_pair_accuracy_mean": _mean(runs, "blimp.pair_accuracy.overall"),
        "blimp_type_accuracy_mean": _mean(runs, "blimp.type_accuracy.overall"),
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(text)
        temporary_path = Path(stream.name)
    temporary_path.replace(path)


def _write_benchmark_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=BENCHMARK_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{row[key]:.12f}" if isinstance(row[key], float) else row[key]
                    for key in BENCHMARK_COLUMNS
                }
            )
    temporary_path.replace(path)


def _svg_bar_chart(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    title: str,
    subtitle: str,
    ci_keys: tuple[str, str] | None = None,
) -> str:
    width, height = 900, 560
    left, right, top, bottom = 105, 40, 90, 100
    plot_width = width - left - right
    plot_height = height - top - bottom
    colors = ("#6B7280", "#3B82F6", "#8B5CF6", "#10B981")
    bar_slot = plot_width / len(rows)
    bar_width = min(105.0, bar_slot * 0.58)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.grid{stroke:#E5E7EB;stroke-width:1}.axis{stroke:#374151;stroke-width:1.5}.bar{rx:5}.error{stroke:#111827;stroke-width:2}</style>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="23" font-weight="700">{html.escape(title)}</text>',
        f'<text x="{width / 2}" y="60" text-anchor="middle" font-size="13" fill="#4B5563">{html.escape(subtitle)}</text>',
    ]
    for tick in range(0, 11, 2):
        value = tick / 10
        y = top + plot_height * (1.0 - value)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 5:.2f}" text-anchor="end" font-size="12">{value:.1f}</text>')
    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
        ]
    )

    for index, row in enumerate(rows):
        value = float(row[value_key])
        center = left + bar_slot * (index + 0.5)
        x = center - bar_width / 2
        y = top + plot_height * (1.0 - value)
        bar_height = plot_height * value
        label_y = y - 10
        parts.append(
            f'<rect class="bar" x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{colors[index]}"/>'
        )

        if ci_keys is not None:
            lower = float(row[ci_keys[0]])
            upper = float(row[ci_keys[1]])
            y_lower = top + plot_height * (1.0 - lower)
            y_upper = top + plot_height * (1.0 - upper)
            label_y = y_upper - 10
            parts.extend(
                [
                    f'<line class="error" x1="{center:.2f}" y1="{y_upper:.2f}" x2="{center:.2f}" y2="{y_lower:.2f}"/>',
                    f'<line class="error" x1="{center - 9:.2f}" y1="{y_upper:.2f}" x2="{center + 9:.2f}" y2="{y_upper:.2f}"/>',
                    f'<line class="error" x1="{center - 9:.2f}" y1="{y_lower:.2f}" x2="{center + 9:.2f}" y2="{y_lower:.2f}"/>',
                ]
            )
        parts.append(f'<text x="{center:.2f}" y="{label_y:.2f}" text-anchor="middle" font-size="13" font-weight="700">{value:.3f}</text>')
        parts.append(f'<text x="{center:.2f}" y="{top + plot_height + 30}" text-anchor="middle" font-size="14">{html.escape(DISPLAY_NAMES[str(row["model"])])}</text>')

    parts.append(f'<text x="25" y="{top + plot_height / 2}" transform="rotate(-90 25 {top + plot_height / 2})" text-anchor="middle" font-size="14">Accuracy / score</text>')
    parts.append("</svg>\n")
    return "".join(parts)


def write_figures(rows: Sequence[Mapping[str, Any]], figures_dir: str | Path) -> tuple[Path, Path]:
    """Write the two required dependency-free SVG benchmark charts."""

    output_dir = Path(figures_dir)
    macro_path = output_dir / "macro_f1.svg"
    blimp_path = output_dir / "blimp_pair_accuracy.svg"
    macro_svg = _svg_bar_chart(
        rows,
        value_key="macro_f1_mean",
        title="GrammarLens Test Macro-F1",
        subtitle="Bars: mean across required seeds; whiskers: 95% bootstrap CI",
        ci_keys=("macro_f1_ci_lower", "macro_f1_ci_upper"),
    )
    blimp_svg = _svg_bar_chart(
        rows,
        value_key="blimp_pair_accuracy_mean",
        title="GrammarLens BLiMP Pair Accuracy",
        subtitle="Mean across required seeds; strict P(CORRECT|good) > P(CORRECT|bad)",
    )
    _atomic_write(macro_path, macro_svg)
    _atomic_write(blimp_path, blimp_svg)
    return macro_path, blimp_path


def build_benchmark(
    *,
    results_dir: str | Path = RESULTS_DIR,
    test_path: str | Path = TEST_PATH,
    benchmark_path: str | Path = BENCHMARK_PATH,
    figures_dir: str | Path = FIGURES_DIR,
    n_resamples: int = 1_000,
    random_state: int = 42,
) -> tuple[Path, tuple[Path, Path]]:
    """Validate all frozen inputs, then atomically write table and figures."""

    results_path = Path(results_dir)
    _require_complete_inputs(results_path)
    expected_ids, expected_labels = _load_test_reference(Path(test_path))
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        runs = [
            load_run(
                model,
                seed,
                results_dir=results_path,
                expected_ids=expected_ids,
                expected_labels=expected_labels,
            )
            for seed in MODEL_SEEDS[model]
        ]
        rows.append(
            _summarize_model(
                model,
                runs,
                n_resamples=n_resamples,
                random_state=random_state,
            )
        )

    benchmark_output = Path(benchmark_path)
    figure_paths = write_figures(rows, figures_dir)
    _write_benchmark_csv(rows, benchmark_output)
    return benchmark_output, figure_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate frozen results and build benchmark.csv plus two SVG figures."
    )
    parser.add_argument("--n-resamples", type=int, default=1_000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        benchmark_path, figure_paths = build_benchmark(
            n_resamples=args.n_resamples,
            random_state=args.random_state,
        )
    except MissingBenchmarkInputsError as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from None
    print(f"benchmark={benchmark_path}")
    for figure_path in figure_paths:
        print(f"figure={figure_path}")


if __name__ == "__main__":
    main()
