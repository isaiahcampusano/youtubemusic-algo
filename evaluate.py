"""Leakage-safe offline evaluation for the transparent recommenders."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from recommend import (
    DEFAULT_WEIGHTS,
    Event,
    ModelState,
    fit_model,
    item_similarity,
    list_diversity,
    load_events,
    rank_cooccurrence,
    rank_cooccurrence_penalty,
    rank_most_played,
    rank_most_recent,
    rank_scoring,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "public" / "listening_events.csv"
DEFAULT_RESULTS = ROOT / "evaluation_results.json"
DEFAULT_REPORT = ROOT / "EVALUATION_REPORT.md"
K_VALUES = (5, 10, 20)
MAX_K = max(K_VALUES)

Algorithm = Callable[[Sequence[Event], int], list[str]]


WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "balanced": dict(DEFAULT_WEIGHTS),
    "relevance_heavy": {
        "relevance": 0.55,
        "novelty": 0.10,
        "recency": 0.15,
        "repeat_avoidance": 0.10,
        "diversity": 0.10,
    },
    "novelty_heavy": {
        "relevance": 0.25,
        "novelty": 0.35,
        "recency": 0.10,
        "repeat_avoidance": 0.20,
        "diversity": 0.10,
    },
    "repeat_avoidance_heavy": {
        "relevance": 0.30,
        "novelty": 0.15,
        "recency": 0.10,
        "repeat_avoidance": 0.35,
        "diversity": 0.10,
    },
}
PENALTY_HOURS_GRID = (6.0, 24.0, 72.0)


def temporal_split(
    events: Sequence[Event],
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
) -> tuple[list[Event], list[Event], list[Event]]:
    """Return a chronological 80/10/10 split with no shuffled events."""
    total = len(events)
    train_end = round(total * train_fraction)
    validation_end = train_end + int(total * validation_fraction)
    if train_end < 1 or validation_end >= total:
        raise ValueError("The event sequence is too short for this temporal split.")
    return (
        list(events[:train_end]),
        list(events[train_end:validation_end]),
        list(events[validation_end:]),
    )


def _inverse_popularity_novelty(state: ModelState, items: Sequence[str]) -> float:
    if not items:
        return 0.0
    total_plays = sum(state.play_counts.values())
    denominator = math.log2(total_plays) if total_plays > 1 else 1.0
    values = [
        -math.log2(state.play_counts[item] / total_plays) / denominator
        for item in items
    ]
    return sum(values) / len(values)


def evaluate_algorithm(
    algorithm_fn: Algorithm,
    train_events: Sequence[Event],
    test_events: Sequence[Event],
    k_values: Sequence[int] = K_VALUES,
    *,
    model_state: ModelState | None = None,
    initial_history: Sequence[Event] | None = None,
) -> dict[str, object]:
    """Evaluate one ranker sequentially while keeping fitted state fixed."""
    state = fit_model(train_events) if model_state is None else model_state
    history = list(train_events if initial_history is None else initial_history)
    maximum_k = max(k_values)
    hits = {k: 0 for k in k_values}
    recommended_slots: list[str] = []
    diversity_values: list[float] = []
    list_sizes: list[int] = []
    cold_start_targets = 0
    duplicate_lists = 0
    outside_catalog = 0

    for target in test_events:
        recommendations = algorithm_fn(history, maximum_k)
        if len(recommendations) != len(set(recommendations)):
            duplicate_lists += 1
        outside_catalog += sum(item not in state.play_counts for item in recommendations)
        target_item = str(target["video_id"])
        if target_item not in state.play_counts:
            cold_start_targets += 1
        for k in k_values:
            hits[k] += int(target_item in recommendations[:k])
        recommended_slots.extend(recommendations)
        diversity_values.append(list_diversity(state, recommendations))
        list_sizes.append(len(recommendations))
        history.append(target)

    target_count = len(test_events)
    unique_recommendations = set(recommended_slots)
    slot_count = len(recommended_slots)
    return {
        **{
            f"hit_rate_at_{k}": round(hits[k] / target_count, 6)
            for k in k_values
        },
        "catalog_coverage": round(
            len(unique_recommendations) / len(state.catalog), 6
        ),
        "recommendation_slot_repetition_rate": round(
            1.0 - len(unique_recommendations) / slot_count, 6
        ) if slot_count else 0.0,
        "recommendations_per_unique_item": round(
            slot_count / len(unique_recommendations), 6
        ) if unique_recommendations else 0.0,
        "inverse_popularity_novelty": round(
            _inverse_popularity_novelty(state, recommended_slots), 6
        ),
        "mean_list_diversity": round(
            sum(diversity_values) / len(diversity_values), 6
        ) if diversity_values else 0.0,
        "cold_start_target_count": cold_start_targets,
        "cold_start_target_rate": round(cold_start_targets / target_count, 6),
        "mean_returned_list_size": round(sum(list_sizes) / len(list_sizes), 6),
        "evaluated_target_count": target_count,
        "recommendation_slot_count": slot_count,
        "unique_recommended_item_count": len(unique_recommendations),
        "duplicate_list_count": duplicate_lists,
        "outside_training_catalog_count": outside_catalog,
    }


def _algorithm_functions(
    state: ModelState,
    *,
    composite_weights: Mapping[str, float],
    penalty_hours: float,
) -> dict[str, Algorithm]:
    def most_played(history: Sequence[Event], k: int) -> list[str]:
        del history
        return rank_most_played(state, k)

    def most_recent(history: Sequence[Event], k: int) -> list[str]:
        return rank_most_recent(history, state.catalog, k)

    def cooccurrence(history: Sequence[Event], k: int) -> list[str]:
        return rank_cooccurrence(state, str(history[-1]["video_id"]), k)

    def cooccurrence_penalty(history: Sequence[Event], k: int) -> list[str]:
        return rank_cooccurrence_penalty(
            state,
            str(history[-1]["video_id"]),
            history,
            k,
            penalty_hours,
        )

    def proposed_scoring(history: Sequence[Event], k: int) -> list[str]:
        return rank_scoring(
            state,
            history,
            k,
            composite_weights,
            penalty_hours,
            candidate_pool_size=100,
        )

    return {
        "most_played": most_played,
        "most_recent": most_recent,
        "cooccurrence": cooccurrence,
        "cooccurrence_penalty": cooccurrence_penalty,
        "proposed_scoring": proposed_scoring,
    }


def select_composite_configuration(
    train_events: Sequence[Event],
    validation_events: Sequence[Event],
    state: ModelState,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Select by validation Hit@10, then novelty, coverage, and diversity."""
    trials: list[dict[str, object]] = []
    for preset_name, weights in WEIGHT_PRESETS.items():
        for penalty_hours in PENALTY_HOURS_GRID:
            ranker = _algorithm_functions(
                state,
                composite_weights=weights,
                penalty_hours=penalty_hours,
            )["proposed_scoring"]
            metrics = evaluate_algorithm(
                ranker,
                train_events,
                validation_events,
                model_state=state,
                initial_history=train_events,
            )
            trials.append(
                {
                    "preset": preset_name,
                    "weights": weights,
                    "penalty_hours": penalty_hours,
                    "metrics": metrics,
                }
            )
    selected = max(
        trials,
        key=lambda trial: (
            trial["metrics"]["hit_rate_at_10"],
            trial["metrics"]["inverse_popularity_novelty"],
            trial["metrics"]["catalog_coverage"],
            trial["metrics"]["mean_list_diversity"],
            trial["preset"],
            -trial["penalty_hours"],
        ),
    )
    return selected, trials


def _percentage(value: float) -> str:
    return f"{100 * value:.2f}%"


def build_report(payload: Mapping[str, object]) -> str:
    results = payload["algorithms"]
    selected = payload["validation"]["selected_composite_configuration"]
    names = {
        "most_played": "Most played",
        "most_recent": "Most recent",
        "cooccurrence": "Item transition",
        "cooccurrence_penalty": "Transition + repeat penalty",
        "proposed_scoring": "Composite scorer",
    }
    rows = []
    for key, label in names.items():
        metrics = results[key]
        rows.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                label,
                _percentage(metrics["hit_rate_at_5"]),
                _percentage(metrics["hit_rate_at_10"]),
                _percentage(metrics["hit_rate_at_20"]),
                _percentage(metrics["catalog_coverage"]),
                f'{metrics["inverse_popularity_novelty"]:.4f}',
                f'{metrics["mean_list_diversity"]:.4f}',
            )
        )

    best_accuracy_key = max(
        results, key=lambda key: results[key]["hit_rate_at_10"]
    )
    best_coverage_key = max(
        results, key=lambda key: results[key]["catalog_coverage"]
    )
    composite = results["proposed_scoring"]
    best_baseline_hit = max(
        metrics["hit_rate_at_10"]
        for key, metrics in results.items()
        if key != "proposed_scoring"
    )
    accuracy_delta = composite["hit_rate_at_10"] - best_baseline_hit
    conclusion = (
        "The composite scorer beat the strongest baseline on Hit Rate@10."
        if accuracy_delta > 0
        else "The composite scorer did not beat the strongest baseline on Hit Rate@10."
    )

    split = payload["split"]
    integrity = payload["integrity"]
    return "\n".join(
        [
            "# Offline Recommendation Evaluation",
            "",
            "## Executive summary",
            "",
            f"The strongest test Hit Rate@10 came from **{names[best_accuracy_key]}**; "
            f"the widest catalog coverage came from **{names[best_coverage_key]}**. "
            f"{conclusion} Its difference from the strongest baseline was "
            f"{accuracy_delta:+.4f}. These are offline next-item results, not evidence "
            "about YouTube Music's internal ranking objective.",
            "",
            "## Setup",
            "",
            f"The {split['total_events']:,} ordered events were split into "
            f"{split['train_events']:,} training, {split['validation_events']:,} "
            f"validation, and {split['test_events']:,} test events. Fitted popularity "
            "and transition statistics used training only. Validation selected the "
            f"**{selected['preset']}** weight preset and a "
            f"{selected['penalty_hours']:g}-hour repeat window; the locked test segment "
            "was used once for the table below.",
            "",
            "## Locked test results",
            "",
            "| Algorithm | Hit@5 | Hit@10 | Hit@20 | Coverage | Novelty | Diversity |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "Hit Rate is equivalent to Recall for this single-next-item task. Coverage "
            "is the fraction of the training catalog recommended at least once. Novelty "
            "is normalized self-information under training popularity. Diversity is "
            "mean pairwise cosine distance between training-only transition profiles.",
            "",
            "## Findings",
            "",
            f"- **Accuracy:** {names[best_accuracy_key]} achieved the highest Hit@10 "
            f"({_percentage(results[best_accuracy_key]['hit_rate_at_10'])}).",
            f"- **Catalog reach:** {names[best_coverage_key]} reached "
            f"{_percentage(results[best_coverage_key]['catalog_coverage'])} of the "
            "training catalog.",
            f"- **Cold start:** {composite['cold_start_target_count']} of "
            f"{composite['evaluated_target_count']} test targets "
            f"({_percentage(composite['cold_start_target_rate'])}) were absent from "
            "the training catalog and could not be hit by any evaluated method.",
            f"- **Integrity:** duplicate recommendation lists = "
            f"{integrity['duplicate_list_count_total']}; recommendations outside the "
            f"training catalog = {integrity['outside_training_catalog_count_total']}.",
            "",
            "## Limitations",
            "",
            "This single-user Takeout history records consumption, not recommendation "
            "impressions. It has no rejected candidates, ranks, skips, completion, "
            "listening duration, or counterfactual feedback. Autoplay and deliberate "
            "selection cannot be separated. Video IDs are uploads rather than canonical "
            "songs, and transition-profile distance is only a behavioral proxy for "
            "musical diversity. The small test set makes fine-grained differences noisy.",
            "",
            "## MVP conclusion",
            "",
            "The repository now supports a reproducible, leakage-aware comparison of "
            "five transparent recommenders. It can reveal tradeoffs among next-item "
            "accuracy, novelty, coverage, repetition, and diversity for this history. "
            "It cannot recover YouTube Music's algorithm or loss function. The next "
            "stronger experiment would collect recommendation impressions and explicit "
            "outcomes, then evaluate a pre-registered objective on future data.",
            "",
        ]
    )


def run(input_path: Path, results_path: Path, report_path: Path) -> dict[str, object]:
    events = load_events(input_path)
    train_events, validation_events, test_events = temporal_split(events)
    state = fit_model(train_events)
    selected, trials = select_composite_configuration(
        train_events, validation_events, state
    )
    functions = _algorithm_functions(
        state,
        composite_weights=selected["weights"],
        penalty_hours=selected["penalty_hours"],
    )
    test_history = [*train_events, *validation_events]
    results = {
        name: evaluate_algorithm(
            ranker,
            train_events,
            test_events,
            model_state=state,
            initial_history=test_history,
        )
        for name, ranker in functions.items()
    }
    duplicate_total = sum(
        metrics["duplicate_list_count"] for metrics in results.values()
    )
    outside_total = sum(
        metrics["outside_training_catalog_count"] for metrics in results.values()
    )
    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(input_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        },
        "split": {
            "method": "chronological_80_10_10",
            "total_events": len(events),
            "train_events": len(train_events),
            "validation_events": len(validation_events),
            "test_events": len(test_events),
            "training_catalog_size": len(state.catalog),
        },
        "evaluation": {
            "k_values": list(K_VALUES),
            "candidate_set": "training_catalog",
            "session_gap_minutes": 30,
            "fitted_statistics": "training_only",
            "test_context_updates": "prior_events_only; fitted counts remain frozen",
        },
        "validation": {
            "selection_rule": (
                "maximize Hit@10, then novelty, coverage, diversity; deterministic ties"
            ),
            "selected_composite_configuration": {
                "preset": selected["preset"],
                "weights": selected["weights"],
                "penalty_hours": selected["penalty_hours"],
                "candidate_pool_size": 100,
            },
            "trials": trials,
        },
        "algorithms": results,
        "integrity": {
            "split_sizes_reconcile": (
                len(train_events) + len(validation_events) + len(test_events)
                == len(events)
            ),
            "duplicate_list_count_total": duplicate_total,
            "outside_training_catalog_count_total": outside_total,
            "all_lists_unique_and_in_catalog": duplicate_total == 0 and outside_total == 0,
        },
    }
    results_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(build_report(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run(args.input.resolve(), args.results.resolve(), args.report.resolve())
    print("Offline recommendation evaluation complete")
    print(
        "Split: {train_events} train / {validation_events} validation / "
        "{test_events} test".format(**payload["split"])
    )
    selected = payload["validation"]["selected_composite_configuration"]
    print(
        f"Selected composite: {selected['preset']} "
        f"({selected['penalty_hours']:g}h penalty)"
    )
    for name, metrics in payload["algorithms"].items():
        print(
            f"{name:24} Hit@10={metrics['hit_rate_at_10']:.4f}  "
            f"Coverage={metrics['catalog_coverage']:.4f}  "
            f"Novelty={metrics['inverse_popularity_novelty']:.4f}"
        )
    print(f"Wrote {args.results.resolve()}")
    print(f"Wrote {args.report.resolve()}")


if __name__ == "__main__":
    main()
