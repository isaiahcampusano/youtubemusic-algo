"""Transparent single-user recommendation baselines.

The module uses only the checked-in event fields. It does not call external
services, infer YouTube recommendation impressions, or claim to reproduce the
YouTube Music production system.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DEFAULT_SESSION_GAP_MINUTES = 30
DEFAULT_PENALTY_HOURS = 24.0
DEFAULT_WEIGHTS = {
    "relevance": 0.35,
    "novelty": 0.20,
    "recency": 0.15,
    "repeat_avoidance": 0.20,
    "diversity": 0.10,
}


Event = dict[str, object]


@dataclass(frozen=True)
class ModelState:
    """Statistics fitted strictly from a declared training sequence."""

    catalog: tuple[str, ...]
    play_counts: Counter[str]
    last_position: dict[str, int]
    last_elapsed_seconds: dict[str, float]
    transitions: dict[str, Counter[str]]
    popularity_order: tuple[str, ...]
    item_profiles: dict[str, Counter[str]]
    profile_norms: dict[str, float]


def load_events(
    csv_path: str | Path,
    session_gap_minutes: int = DEFAULT_SESSION_GAP_MINUTES,
) -> list[Event]:
    """Load the public event CSV and reconstruct relative elapsed time."""
    path = Path(csv_path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no events.")

    required = {"video_id", "seconds_since_previous"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    threshold_seconds = session_gap_minutes * 60
    elapsed_seconds = 0.0
    session_id = 0
    events: list[Event] = []
    for position, row in enumerate(rows):
        raw_gap = (row.get("seconds_since_previous") or "").strip()
        gap_seconds = 0.0 if position == 0 and not raw_gap else float(raw_gap or 0)
        if gap_seconds < 0:
            raise ValueError(f"Negative gap at CSV row {position + 2}.")
        if position > 0 and gap_seconds > threshold_seconds:
            session_id += 1
        elapsed_seconds += gap_seconds

        video_id = (row.get("video_id") or "").strip()
        if not video_id:
            raise ValueError(f"Blank video_id at CSV row {position + 2}.")
        events.append(
            {
                "event_index": int(row.get("event_index") or position + 1),
                "video_id": video_id,
                "track_title": row.get("track_title") or "",
                "channel_name": row.get("channel_name") or "",
                "gap_seconds": gap_seconds,
                "elapsed_seconds": elapsed_seconds,
                "session_id": session_id,
            }
        )
    return events


def fit_model(train_events: Sequence[Event]) -> ModelState:
    """Fit counts, transitions, and transition-profile vectors."""
    if not train_events:
        raise ValueError("At least one training event is required.")

    play_counts: Counter[str] = Counter()
    last_position: dict[str, int] = {}
    last_elapsed_seconds: dict[str, float] = {}
    transitions: dict[str, Counter[str]] = defaultdict(Counter)

    for position, event in enumerate(train_events):
        video_id = str(event["video_id"])
        play_counts[video_id] += 1
        last_position[video_id] = position
        last_elapsed_seconds[video_id] = float(event["elapsed_seconds"])
        if position:
            previous = train_events[position - 1]
            if event["session_id"] == previous["session_id"]:
                transitions[str(previous["video_id"])][video_id] += 1

    catalog = tuple(sorted(play_counts))
    popularity_order = tuple(
        sorted(catalog, key=lambda item: (-play_counts[item], item))
    )

    profiles: dict[str, Counter[str]] = {item: Counter() for item in catalog}
    for source, successors in transitions.items():
        for target, count in successors.items():
            profiles[source][f"out:{target}"] += count
            profiles[target][f"in:{source}"] += count
    norms = {
        item: math.sqrt(sum(value * value for value in profile.values()))
        for item, profile in profiles.items()
    }

    return ModelState(
        catalog=catalog,
        play_counts=play_counts,
        last_position=last_position,
        last_elapsed_seconds=last_elapsed_seconds,
        transitions={item: Counter(counts) for item, counts in transitions.items()},
        popularity_order=popularity_order,
        item_profiles=profiles,
        profile_norms=norms,
    )


def _unique_ranked(items: Iterable[str], k: int) -> list[str]:
    if k < 1:
        return []
    seen: set[str] = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
        if len(result) == k:
            break
    return result


def rank_most_played(state: ModelState, k: int = 10) -> list[str]:
    return list(state.popularity_order[:k])


def rank_most_recent(
    history_events: Sequence[Event],
    catalog: Iterable[str],
    k: int = 10,
) -> list[str]:
    allowed = set(catalog)
    recent = (
        str(event["video_id"])
        for event in reversed(history_events)
        if str(event["video_id"]) in allowed
    )
    return _unique_ranked(recent, k)


def rank_cooccurrence(
    state: ModelState,
    last_video_id: str,
    k: int = 10,
) -> list[str]:
    successors = state.transitions.get(last_video_id, Counter())
    ordered = sorted(successors, key=lambda item: (-successors[item], item))
    return _unique_ranked(
        [*ordered, *state.popularity_order],
        min(k, len(state.catalog)),
    )


def _recent_items(
    history_events: Sequence[Event],
    penalty_hours: float,
) -> set[str]:
    if not history_events:
        return set()
    current_time = float(history_events[-1]["elapsed_seconds"])
    current_session = history_events[-1]["session_id"]
    threshold = penalty_hours * 3600
    recent: set[str] = set()
    for event in reversed(history_events):
        age = current_time - float(event["elapsed_seconds"])
        same_session = event["session_id"] == current_session
        if not same_session and age > threshold:
            break
        recent.add(str(event["video_id"]))
    return recent


def rank_cooccurrence_penalty(
    state: ModelState,
    last_video_id: str,
    history_events: Sequence[Event],
    k: int = 10,
    penalty_hours: float = DEFAULT_PENALTY_HOURS,
) -> list[str]:
    successors = state.transitions.get(last_video_id, Counter())
    recent = _recent_items(history_events, penalty_hours)
    penalized_successors = sorted(
        successors,
        key=lambda item: (
            -(successors[item] * (0.05 if item in recent else 1.0)),
            item,
        ),
    )
    return _unique_ranked(
        [*penalized_successors, *state.popularity_order],
        min(k, len(state.catalog)),
    )


def item_similarity(state: ModelState, first: str, second: str) -> float:
    if first == second:
        return 1.0
    first_norm = state.profile_norms.get(first, 0.0)
    second_norm = state.profile_norms.get(second, 0.0)
    if not first_norm or not second_norm:
        return 0.0
    first_profile = state.item_profiles[first]
    second_profile = state.item_profiles[second]
    if len(first_profile) > len(second_profile):
        first_profile, second_profile = second_profile, first_profile
    dot_product = sum(
        value * second_profile.get(key, 0.0)
        for key, value in first_profile.items()
    )
    return dot_product / (first_norm * second_norm)


def list_diversity(state: ModelState, recommendations: Sequence[str]) -> float:
    if len(recommendations) < 2:
        return 0.0
    distances = []
    for left_index, left in enumerate(recommendations):
        for right in recommendations[left_index + 1 :]:
            distances.append(1.0 - item_similarity(state, left, right))
    return sum(distances) / len(distances)


def _validate_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    selected = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if set(selected) != set(DEFAULT_WEIGHTS):
        raise ValueError(f"Weights must contain {sorted(DEFAULT_WEIGHTS)}")
    if any(value < 0 for value in selected.values()):
        raise ValueError("Weights must be non-negative.")
    if not math.isclose(sum(selected.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Weights must sum to 1.0.")
    return selected


def rank_scoring(
    state: ModelState,
    history_events: Sequence[Event],
    k: int = 10,
    weights: Mapping[str, float] | None = None,
    penalty_hours: float = DEFAULT_PENALTY_HOURS,
    recency_half_life_hours: float = 72.0,
    candidate_pool_size: int = 100,
) -> list[str]:
    selected_weights = _validate_weights(weights)
    if not history_events:
        return rank_most_played(state, k)

    last_video_id = str(history_events[-1]["video_id"])
    current_time = float(history_events[-1]["elapsed_seconds"])
    successors = state.transitions.get(last_video_id, Counter())
    max_transition = max(successors.values(), default=0)
    max_popularity = max(state.play_counts.values())
    recent = _recent_items(history_events, penalty_hours)
    last_seen_by_item = {
        str(event["video_id"]): float(event["elapsed_seconds"])
        for event in history_events
    }

    base_scores: dict[str, float] = {}
    for item in state.catalog:
        if max_transition:
            relevance = successors[item] / max_transition
        else:
            relevance = state.play_counts[item] / max_popularity
        novelty = 1.0 / state.play_counts[item]

        last_seen = last_seen_by_item.get(item)
        if last_seen is None:
            recency_affinity = 0.0
        else:
            age_hours = max(0.0, (current_time - last_seen) / 3600)
            recency_affinity = math.exp(
                -math.log(2) * age_hours / recency_half_life_hours
            )
        repeat_avoidance = 0.0 if item in recent else 1.0
        base_scores[item] = (
            selected_weights["relevance"] * relevance
            + selected_weights["novelty"] * novelty
            + selected_weights["recency"] * recency_affinity
            + selected_weights["repeat_avoidance"] * repeat_avoidance
        )

    recommendations: list[str] = []
    pool_limit = max(k, candidate_pool_size)
    base_order = sorted(
        state.catalog,
        key=lambda item: (-base_scores[item], item),
    )
    remaining = set(base_order[:pool_limit])
    target_size = min(k, len(remaining))
    while len(recommendations) < target_size:
        scored = []
        for item in remaining:
            if recommendations:
                diversity_gain = 1.0 - max(
                    item_similarity(state, item, chosen)
                    for chosen in recommendations
                )
            else:
                diversity_gain = 1.0
            score = (
                base_scores[item]
                + selected_weights["diversity"] * diversity_gain
            )
            scored.append((score, item))
        _, chosen = sorted(scored, key=lambda pair: (-pair[0], pair[1]))[0]
        recommendations.append(chosen)
        remaining.remove(chosen)
    return recommendations


def recommend_most_played(train_events: Sequence[Event], k: int = 10) -> list[str]:
    """Return top-k most-played video IDs from training data."""
    return rank_most_played(fit_model(train_events), k)


def recommend_most_recent(train_events: Sequence[Event], k: int = 10) -> list[str]:
    """Return top-k most-recent unique video IDs."""
    state = fit_model(train_events)
    return rank_most_recent(train_events, state.catalog, k)


def recommend_cooccurrence(
    train_events: Sequence[Event],
    last_video_id: str,
    k: int = 10,
    session_gap_minutes: int = DEFAULT_SESSION_GAP_MINUTES,
) -> list[str]:
    """Return top-k trained successors of ``last_video_id``."""
    del session_gap_minutes  # sessions are assigned during event preparation
    return rank_cooccurrence(fit_model(train_events), last_video_id, k)


def recommend_cooccurrence_penalty(
    train_events: Sequence[Event],
    last_video_id: str,
    k: int = 10,
    session_gap_minutes: int = DEFAULT_SESSION_GAP_MINUTES,
    penalty_hours: float = DEFAULT_PENALTY_HOURS,
    history_events: Sequence[Event] | None = None,
) -> list[str]:
    """Return successors with recent and current-session items penalized."""
    del session_gap_minutes
    history = train_events if history_events is None else history_events
    return rank_cooccurrence_penalty(
        fit_model(train_events), last_video_id, history, k, penalty_hours
    )


def recommend_scoring(
    train_events: Sequence[Event],
    context: Mapping[str, object],
    k: int = 10,
    weights: Mapping[str, float] | None = None,
) -> list[str]:
    """Return top-k candidates under the composite transparent objective."""
    history = context.get("history_events", train_events)
    if not isinstance(history, Sequence):
        raise TypeError("context['history_events'] must be a sequence.")
    penalty_hours = float(context.get("penalty_hours", DEFAULT_PENALTY_HOURS))
    recency_half_life = float(context.get("recency_half_life_hours", 72.0))
    candidate_pool_size = int(context.get("candidate_pool_size", 100))
    return rank_scoring(
        fit_model(train_events),
        history,
        k=k,
        weights=weights,
        penalty_hours=penalty_hours,
        recency_half_life_hours=recency_half_life,
        candidate_pool_size=candidate_pool_size,
    )
