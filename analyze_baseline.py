"""Calculate descriptive listening baselines from the existing public CSV.

No external services or enrichment data are used. The public event file omits
absolute timestamps, so elapsed time is reconstructed exactly from its
``seconds_since_previous`` column.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


OUTPUT_PATH = Path("analysis_baseline.json")
SESSION_GAP_SECONDS = 30 * 60
TIMESTAMP_COLUMNS = ("timestamp", "played_at", "event_time", "time")
TRACK_ID_COLUMNS = ("video_id", "track_id", "track", "track_title")


def find_listening_csv(repo_root: Path) -> tuple[Path, list[str]]:
    """Find the event-level CSV by checking its schema, not its filename alone."""
    preferred = repo_root / "data" / "public" / "listening_events.csv"
    candidates = [preferred] if preferred.exists() else []
    candidates.extend(
        path
        for path in sorted(repo_root.rglob("*.csv"))
        if path != preferred and ".git" not in path.parts
    )

    for path in candidates:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            headers = next(reader, [])
        has_track = any(column in headers for column in TRACK_ID_COLUMNS)
        has_time = (
            "seconds_since_previous" in headers
            or any(column in headers for column in TIMESTAMP_COLUMNS)
        )
        if has_track and has_time:
            return path, headers

    raise FileNotFoundError(
        "No event-level CSV with track and time/gap columns was found."
    )


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no data rows.")
    return rows


def first_present(columns: Iterable[str], fieldnames: Iterable[str]) -> str | None:
    available = set(fieldnames)
    return next((column for column in columns if column in available), None)


def parse_iso_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp has no timezone: {value!r}")
    return parsed


def reconstruct_elapsed_seconds(
    rows: list[dict[str, str]], fieldnames: list[str]
) -> tuple[list[float], list[float], str]:
    """Return elapsed seconds, inter-event gaps, and the time basis used."""
    if "seconds_since_previous" in fieldnames:
        elapsed: list[float] = []
        gaps: list[float] = []
        running_total = 0.0
        for index, row in enumerate(rows):
            raw_gap = row.get("seconds_since_previous", "").strip()
            gap = 0.0 if index == 0 and not raw_gap else float(raw_gap or 0)
            if gap < 0:
                raise ValueError(f"Negative time gap at data row {index + 2}.")
            running_total += gap
            gaps.append(gap)
            elapsed.append(running_total)
        return (
            elapsed,
            gaps,
            "relative elapsed time reconstructed from seconds_since_previous",
        )

    timestamp_column = first_present(TIMESTAMP_COLUMNS, fieldnames)
    if timestamp_column is None:
        raise ValueError("No supported timestamp or gap column was found.")

    parsed = [parse_iso_timestamp(row[timestamp_column]) for row in rows]
    if parsed != sorted(parsed):
        raise ValueError(f"Rows are not ordered by {timestamp_column}.")
    origin = parsed[0]
    elapsed = [(value - origin).total_seconds() for value in parsed]
    gaps = [0.0] + [
        (current - previous).total_seconds()
        for previous, current in zip(parsed, parsed[1:])
    ]
    return elapsed, gaps, f"absolute timestamps from {timestamp_column}"


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else 0.0


def rounded(value: float, digits: int = 4) -> float:
    return round(value, digits)


def calculate_metrics(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    source_path: Path,
    repo_root: Path,
) -> dict:
    track_column = first_present(TRACK_ID_COLUMNS, fieldnames)
    if track_column is None:
        raise ValueError("No supported track identifier column was found.")

    elapsed, event_gaps, time_basis = reconstruct_elapsed_seconds(rows, fieldnames)
    track_ids = [row[track_column].strip() for row in rows]
    if any(not track_id for track_id in track_ids):
        raise ValueError(f"Blank {track_column} value found.")

    play_counts = Counter(track_ids)
    total_plays = len(rows)
    unique_tracks = len(play_counts)
    sorted_counts = sorted(play_counts.values(), reverse=True)

    top_concentration = {}
    for size in (1, 5, 10):
        occupied = sum(sorted_counts[: min(size, unique_tracks)])
        top_concentration[f"top_{size}_percent"] = rounded(
            occupied / total_plays * 100
        )

    track_times: dict[str, list[float]] = defaultdict(list)
    metadata: dict[str, dict[str, str]] = {}
    for row, event_time, track_id in zip(rows, elapsed, track_ids):
        track_times[track_id].append(event_time)
        metadata[track_id] = {
            "track_title": row.get("track_title", ""),
            "channel_name": row.get("channel_name", row.get("artist", "")),
        }

    repeat_gap_rows = []
    all_repeat_gaps = []
    for track_id, times in track_times.items():
        if len(times) < 2:
            continue
        gaps = [
            current - previous for previous, current in zip(times, times[1:])
        ]
        all_repeat_gaps.extend(gaps)
        average_seconds = mean(gaps)
        repeat_gap_rows.append(
            {
                "track_id": track_id,
                **metadata[track_id],
                "play_count": len(times),
                "repeat_gap_count": len(gaps),
                "average_gap_hours": rounded(average_seconds / 3600),
                "average_gap_days": rounded(average_seconds / 86400),
            }
        )
    repeat_gap_rows.sort(key=lambda item: (-item["play_count"], item["track_id"]))

    single_play_tracks = sum(count == 1 for count in play_counts.values())

    sessions: list[list[str]] = []
    current_session: list[str] = []
    for track_id, gap in zip(track_ids, event_gaps):
        if current_session and gap > SESSION_GAP_SECONDS:
            sessions.append(current_session)
            current_session = []
        current_session.append(track_id)
    if current_session:
        sessions.append(current_session)

    session_lengths = [len(session) for session in sessions]
    session_unique_counts = [len(set(session)) for session in sessions]
    session_unique_ratios = [
        unique_count / length
        for unique_count, length in zip(session_unique_counts, session_lengths)
    ]

    return {
        "source_csv": source_path.relative_to(repo_root).as_posix(),
        "time_basis": time_basis,
        "session_definition": "A new session starts when the preceding gap exceeds 30 minutes.",
        "schema": fieldnames,
        "data_summary": {
            "total_plays": total_plays,
            "unique_tracks": unique_tracks,
        },
        "repetition_rate": {
            "definition": "total plays divided by total unique tracks",
            "plays_per_unique_track": rounded(total_plays / unique_tracks),
        },
        "repeat_gaps": {
            "tracks_with_repeats": len(repeat_gap_rows),
            "repeat_gap_observations": len(all_repeat_gaps),
            "overall_average_hours": rounded(mean(all_repeat_gaps) / 3600),
            "overall_average_days": rounded(mean(all_repeat_gaps) / 86400),
            "median_gap_hours": rounded(
                statistics.median(all_repeat_gaps) / 3600
                if all_repeat_gaps
                else 0
            ),
            "per_track": repeat_gap_rows,
        },
        "top_track_concentration": {
            "definition": "cumulative share of all plays held by the top N tracks",
            **top_concentration,
        },
        "discovery_rate": {
            "definition": "share of unique tracks that appear exactly once",
            "single_play_tracks": single_play_tracks,
            "unique_tracks": unique_tracks,
            "percent": rounded(single_play_tracks / unique_tracks * 100),
        },
        "session_diversity": {
            "session_gap_minutes": SESSION_GAP_SECONDS // 60,
            "session_count": len(sessions),
            "average_tracks_per_session": rounded(mean(session_lengths)),
            "average_unique_tracks_per_session": rounded(
                mean(session_unique_counts)
            ),
            "average_unique_track_ratio": rounded(mean(session_unique_ratios)),
            "average_unique_track_percent": rounded(
                mean(session_unique_ratios) * 100
            ),
        },
    }


def print_summary(metrics: dict, rows: list[dict[str, str]]) -> None:
    print("=== Data Loading & Inspection ===")
    print(f"Source CSV: {metrics['source_csv']}")
    print(f"Columns: {metrics['schema']}")
    print("First 5 rows:")
    for row in rows[:5]:
        print(row)

    data = metrics["data_summary"]
    repetition = metrics["repetition_rate"]
    repeat_gaps = metrics["repeat_gaps"]
    concentration = metrics["top_track_concentration"]
    discovery = metrics["discovery_rate"]
    sessions = metrics["session_diversity"]

    print("\n=== Baseline Summary ===")
    print(f"Time basis: {metrics['time_basis']}")
    print(f"Total plays: {data['total_plays']:,}")
    print(f"Unique tracks: {data['unique_tracks']:,}")
    print(
        "Repetition rate: "
        f"{repetition['plays_per_unique_track']:.4f} plays per unique track"
    )
    print(
        "Overall repeat gap: "
        f"{repeat_gaps['overall_average_hours']:.4f} hours "
        f"({repeat_gaps['overall_average_days']:.4f} days)"
    )
    print(
        f"Median repeat gap: {repeat_gaps['median_gap_hours']:.4f} hours"
    )
    print(
        "Top-track concentration: "
        f"top 1 = {concentration['top_1_percent']:.4f}%, "
        f"top 5 = {concentration['top_5_percent']:.4f}%, "
        f"top 10 = {concentration['top_10_percent']:.4f}%"
    )
    print(
        "Discovery rate: "
        f"{discovery['percent']:.4f}% "
        f"({discovery['single_play_tracks']:,} of "
        f"{discovery['unique_tracks']:,} unique tracks played once)"
    )
    print(
        "Sessions: "
        f"{sessions['session_count']:,}; "
        f"average tracks/session = "
        f"{sessions['average_tracks_per_session']:.4f}; "
        f"average unique tracks/session = "
        f"{sessions['average_unique_tracks_per_session']:.4f}; "
        f"average unique ratio = "
        f"{sessions['average_unique_track_ratio']:.4f} "
        f"({sessions['average_unique_track_percent']:.4f}%)"
    )

    print("\n=== Per-Track Repeat Gaps ===")
    print(
        "play_count | average_hours | average_days | "
        "track_title | channel_name | track_id"
    )
    for item in repeat_gaps["per_track"]:
        print(
            f"{item['play_count']:>10} | "
            f"{item['average_gap_hours']:>13.4f} | "
            f"{item['average_gap_days']:>12.4f} | "
            f"{item['track_title']} | {item['channel_name']} | "
            f"{item['track_id']}"
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    repo_root = Path(__file__).resolve().parent
    source_path, fieldnames = find_listening_csv(repo_root)
    rows = load_rows(source_path)
    metrics = calculate_metrics(rows, fieldnames, source_path, repo_root)
    output_path = repo_root / OUTPUT_PATH
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print_summary(metrics, rows)
    print(f"\nSaved metrics: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
