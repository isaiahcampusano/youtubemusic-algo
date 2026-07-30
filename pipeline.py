"""Build privacy-reduced YouTube Music CSVs from a Google Takeout ZIP.

The raw archive is read in place and is never extracted by this script.
Only watch-history records explicitly labeled "YouTube Music" are included.
Absolute timestamps, account URLs, searches, subscriptions, and playlists are
excluded from the public outputs.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlparse
from zipfile import BadZipFile, ZipFile


WATCH_HISTORY_SUFFIX = PurePosixPath(
    "Takeout/YouTube and YouTube Music/history/watch-history.json"
)
LIBRARY_SUFFIX = PurePosixPath(
    "Takeout/YouTube and YouTube Music/"
    "music (library and uploads)/music library songs.csv"
)

EVENT_FIELDS = [
    "event_index",
    "relative_day",
    "seconds_since_previous",
    "hour_utc",
    "weekday_utc",
    "video_id",
    "track_title",
    "channel_name",
    "in_music_library",
]

TRACK_FIELDS = [
    "video_id",
    "track_title",
    "channel_name",
    "play_count",
    "first_event_index",
    "last_event_index",
    "first_relative_day",
    "last_relative_day",
    "in_music_library",
]


def _find_member(zip_file: ZipFile, suffix: PurePosixPath) -> str:
    """Return the single normalized member matching a required Takeout path."""
    normalized = {
        PurePosixPath(info.filename): info.filename
        for info in zip_file.infolist()
        if not info.is_dir()
    }
    matches = [
        original
        for path, original in normalized.items()
        if path == suffix or str(path).endswith("/" + str(suffix))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {suffix.name!r} file, found {len(matches)}."
        )
    return matches[0]


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp has no timezone: {value!r}")
    return parsed


def _video_id(url: str) -> str:
    values = parse_qs(urlparse(url).query).get("v", [])
    return values[0] if values else ""


def _track_title(raw_title: str) -> str:
    prefix = "Watched "
    return raw_title[len(prefix) :] if raw_title.startswith(prefix) else raw_title


def _channel_name(record: dict) -> str:
    subtitles = record.get("subtitles") or []
    if not subtitles:
        return ""
    return str(subtitles[0].get("name") or "")


def _safe_csv_text(value: object) -> object:
    """Neutralize spreadsheet formulas while preserving ordinary text."""
    if not isinstance(value, str):
        return value
    if value.startswith(("=", "+", "@", "\t", "\r", "\n")):
        return "'" + value
    return value


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _safe_csv_text(row.get(field, "")) for field in fieldnames}
            )


def build_dataset(takeout_zip: Path, output_dir: Path) -> tuple[Path, Path, dict]:
    try:
        with ZipFile(takeout_zip) as archive:
            watch_member = _find_member(archive, WATCH_HISTORY_SUFFIX)
            library_member = _find_member(archive, LIBRARY_SUFFIX)

            with archive.open(watch_member) as raw:
                watch_history = json.load(io.TextIOWrapper(raw, encoding="utf-8"))

            with archive.open(library_member) as raw:
                library_reader = csv.DictReader(
                    io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                )
                library_ids = {
                    row.get("Video ID", "").strip()
                    for row in library_reader
                    if row.get("Video ID", "").strip()
                }
    except BadZipFile as exc:
        raise ValueError(f"Not a valid ZIP archive: {takeout_zip}") from exc

    selected = []
    for record in watch_history:
        if record.get("header") != "YouTube Music":
            continue
        video_id = _video_id(str(record.get("titleUrl") or ""))
        if not video_id:
            continue
        selected.append(
            {
                "time": _parse_time(str(record["time"])),
                "video_id": video_id,
                "track_title": _track_title(str(record.get("title") or "")),
                "channel_name": _channel_name(record),
                "in_music_library": video_id in library_ids,
            }
        )

    selected.sort(key=lambda row: row["time"])
    if not selected:
        raise ValueError("No watch-history records labeled 'YouTube Music' were found.")

    first_time = selected[0]["time"]
    previous_time = None
    event_rows: list[dict] = []
    track_events: dict[str, list[dict]] = defaultdict(list)

    for index, row in enumerate(selected, start=1):
        current_time = row["time"]
        relative_day = (current_time - first_time).days
        seconds_since_previous = (
            ""
            if previous_time is None
            else max(0, round((current_time - previous_time).total_seconds()))
        )
        event = {
            "event_index": index,
            "relative_day": relative_day,
            "seconds_since_previous": seconds_since_previous,
            "hour_utc": current_time.hour,
            "weekday_utc": current_time.strftime("%A"),
            "video_id": row["video_id"],
            "track_title": row["track_title"],
            "channel_name": row["channel_name"],
            "in_music_library": row["in_music_library"],
        }
        event_rows.append(event)
        track_events[row["video_id"]].append(event)
        previous_time = current_time

    track_rows = []
    for video_id, events in track_events.items():
        title_counts = Counter(
            event["track_title"] for event in events if event["track_title"]
        )
        channel_counts = Counter(
            event["channel_name"] for event in events if event["channel_name"]
        )
        track_rows.append(
            {
                "video_id": video_id,
                "track_title": (
                    title_counts.most_common(1)[0][0] if title_counts else ""
                ),
                "channel_name": (
                    channel_counts.most_common(1)[0][0] if channel_counts else ""
                ),
                "play_count": len(events),
                "first_event_index": events[0]["event_index"],
                "last_event_index": events[-1]["event_index"],
                "first_relative_day": events[0]["relative_day"],
                "last_relative_day": events[-1]["relative_day"],
                "in_music_library": any(
                    event["in_music_library"] for event in events
                ),
            }
        )

    track_rows.sort(key=lambda row: (-row["play_count"], row["video_id"]))

    events_path = output_dir / "listening_events.csv"
    tracks_path = output_dir / "track_summary.csv"
    _write_csv(events_path, EVENT_FIELDS, event_rows)
    _write_csv(tracks_path, TRACK_FIELDS, track_rows)

    stats = {
        "youtube_music_events": len(event_rows),
        "unique_tracks": len(track_rows),
        "relative_days": event_rows[-1]["relative_day"] + 1,
        "library_matched_events": sum(
            bool(row["in_music_library"]) for row in event_rows
        ),
        "library_matched_tracks": sum(
            bool(row["in_music_library"]) for row in track_rows
        ),
    }
    return events_path, tracks_path, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create privacy-reduced YouTube Music CSVs from Takeout."
    )
    parser.add_argument("takeout_zip", type=Path, help="Original Google Takeout ZIP")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/public"),
        help="Output directory (default: data/public)",
    )
    args = parser.parse_args()

    events_path, tracks_path, stats = build_dataset(
        args.takeout_zip.resolve(), args.output_dir.resolve()
    )
    print(f"Wrote {events_path}")
    print(f"Wrote {tracks_path}")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
