# youtubemusic-algo

An evidence-first exploration of a personal question:

> Why does YouTube Music repeat familiar tracks, and what would a
> less-conservative playlist policy look like on my own listening history?

## What this project can and cannot establish

Google Takeout gives us a record of consumption. It does **not** include the
recommendations that were shown but ignored, their ranks, the candidate pool,
model features, or YouTube's loss function.

That means this repository can measure repeat behavior in one account and test
alternative playlist policies offline. It cannot reverse engineer or prove
YouTube Music's production algorithm from Takeout alone.

## Current dataset

The first public dataset is derived from a Google Takeout export created on
July 29, 2026.

- 1,916 events explicitly labeled `YouTube Music`
- 835 unique video IDs
- 310 events matching 62 tracks in the exported music library
- relative timing and chronological order retained
- absolute dates and unrelated Google/YouTube activity removed

Files:

- `data/public/listening_events.csv`: event sequence for repetition, recency,
  novelty, and session analysis
- `data/public/track_summary.csv`: one row per track with play counts and
  first/last positions

The raw Takeout ZIP is deliberately not committed because this is a public
repository.

## Build the CSV files

Python 3.10+ is enough; the pipeline uses only the standard library.

```powershell
python pipeline.py "C:\path\to\takeout-YYYYMMDDTHHMMSSZ-1-001.zip"
```

The command reads the ZIP without extracting it and writes both CSV files to
`data/public/`.

## Published event columns

| Column | Meaning |
| --- | --- |
| `event_index` | Chronological position, starting at 1 |
| `relative_day` | Days since the first retained event |
| `seconds_since_previous` | Gap from the preceding retained event |
| `hour_utc` | UTC hour, with the calendar date removed |
| `weekday_utc` | UTC weekday |
| `video_id` | Public YouTube video identifier |
| `track_title` | Title recorded by Takeout |
| `channel_name` | Public channel/artist label recorded by Takeout |
| `in_music_library` | Whether the video ID appears in the exported library |

## Research direction

1. Describe the observed sequence: concentration, repeat gaps, catalog coverage,
   and session-level diversity.
2. Create transparent baselines such as most-played, recency-aware, and
   item-to-item co-occurrence.
3. Add controlled exploration using a novelty bonus or maximum marginal
   relevance.
4. Evaluate with temporal holdouts and clearly separate observed behavior from
   claims about YouTube's internal system.

Spotify audio-feature enrichment is not part of this pipeline. Spotify
deprecated its audio-features endpoint for new use cases, and Spotify content
should not be used to train an ML model. Future metadata work should use
sources whose access and licensing fit the research purpose.
