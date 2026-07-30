# Data policy

`public/` contains derived YouTube Music listening data that is suitable for
this public repository.

The published files intentionally exclude:

- the original Google Takeout ZIP;
- general YouTube watch history;
- YouTube search history;
- subscriptions and playlist names;
- account identifiers and URLs; and
- absolute timestamps.

`listening_events.csv` preserves chronological order, relative time, repeat
behavior, track identity, and public track/channel metadata. It can support
sequence, novelty, coverage, and repetition analyses without exposing the
account's exact listening dates.

`track_summary.csv` is a track-level aggregation derived from those events.

Regenerate both files from a local Takeout ZIP:

```powershell
python pipeline.py "C:\path\to\takeout.zip"
```

Raw exports belong in `data/raw/` or outside the repository. Both locations and
common Takeout ZIP names are ignored by Git.
