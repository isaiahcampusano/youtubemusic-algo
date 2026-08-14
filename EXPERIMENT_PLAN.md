# Offline Recommender Experiment Plan

## Research question

Can a transparent, sequence-aware recommender trained on one user's YouTube
Music consumption history increase novelty and catalog coverage relative to
simple popularity and recency baselines without giving up all next-item
predictive accuracy?

This is an offline experiment on observed consumption. It does not recreate,
identify, or causally evaluate YouTube Music's production recommender or loss
function.

## Evidence available

The repository contains 1,916 chronologically ordered events covering 835
unique YouTube video IDs. It supports measurement of historical repetition,
concentration, recency, session structure, transitions, and offline next-item
prediction.

It does not contain recommendation impressions, ranks, rejected candidates,
skip or completion signals, listening duration, counterfactual outcomes, or
other users. A Takeout event may represent a deliberate selection or autoplay.
Different uploads and versions of the same song remain different video IDs,
which can overstate musical diversity.

## Verified descriptive baseline

| Metric | Value |
| --- | ---: |
| Events | 1,916 |
| Unique video IDs | 835 |
| Plays per unique video | 2.2946 |
| Top-10 share of events | 16.2317% |
| Video IDs appearing once | 72.2156% |
| Sessions using a 30-minute boundary | 188 |
| Average within-session unique-item ratio | 99.6125% |

The last value is an average ratio, not the percentage of sessions that are
completely unique. It suggests that exact-video repetition is primarily a
cross-session phenomenon in this dataset. It does not establish why the user
replayed those items.

## Testable hypotheses

1. A most-played ranking is a competitive next-item baseline because repeat
   consumption is concentrated in a small catalog head.
2. Recent listening context contains predictive information beyond static play
   counts.
3. Immediate item-transition counts improve next-item prediction when the last
   item has sufficient transition support.
4. Penalizing items heard in the current session increases recommendation
   coverage and reduces repeated exposure, with a measurable accuracy tradeoff.
5. A composite ranker can occupy a different accuracy/novelty frontier than
   any single-signal baseline.

These hypotheses concern this offline dataset only. Failure to beat a baseline
is a valid result and must not be hidden or tuned away on the test set.

## Temporal evaluation

Split the ordered event sequence into:

- first 80%: training;
- next 10%: validation and composite-weight selection;
- final 10%: locked test set.

Model counts and transition tables are built from training events only.
Hyperparameters are selected on validation data. Test events are processed in
order: previously observed test events may enter the live recency/session
context, but they must not update fitted popularity or transition counts.

For every evaluation target, algorithms rank candidates drawn from the
training catalog. An item absent from the training catalog is therefore an
unavoidable miss and is reported as such.

## Algorithms

### Most played

Rank the training catalog by descending training play count, with video ID as
a deterministic tie-breaker.

### Most recent

Rank known items by their latest occurrence in the history available before
the target event.

### Item transition

Rank the observed immediate successors of the current item by transition
count. Fall back to most played when the current item has no trained outgoing
transitions.

### Item transition with repeat penalty

Start from the transition score and down-weight candidates heard in the
current session or within a configured recent-time window. Keep the same
most-played fallback.

### Composite scorer

For candidate `j` at time `t`, combine normalized signals:

```text
score(j | t) =
    w_relevance * transition_relevance(j)
  + w_novelty * inverse_popularity(j)
  + w_recency * recency_affinity(j)
  + w_repeat * repeat_avoidance(j)
  + w_diversity * list_diversity_gain(j)
```

The five non-negative weights sum to 1. Validation selects one preset from a
small declared grid. Diversity uses cosine similarity between item transition
profiles; no external embeddings or metadata are introduced.

## Metrics

Report all final metrics on the locked test segment:

- Hit Rate@5, @10, and @20 (equivalent to single-target Recall@K here);
- catalog coverage over the training catalog;
- recommendation-slot repetition rate;
- recommendations per unique recommended item;
- inverse-popularity novelty;
- mean pairwise list diversity from transition-profile cosine distance;
- cold-start target count and rate;
- mean returned-list size.

Accuracy, novelty, coverage, and diversity are separate outcomes. No single
metric is treated as proof of user satisfaction.

## Required artifacts

- `recommend.py`: data preparation, fitted training state, four baselines, and
  the composite reranker;
- `evaluate.py`: temporal split, validation selection, locked test evaluation,
  metric calculation, and report generation;
- `evaluation_results.json`: complete machine-readable configuration and
  results;
- `EVALUATION_REPORT.md`: concise methodology, result table, findings,
  limitations, and MVP conclusion.

All implementation code uses the Python standard library and reads only the
existing public event CSV.

## Completion criteria

- all five algorithms return deterministic ranked lists with no duplicates;
- the split is chronological and its sizes reconcile to all 1,916 events;
- training statistics never incorporate validation or test targets;
- validation selects composite settings without consulting test metrics;
- all required metrics are emitted for every algorithm;
- result counts, recommendation slots, and catalog denominators reconcile;
- `evaluate.py` regenerates both result artifacts successfully;
- the report states the measured tradeoffs even if the composite method loses;
- the branch is mergeable and contains no raw Takeout archive or new external
  data.

The MVP is complete when the pipeline, descriptive baseline, five recommenders,
locked offline evaluation, and documented limitations all run from the checked-
in repository state.
