# Offline Recommendation Evaluation

## Executive summary

The strongest test Hit Rate@10 came from **Item transition**; the widest catalog coverage came from **Composite scorer**. The composite scorer did not beat the strongest baseline on Hit Rate@10. Its difference from the strongest baseline was -0.0260. These are offline next-item results, not evidence about YouTube Music's internal ranking objective.

## Setup

The 1,916 ordered events were split into 1,533 training, 191 validation, and 192 test events. Fitted popularity and transition statistics used training only. Validation selected the **relevance_heavy** weight preset and a 6-hour repeat window; the locked test segment was used once for the table below.

## Locked test results

| Algorithm | Hit@5 | Hit@10 | Hit@20 | Coverage | Novelty | Diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Most played | 10.94% | 15.62% | 27.60% | 2.66% | 0.6066 | 0.9272 |
| Most recent | 1.56% | 4.69% | 13.02% | 10.37% | 0.7292 | 0.9319 |
| Item transition | 22.40% | 26.56% | 32.81% | 17.29% | 0.6235 | 0.9266 |
| Transition + repeat penalty | 23.96% | 26.56% | 32.81% | 17.29% | 0.6235 | 0.9266 |
| Composite scorer | 19.79% | 23.96% | 27.08% | 20.08% | 0.8253 | 0.9703 |

Hit Rate is equivalent to Recall for this single-next-item task. Coverage is the fraction of the training catalog recommended at least once. Novelty is normalized self-information under training popularity. Diversity is mean pairwise cosine distance between training-only transition profiles.

## Findings

- **Accuracy:** Item transition achieved the highest Hit@10 (26.56%).
- **Catalog reach:** Composite scorer reached 20.08% of the training catalog.
- **Cold start:** 50 of 192 test targets (26.04%) were absent from the training catalog and could not be hit by any evaluated method.
- **Integrity:** duplicate recommendation lists = 0; recommendations outside the training catalog = 0.

## Limitations

This single-user Takeout history records consumption, not recommendation impressions. It has no rejected candidates, ranks, skips, completion, listening duration, or counterfactual feedback. Autoplay and deliberate selection cannot be separated. Video IDs are uploads rather than canonical songs, and transition-profile distance is only a behavioral proxy for musical diversity. The small test set makes fine-grained differences noisy.

## MVP conclusion

The repository now supports a reproducible, leakage-aware comparison of five transparent recommenders. It can reveal tradeoffs among next-item accuracy, novelty, coverage, repetition, and diversity for this history. It cannot recover YouTube Music's algorithm or loss function. The next stronger experiment would collect recommendation impressions and explicit outcomes, then evaluate a pre-registered objective on future data.
