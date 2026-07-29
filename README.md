# youtubemusic-algo
i keep getting the same songs recommended and not receiving any adjacent songs like i would in infinite play using apple music

turns out its baked in its loss functions
-im thinking this means that they minimize the loss of having someone just stop listening to the next song so they know what you want 

the loss function is understandable but there is a negative externality for a user who spends an afternoon playing their mix

if you really want this: 

1. 2016 Deep Neural Networks paper, 2019 Multitask Ranking System paper.



Phase 2: Data setup (1–2 weeks)
Export your history via Google Takeout (YouTube Music data)
Set up enrichment: Write a script to pull metadata from Spotify API and MusicBrainz for every track
Explore public datasets: Grab LFM-1b or Million Song Dataset for collaborative filtering baseline

Deliverable: A cleaned dataset of your listening history + rich features, ready for modeling.

Phase 3: Build the alternative (3–4 weeks)
Candidate generation: Train an item-based collaborative filtering model (I'd start with LightFM in Python)
Ranking layer: XGBoost ranker with similarity, novelty bonus, time-since-last-play, etc.
Diversity post-processing: Use a simple diversity constraint (e.g., MMR or a DPP library)
Offline evaluation: Compute Recall@k, NDCG, catalog coverage on held-out data

Deliverable: A playlist generator that shows "what a less conservative algorithm would recommend" without you ever listening to anything ne
