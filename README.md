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

Deliverable: A playlist generator that shows "what a less conservative algorithm would recommend" without you ever listening to anything new


#first deliverable

- deepseek has the instructions for how i can track my own youtube data

- Codex Prompt / Plan:

You have full read/write access to this repository. Locate the YouTube Music listening history CSV file (search the root directory or use glob to find any .csv file if the name is unknown). Do not build or intake new information—we are purely evaluating the existing data.

Write and execute a single Python script that loads this CSV, parses the timestamps, and calculates the baseline metrics previously discussed. Specifically:

Data Loading & Inspection – Print the column names and first 5 rows to confirm the schema (track, artist, timestamp, etc.).
Repetition Rate – Calculate total plays divided by total unique tracks.
Repeat Gaps – For each track played more than once, compute the average time (in hours/days) between consecutive replays.
Top-Track Concentration – Calculate what percentage of total streams are occupied by the pipeline #1, #5, and #10 most-played tracks.
Discovery Rate – Calculate the percentage of tracks that appear only once in the entire dataset.
Session Diversity – Define a session as plays occurring within 30 minutes of each other (gap > 30 min = new session). For each session, calculate the average number of tracks per session and the average ratio of unique tracks per session (unique/total).
Formatted Output – Print all results clearly to the terminal in a human-readable summary.
Save Artifact – Write the final calculated metrics to a new JSON file named analysis_baseline.json in the repo root for future reference.
Execute the script immediately and show me the terminal output.



  
