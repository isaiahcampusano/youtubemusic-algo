# youtubemusic-algo

YouTube Music kept narrowing my listening to roughly the same half-dozen songs—usually tracks I had already played heavily that year. I wanted to understand why that pattern emerged and whether a recommender could make more room for discovery.

## What I tested

I converted my Google Takeout history into a privacy-reduced dataset containing:

- 1,916 YouTube Music plays
- 835 unique video IDs
- chronological order and relative timing
- no absolute dates or unrelated Google activity

I then compared five transparent recommendation strategies, ranging from “play what I usually play” to a composite strategy that rewards novelty, diversity, and repeat avoidance.

## What I learned

The song-to-song transition model predicted my next play best, reaching a 26.56% Hit Rate@10.

The discovery-oriented composite model was slightly less accurate at 23.96%, but it recommended a broader and less-popular selection of music.

In short: **more discovery was possible, but it cost some predictive accuracy.**

This does not prove that YouTube Music deliberately favors repetition, nor does it reveal YouTube’s loss function. My Takeout history records what I consumed—not what YouTube recommended, what I ignored, whether a song came from autoplay, or why I selected it.

## Why it matters

This project became an introduction to recommender-system research: defining objectives, building simple baselines, preventing future-data leakage, and measuring the tradeoff between predicting familiar behavior and creating opportunities for exploration.

It also gave me a more precise version of my original question: not “What is YouTube’s algorithm?” but “What evidence would let me measure how my actions and YouTube’s recommendations influence each other?”

## Explore the results

- [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md): findings in plain language
- [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md): hypotheses and evaluation design
- [`evaluation_results.json`](evaluation_results.json): complete calculated results
- [`data/public/listening_events.csv`](data/public/listening_events.csv): privacy-reduced event history

Run the evaluation with Python 3.10 or newer:

```powershell
python evaluate.py
```

The raw Google Takeout archive is not included.

## Next? 

**For one to two weeks, record what YouTube recommends before each listening session and what I play, skip, finish, or receive through autoplay—then test whether those actions predict how the next recommendations change.**
