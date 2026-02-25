# ⚾ Pitcher Analytics Dashboard

An end-to-end baseball analytics platform that discovers pitcher archetypes, surfaces traditional and expected stats, predicts regression candidates, and scores pitcher quality — all through an interactive Streamlit dashboard.

**Why?** Pitchers are hard to value. ERA mixes skill and luck; two guys with the same ERA can have wildly different arsenals. This project focuses on *how* pitchers throw (arsenal features) and *what should have happened* (expected stats), so we can separate signal from noise.

## What it does

| Tab | What you get |
|-----|-------------|
| **Classic Clustering** | KMeans pitcher archetypes from Statcast arsenal features |
| **Modern Clustering** | UMAP + HDBSCAN density-based clusters (4D by default) |
| **Similarity Search** | Nearest-neighbour pitchers by arsenal profile (Euclidean distance) + compare two pitchers |
| **Cluster Rosters & Performance** | Which pitchers belong to each cluster; cluster avg ERA/WHIP/SO9 by year |
| **Traditional & Expected Stats** | ERA / WHIP / K/9 vs. xERA — all years 2015–2025, with trend lines |
| **Pitcher Quality Scores** | Composite 0–100 score from Savant percentile ranks (A+…F grades) |
| **Regression Predictor** | Six models predict next-season ERA; buy-low / sell-high candidate lists |
| **Raw Data** | Filterable, downloadable Statcast feature table |

## Project Structure

```
baseball-trends/
├── run_pipeline.py                         # One-command orchestrator
├── requirements.txt
├── src/
│   ├── backend/
│   │   ├── preprocess_data.py              # Pitch-by-pitch → pitcher feature vectors
│   │   ├── clustering.py                   # KMeans + UMAP/HDBSCAN + similarity search
│   │   ├── preprocess_pitching_bref.py     # bref ERA/WHIP/SO9 + Savant expected/percentile join
│   │   ├── predict_actual_expected_residual.py  # ERA regression models + quality score
│   │   ├── llm_summaries.py                # Rule-based cluster narratives
│   │   ├── text_insights.py                # Human-readable stat interpretation helpers
│   │   └── download_statcast.py            # Statcast data download utilities
│   └── frontend/
│       └── app.py                          # Streamlit dashboard (7 tabs)
├── data/
│   ├── statcast_pitches/                   # Hive-partitioned: game_date=YYYY-MM-DD/
│   ├── pitching_stats_bref/                # Baseball Reference traditional stats by year
│   ├── statcast_pitcher_expected_stats/    # Savant xERA / expected wOBA by year
│   ├── statcast_pitcher_percentile_ranks/  # Savant 0-100 percentile ranks by year
│   ├── statcast_pitcher_exitvelo_barrels/  # Savant barrel / exit velo by year
│   ├── statcast_pitcher_arsenal_stats/     # Savant pitch-type breakdowns by year
│   ├── chadwick_register.parquet           # MLBAM ↔ Baseball Reference ID crosswalk
│   └── processed/                          # Generated outputs (gitignored)
└── figures/                                # Generated plots (gitignored)
```

## Setup

```bash
conda activate baseball          # or: source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### 1. Build pitching stats (fastest — no pitch-by-pitch data needed)
```bash
python src/backend/preprocess_pitching_bref.py --years 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
python src/backend/predict_actual_expected_residual.py
streamlit run src/frontend/app.py
```
This gives you the Traditional Stats, Quality Scores, and Regression Predictor tabs immediately.

### 2. Full pipeline (requires pitch-by-pitch Statcast data)
```bash
python run_pipeline.py           # 2024 season, full pipeline
python run_pipeline.py --skip-pitching-stats   # clustering only
python run_pipeline.py --skip-clustering       # pitching stats only
```

### 3. Cluster narratives
Cluster summaries are generated automatically using a rule-based template (velocity tier, spin rate, pitch mix). No API key required.

## Pipeline Cadence (during the MLB season)

| Job | Frequency | Command |
|-----|-----------|---------|
| Download Statcast pitches | As needed | `python src/backend/download_statcast.py` |
| Refresh traditional + expected stats | Weekly | `preprocess_pitching_bref.py` |
| Re-cluster pitcher profiles | Weekly | `run_pipeline.py --skip-pitching-stats` |
| Re-fit regression models | Weekly | `predict_actual_expected_residual.py` |

## Key Data Sources

- **Baseball Savant** (via pybaseball) — pitch-by-pitch Statcast, expected stats, percentile ranks
- **Baseball Reference** (via pybaseball) — traditional pitching stats (ERA, WHIP, K/9, IP)
- **Chadwick Register** — MLBAM ↔ Baseball Reference player ID crosswalk
