# Pitcher Performance Clustering

Discover pitcher archetypes from Statcast pitch-level data using classical and modern clustering techniques, with LLM-powered narrative summaries.

## Project Structure

```
baseball-trends/
├── run_pipeline.py              # One-command pipeline orchestrator
├── requirements.txt
├── src/
│   ├── backend/
│   │   ├── preprocess_data.py   # Data loading + pitcher feature engineering
│   │   ├── clustering.py        # KMeans baseline + UMAP/HDBSCAN advanced
│   │   ├── llm_summaries.py     # LLM-generated cluster narratives
│   │   └── download_statcast.py # Statcast data downloader (hive-partitioned)
│   └── frontend/
│       └── app.py               # Streamlit dashboard
├── data/
│   ├── statcast_pitches/        # Hive-partitioned: game_date=YYYY-MM-DD/pitches.parquet
│   ├── processed/               # Generated pitcher profiles + clustered outputs
│   └── ...                      # Additional Statcast leaderboard data
└── figures/                     # Generated visualizations
```

## Setup

```bash
conda activate baseball
pip install -r requirements.txt
```

## Usage

### Full pipeline (2024 season)
```bash
python run_pipeline.py
```

### Custom date range
```bash
python run_pipeline.py --start 2024-06-01 --end 2024-06-30 --min-pitches 100
```

### Skip LLM summaries (no API key needed)
```bash
python run_pipeline.py --skip-llm
```

### Launch dashboard
```bash
streamlit run src/frontend/app.py
```

## Pipeline Overview

1. **Data Loading** — Reads hive-partitioned Statcast parquet files for a date range
2. **Feature Engineering** — Builds pitcher-level profiles: velocity, spin, movement, pitch mix, command, batted-ball outcomes
3. **Baseline Clustering** — KMeans with elbow/silhouette analysis
4. **Advanced Clustering** — UMAP dimensionality reduction + HDBSCAN density-based clustering
5. **LLM Summaries** — OpenAI-powered (or rule-based fallback) scouting narratives per cluster
6. **Dashboard** — Interactive Streamlit app with cluster comparison, similarity search, and data explorer

## Key Features

- **Pitcher profiles** built from 50+ Statcast metrics, aggregated per pitcher
- **9 pitch types tracked**: FF, SI, SL, CH, CU, FC, ST, KC, FS with per-type velocity, spin, and movement
- **Similarity search**: find nearest-neighbor pitchers in feature space
- **Dual clustering**: compare classical KMeans vs density-based HDBSCAN
- **LLM integration**: GPT-powered cluster narratives with rule-based fallback
