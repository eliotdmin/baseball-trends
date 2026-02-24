# Technical README — Pitcher Analytics Platform

A deep-dive into every step of the system: data sources, feature engineering, modelling decisions, and the Streamlit dashboard architecture.

---

## Table of Contents

1. [Data Architecture](#1-data-architecture)
2. [Pitch-by-Pitch Feature Engineering](#2-pitch-by-pitch-feature-engineering)
3. [Pitcher Clustering Pipeline](#3-pitcher-clustering-pipeline)
4. [Traditional Stats + Expected Stats Join](#4-traditional-stats--expected-stats-join)
5. [Pitcher Quality Score](#5-pitcher-quality-score)
6. [ERA Regression Models](#6-era-regression-models)
7. [LLM Cluster Narratives](#7-llm-cluster-narratives)
8. [Text Insights Layer](#8-text-insights-layer)
9. [Streamlit Dashboard Architecture](#9-streamlit-dashboard-architecture)
10. [Pipeline Orchestration](#10-pipeline-orchestration)
11. [Data ID Crosswalk](#11-data-id-crosswalk)
12. [Extending the System](#12-extending-the-system)

---

## 1. Data Architecture

### Directory layout

```
data/
├── statcast_pitches/
│   └── game_date=YYYY-MM-DD/
│       └── pitches.parquet          # Hive-partitioned pitch-by-pitch Statcast
│
├── pitching_stats_bref/
│   └── data_YYYY.parquet            # Baseball Reference seasonal pitching stats
│
├── statcast_pitcher_expected_stats/
│   └── data_YYYY.parquet            # Savant: xERA, xwOBA, ERA−xERA diff
│
├── statcast_pitcher_percentile_ranks/
│   └── data_YYYY.parquet            # Savant: 0-100 percentile ranks vs. league
│
├── statcast_pitcher_exitvelo_barrels/
│   └── data_YYYY.parquet            # Savant: avg exit velo, barrel%, hard hit%
│
├── statcast_pitcher_arsenal_stats/
│   └── data_YYYY.parquet            # Savant: per-pitch-type run value, usage%
│
├── chadwick_register.parquet        # MLBAM ID ↔ Baseball Reference ID mapping
│
└── processed/                       # Generated outputs (not committed to git)
    ├── pitcher_profiles.parquet          # Raw per-pitcher Statcast feature vectors
    ├── pitcher_profiles_clustered.parquet # + KMeans and HDBSCAN cluster labels
    ├── pitching_profiles_YYYY.parquet    # Single-year enriched traditional stats
    ├── pitching_trends.parquet           # All years stacked (multi-season table)
    ├── regression_model_comparison.parquet
    ├── regression_feature_importances.parquet
    ├── regression_buy_low.parquet        # Unlucky pitcher candidates (ERA > xERA)
    ├── regression_sell_high.parquet      # Lucky pitcher candidates (ERA < xERA)
    ├── regression_projections.parquet    # Next-season ERA projections
    ├── quality_leaderboard.parquet       # 0-100 pitcher quality scores
    └── summaries/
        ├── cluster_summaries.json         # HDBSCAN cluster narratives
        └── cluster_summaries_kmeans.json # KMeans cluster narratives
```

### Storage format

All raw and processed data uses **Apache Parquet** via PyArrow. The pitch-by-pitch data is **hive-partitioned** by `game_date`, which means Pandas can read a subset of dates without scanning the entire dataset by passing a list of file paths.

---

## 2. Pitch-by-Pitch Feature Engineering

**File:** `src/backend/preprocess_data.py`

### Input

Pitch-level Statcast rows loaded from `data/statcast_pitches/game_date=YYYY-MM-DD/pitches.parquet`. Key columns used:

| Column | Meaning |
|--------|---------|
| `pitcher` | MLBAM player ID |
| `player_name` | Display name |
| `pitch_type` | FF / SI / SL / CH / CU / FC / ST / KC / FS |
| `release_speed` | Pitch velocity (mph) |
| `release_spin_rate` | Spin rate (rpm) |
| `pfx_x` / `pfx_z` | Horizontal / vertical movement (inches, gravity-free) |
| `release_pos_x/z` | Horizontal / vertical release point |
| `release_extension` | Extension toward home plate (ft) |
| `arm_angle` | Arm slot angle (degrees) |
| `plate_x` / `plate_z` | Location at front of plate |
| `launch_speed` | Exit velocity on batted balls |
| `launch_angle` | Launch angle on batted balls |
| `p_throws` | Handedness (R / L) |

### Aggregation logic (`build_pitcher_profiles`)

For each pitcher with ≥200 pitches (default), one feature vector is constructed:

**Arsenal-weighted global features:**
- `avg_velo`, `max_velo`, `std_velo` — velocity distribution across all pitches
- `avg_spin` — mean spin rate across all pitches
- `avg_extension` — how far toward home plate the pitcher releases the ball
- `avg_arm_angle` — mean arm slot
- `avg_pfx_x`, `avg_pfx_z` — mean horizontal/vertical movement (arsenal average)
- `release_x_mean`, `release_z_mean`, `release_x_std`, `release_z_std` — release point consistency
- `cmd_plate_x_std`, `cmd_plate_z_std` — command spread (lower std = more consistent location)
- `n_pitch_types` — arsenal breadth
- `p_throws` — handedness

**Per-pitch-type features (9 canonical types × 8 features = 72 columns):**
- `pct_{PT}` — proportion of all pitches this type represents (0–1)
- `velo_{PT}` — mean velocity for this pitch type (NaN if <10 thrown)
- `spin_{PT}` — mean spin rate
- `pfx_x_{PT}`, `pfx_z_{PT}` — mean movement (gravity-free)
- `spin_axis_{PT}` — mean spin axis (degrees)
- `break_z_{PT}` — mean induced vertical break (IVB, from `api_break_z_with_gravity`)
- `break_x_{PT}` — mean horizontal break (from `api_break_x_arm`)

**Batted-ball outcomes (for pitchers with ≥20 batted balls):**
- `avg_exit_velo_against` — mean exit velocity allowed
- `avg_launch_angle_against` — mean launch angle
- `hard_hit_pct` — proportion of batted balls ≥95 mph

### Output

`data/processed/pitcher_profiles.parquet` — one row per pitcher, ~50+ columns.

---

## 3. Pitcher Clustering Pipeline

**File:** `src/backend/clustering.py`

The pipeline runs two approaches in parallel.

### 3a. Feature modes: Two-layer vs single-layer

By default the pipeline uses **two-layer** clustering: Layer 1 clusters each pitch type by velocity/spin/movement into `quality_{PT}` archetypes (0..k−1); Layer 2 clusters pitchers by pitch mix (`pct_*`) and quality tiers (`quality_*`). Use `--no-two-layer` for legacy **single-layer** mode: full per-pitch velo/spin/movement/break features (no quality condensation).

**Quality clusters are one-hot encoded** before clustering: `quality_FF` (0, 1, 2) becomes `quality_FF_0`, `quality_FF_1`, `quality_FF_2`, since cluster IDs have no ordinal meaning — cluster 0 vs 2 should be the same distance as cluster 0 vs 1.

### 3b. Baseline: KMeans

1. **Feature selection** (`get_clustering_features`): depends on mode — two-layer uses `pct_*` + `quality_*`; single-layer uses `get_similarity_features` (pct + velo/spin/movement/break per pitch).
2. **Preprocessing** (`prepare_clustering_matrix`): fill NaN with column medians; `StandardScaler` normalisation.
3. **Elbow analysis** (`find_optimal_k`): fit KMeans for k = 3–11; record inertia and silhouette score.
4. **Fit** (`run_kmeans`): `sklearn.cluster.KMeans` with `n_init=10`, `random_state=42`.
5. Default k=5, configurable via `--k` flag in `run_pipeline.py`.

**Why KMeans:** Produces a clean, fixed-size partition. Every pitcher gets exactly one label. Good for interpretability.

**Limitation:** Assumes spherical clusters of roughly equal size; sensitive to the choice of k; assigns every pitcher to a cluster even if they don't fit well.

### 3c. Advanced: UMAP + HDBSCAN

1. **UMAP reduction** (`compute_umap_embedding`): compress the high-dimensional feature matrix to **6D by default** (configurable 2–10) using `umap-learn`. Higher dimensions preserve more structure. Hyperparameters: `n_neighbors=30`, `min_dist=0.1`, `metric='euclidean'`.
2. **HDBSCAN clustering** (`run_hdbscan`): fit on the UMAP embedding (or optionally on the **original scaled feature space** via `hdbscan_use_original_space`). Defaults: `min_cluster_size=5`, `min_samples=1`, `cluster_selection_method='leaf'`, `cluster_selection_epsilon=0.1`. Pitchers in sparse regions are labelled −1 (noise/outliers).
3. **Noise mitigation options:**
   - **min_samples=1** — per HDBSCAN docs, yields the fewest noise points.
   - **cluster_selection_epsilon** — merge clusters within a distance; absorbs borderline points (0.1–0.3 typical).
   - **hdbscan_use_original_space** — cluster in scaled feature space instead of UMAP; preserves full structure.
   - **hdbscan_assign_noise_via_soft** — assign each −1 to its most likely cluster via HDBSCAN soft clustering; yields 0% noise.
4. **Embedding storage**: `umap_0`, `umap_1`, … are saved in the parquet so the dashboard can plot any 2 axes (e.g. Dim 0 vs Dim 2).

**Why 6D:** 2D loses a lot of information when compressing ~100+ features. 6D preserves more structure for better clustering while still allowing 2D projection for exploration.

### 3d. Similarity Search and Compare Two Pitchers

`find_similar_pitchers`: compute **Euclidean distances** between standardized feature vectors (default). Euclidean penalizes magnitude differences (e.g. 92 vs 95 mph), so Abbott and Snell no longer appear overly similar. Cosine distance was deprecated because it measures direction only — similar pitch mix could group very different velocities.

The dashboard includes a **Compare Two Pitchers** view: side-by-side table of velocity, spin, break, spin axis, and usage by pitch type, plus Euclidean distance in feature space.

**Clustering and similarity use the same pitch-type-aware feature set** (`get_clustering_features` → `get_similarity_features`):
- **Excluded**: `avg_velo`, `avg_spin`, `avg_pfx_*` — these blend across pitch types (e.g. 95 mph FF + 82 mph SL → 88.5 mph), losing structure.
- **Included**: Per-pitch-type `velo`, `spin`, `pfx_x`, `pfx_z`, `spin_axis`, `break_z` (IVB), `break_x` for all pitch types — FF compared to FF, SL to SL, etc.
- Plus pitch mix (`pct_*`), extension, arm angle, command.

### Output

`data/processed/pitcher_profiles_clustered.parquet` — same as pitcher_profiles plus `cluster_kmeans`, `cluster_hdbscan`, and `umap_0`–`umap_{n-1}` columns.

---

## 4. Traditional Stats + Expected Stats Join

**File:** `src/backend/preprocess_pitching_bref.py`

### Data sources

| Source | File | Join key |
|--------|------|----------|
| Baseball Reference stats | `pitching_stats_bref/data_YYYY.parquet` | `mlbID` (MLBAM ID) |
| Savant expected stats | `statcast_pitcher_expected_stats/data_YYYY.parquet` | `player_id` (MLBAM ID) |
| Savant percentile ranks | `statcast_pitcher_percentile_ranks/data_YYYY.parquet` | `player_id` |
| Savant exit velo | `statcast_pitcher_exitvelo_barrels/data_YYYY.parquet` | `player_id` |

**Critical:** The `mlbID` column in Baseball Reference stats is the **MLBAM ID** (same ID space used by Baseball Savant), so joins are direct without needing the Chadwick crosswalk. The Chadwick register (`chadwick_register.parquet`) maps `key_mlbam` ↔ `key_bbref` and is available for other joins (e.g., linking to FanGraphs data).

### Join procedure (`build_pitching_profiles`)

1. Load all years of Baseball Reference stats → `bref` DataFrame.
2. Rename `season` → `year` for consistency.
3. Left-join `statcast_pitcher_expected_stats` on `(mlbID, year)`.
4. Left-join percentile ranks on `(mlbID, year)` (columns prefixed `pct_`).
5. Left-join exit velo / barrels on `(mlbID, year)`.
6. Filter: `IP >= min_ip` (default 20).

### Key columns in output

| Column | Source | Meaning |
|--------|--------|---------|
| `ERA` | bref | Actual earned run average |
| `WHIP` | bref | Walks + hits per inning pitched |
| `SO9` | bref | Strikeouts per 9 innings |
| `xera` | Savant | Expected ERA from quality of contact |
| `era_minus_xera` | Savant | ERA − xERA (positive = unlucky) |
| `pct_xera` | Savant | xERA percentile rank (0–100 vs. league) |
| `pct_k_percent` | Savant | K% percentile |
| `pct_bb_percent` | Savant | BB% percentile |
| `pct_whiff_percent` | Savant | Whiff% percentile |
| `pct_hard_hit_percent` | Savant | Hard hit% percentile |
| `pct_fb_velocity` | Savant | Fastball velocity percentile |
| `avg_hit_speed` | Savant | Average exit velocity allowed |
| `brl_percent` | Savant | Barrel rate allowed |

### Outputs

- `data/processed/pitching_profiles_YYYY.parquet` — snapshot for the most recent year requested
- `data/processed/pitching_trends.parquet` — all years stacked (one row per pitcher-season)

---

## 5. Pitcher Quality Score

**File:** `src/backend/predict_actual_expected_residual.py` → `compute_pitcher_quality_score()`

### Motivation

ERA and WHIP are affected by defense, sequencing luck (strand rate), and BABIP luck. A quality score derived from **percentile ranks of process metrics** (what the pitcher controls) is a better gauge of true talent.

### Formula

```
Quality Score = weighted_average(adjusted_percentiles)
```

Where:

| Component | Raw column | Direction | Weight |
|-----------|-----------|-----------|--------|
| Expected ERA | `pct_xera` | Higher = better | 30% |
| Strikeout rate | `pct_k_percent` | Higher = better | 20% |
| Whiff rate | `pct_whiff_percent` | Higher = better | 15% |
| Walk rate | `pct_bb_percent` | **Inverted** (100 − pct) | 15% |
| Hard hit rate | `pct_hard_hit_percent` | **Inverted** (100 − pct) | 10% |
| FB velocity | `pct_fb_velocity` | Higher = better | 10% |

A pitcher must have data covering at least 30% of total weight to receive a score (prevents scores from a single data point). Percentile ranks are provided by Baseball Savant and require a minimum PA threshold (~40 IP equivalent), so very-low-leverage relievers may not appear.

### Grade scale

| Score | Grade |
|-------|-------|
| ≥ 90 | A+ |
| ≥ 80 | A |
| ≥ 70 | B+ |
| ≥ 60 | B |
| ≥ 50 | C+ |
| ≥ 40 | C |
| ≥ 30 | D |
| < 30 | F |

---

## 6. ERA Regression Models

**File:** `src/backend/predict_actual_expected_residual.py`

### Problem statement

Given a pitcher's **current-season statistics**, predict their **ERA the following season**. This allows us to identify:
- **Buy-low candidates** — pitchers whose ERA greatly exceeds xERA (likely unlucky, expect improvement)
- **Sell-high candidates** — pitchers whose ERA greatly beats xERA (likely lucky, expect regression)
- **Next-season ERA projections** — forward-looking ERA estimates for all qualified pitchers

### Dataset construction (`build_regression_dataset`)

For each pitcher who appears in consecutive seasons (year N and year N+1) with xERA available in year N and ≥20 IP in year N+1:

- **Features (year N):** `xera`, `era_minus_xera`, `ERA`, `SO9`, `WHIP`, `brl_pa`, `pct_k_percent`, `pct_bb_percent`, `pct_hard_hit_percent`, `pct_whiff_percent`, `IP`
- **Target:** ERA in year N+1

Result: ~2,600 paired rows across 10 seasons (2015→2016 through 2024→2025).

### Models compared

| Model | Library | Notes |
|-------|---------|-------|
| Naïve baseline | — | Predict the training-set mean ERA for every pitcher |
| xERA only | — | Use current xERA directly as the prediction |
| Linear OLS | `sklearn.linear_model.LinearRegression` | Interpretable; preceded by `StandardScaler` |
| Ridge | `sklearn.linear_model.Ridge` | L2 regularisation; handles correlated predictors |
| Lasso | `sklearn.linear_model.Lasso` | L1 regularisation; automatic feature selection |
| Random Forest | `sklearn.ensemble.RandomForestRegressor` | 200 trees, max_depth=5 |
| Gradient Boost | `sklearn.ensemble.GradientBoostingRegressor` | 200 trees, depth=3, lr=0.05, subsample=0.8 |

### Evaluation

**Time-series cross-validation:** train on all seasons before the test season, evaluate on the test season. This is critical — using k-fold CV on this data would leak future information (a 2024 pitcher appearing in both train and test splits).

Test season: 2024 (train on 2015–2023).

**Results (2024 test):**

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Naïve mean | 1.366 | 1.075 | −0.003 |
| xERA only | 1.362 | 1.034 | 0.002 |
| Linear OLS | 1.278 | 1.020 | 0.122 |
| Ridge | 1.278 | 1.020 | 0.122 |
| Lasso | 1.283 | 1.015 | 0.115 |
| **Random Forest** | **1.275** | **1.018** | **0.126** |
| Gradient Boost | 1.298 | 1.034 | 0.093 |

**Interpretation:** RMSE ~1.28 means predictions are off by ~1.28 ERA points on average. ERA is genuinely noisy year-to-year; the known true correlation from academic literature is ~0.35–0.45 for a single prior season. Our models are within a normal range for this problem.

**Key finding from feature importances:** xERA (0.245) and K% percentile (0.225) dominate. The ERA−xERA luck gap alone explains very little variance year-over-year — regression toward xERA happens slowly and noisily, not in a single season.

### Regression candidates (`identify_regression_candidates`)

Applied to the **most recent season** only. Filters to pitchers with ≥40 IP and xERA data.

- **Buy low:** sorted by `ERA − xERA` descending (ERA most above xERA = most unlucky)
- **Sell high:** sorted by `ERA − xERA` ascending (ERA most below xERA = most lucky)

### Next-season projections (`predict_next_season_era`)

Trains the chosen model on **all available historical pairs** (not just up to a test season), then projects next-year ERA for all current-season pitchers with ≥40 IP and xERA. This uses all data to maximise the training set for the forward projection.

---

## 7. LLM Cluster Narratives

**File:** `src/backend/llm_summaries.py`

### Prompt construction (`build_cluster_stats_prompt`)

For each cluster, the prompt includes:
- Number of pitchers and handedness split
- Average pitch mix (proportions > 1%)
- Mean ± std for all numeric features in the feature list
- Example pitcher names (first 8 in the cluster)

### LLM providers

| Provider | Model | Cost | Setup |
|----------|-------|------|-------|
| OpenAI | `gpt-4o-mini` | ~$0.003/full run | `OPENAI_API_KEY` in `.streamlit/secrets.toml` |
| Groq | `llama-3.1-8b-instant` | Free tier | `GROQ_API_KEY` in `.streamlit/secrets.toml` |
| None | Rule-based | Free | No setup required |

Provider is selected in the Streamlit sidebar; API keys can be stored in `.streamlit/secrets.toml` (gitignored) or entered at runtime.

### Rule-based fallback (`_fallback_summary`)

When no LLM key is available, generates a structured narrative using:

1. **Velocity tier** classification (elite/plus/above-average/average/finesse)
2. **Spin tier** classification
3. **Relative pitch mix** — computes cluster-average `pct_{PT}` vs. league-average `pct_{PT}` across all pitchers in the profiles DataFrame. Highlights pitch types ≥5 percentage points above league average, e.g.: *"relies disproportionately on the sweeper (28%, +15pp vs. avg)"*. If no pitch stands out, describes the top two by volume.
4. **Extension** and **hard-hit rate** when available.

---

## 8. Text Insights Layer

**File:** `src/backend/text_insights.py`

A library of interpretation functions used throughout the app to convert raw numbers into contextual language.

### Threshold-based classifiers

```python
describe_velo(mph)   → "elite (97+ mph)" / "plus (95-97 mph)" / ...
describe_spin(rpm)   → "elite spin (2500+ rpm)" / ...
describe_era(era)    → "ace-caliber (sub-2.50)" / "frontline starter" / ...
describe_whip(whip)  → "elite (sub-1.00)" / "plus (1.00-1.15)" / ...
describe_so9(so9)    → "elite strikeout pitcher (11+ K/9)" / ...
describe_hard_hit(pct) → "soft contact allowed (<38%)" / ...
```

Thresholds are based on 2021–2024 MLB leaderboard distributions.

### `describe_xera_gap(era, xera)`

Interprets the luck component: gap < −0.75 = lucky (regression risk); gap > +0.75 = unlucky (improvement candidate). Returns a plain-English sentence.

### `build_pitcher_narrative(row, league_avg_pct=None)`

Given a pitcher row from the Statcast profiles DataFrame, produces a 2–4 sentence narrative covering:
- Velocity with tier label
- Pitch mix with **relative usage** vs. league average: `"four-seam fastball 52% (+14pp vs. avg), slider 24% (+6pp vs. avg)"`
- Spin rate with tier label
- Hard-hit rate with tier label

`league_avg_pct` is a dict `{pitch_type: mean_proportion}` computed from the full profiles DataFrame (mean of `pct_{PT}` across all pitchers). Passed in from the app to avoid recomputing per pitcher.

### `build_traditional_stats_narrative(row)`

Formats ERA + tier, WHIP + tier, K/9 + tier, plus the xERA gap sentence into a single paragraph.

### `build_trend_annotation(values, years, stat)`

Computes first-to-last delta for a multi-year stat sequence and returns a human-readable trend description (e.g., *"ERA has improved modestly (−0.43) since 2022."*).

---

## 9. Streamlit Dashboard Architecture

**File:** `src/frontend/app.py`

### Data loading

All data loading uses `@st.cache_data` decorators so files are read from disk only once per session:

```python
load_cluster_data()       → pitcher_profiles_clustered.parquet
load_summaries()          → summaries/cluster_summaries.json (HDBSCAN)
load_summaries_kmeans()   → summaries/cluster_summaries_kmeans.json
load_pitching_profiles()  → pitching_profiles_YYYY.parquet
load_pitching_trends()    → pitching_trends.parquet
load_regression_outputs() → 6 regression + quality parquet files
```

### Graceful degradation

The clustering tabs require `pitcher_profiles_clustered.parquet`, which is only generated after running the full pitch-by-pitch pipeline. If absent, those tabs show an info banner with run instructions. The Traditional Stats, Quality Scores, and Regression Predictor tabs work independently and only require the outputs of `preprocess_pitching_bref.py` and `predict_actual_expected_residual.py`.

### Tabs: Clustering and Rosters

- **Clustering**: Unified tab with algorithm selector (HDBSCAN, KMeans, or Agglomerative Ward if available). UMAP scatter with X/Y axis selectors, cluster sizes bar chart, pie chart, and cluster narratives. HDBSCAN may label pitchers as noise (−1); mitigations (min_samples=1, epsilon, cluster in original space, assign noise soft) are available in the sidebar.
- **Rosters & Performance**: Year selector, cluster method (HDBSCAN vs KMeans), performance table (ERA/WHIP/SO9 by cluster), expandable rosters per cluster.

### Tabs: Similarity, Raw Data

- **Similarity**: Pitcher search, 10 most similar by Euclidean distance, Compare Two Pitchers (side-by-side velo/spin/break/usage table).
- **Raw Data**: Full profiles dataframe. **Clustering Feature Matrix** (expander): exact features used for clustering; filter by cluster (including noise −1) or feature name; downloadable CSV. Useful for diagnosing why pitchers are outliers.

### Sidebar

- **LLM configuration**: provider (OpenAI / Groq / none), API key input, regenerate summaries button
- **Re-run Clustering** (year-level): cluster year(s) multiselect, KMeans k, HDBSCAN min cluster size, UMAP dimensions, HDBSCAN epsilon, cluster selection (leaf/eom), "Cluster in original space" checkbox, min_samples (1–5), "Assign noise to nearest cluster (soft)" checkbox
- **Season selector**: "All Years (2015–2025)" or individual season; controls the Traditional Stats tab

### Tab: Traditional & Expected Stats

Three display modes depending on input:
1. **Single pitcher match** (exact search) — stat card, percentile bar chart, year-over-year trend lines
2. **Multiple matches in all-years mode** — per-pitcher trend sparklines
3. **Leaderboard** (default) — sortable table + ERA vs xERA scatter

The ERA vs xERA scatter uses a diagonal reference line (ERA = xERA). Points above the line are unlucky; points below are lucky. Color encodes the ERA−xERA gap.

### Tab: Pitcher Quality Scores

Grade distribution summary (A+ through F counts), filterable leaderboard with `background_gradient` color coding, score distribution histogram, and methodology expander.

### Tab: Regression Predictor

1. **Model comparison table** (highlighted best RMSE/MAE/R²)
2. **Feature importance bar chart** (Random Forest)
3. **Next-Season ERA Projections** table (color-coded by projected ERA)
4. **Buy Low / Sell High** two-column layout with color-coded ERA−xERA gradients
5. **Methodology expander** explaining dataset, CV, and why R² is low

---

## 10. Pipeline Orchestration

**File:** `run_pipeline.py`

Combines all backend steps with CLI flags:

**Clustering (year-level):**
```
--cluster-years          Year(s) to cluster on (default: 2022 2023 2024)
--start / --end          Override Statcast date range (ignored if --cluster-years set)
--min-pitches            Minimum pitcher pitch count (default: 200)
--k                      KMeans clusters (default: 5)
--pitch-quality-k        Layer 1: clusters per pitch type when using two-layer (default: 3)
--no-two-layer           Use legacy single-layer clustering (pct_* + velo/spin/movement per pitch)
--umap-neighbors         UMAP n_neighbors (default: 30)
--umap-min-dist         UMAP min_dist (default: 0.1)
--umap-components       UMAP n_components (default: 6)
--hdbscan-min-cluster-size  HDBSCAN min_cluster_size (default: 5)
--hdbscan-min-samples    HDBSCAN min_samples (default: 1; smaller = less noise)
--hdbscan-cluster-selection  leaf or eom (default: leaf)
--hdbscan-epsilon        HDBSCAN cluster_selection_epsilon (default: 0.1)
--hdbscan-original-space  Cluster in scaled feature space instead of UMAP
--hdbscan-assign-noise-soft  Assign −1 to nearest cluster via soft clustering (0% noise)
--skip-clustering        Skip pitch-by-pitch pipeline
--skip-llm               Use rule-based fallback instead of LLM
```

**Pitching stats:**
```
--years                  Seasons for pitching stats (default: 2015-2025)
--min-ip                 Minimum IP for qualifying pitchers (default: 20)
--skip-pitching-stats    Skip traditional stats pipeline
--skip-regression        Skip regression / quality-score pipeline
```

### Step ordering

```
run_pipeline.py
  ├── [if not --skip-clustering]
  │   ├── preprocess_data.py::run_preprocessing()
  │   ├── clustering.py::run_clustering_pipeline()
  │   └── llm_summaries.py::generate_all_summaries()
  │
  └── [if not --skip-pitching-stats]
      └── preprocess_pitching_bref.py::save_pitching_profiles()
```

`predict_actual_expected_residual.py` is run independently (not part of `run_pipeline.py`) since it reads the output of `preprocess_pitching_bref.py`.

---

## 11. Data ID Crosswalk

### Player ID systems in use

| System | Used by | Column name |
|--------|---------|-------------|
| MLBAM ID | Baseball Savant, pybaseball, all Statcast files | `player_id` (Savant) / `pitcher` (pitch-by-pitch) / `mlbID` (bref stats) |
| Baseball Reference ID | Baseball Reference | `key_bbref` (chadwick) |

### Why no Chadwick join is needed for the primary pipeline

`pitching_stats_bref.mlbID` is already the MLBAM ID. Savant tables use `player_id` which is also MLBAM. The join is direct: `pitching_stats_bref.mlbID == statcast_*.player_id`.

Chadwick (`chadwick_register.parquet`) would be needed to join Baseball Reference data that uses the BBref string key (e.g., `verlaju01`) against Statcast data. It maps `key_bbref` ↔ `key_mlbam`.

---

## 12. Extending the System

### Batter analytics

All the infrastructure exists. Mirror `preprocess_pitching_bref.py` for batters:
- `batting_stats_bref/data_YYYY.parquet` — BA, OBP, SLG, HR, SB
- `statcast_batter_expected_stats/data_YYYY.parquet` — xBA, xSLG, xwOBA
- `statcast_batter_percentile_ranks/data_YYYY.parquet` — percentile ranks
- `statcast_batter_exitvelo_barrels/data_YYYY.parquet` — barrel%, max EV
- `statcast_sprint_speed/data_YYYY.parquet` — Sprint Speed (ft/s)

### Multi-season pitcher clustering

Run clustering year-by-year and track `cluster_hdbscan` labels across seasons per pitcher. A pitcher moving from a "finesse" cluster to a "power" cluster signals a breakout. Requires consistent cluster labelling across years (e.g., matching clusters by centroid distance).

### Daily Statcast scraper (`scrape_daily_data.py`)

Current state: stub. Implementation pattern:
```python
from pybaseball import statcast
import pandas as pd
from pathlib import Path

def download_daily_data(date: str):
    df = statcast(start_dt=date, end_dt=date)
    out = Path(f"data/statcast_pitches/game_date={date}/pitches.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
```

Schedule with `cron`, GitHub Actions, or Airflow. Run at ~6am ET after the previous day's games are ingested.

### Win/Loss simulation (`misc_simulations.py`)

`simulate_win_loss_multiple` already exists — runs N Monte Carlo seasons given mean/std of runs scored/allowed. Next step: add a Streamlit tab that takes team ERA inputs (pulling from the pitching leaderboard) and displays the projected record distribution.

### Regression model improvements

- Add **multiple prior seasons** as features (xERA_N, xERA_{N-1}, ERA_N, ERA_{N-1}) — improves stability
- Add **FIP** or **SIERA** as features if available
- Train a **quantile regression** to output prediction intervals (e.g., 80% confidence ERA range)
- Use **isotonic calibration** to ensure projected ERAs are well-calibrated at the tails
