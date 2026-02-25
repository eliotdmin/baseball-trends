"""
Pitcher Performance Preprocessing Pipeline
==========================================
Loads hive-partitioned Statcast pitch data, engineers pitcher-level feature
vectors for clustering. Each pitcher is summarized by their pitch arsenal
characteristics: velocity, spin, movement, release point, command, and
pitch mix proportions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
STATCAST_DIR = DATA_DIR / "statcast_pitches"
PROCESSED_DIR = DATA_DIR / "processed"

PITCH_FEATURES = [
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "plate_x",
    "plate_z",
    "spin_axis",
    "arm_angle",
    "effective_speed",
    "api_break_z_with_gravity",
    "api_break_x_arm",
]


def load_statcast(
    start_date: str = "2015-03-01",
    end_date: str = "2025-11-30",
    game_type: str = "R",
) -> pd.DataFrame:
    """Load hive-partitioned Statcast data for a date range.

    Reads parquet files from data/statcast_pitches/game_date=YYYY-MM-DD/.
    Filters to regular season by default.
    """
    dates = pd.date_range(start=start_date, end=end_date)
    frames = []

    for dt in dates:
        date_str = dt.strftime("%Y-%m-%d")
        parquet_path = STATCAST_DIR / f"game_date={date_str}" / "pitches.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            df["game_date"] = date_str
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No parquet files found between {start_date} and {end_date}"
        )

    data = pd.concat(frames, ignore_index=True)

    if game_type:
        data = data[data["game_type"] == game_type].reset_index(drop=True)

    print(f"Loaded {len(data):,} pitches across {data['game_date'].nunique()} game days")
    return data


def build_pitcher_profiles(
    df: pd.DataFrame, min_pitches: int = 200
) -> pd.DataFrame:
    """Build a single feature vector per pitcher for clustering.

    Features:
    - Arsenal-weighted velocity, spin, movement
    - Pitch mix proportions (top N pitch types)
    - Release point consistency
    - Command metrics (plate location spread)
    - Velocity variance (within-pitch-type)
    """
    pitch_counts = df.groupby("pitcher")["release_speed"].count()
    qualified = pitch_counts[pitch_counts >= min_pitches].index
    df = df[df["pitcher"].isin(qualified)].copy()

    pitcher_names = (
        df.groupby("pitcher")["player_name"].first().to_dict()
    )

    CANONICAL_TYPES = ["FF", "SI", "SL", "CH", "CU", "FC", "ST", "KC", "FS"]

    profiles = []
    for pid, grp in df.groupby("pitcher"):
        total = len(grp)
        row = {"pitcher": pid, "player_name": pitcher_names[pid], "n_pitches": total}

        for pt in CANONICAL_TYPES:
            pt_data = grp[grp["pitch_type"] == pt]
            pct = len(pt_data) / total
            row[f"pct_{pt}"] = pct

            if len(pt_data) >= 10:
                row[f"velo_{pt}"] = pt_data["release_speed"].mean()
                row[f"spin_{pt}"] = pt_data["release_spin_rate"].mean()
                row[f"pfx_x_{pt}"] = pt_data["pfx_x"].mean()
                row[f"pfx_z_{pt}"] = pt_data["pfx_z"].mean()
                if "spin_axis" in pt_data.columns and pt_data["spin_axis"].notna().any():
                    row[f"spin_axis_{pt}"] = pt_data["spin_axis"].mean()
                else:
                    row[f"spin_axis_{pt}"] = np.nan
                if "api_break_z_with_gravity" in pt_data.columns and pt_data["api_break_z_with_gravity"].notna().any():
                    row[f"break_z_{pt}"] = pt_data["api_break_z_with_gravity"].mean()
                else:
                    row[f"break_z_{pt}"] = np.nan
                if "api_break_x_arm" in pt_data.columns and pt_data["api_break_x_arm"].notna().any():
                    row[f"break_x_{pt}"] = pt_data["api_break_x_arm"].mean()
                else:
                    row[f"break_x_{pt}"] = np.nan
            else:
                row[f"velo_{pt}"] = np.nan
                row[f"spin_{pt}"] = np.nan
                row[f"pfx_x_{pt}"] = np.nan
                row[f"pfx_z_{pt}"] = np.nan
                row[f"spin_axis_{pt}"] = np.nan
                row[f"break_z_{pt}"] = np.nan
                row[f"break_x_{pt}"] = np.nan

        row["avg_velo"] = grp["release_speed"].mean()
        row["max_velo"] = grp["release_speed"].max()
        row["std_velo"] = grp["release_speed"].std()
        row["avg_spin"] = grp["release_spin_rate"].mean()
        row["avg_extension"] = grp["release_extension"].mean()
        row["avg_arm_angle"] = grp["arm_angle"].mean()
        row["avg_pfx_x"] = grp["pfx_x"].mean()
        row["avg_pfx_z"] = grp["pfx_z"].mean()

        row["release_x_mean"] = grp["release_pos_x"].mean()
        row["release_z_mean"] = grp["release_pos_z"].mean()
        row["release_x_std"] = grp["release_pos_x"].std()
        row["release_z_std"] = grp["release_pos_z"].std()

        row["cmd_plate_x_std"] = grp["plate_x"].std()
        row["cmd_plate_z_std"] = grp["plate_z"].std()

        row["n_pitch_types"] = grp["pitch_type"].nunique()

        row["p_throws"] = grp["p_throws"].mode().iloc[0] if len(grp) > 0 else "R"

        batted = grp.dropna(subset=["launch_speed", "launch_angle"])
        if len(batted) >= 20:
            row["avg_exit_velo_against"] = batted["launch_speed"].mean()
            row["avg_launch_angle_against"] = batted["launch_angle"].mean()
            row["hard_hit_pct"] = (batted["launch_speed"] >= 95).mean()
        else:
            row["avg_exit_velo_against"] = np.nan
            row["avg_launch_angle_against"] = np.nan
            row["hard_hit_pct"] = np.nan

        profiles.append(row)

    profiles_df = pd.DataFrame(profiles)
    print(
        f"Built profiles for {len(profiles_df)} pitchers "
        f"(min {min_pitches} pitches)"
    )
    return profiles_df


# Pitch types where velocity is the primary weapon → include per-pitch-type velo
FASTBALL_TYPES = {"FF", "SI", "FC"}
# Breaking balls / offspeed → velocity is derived from arm speed and not itself meaningful;
# what matters is spin, movement, and usage
BREAKING_TYPES = {"SL", "CH", "CU", "ST", "KC", "FS"}
CANONICAL_TYPES_SET = FASTBALL_TYPES | BREAKING_TYPES


def get_clustering_features(profiles: pd.DataFrame) -> list[str]:
    """Return features for clustering and similarity. Same as get_similarity_features."""
    return get_similarity_features(profiles)


def get_similarity_features(profiles: pd.DataFrame) -> list[str]:
    """
    Pitch-type-aware arsenal features: pct_*, velo_*, spin_*, movement, break per pitch type,
    plus extension, arm angle, command, n_pitch_types, hard_hit_pct.
    """
    wanted: list[str] = []

    # Pitch mix — defines arsenal identity
    for pt in sorted(CANONICAL_TYPES_SET):
        wanted.append(f"pct_{pt}")

    # Per-pitch-type velocity for ALL types (not just fastballs)
    for pt in sorted(CANONICAL_TYPES_SET):
        wanted.append(f"velo_{pt}")

    # Per-pitch-type spin, movement, break, and spin axis
    for pt in sorted(CANONICAL_TYPES_SET):
        for stat in ("spin", "pfx_x", "pfx_z"):
            wanted.append(f"{stat}_{pt}")
        wanted.append(f"spin_axis_{pt}")
        wanted.append(f"break_z_{pt}")
        wanted.append(f"break_x_{pt}")

    # Whole-arsenal stats that are NOT pitch-blended
    wanted += [
        "avg_extension",
        "avg_arm_angle",
        "cmd_plate_x_std",
        "cmd_plate_z_std",
        "n_pitch_types",
        "hard_hit_pct",
    ]

    return [c for c in wanted if c in profiles.columns]


PITCH_USAGE_THRESHOLD = 0.01  # pct_X below this = don't throw; no imputation for those pitch chars


def _feature_to_pitch_type(feat: str) -> Optional[str]:
    """Return pitch type for pitch-specific feature (velo_CH, pct_FF -> CH, FF), else None."""
    for pt in CANONICAL_TYPES_SET:
        if feat in (f"pct_{pt}",):
            return pt
        for stem in ("velo", "spin", "pfx_x", "pfx_z", "spin_axis", "break_z", "break_x"):
            if feat == f"{stem}_{pt}":
                return pt
    return None


def prepare_clustering_matrix(
    profiles: pd.DataFrame,
    features: Optional[list[str]] = None,
    fill_strategy: str = "median",
    pitch_usage_threshold: float = PITCH_USAGE_THRESHOLD,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Prepare the feature matrix for clustering.

    When pct_X < threshold, pitch characteristics (velo_X, spin_X, etc.) are set to 0
    rather than median-imputed — avoids inventing fake characteristics. These dimensions
    are downweighted in the weighted distance (weight = min(pct_X_i, pct_X_j)).
    """
    from sklearn.preprocessing import StandardScaler

    if features is None:
        features = get_clustering_features(profiles)

    X = profiles[features].copy()

    # For pitch-specific characteristic cols: no imputation when pitcher doesn't throw
    for pt in CANONICAL_TYPES_SET:
        pct_col = f"pct_{pt}"
        if pct_col not in X.columns:
            continue
        low_usage = (profiles[pct_col].fillna(0) < pitch_usage_threshold).values
        for stem in ("velo", "spin", "pfx_x", "pfx_z", "spin_axis", "break_z", "break_x"):
            col = f"{stem}_{pt}"
            if col in X.columns:
                X.loc[low_usage, col] = 0  # sentinel; weighted dist will ignore (weight=0)

    # Median-fill any remaining NaN (e.g. avg_extension, cmd_*)
    if fill_strategy == "median":
        X = X.fillna(X.median())
    elif fill_strategy == "zero":
        X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, features, profiles


def compute_weighted_distance_matrix(
    profiles: pd.DataFrame,
    X_scaled: np.ndarray,
    feature_names: list[str],
    pitch_usage_threshold: float = PITCH_USAGE_THRESHOLD,
) -> np.ndarray:
    """Pairwise weighted Euclidean distance. Pitch-type characteristic dimensions use
    weight = min(pct_X_i, pct_X_j) so pitches a pitcher doesn't throw don't contribute.
    """
    n = len(profiles)
    weights = np.ones((n, len(feature_names)))

    for j, feat in enumerate(feature_names):
        pt = _feature_to_pitch_type(feat)
        if pt is not None and feat != f"pct_{pt}":
            pct_col = f"pct_{pt}"
            pct = profiles[pct_col].fillna(0).values if pct_col in profiles.columns else np.zeros(n)
            weights[:, j] = pct

    # d_ij = sqrt(sum_d min(w_id,w_jd) * (x_id - x_jd)^2)
    w_min = np.minimum(weights[:, np.newaxis, :], weights[np.newaxis, :, :])
    diff = X_scaled[:, np.newaxis, :] - X_scaled[np.newaxis, :, :]
    D = np.sqrt(np.sum(w_min * diff**2, axis=2).astype(float))
    return D


def compute_weighted_distances_from_row(
    profiles: pd.DataFrame,
    X_scaled: np.ndarray,
    feature_names: list[str],
    row_idx: int,
    pitch_usage_threshold: float = PITCH_USAGE_THRESHOLD,
) -> np.ndarray:
    """Distances from row_idx to all rows (for similarity search without full matrix)."""
    n = X_scaled.shape[0]
    weights = np.ones((n, len(feature_names)))

    for j, feat in enumerate(feature_names):
        pt = _feature_to_pitch_type(feat)
        if pt is not None and feat != f"pct_{pt}":
            pct_col = f"pct_{pt}"
            pct = profiles[pct_col].fillna(0).values if pct_col in profiles.columns else np.zeros(n)
            weights[:, j] = pct

    w_min = np.minimum(weights[row_idx : row_idx + 1], weights)
    diff = X_scaled[row_idx : row_idx + 1] - X_scaled
    return np.sqrt(np.sum(w_min * diff**2, axis=1).astype(float))


def run_preprocessing(
    years: list[int] | int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_pitches: int = 200,
) -> pd.DataFrame:
    """End-to-end: load data -> build profiles -> save.

    Either years or (start_date, end_date) must be provided.
    years: e.g. [2024] or [2023, 2024] — uses full season(s) for each year.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if years is not None:
        if isinstance(years, int):
            years = [years]
        start_date = f"{min(years)}-03-01"
        end_date = f"{max(years)}-11-30"
        print(f"Clustering years: {years} → {start_date} to {end_date}")
    elif start_date is None or end_date is None:
        start_date = start_date or "2015-03-01"
        end_date = end_date or "2025-11-30"

    print("Loading Statcast data...")
    data = load_statcast(start_date, end_date)

    print("Building pitcher profiles...")
    profiles = build_pitcher_profiles(data, min_pitches=min_pitches)
    # Add year for merging with outcomes (use end of date range)
    end_year = int(end_date[:4])
    profiles["year"] = end_year

    out_path = PROCESSED_DIR / "pitcher_profiles.parquet"
    profiles.to_parquet(out_path, index=False)
    print(f"Saved pitcher profiles to {out_path}")

    return profiles


if __name__ == "__main__":
    profiles = run_preprocessing()
    print(profiles.head())
    print(f"\nFeature columns: {get_clustering_features(profiles)}")
