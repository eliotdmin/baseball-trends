"""
Arsenal → Outcome Predictive Model
===================================
Predicts pitcher performance (xERA, ERA) from Statcast arsenal features.
Uses Random Forest for interpretable feature importances.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_DIR / "processed"

DEFAULT_OUTCOME = "xera"  # xERA is process-based; ERA has more noise


def build_arsenal_outcome_dataset(
    profiles: pd.DataFrame,
    trends: pd.DataFrame,
    outcome_col: str = "xera",
    min_ip: float = 20.0,
) -> pd.DataFrame:
    """
    Merge Statcast pitcher profiles (arsenal features) with outcome (xERA, ERA).
    profiles: pitcher, player_name, pct_*, velo_*, spin_*, ...
    trends: mlbID, Name, year, ERA, xera, IP, ...
    """
    from preprocess_data import get_similarity_features

    # profiles uses 'pitcher' (MLBAM ID); trends uses 'mlbID'
    profiles = profiles.copy()
    profiles["mlbID"] = pd.to_numeric(profiles["pitcher"], errors="coerce")
    profiles = profiles.dropna(subset=["mlbID"])
    profiles["mlbID"] = profiles["mlbID"].astype(int)

    # Use most recent year if profiles has year; else assume trends year
    if "year" in profiles.columns:
        primary_year = int(profiles["year"].max())
    else:
        primary_year = int(trends["year"].max())
        profiles["year"] = primary_year

    trends_yr = trends[
        (trends["year"] == primary_year) &
        (trends["IP"].fillna(0) >= min_ip)
    ].copy()

    if outcome_col not in trends_yr.columns:
        # Handle lowercase col names
        outcome_col = outcome_col.lower()
        if outcome_col not in trends_yr.columns:
            raise ValueError(f"Outcome '{outcome_col}' not in trends. Available: {list(trends_yr.columns)}")

    merged = profiles.merge(
        trends_yr[["mlbID", "year", outcome_col, "ERA", "IP"]],
        on=["mlbID", "year"],
        how="inner",
    )

    merged = merged.dropna(subset=[outcome_col])
    return merged


# Outcome-related stats (hard_hit_pct, etc.) should not be features — we want arsenal → outcome, not outcome → outcome
ARSENAL_PREDICTOR_EXCLUDE = {"hard_hit_pct"}


def train_arsenal_model(
    df: pd.DataFrame,
    outcome_col: str = "xera",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train Random Forest: arsenal features → outcome.
    Returns model, scaler, feature_names, importances, metrics.
    Excludes hard_hit_pct (outcome) so we measure purely arsenal → expected performance.
    """
    from preprocess_data import get_similarity_features

    features = get_similarity_features(df)
    features = [c for c in features if c in df.columns and c not in ARSENAL_PREDICTOR_EXCLUDE]

    X = df[features].fillna(df[features].median())
    y = df[outcome_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=random_state,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    importances = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": features,
        "importances": importances,
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def run_arsenal_predictor(
    profiles: pd.DataFrame | None = None,
    outcome_col: str = "xera",
    min_ip: float = 20.0,
) -> dict:
    """
    End-to-end: load data, merge, train, save importances.
    """
    if profiles is None:
        profiles = pd.read_parquet(PROCESSED_DIR / "pitcher_profiles.parquet")
    trends = pd.read_parquet(PROCESSED_DIR / "pitching_trends.parquet")

    df = build_arsenal_outcome_dataset(
        profiles, trends, outcome_col=outcome_col, min_ip=min_ip
    )

    if len(df) < 50:
        raise ValueError(f"Only {len(df)} pitcher-season rows after merge. Need pitching_trends and profiles for same year.")

    result = train_arsenal_model(df, outcome_col=outcome_col)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result["importances"].to_parquet(
        PROCESSED_DIR / "arsenal_feature_importances.parquet",
        index=False,
    )
    result["merged_df"] = df
    print(f"Arsenal → {outcome_col}: RMSE={result['rmse']:.3f}, MAE={result['mae']:.3f}, R²={result['r2']:.3f}")
    print(f"Top 5 features: {list(result['importances']['feature'].head(5))}")
    return result
