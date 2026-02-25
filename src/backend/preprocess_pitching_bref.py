"""
Pitching Stats Preprocessing
==============================
Loads and joins:
  - Baseball Reference traditional stats (ERA, WHIP, SO9, IP, BB, K, HR)
  - Savant pitcher expected stats (xERA, xwOBA — actual vs expected)
  - Savant pitcher percentile ranks (0-100 vs league)
  - Savant pitcher exit velo / barrel data (contact quality allowed)

Join key: pitching_stats_bref.mlbID == statcast tables.player_id (MLBAM ID)

The chadwick_register.parquet (key_mlbam / key_bbref) is also loaded
for name normalisation but is not required for the primary join.

Outputs
-------
data/processed/pitching_profiles_YYYY.parquet  — single-season snapshot
data/processed/pitching_trends.parquet          — multi-season long table
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_DIR / "processed"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_pitching_bref(years: list[int]) -> pd.DataFrame:
    """Load Baseball Reference traditional pitching stats for given years."""
    frames = []
    for yr in years:
        path = DATA_DIR / "pitching_stats_bref" / f"data_{yr}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["season"] = yr
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No pitching_stats_bref data found for {years}")

    combined = pd.concat(frames, ignore_index=True)

    # Keep numeric mlbID only; some rows have NaN (minor leaguers, etc.)
    combined = combined.dropna(subset=["mlbID"])
    combined["mlbID"] = combined["mlbID"].astype(int)

    # Coerce key numerics
    for col in ["ERA", "WHIP", "SO9", "IP", "BB", "SO", "HR", "BF", "G", "GS"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Minimum innings pitched filter
    combined = combined[combined["IP"].fillna(0) >= 1]

    return combined


def load_statcast_expected(years: list[int]) -> pd.DataFrame:
    """Load Savant pitcher expected stats (xERA, ERA, xwOBA) for given years."""
    frames = []
    for yr in years:
        path = DATA_DIR / "statcast_pitcher_expected_stats" / f"data_{yr}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "year" not in df.columns:
            df["year"] = yr
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={
        "player_id": "mlbID",
        "era": "savant_era",
        "xera": "xera",
        "era_minus_xera_diff": "era_minus_xera",
        "woba": "savant_woba",
        "est_woba": "est_woba",
    })
    combined["mlbID"] = combined["mlbID"].astype(int)
    return combined


def load_statcast_percentiles(years: list[int]) -> pd.DataFrame:
    """Load Savant percentile rank table (0-100 per stat vs league)."""
    frames = []
    for yr in years:
        path = DATA_DIR / "statcast_pitcher_percentile_ranks" / f"data_{yr}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "year" not in df.columns:
            df["year"] = yr
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"player_id": "mlbID"})
    combined["mlbID"] = combined["mlbID"].astype(int)
    # Prefix percentile columns to avoid collision
    pct_cols = [c for c in combined.columns if c not in ("mlbID", "player_name", "year")]
    combined = combined.rename(columns={c: f"pct_{c}" for c in pct_cols})
    return combined


def load_statcast_exitvelo(years: list[int]) -> pd.DataFrame:
    """Load Savant pitcher exit velo / barrel data."""
    frames = []
    for yr in years:
        path = DATA_DIR / "statcast_pitcher_exitvelo_barrels" / f"data_{yr}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["year"] = yr
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"player_id": "mlbID"})
    combined["mlbID"] = combined["mlbID"].astype(int)
    return combined


# ---------------------------------------------------------------------------
# Join & build profiles
# ---------------------------------------------------------------------------

def build_pitching_profiles(
    years: list[int],
    min_ip: float = 20.0,
) -> pd.DataFrame:
    """
    Build enriched pitcher profiles by joining all four data sources.
    Returns one row per (pitcher, season).
    """
    bref = load_pitching_bref(years)
    expected = load_statcast_expected(years)
    percentiles = load_statcast_percentiles(years)
    exitvelo = load_statcast_exitvelo(years)

    # Start with bref as base
    df = bref.copy()
    df = df.rename(columns={"season": "year"})

    # Merge expected stats
    if not expected.empty:
        keep_exp = ["mlbID", "year", "xera", "era_minus_xera",
                    "savant_woba", "est_woba", "savant_era"]
        keep_exp = [c for c in keep_exp if c in expected.columns]
        df = df.merge(expected[keep_exp], on=["mlbID", "year"], how="left")

    # Merge percentiles
    if not percentiles.empty:
        pct_cols = ["mlbID", "year"] + [c for c in percentiles.columns
                                          if c.startswith("pct_") and c not in ("pct_player_name",)]
        df = df.merge(percentiles[pct_cols], on=["mlbID", "year"], how="left")

    # Merge exit velo
    if not exitvelo.empty:
        ev_cols = ["mlbID", "year", "avg_hit_speed", "brl_percent", "brl_pa",
                   "ev95percent", "anglesweetspotpercent"]
        ev_cols = [c for c in ev_cols if c in exitvelo.columns]
        df = df.merge(exitvelo[ev_cols], on=["mlbID", "year"], how="left")

    # Apply IP filter
    df = df[df["IP"].fillna(0) >= min_ip].reset_index(drop=True)

    return df


def build_pitching_trends(
    years: list[int],
    min_ip_per_season: float = 20.0,
) -> pd.DataFrame:
    """Long-format table of traditional stats per (pitcher, season) — for trend charts."""
    bref = load_pitching_bref(years)
    bref = bref.rename(columns={"season": "year"})
    bref = bref[bref["IP"].fillna(0) >= min_ip_per_season].reset_index(drop=True)
    return bref


def save_pitching_profiles(years: list[int], min_ip: float = 20.0) -> pd.DataFrame:
    """Build and persist the pitching profiles to processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = build_pitching_profiles(years, min_ip=min_ip)

    # Single most recent year snapshot
    most_recent = max(years)
    snapshot = df[df["year"] == most_recent].copy()
    snap_path = PROCESSED_DIR / f"pitching_profiles_{most_recent}.parquet"
    snapshot.to_parquet(snap_path, index=False)
    print(f"Saved {len(snapshot)} pitcher profiles ({most_recent}) to {snap_path}")

    # Full multi-year trends table
    trends_path = PROCESSED_DIR / "pitching_trends.parquet"
    df.to_parquet(trends_path, index=False)
    print(f"Saved {len(df)} pitcher-season rows to {trends_path}")

    return df


# ---------------------------------------------------------------------------
# Convenience: load latest available
# ---------------------------------------------------------------------------

def load_pitching_profiles(year: Optional[int] = None) -> pd.DataFrame:
    """Load the most recent processed pitching profiles parquet."""
    if year:
        path = PROCESSED_DIR / f"pitching_profiles_{year}.parquet"
        if path.exists():
            return pd.read_parquet(path)

    # Fall back: find most recent
    candidates = sorted(PROCESSED_DIR.glob("pitching_profiles_*.parquet"))
    if candidates:
        return pd.read_parquet(candidates[-1])

    raise FileNotFoundError(
        "No processed pitching profiles found. "
        "Run: python src/backend/preprocess_pitching_bref.py"
    )


def load_pitching_trends() -> pd.DataFrame:
    path = PROCESSED_DIR / "pitching_trends.parquet"
    if path.exists():
        return pd.read_parquet(path)
    raise FileNotFoundError(
        "No pitching trends file found. "
        "Run: python src/backend/preprocess_pitching_bref.py"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build pitching profiles from bref + statcast data")
    parser.add_argument("--years", nargs="+", type=int,
                        default=list(range(2015, 2026)),
                        help="Seasons to include (default: 2015-2025)")
    parser.add_argument("--min-ip", type=float, default=20.0,
                        help="Minimum IP to qualify (default: 20)")
    args = parser.parse_args()

    print(f"Building pitching profiles for {args.years}, min IP={args.min_ip}")
    df = save_pitching_profiles(args.years, min_ip=args.min_ip)

    print(f"\nSample output ({df['year'].max()} season):")
    latest = df[df["year"] == df["year"].max()]
    display_cols = ["Name", "Tm", "ERA", "WHIP", "SO9", "IP", "xera", "era_minus_xera"]
    display_cols = [c for c in display_cols if c in latest.columns]
    print(latest[display_cols].sort_values("ERA").head(15).to_string(index=False))
