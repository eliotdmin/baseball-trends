"""
Multi-Target ERA / wOBA / BA Regression Models
================================================
Two prediction groups:

Group A — Absolute performance next season
  ERA, wOBA against, BA against

Group B — Luck / residual next season (central question)
  ERA−xERA, wOBA−est.wOBA, BA−est.BA
  These are actual − expected; positive = pitcher underperformed (unlucky).
  Key question: can we predict luck? Answer: mostly no — residuals reset year-over-year.

Source data
-----------
 pitching_trends.parquet, statcast_pitcher_expected_stats

Models (consolidated)
---------------------
  Naïve (mean), Linear OLS, Random Forest
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# ---------------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------------

TARGETS = [
    # Group A: predict the raw statistic next season
    dict(key="ERA",        col="era",         label="ERA",             group="A",
         baseline_col="xera",      baseline_label="xERA",      lower_better=True),
    dict(key="wOBA",       col="woba",        label="wOBA against",    group="A",
         baseline_col="est_woba",  baseline_label="est.wOBA",  lower_better=True),
    dict(key="BA",         col="ba",          label="BA against",      group="A",
         baseline_col="est_ba",    baseline_label="est.BA",    lower_better=True),
    # Group B: predict the residual (actual − expected) next season
    dict(key="ERA_resid",  col="era_vs_xera",   label="ERA − xERA",      group="B",
         baseline_col=None, baseline_label=None, lower_better=True),
    dict(key="wOBA_resid", col="woba_vs_est",   label="wOBA − est.wOBA", group="B",
         baseline_col=None, baseline_label=None, lower_better=True),
    dict(key="BA_resid",   col="ba_vs_est",     label="BA − est.BA",     group="B",
         baseline_col=None, baseline_label=None, lower_better=True),
]

FEATURE_COLS = [
    # Expected-stat anchors — strongest single predictors
    "xera", "est_woba", "est_ba",
    # Current-season residuals — how lucky was the pitcher?
    "era_vs_xera", "woba_vs_est", "ba_vs_est",
    # Current-season actuals
    "era", "woba", "ba",
    # Skill indicators (durable, repeatable)
    "SO9", "WHIP", "IP",
    # Savant percentile ranks
    "pct_k_percent", "pct_bb_percent", "pct_whiff_percent",
    "pct_hard_hit_percent", "pct_xera",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_expected_stats(years: list[int]) -> pd.DataFrame:
    """Load Savant pitcher expected stats with full BA/wOBA/ERA columns."""
    frames = []
    for yr in years:
        p = DATA_DIR / "statcast_pitcher_expected_stats" / f"data_{yr}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["year"] = yr
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"player_id": "mlbID"})
    combined["mlbID"] = combined["mlbID"].astype(int)
    return combined


def build_comprehensive_dataset(
    years: list[int],
    min_ip: float = 30.0,
    min_ip_next: float = 20.0,
) -> pd.DataFrame:
    """
    Join pitching_trends with full expected stats, compute residuals,
    and create year-over-year pairs for all 6 targets.
    """
    trends = pd.read_parquet(PROCESSED_DIR / "pitching_trends.parquet")
    trends = trends.rename(columns={"year": "season"})

    exp = load_expected_stats(years)
    if exp.empty:
        raise FileNotFoundError("No expected stats data found. Run preprocess_pitching_bref.py first.")
    exp = exp.rename(columns={"year": "season"})

    # Consistent residual columns: actual − expected (positive = unlucky)
    exp["era_vs_xera"]  = exp["era_minus_xera_diff"]          # era − xera
    exp["ba_vs_est"]    = -(exp["est_ba_minus_ba_diff"])       # ba − est_ba
    exp["woba_vs_est"]  = -(exp["est_woba_minus_woba_diff"])   # woba − est_woba

    exp_cols = ["mlbID", "season", "ba", "est_ba", "ba_vs_est",
                "woba", "est_woba", "woba_vs_est",
                "era", "xera", "era_vs_xera"]
    exp_sub = exp[[c for c in exp_cols if c in exp.columns]].copy()

    # Merge expected stats into trends
    merged = trends.merge(exp_sub, on=["mlbID", "season"], how="left")

    # Filter by IP
    merged = merged[merged["IP"].fillna(0) >= min_ip].copy()

    # Build lookup: (mlbID, season+1) → values for all targets
    next_cols = {t["col"]: t["col"] + "_next" for t in TARGETS}
    all_target_cols = list(next_cols.keys())

    lookup_df = merged[["mlbID", "season"] + [c for c in all_target_cols if c in merged.columns]].copy()
    lookup_df = lookup_df.rename(columns={c: c + "_next" for c in all_target_cols})
    lookup_df["season"] = lookup_df["season"] - 1  # shift: these become "next year" values

    # Left join next-year targets
    paired = merged.merge(lookup_df, on=["mlbID", "season"], how="left")

    # Drop rows where all Group-A targets are missing
    group_a_next = ["era_next", "woba_next", "ba_next"]
    group_a_next = [c for c in group_a_next if c in paired.columns]
    paired = paired.dropna(subset=group_a_next[:1])  # require at least ERA_next

    return paired


# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------

def _feature_matrix(df: pd.DataFrame, feature_cols: list[str]):
    """Select available features and median-fill NaN."""
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy().fillna(df[available].median())
    return X.values, available


def _make_models():
    """Consolidated: Naïve baseline, Linear (interpretable), RF (best performer)."""
    return {
        "Naïve (mean)":    None,
        "Linear OLS":      Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
        "Random Forest":   RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1),
    }


MIN_TRAIN_SEASONS = 3  # Minimum seasons for first walk-forward fold


def _evaluate_single_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    models: dict,
) -> dict:
    """Evaluate all models on one train/test fold. Returns {model_name: {RMSE, MAE, R²}}."""
    X_train, feats = _feature_matrix(train, feature_cols)
    X_test, _ = _feature_matrix(test, feats)
    y_train = train[target_col].values
    y_test = test[target_col].values

    fold_results = {}
    for name, model in models.items():
        if name == "Naïve (mean)":
            preds = np.full(len(y_test), y_train.mean())
        else:
            m = clone(model)  # Fresh model per fold
            m.fit(X_train, y_train)
            preds = m.predict(X_test)

        fold_results[name] = {
            "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
            "MAE": float(mean_absolute_error(y_test, preds)),
            "R²": float(r2_score(y_test, preds)),
        }
    return fold_results, feats


def evaluate_target(
    dataset: pd.DataFrame,
    target: dict,
    test_season: Optional[int] = None,
    feature_cols: list[str] = FEATURE_COLS,
) -> dict:
    """
    Walk-forward validation: expanding window, each year (after min train seasons) is a test fold.
    Aggregates mean ± std across folds. For feature importance, fits one full-data model (best type).
    """
    target_col = target["col"] + "_next"
    if target_col not in dataset.columns:
        return {}

    seasons = sorted(dataset["season"].dropna().unique().astype(int))
    if len(seasons) < MIN_TRAIN_SEASONS + 1:
        return {}

    models = _make_models()
    model_names = [n for n in models if n != "Naïve (mean)"]

    # Walk-forward: test on seasons min_train+1 .. max
    first_test_season = int(seasons[MIN_TRAIN_SEASONS - 1]) + 1
    test_seasons = [s for s in seasons if s >= first_test_season]

    if not test_seasons:
        return {}

    all_folds = []
    for ts in test_seasons:
        train = dataset[(dataset["season"] < ts) & dataset[target_col].notna()].copy()
        test = dataset[(dataset["season"] == ts) & dataset[target_col].notna()].copy()
        if len(train) < 20 or len(test) < 5:
            continue
        fold_res, feats = _evaluate_single_fold(train, test, target_col, feature_cols, models)
        all_folds.append(fold_res)

    if not all_folds:
        return {}

    # Aggregate: mean ± std per model
    results = {}
    for name in model_names:
        rmse_vals = [f[name]["RMSE"] for f in all_folds]
        mae_vals = [f[name]["MAE"] for f in all_folds]
        r2_vals = [f[name]["R²"] for f in all_folds]
        results[name] = {
            "RMSE": round(float(np.mean(rmse_vals)), 4),
            "RMSE_std": round(float(np.std(rmse_vals)), 4),
            "MAE": round(float(np.mean(mae_vals)), 4),
            "MAE_std": round(float(np.std(mae_vals)), 4),
            "R²": round(float(np.mean(r2_vals)), 4),
            "R²_std": round(float(np.std(r2_vals)), 4),
            "_n_folds": len(all_folds),
        }

    # Naïve
    naive_rmse = [f["Naïve (mean)"]["RMSE"] for f in all_folds]
    results["Naïve (mean)"] = {
        "RMSE": round(float(np.mean(naive_rmse)), 4),
        "RMSE_std": round(float(np.std(naive_rmse)), 4),
        "_n_folds": len(all_folds),
    }

    # Best model (lowest mean RMSE, excluding naïve)
    non_naive = {k: v for k, v in results.items() if k != "Naïve (mean)"}
    best_name = min(non_naive, key=lambda k: non_naive[k]["RMSE"])
    results["_best"] = best_name
    results["_feats"] = feats
    results["_n_folds"] = len(all_folds)

    # Full-data model for feature importance & projections (canonical, avoids fold-to-fold variation)
    train_full = dataset[dataset[target_col].notna()].copy()
    if len(train_full) >= 20:
        X_full, feats_full = _feature_matrix(train_full, feature_cols)
        y_full = train_full[target_col].values
        best_model = _make_models().get(best_name)
        if best_model is not None:
            best_model.fit(X_full, y_full)
            results["_full_model"] = best_model
            results["_full_feats"] = feats_full

    return results


def get_feature_importances(model_result: dict, model_name: Optional[str] = None) -> pd.DataFrame:
    """
    Extract feature importances. Uses _full_model (canonical full-data fit) when available,
    so importances are stable across folds and match the model used for projections.
    """
    model_name = model_name or model_result.get("_best", "Random Forest")
    m = model_result.get("_full_model")
    feats = model_result.get("_full_feats") or model_result.get("_feats", [])
    if m is None:
        res = model_result.get(model_name, {})
        m = res.get("_model")
    if m is None or not feats:
        return pd.DataFrame()
    inner = getattr(m, "named_steps", {}).get("m", m)
    if not hasattr(inner, "feature_importances_"):
        return pd.DataFrame()
    return (
        pd.DataFrame({"feature": feats, "importance": inner.feature_importances_})
        .sort_values("importance", ascending=False).reset_index(drop=True)
    )


def evaluate_all_targets(
    dataset: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
) -> dict:
    """Evaluate all 6 targets with walk-forward CV. Returns {target_key: model_results}."""
    all_results = {}
    for t in TARGETS:
        res = evaluate_target(dataset, t, feature_cols=feature_cols)
        if res:
            all_results[t["key"]] = res
            best = res.get("_best", "?")
            rmse = res.get(best, {}).get("RMSE", "?")
            std = res.get(best, {}).get("RMSE_std", "")
            nf = res.get("_n_folds", "")
            print(f"  {t['label']:25s} best={best:15s}  RMSE={rmse} ± {std}  ({nf} folds)")
    return all_results


# ---------------------------------------------------------------------------
# Projections & candidates
# ---------------------------------------------------------------------------

def project_next_season(
    current: pd.DataFrame,
    dataset: pd.DataFrame,
    target: dict,
    model_name: str = "Random Forest",
    min_ip: float = 30.0,
    feature_cols: list[str] = FEATURE_COLS,
) -> pd.DataFrame:
    """Train on all historical data, project next-season value for current pitchers."""
    target_col = target["col"] + "_next"
    train = dataset[dataset[target_col].notna()].copy()
    if len(train) < 20:
        return pd.DataFrame()

    X_train, feats = _feature_matrix(train, feature_cols)
    y_train = train[target_col].values

    model = _make_models().get(model_name)
    if model is None:
        return pd.DataFrame()
    model.fit(X_train, y_train)

    cur = current[current["IP"].fillna(0) >= min_ip].copy()
    # Backfill era/xera from ERA/xERA when statcast merge omitted them (e.g. injured pitchers)
    if target["col"] == "era" and "era" not in cur.columns and "ERA" in cur.columns:
        cur["era"] = cur["ERA"]
    X_pred, _ = _feature_matrix(cur, feats)
    cur[f"proj_{target['col']}"] = model.predict(X_pred)

    display = ["Name", "Tm", "IP", target["col"], f"proj_{target['col']}"]
    if target.get("baseline_col") and target["baseline_col"] in cur.columns:
        display.insert(4, target["baseline_col"])
    display = [c for c in display if c in cur.columns]

    sort_asc = target.get("lower_better", True)
    return cur[display].sort_values(f"proj_{target['col']}", ascending=sort_asc).reset_index(drop=True)


def identify_candidates(
    current: pd.DataFrame,
    target: dict,
    min_ip: float = 40.0,
    top_n: int = 20,
) -> dict:
    """
    For residual targets: split into 'lucky' (residual < 0, likely to regress)
    and 'unlucky' (residual > 0, likely to improve).
    """
    col = target["col"]
    if col not in current.columns:
        return {}

    df = current.dropna(subset=[col])
    df = df[df["IP"].fillna(0) >= min_ip].copy()

    show = ["Name", "Tm", "IP"] + [c for c in [target.get("baseline_col"), col, "WHIP", "SO9"]
                                    if c and c in df.columns]

    unlucky = df[df[col] > 0].sort_values(col, ascending=False).head(top_n)[show].reset_index(drop=True)
    lucky   = df[df[col] < 0].sort_values(col, ascending=True).head(top_n)[show].reset_index(drop=True)
    return {"unlucky": unlucky, "lucky": lucky}


# ---------------------------------------------------------------------------
# Summary comparison table
# ---------------------------------------------------------------------------

def collect_pred_vs_actual(
    dataset: pd.DataFrame,
    all_results: dict,
    feature_cols: list[str] = FEATURE_COLS,
) -> dict[str, pd.DataFrame]:
    """
    Collect (actual, predicted) for each Group B target across all walk-forward folds.
    Used for predicted-vs-actual scatter plots to show predictive power (or lack thereof).
    Returns {target_key: DataFrame with cols [actual, predicted, Name, season, mlbID]}.
    """
    out = {}
    seasons = sorted(dataset["season"].dropna().unique().astype(int))
    if len(seasons) < MIN_TRAIN_SEASONS + 1:
        return out

    first_test = int(seasons[MIN_TRAIN_SEASONS - 1]) + 1
    test_seasons = [s for s in seasons if s >= first_test]
    models = _make_models()

    for t in TARGETS:
        if t["group"] != "B":
            continue
        target_col = t["col"] + "_next"
        if target_col not in dataset.columns:
            continue
        res = all_results.get(t["key"], {})
        best_name = res.get("_best", "Linear OLS")
        best_model = models.get(best_name)
        if best_model is None:
            continue

        rows = []
        for ts in test_seasons:
            train = dataset[(dataset["season"] < ts) & dataset[target_col].notna()].copy()
            test = dataset[(dataset["season"] == ts) & dataset[target_col].notna()].copy()
            if len(train) < 20 or len(test) < 5:
                continue
            X_train, feats = _feature_matrix(train, feature_cols)
            X_test, _ = _feature_matrix(test, feats)
            y_train = train[target_col].values
            y_test = test[target_col].values

            m = clone(best_model)
            m.fit(X_train, y_train)
            preds = m.predict(X_test)

            for i in range(len(y_test)):
                rows.append({
                    "actual": float(y_test[i]),
                    "predicted": float(preds[i]),
                    "Name": test["Name"].iloc[i] if "Name" in test.columns else None,
                    "season": int(ts),
                    "mlbID": test["mlbID"].iloc[i] if "mlbID" in test.columns else None,
                })

        if rows:
            out[t["key"]] = pd.DataFrame(rows)
    return out


def build_comparison_table(all_results: dict) -> pd.DataFrame:
    """Wide table: rows=targets, cols=models, values=RMSE (mean over folds)."""
    model_names = list(_make_models().keys())
    rows = []
    for tkey, res in all_results.items():
        tdef = next(t for t in TARGETS if t["key"] == tkey)
        row = {"Target": tdef["label"], "Group": tdef["group"]}
        for model_name in model_names:
            mres = res.get(model_name, {})
            row[model_name] = mres.get("RMSE")
            if mres.get("RMSE_std") is not None:
                row[f"{model_name}_std"] = mres.get("RMSE_std")
        row["Best Model"] = res.get("_best", "")
        row["Folds"] = res.get("_n_folds", "")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Quality score (z-scored raw stats)
# ---------------------------------------------------------------------------

# Z-score components: (col, lower_better) — lower_better=True means negate z for "good"
ZSCORE_COLS = [
    ("xera", True),        # lower better
    ("SO9", False),        # higher better
    ("est_woba", True),    # lower better
    ("bb_rate", True),     # BB/BF, lower better
    ("brl_percent", True), # lower better (barrels allowed)
    ("IP", False),         # higher better (workload)
]
ZSCORE_WEIGHTS = {"xera": 0.25, "SO9": 0.22, "est_woba": 0.18, "bb_rate": 0.12, "brl_percent": 0.08, "IP": 0.15}
GRADE_THRESHOLDS = [
    (90, "A+"), (80, "A"), (70, "B+"), (60, "B"),
    (50, "C+"), (40, "C"), (30, "D"), (0,  "F"),
]


def _zscore(series: pd.Series) -> pd.Series:
    m = series.mean()
    s = series.std()
    if s is None or s == 0 or pd.isna(s):
        return series - m
    return (series - m) / s


def compute_pitcher_quality_score_zscore(row: pd.Series, z_mean: dict, z_std: dict) -> Optional[float]:
    """Weighted sum of z-scored components (normalized so higher = better)."""
    total = 0.0
    total_w = 0.0
    for col, lower_better in ZSCORE_COLS:
        v = row.get(col)
        w = ZSCORE_WEIGHTS.get(col, 0)
        if pd.isna(v) or w <= 0 or col not in z_mean:
            continue
        mu, s = z_mean[col], z_std[col]
        if s is None or s == 0 or pd.isna(s):
            z = 0.0
        else:
            z = (float(v) - mu) / s
        if lower_better:
            z = -z  # invert so higher z = better
        total += z * w
        total_w += w
    if total_w < 0.2:
        return None
    raw = total / total_w
    return round(raw, 2)


def quality_grade(score: Optional[float]) -> str:
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "N/A"
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


def build_quality_leaderboard(df: pd.DataFrame, min_ip: float = 20.0) -> pd.DataFrame:
    q = df[df["IP"].fillna(0) >= min_ip].copy()
    if "BF" in q.columns and "BB" in q.columns:
        q["bb_rate"] = q["BB"] / q["BF"].replace(0, np.nan)
    elif "bb_rate" not in q.columns:
        q["bb_rate"] = np.nan
    for col, _ in ZSCORE_COLS:
        if col not in q.columns:
            q[col] = np.nan
    z_mean, z_std = {}, {}
    for col, _ in ZSCORE_COLS:
        s = q[col].dropna()
        if len(s) >= 5:
            z_mean[col] = float(s.mean())
            z_std[col] = float(s.std()) if s.std() and not pd.isna(s.std()) else 1.0
    q["quality_score"] = q.apply(
        lambda r: compute_pitcher_quality_score_zscore(r, z_mean, z_std), axis=1
    )
    q["quality_score"] = _scale_z_to_0_100(q["quality_score"])
    q["grade"] = q["quality_score"].apply(quality_grade)
    base = ["mlbID", "Name", "Tm", "IP", "ERA", "xera", "WHIP", "SO9", "quality_score", "grade"]
    raw_cols = [c for c, _ in ZSCORE_COLS if c in q.columns and c not in base]
    # Include wOBA and hard-hit for correlation analysis in UI
    extra = [c for c in ("woba", "savant_woba", "ev95percent") if c in q.columns and c not in base + raw_cols]
    cols = list(dict.fromkeys(c for c in base + raw_cols + extra if c in q.columns))
    return q[cols].sort_values("quality_score", ascending=False, na_position="last").reset_index(drop=True)


def _scale_z_to_0_100(z_series: pd.Series) -> pd.Series:
    """Scale z-score-like values (roughly -3 to +3) to 0-100."""
    z = z_series.dropna()
    if len(z) < 2:
        return z_series
    lo, hi = z.quantile(0.02), z.quantile(0.98)
    if hi - lo == 0:
        return z_series
    scaled = (z_series - lo) / (hi - lo) * 100
    return scaled.clip(0, 100).round(1)


def quality_component_correlations(df: pd.DataFrame, min_ip: float = 30.0) -> pd.DataFrame:
    """
    Compute Pearson correlation of each z-score component (raw stat) with
    ERA, WHIP, and SO9. Negative with ERA/WHIP = good; positive with SO9 = good.
    """
    q = df[df["IP"].fillna(0) >= min_ip].copy()
    if "bb_rate" not in q.columns and "BF" in q.columns and "BB" in q.columns:
        q["bb_rate"] = q["BB"] / q["BF"].replace(0, np.nan)
    targets = [c for c in ["ERA", "WHIP", "SO9"] if c in q.columns]
    available = [c for c, _ in ZSCORE_COLS if c in q.columns]
    rows = []
    labels = {"xera": "xERA", "SO9": "SO9", "est_woba": "est.wOBA", "bb_rate": "BB%", "brl_percent": "Barrel%"}
    for col in available:
        row = {"component": labels.get(col, col)}
        for t in targets:
            sub = q[[col, t]].dropna()
            if len(sub) > 10:
                row[t] = round(sub[col].corr(sub[t]), 3)
            else:
                row[t] = None
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_regression_pipeline(
    years: list[int] = None,
    save: bool = True,
) -> dict:
    if years is None:
        years = list(range(2015, 2026))

    print("Building comprehensive dataset…")
    dataset = build_comprehensive_dataset(years)
    n_seasons = dataset["season"].nunique()
    print(f"  {len(dataset)} rows across {n_seasons} seasons "
          f"({sorted(dataset['season'].unique())})")

    if n_seasons < 2:
        print("Need ≥2 seasons with xERA data. Exiting.")
        return {"dataset": dataset}

    test_season = int(dataset["season"].max())
    print("\nEvaluating all 6 targets (walk-forward CV)…")
    all_results = evaluate_all_targets(dataset)

    comparison = build_comparison_table(all_results)
    print(f"\n--- RMSE Overview ---")
    print(comparison.to_string(index=False))

    # Current season analysis
    latest_season = int(
        pd.read_parquet(PROCESSED_DIR / "pitching_trends.parquet")["year"].max()
    )
    latest = pd.read_parquet(PROCESSED_DIR / f"pitching_profiles_{latest_season}.parquet")

    # Merge expected stats into latest (only cols not already in latest to avoid _x/_y suffixes)
    exp = load_expected_stats([latest_season])
    if not exp.empty:
        exp["era_vs_xera"]  = exp["era_minus_xera_diff"]
        exp["ba_vs_est"]    = -(exp["est_ba_minus_ba_diff"])
        exp["woba_vs_est"]  = -(exp["est_woba_minus_woba_diff"])
        exp_cols = ["mlbID", "ba", "est_ba", "ba_vs_est",
                    "woba", "est_woba", "woba_vs_est",
                    "era", "xera", "era_vs_xera"]
        exp_sub = exp[[c for c in exp_cols if c in exp.columns]].copy()
        # Only merge columns that latest doesn't already have (avoids xera_x/xera_y duplicates)
        merge_cols = ["mlbID"] + [c for c in exp_sub.columns if c != "mlbID" and c not in latest.columns]
        if len(merge_cols) > 1:
            latest = latest.merge(exp_sub[merge_cols], on="mlbID", how="left")

    # Quality leaderboard
    quality_lb = build_quality_leaderboard(latest, min_ip=20)

    # Next-season projections for each target (use best model per target from walk-forward)
    projections = {}
    for t in TARGETS:
        best = all_results.get(t["key"], {}).get("_best", "Random Forest")
        proj = project_next_season(latest, dataset, t, model_name=best, min_ip=30)
        if not proj.empty:
            projections[t["key"]] = proj
            print(f"  Projected {len(proj)} pitchers for {t['label']}")

    # Regression candidates for Group B residuals
    candidates = {}
    for t in [t for t in TARGETS if t["group"] == "B"]:
        cands = identify_candidates(latest, t, min_ip=40)
        if cands:
            candidates[t["key"]] = cands
            print(f"  {t['label']:25s}  unlucky={len(cands.get('unlucky', []))}  "
                  f"lucky={len(cands.get('lucky', []))}")

    # Predicted vs actual for Group B (for scatter plots)
    pred_vs_actual = collect_pred_vs_actual(dataset, all_results)

    output = {
        "dataset": dataset,
        "all_results": all_results,
        "comparison": comparison,
        "projections": projections,
        "candidates": candidates,
        "quality_leaderboard": quality_lb,
        "pred_vs_actual": pred_vs_actual,
        "test_season": test_season,
        "latest_season": latest_season,
    }

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        comparison.to_parquet(PROCESSED_DIR / "regression_comparison.parquet", index=False)
        quality_lb.to_parquet(PROCESSED_DIR / "quality_leaderboard.parquet", index=False)
        (PROCESSED_DIR / "regression_metadata.json").write_text(json.dumps({"latest_season": latest_season}))

        # This-year vs next-year luck (ERA − xERA) for direct scatter
        luck_cols = ["era_vs_xera", "era_vs_xera_next", "Name", "season", "mlbID"]
        luck_cols = [c for c in luck_cols if c in dataset.columns]
        if len(luck_cols) >= 2:
            luck_pairs = dataset[luck_cols].dropna(subset=["era_vs_xera", "era_vs_xera_next"])
            if not luck_pairs.empty:
                luck_pairs = luck_pairs.rename(columns={
                    "era_vs_xera": "luck_this_year",
                    "era_vs_xera_next": "luck_next_year",
                })
                luck_pairs.to_parquet(PROCESSED_DIR / "luck_this_vs_next.parquet", index=False)

        for tkey, pva in pred_vs_actual.items():
            pva.to_parquet(PROCESSED_DIR / f"pred_vs_actual_{tkey}.parquet", index=False)

        for t in TARGETS:
            tkey = t["key"]
            if tkey in projections:
                projections[tkey].to_parquet(
                    PROCESSED_DIR / f"proj_{tkey}.parquet", index=False)
            if tkey in candidates:
                candidates[tkey]["unlucky"].to_parquet(
                    PROCESSED_DIR / f"cand_{tkey}_unlucky.parquet", index=False)
                candidates[tkey]["lucky"].to_parquet(
                    PROCESSED_DIR / f"cand_{tkey}_lucky.parquet", index=False)

            # Feature importances (from full-data model; uses best model type per target)
            if tkey in all_results:
                fi = get_feature_importances(all_results[tkey])
                if not fi.empty:
                    fi.to_parquet(PROCESSED_DIR / f"fi_{tkey}.parquet", index=False)

        print(f"\nSaved all outputs to {PROCESSED_DIR}")

    return output


if __name__ == "__main__":
    run_regression_pipeline(save=True)
