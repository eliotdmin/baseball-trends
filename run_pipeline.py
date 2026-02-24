"""
Pitcher Analytics Pipeline Runner
====================================
Orchestrates:
  1. Statcast pitch-by-pitch → pitcher profiles
  2. KMeans clustering (simple partition for roster grouping)
  3. Arsenal predictor (RF: features → xERA, feature importances)
  4. Pitching stats (bref + Statcast) + regression / quality scores

Usage:
    python run_pipeline.py
    python run_pipeline.py --cluster-years 2024
    python run_pipeline.py --skip-clustering --skip-pitching-stats  # regression only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "backend"))


def run_clustering_step(args):
    from preprocess_data import run_preprocessing
    from clustering import run_clustering_pipeline

    print("=" * 60)
    print("STEP 1: Preprocessing (Statcast pitch-by-pitch)")
    print("=" * 60)
    if args.cluster_years is not None:
        profiles = run_preprocessing(years=args.cluster_years, min_pitches=args.min_pitches)
    else:
        # Default: full range (2015–2025) to use all available data
        profiles = run_preprocessing(
            start_date=args.start or "2015-03-01",
            end_date=args.end or "2025-11-30",
            min_pitches=args.min_pitches,
        )

    print("\n" + "=" * 60)
    print("STEP 2: Clustering (KMeans)")
    print("=" * 60)
    results = run_clustering_pipeline(profiles, n_clusters=args.k)
    return results


def run_arsenal_predictor_step(args):
    """Train RF: arsenal features → outcome (xERA). Saves feature importances."""
    from arsenal_predictor import run_arsenal_predictor

    print("\n" + "=" * 60)
    print("STEP: Arsenal Predictor (features → xERA)")
    print("=" * 60)
    trends_path = Path(__file__).resolve().parent / "data" / "processed" / "pitching_trends.parquet"
    if not trends_path.exists():
        print("Skipping arsenal predictor: pitching_trends.parquet not found. Run with --skip-pitching-stats off first.")
        return None
    return run_arsenal_predictor(
        outcome_col="xera",
        min_ip=args.min_ip,
    )


def run_pitching_stats_step(args):
    from preprocess_pitching_bref import save_pitching_profiles

    print("\n" + "=" * 60)
    print("STEP: Pitching Stats (bref + Statcast)")
    print("=" * 60)
    df = save_pitching_profiles(args.years, min_ip=args.min_ip)
    return df


def run_regression_step(args):
    from predict_actual_expected_residual import run_regression_pipeline

    print("\n" + "=" * 60)
    print("STEP: Regression (ERA/wOBA/BA + residuals)")
    print("=" * 60)
    return run_regression_pipeline(years=args.years, save=True)


def main():
    parser = argparse.ArgumentParser(description="Pitcher Analytics Pipeline")
    parser.add_argument("--cluster-years", nargs="+", type=int, default=None,
                        help="Year(s) for Statcast profiles (default: all 2015-2025)")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--min-pitches", type=int, default=200)
    parser.add_argument("--k", type=int, default=5, help="KMeans clusters")
    parser.add_argument("--skip-clustering", action="store_true")
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2015, 2026)))
    parser.add_argument("--min-ip", type=float, default=20.0)
    parser.add_argument("--skip-pitching-stats", action="store_true")
    parser.add_argument("--skip-regression", action="store_true")
    parser.add_argument("--skip-arsenal-predictor", action="store_true",
                        help="Skip arsenal → outcome model (requires pitching_trends)")

    args = parser.parse_args()
    results = {}

    if not args.skip_clustering:
        results["clustering"] = run_clustering_step(args)

    if not args.skip_pitching_stats:
        results["pitching_stats"] = run_pitching_stats_step(args)

    if not args.skip_regression:
        results["regression"] = run_regression_step(args)

    if not args.skip_arsenal_predictor:
        try:
            results["arsenal_predictor"] = run_arsenal_predictor_step(args)
        except Exception as e:
            print(f"Arsenal predictor failed: {e}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    if "clustering" in results:
        r = results["clustering"]
        print(f"Clustering: {len(r['profiles'])} pitchers, k={len(set(r['kmeans_labels']))}")
    if "arsenal_predictor" in results and results["arsenal_predictor"]:
        ap = results["arsenal_predictor"]
        print(f"Arsenal predictor: RMSE={ap['rmse']:.3f}, R²={ap['r2']:.3f}")
    if "pitching_stats" in results:
        print(f"Pitching stats: {sorted(results['pitching_stats']['year'].unique())}")
    print(f"\nDashboard: streamlit run src/frontend/app.py")


if __name__ == "__main__":
    main()
