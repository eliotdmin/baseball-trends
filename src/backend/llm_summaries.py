"""
LLM-Powered Cluster Summaries
==============================
Generates natural language descriptions of pitcher clusters.
Supports OpenAI (gpt-4o-mini, cheapest ~$0.003/full run) and Groq (free tier).
Falls back to a rich rule-based narrative when no API key is available.

Key: set OPENAI_API_KEY or GROQ_API_KEY in .streamlit/secrets.toml or as
environment variables. Run `python src/backend/llm_summaries.py` to regenerate.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
SUMMARIES_DIR = PROCESSED_DIR / "summaries"

SYSTEM_PROMPT = """\
You are a baseball analytics expert and scout. Given statistical summaries of
a cluster of MLB pitchers, write a concise scouting report (3-5 sentences)
describing the cluster's archetype. Include:
- The dominant pitching style (power, finesse, breaking-ball heavy, etc.)
- Key statistical traits that define this group
- What type of MLB pitcher typically falls in this cluster
- A comparable well-known pitcher archetype, if applicable

IMPORTANT: When discussing velocity, always refer to FASTBALL velocity
(four-seamer or sinker mph), not an average across all pitch types. A pitcher
who throws a 96 mph fastball and a 78 mph curveball is a "power" pitcher —
the curveball speed is irrelevant to their velocity tier.

Use clear, confident language suitable for a front office audience.\
"""


def build_cluster_stats_prompt(
    cluster_id: int,
    profiles: pd.DataFrame,
    labels: np.ndarray,
    features: list[str],
) -> str:
    """Build a prompt with cluster statistics for LLM summarization."""
    mask = labels == cluster_id
    cluster_df = profiles[mask]
    n_pitchers = len(cluster_df)
    all_pitchers = profiles  # needed for league-average pitch mix

    pitch_types = ["FF", "SI", "SL", "CH", "CU", "FC", "ST", "KC", "FS"]

    # Pitch mix: show cluster vs league average
    pitch_mix = {}
    for pt in pitch_types:
        col = f"pct_{pt}"
        if col in cluster_df.columns:
            cluster_avg = float(cluster_df[col].mean())
            league_avg  = float(all_pitchers[col].mean()) if col in all_pitchers.columns else cluster_avg
            diff = cluster_avg - league_avg
            if cluster_avg > 0.02:
                pitch_mix[pt] = f"{cluster_avg:.1%} ({diff:+.1%} vs. league)"

    # Fastball velocity: use FF, then SI, then FC, then avg_velo as fallback
    fb_velo = _fastball_velo(cluster_df)
    fb_velo_label = fb_velo.name if fb_velo.name else "avg_velo"
    league_fb = _fastball_velo(all_pitchers)
    fb_str = (f"{fb_velo.mean():.1f} mph "
              f"({fb_velo.mean() - league_fb.mean():+.1f} vs. league)"
              if not fb_velo.empty else "n/a")

    # Spin (fastball-specific), extension, outcomes — curated for readability
    fb_spin = _fastball_spin(cluster_df)
    curated_stats = {}
    if not fb_spin.empty:
        v = fb_spin.mean()
        lv = _fastball_spin(profiles).mean() if "spin_FF" in profiles.columns else v
        curated_stats["fastball spin (rpm)"] = f"{v:.0f} ({v - lv:+.0f} vs. league)"
    for col, label in [
        ("avg_extension",    "avg extension (ft)"),
        ("hard_hit_pct",     "hard-hit rate against"),
        ("cmd_plate_x_std",  "horizontal command spread (lower = tighter)"),
    ]:
        if col in cluster_df.columns:
            v = cluster_df[col].mean()
            lv = profiles[col].mean() if col in profiles.columns else v
            curated_stats[label] = f"{v:.2f} ({v - lv:+.2f} vs. league)"

    example_pitchers = cluster_df["player_name"].head(8).tolist() if "player_name" in cluster_df.columns else []
    hand_dist = cluster_df["p_throws"].value_counts().to_dict() if "p_throws" in cluster_df.columns else {}

    prompt = f"""Cluster {cluster_id} contains {n_pitchers} pitchers.
Handedness: {hand_dist}
Example pitchers: {', '.join(example_pitchers)}

FASTBALL velocity ({fb_velo_label}): {fb_str}

Pitch mix (cluster avg vs. league avg):
{json.dumps(pitch_mix, indent=2)}

Other characteristics:
{json.dumps(curated_stats, indent=2)}

Write a scouting-style summary of this pitcher archetype. Lead with fastball velocity as the velocity descriptor."""

    return prompt


def _get_llm_client(provider: str, api_key: Optional[str]):
    """Return a callable (prompt) -> str for the given provider."""
    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            return None
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            return None

        client = OpenAI(api_key=key)

        def call(system, user):
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip()

        return call

    if provider == "groq":
        try:
            from groq import Groq
        except ImportError:
            return None
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            return None

        client = Groq(api_key=key)

        def call(system, user):
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip()

        return call

    return None


def generate_cluster_summary(
    cluster_id: int,
    profiles: pd.DataFrame,
    labels: np.ndarray,
    features: list[str],
    provider: str = "openai",
    api_key: Optional[str] = None,
) -> str:
    """Generate a natural language cluster summary via LLM or rule-based fallback."""
    llm = _get_llm_client(provider, api_key)
    if llm is None:
        if provider != "none":
            print(f"No API key for '{provider}' — using rule-based fallback for cluster {cluster_id}")
        return _fallback_summary(cluster_id, profiles, labels, features)

    user_prompt = build_cluster_stats_prompt(cluster_id, profiles, labels, features)
    try:
        return llm(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        print(f"LLM call failed for cluster {cluster_id}: {e}. Using fallback.")
        return _fallback_summary(cluster_id, profiles, labels, features)


FASTBALL_VELO_COLS = ["velo_FF", "velo_SI", "velo_FC"]   # preference order
FASTBALL_SPIN_COLS = ["spin_FF", "spin_SI", "spin_FC"]   # same order for consistency


def _fastball_velo(df: pd.DataFrame) -> pd.Series:
    """
    Return the best available fastball velocity column as a Series.
    Uses the first column (by FASTBALL_VELO_COLS order) that has non-null data
    for at least half the rows, otherwise falls back to avg_velo.
    """
    for col in FASTBALL_VELO_COLS:
        if col in df.columns and df[col].notna().mean() >= 0.3:
            return df[col]
    if "avg_velo" in df.columns:
        return df["avg_velo"]
    return pd.Series(dtype=float)


def _fastball_spin(df: pd.DataFrame) -> pd.Series:
    """Best available fastball spin column (FF → SI → FC)."""
    for col in FASTBALL_SPIN_COLS:
        if col in df.columns and df[col].notna().mean() >= 0.3:
            return df[col]
    if "avg_spin" in df.columns:
        return df["avg_spin"]
    return pd.Series(dtype=float)


def _compute_league_stats(profiles: pd.DataFrame, pitch_types: list[str]) -> dict:
    """
    Compute all league-wide statistics used for thresholds — driven entirely
    by the data, with no hard-coded mph/rpm numbers.
    Velocity is measured using fastball velocity (FF → SI → FC → avg_velo).
    """
    s = {}

    # Fastball velocity and spin: primary signals; avg_* fallbacks
    fb_velo = _fastball_velo(profiles)
    s["fb_velo_col"]  = fb_velo.name if fb_velo.name else "avg_velo"
    s["fb_velo_mean"] = float(fb_velo.mean()) if not fb_velo.empty else 0.0
    s["fb_velo_std"]  = float(fb_velo.std())  if not fb_velo.empty else 1.0

    fb_spin = _fastball_spin(profiles)
    s["fb_spin_col"]  = fb_spin.name if fb_spin.name else "avg_spin"
    s["fb_spin_mean"] = float(fb_spin.mean()) if not fb_spin.empty else 0.0
    s["fb_spin_std"]  = float(fb_spin.std())  if not fb_spin.empty else 1.0

    for col in ("avg_extension", "hard_hit_pct"):
        if col in profiles.columns:
            s[f"{col}_mean"] = float(profiles[col].mean())
            s[f"{col}_std"]  = float(profiles[col].std())
        else:
            s[f"{col}_mean"] = 0.0
            s[f"{col}_std"]  = 1.0

    for pt in pitch_types:
        col = f"pct_{pt}"
        if col in profiles.columns:
            s[f"pct_{pt}_mean"] = float(profiles[col].mean())
            s[f"pct_{pt}_std"]  = float(profiles[col].std())
        else:
            s[f"pct_{pt}_mean"] = 0.0
            s[f"pct_{pt}_std"]  = 0.01
    return s


def _archetype_name(
    velo_z: float,        # cluster avg_velo expressed as z-score vs. league
    spin_z: float,        # cluster avg_spin expressed as z-score vs. league
    above: list,          # [(pt, pct, diff_z), …] sorted by z-score desc
    pitch_names: dict,
) -> str:
    """
    Derive an archetype label from z-scores — entirely data-driven, no hard-coded
    mph/rpm thresholds.
    Priority: dominant pitch → spin anomaly → velocity tier.
    """
    top_pt, top_pct, top_z = above[0] if above else (None, 0, 0)
    pname = pitch_names.get(top_pt, top_pt or "").title() if top_pt else ""

    # Dominant pitch ≥ 1.5 SD above league average
    if top_z >= 1.5:
        velo_prefix = "Power " if velo_z >= 1.0 else ("Soft-Toss " if velo_z <= -0.5 else "")
        return f"{velo_prefix}{pname} Specialist"

    # Notable pitch lean ≥ 0.75 SD
    if top_z >= 0.75:
        if velo_z >= 1.0:
            return f"Power {pname} Arm"
        elif velo_z <= -0.5:
            return f"Soft-Toss {pname} Arm"
        else:
            return f"Mid-Velocity {pname} Arm"

    # Weak pitch lean (≥ 0.3 SD) — still use the pitch name even if not a "specialist"
    if top_z >= 0.3 and top_pt:
        if spin_z >= 1.5:
            return f"High-Spin {pname} Arm"
        if velo_z >= 1.0:
            return f"Power {pname} Arm"
        if velo_z <= -1.0:
            return f"Finesse {pname} Arm"
        return f"Mid-Velocity {pname} Arm"

    # No discernible pitch lean → distinguish by spin anomaly
    if spin_z >= 1.5:
        return "High-Spin Craftsman" if velo_z < 1.0 else "High-Spin Power Arm"
    if spin_z <= -1.5:
        return "Low-Spin Ground-Ball Arm"

    # Fall back to velocity tier — include dominant pitch when available to reduce duplicates
    suffix = f" {pname}" if (top_pt and top_z >= 0.3) else ""
    if velo_z >= 2.0:
        return f"Elite Power{suffix} Arm" if suffix else "Elite Power Arm"
    if velo_z >= 1.0:
        return f"Power{suffix} Arm" if suffix else "Power Arm"
    if velo_z >= 0.3:
        return f"Above-Average{suffix} Starter" if suffix else "Above-Average Starter"
    if velo_z >= -0.3:
        return f"Mid-Velocity{suffix} Craftsman" if suffix else "Mid-Velocity Craftsman"
    if velo_z >= -1.0:
        return f"Below-Average{suffix} Arm" if suffix else "Below-Average Velocity Arm"
    return f"Finesse{suffix} Pitcher" if suffix else "Finesse/Command Pitcher"


def _fallback_summary(
    cluster_id: int,
    profiles: pd.DataFrame,
    labels: np.ndarray,
    features: list[str],
) -> str:
    """
    Rule-based narrative that emphasises what makes this cluster DISTINCTIVE
    relative to every other cluster.  All thresholds are derived from the
    data's own distribution (z-scores / standard deviations) — no hard-coded
    mph or rpm numbers.
    """
    mask = labels == cluster_id
    cluster_df = profiles[mask]
    n = len(cluster_df)

    pitch_types = ["FF", "SI", "SL", "CH", "CU", "FC", "ST", "KC", "FS"]
    pitch_names = {"FF": "four-seamer", "SI": "sinker", "SL": "slider",
                   "CH": "changeup", "CU": "curveball", "FC": "cutter",
                   "ST": "sweeper", "KC": "knuckle-curve", "FS": "splitter"}

    ls = _compute_league_stats(profiles, pitch_types)

    def _mean(df, col, default=np.nan):
        """Mean of non-null values. Returns default (nan) when all missing — never use 0 for 'not thrown'."""
        if col not in df.columns:
            return default
        vals = df[col].dropna()
        return float(vals.mean()) if len(vals) > 0 else default

    def _best_fastball_stat(df, velo_cols, spin_cols):
        """Pick velo/spin from first column with enough data in THIS cluster. Avoids 0 for 'doesn't throw FF'."""
        for col in velo_cols:
            if col in df.columns and df[col].notna().mean() >= 0.3:
                v = df[col].dropna()
                if len(v) > 0:
                    return col, float(v.mean())
        if "avg_velo" in df.columns and df["avg_velo"].notna().any():
            return "avg_velo", float(df["avg_velo"].dropna().mean())
        return None, np.nan

    def _best_spin(df, spin_cols):
        for col in spin_cols:
            if col in df.columns and df[col].notna().mean() >= 0.3:
                v = df[col].dropna()
                if len(v) > 0:
                    return col, float(v.mean())
        if "avg_spin" in df.columns and df["avg_spin"].notna().any():
            return "avg_spin", float(df["avg_spin"].dropna().mean())
        return None, np.nan

    # Use fastball velocity and spin — pick best available FOR THIS CLUSTER (avoids 0 when they don't throw FF)
    fb_velo_col, avg_velo = _best_fastball_stat(cluster_df, FASTBALL_VELO_COLS, FASTBALL_SPIN_COLS)
    fb_spin_col, avg_spin = _best_spin(cluster_df, FASTBALL_SPIN_COLS)
    if fb_velo_col is None:
        fb_velo_col = ls["fb_velo_col"]
    if fb_spin_col is None:
        fb_spin_col = ls["fb_spin_col"]
    avg_ext  = _mean(cluster_df, "avg_extension") or None
    hard_hit = _mean(cluster_df, "hard_hit_pct") or None

    # Handle clusters with no fastball data (e.g. knuckle-curve specialists who don't throw FF)
    velo_valid = pd.notna(avg_velo) and avg_velo > 50  # 0 or nan = invalid
    spin_valid = pd.notna(avg_spin) and avg_spin > 100
    if velo_valid:
        velo_diff = avg_velo - ls["fb_velo_mean"]
        velo_z = velo_diff / (ls["fb_velo_std"] or 1)
    else:
        velo_z = 0.0  # neutral for archetype
    if spin_valid:
        spin_diff = avg_spin - ls["fb_spin_mean"]
        spin_z = spin_diff / (ls["fb_spin_std"] or 1)
    else:
        spin_z = 0.0

    # For pitch mix, 0 is valid (no usage); nan only when column missing
    cluster_pcts = {pt: _mean(cluster_df, f"pct_{pt}", default=0.0) for pt in pitch_types}
    league_pcts  = {pt: ls[f"pct_{pt}_mean"]           for pt in pitch_types}
    league_stds  = {pt: ls[f"pct_{pt}_std"]            for pt in pitch_types}

    # Pitch mix: z-score each pitch's over/under-use
    diffs = {pt: cluster_pcts[pt] - league_pcts[pt] for pt in pitch_types}
    diff_z = {pt: diffs[pt] / (league_stds[pt] or 0.01) for pt in pitch_types}

    above = [(pt, cluster_pcts[pt], diff_z[pt]) for pt in pitch_types
             if diff_z[pt] >= 0.5 and cluster_pcts[pt] > 0.03]
    below = [(pt, cluster_pcts[pt], diff_z[pt]) for pt in pitch_types
             if diff_z[pt] <= -0.5 and league_pcts[pt] > 0.03]
    above.sort(key=lambda x: -x[2])
    below.sort(key=lambda x: x[2])

    # ---- Velocity description (fastball-specific, z-score based labels) ----
    velo_src = {"velo_FF": "FF", "velo_SI": "SI", "velo_FC": "FC", "avg_velo": "overall"}.get(fb_velo_col, "FB")
    if not velo_valid:
        velo_desc = f"fastball velocity n/a (pitchers in this cluster typically don't feature a primary four-seamer/sinker)"
    elif velo_z >= 2.0:
        velo_desc = f"elite {velo_src} velocity ({avg_velo:.1f} mph, {velo_diff:+.1f} vs. league)"
    elif velo_z >= 1.0:
        velo_desc = f"plus {velo_src} velocity ({avg_velo:.1f} mph, {velo_diff:+.1f} vs. league)"
    elif velo_z >= 0.3:
        velo_desc = f"above-average {velo_src} velocity ({avg_velo:.1f} mph, {velo_diff:+.1f} vs. league)"
    elif velo_z >= -0.3:
        velo_desc = f"average {velo_src} velocity ({avg_velo:.1f} mph, {velo_diff:+.1f} vs. league)"
    elif velo_z >= -1.0:
        velo_desc = f"below-average {velo_src} velocity ({avg_velo:.1f} mph, {velo_diff:+.1f} vs. league)"
    else:
        velo_desc = f"notably low {velo_src} velocity ({avg_velo:.1f} mph, {velo_diff:+.1f} vs. league)"

    # ---- Spin description (fastball-specific, z-score based) ----
    spin_src = {"spin_FF": "FF", "spin_SI": "SI", "spin_FC": "FC", "avg_spin": "overall"}.get(fb_spin_col, "FB")
    if not spin_valid:
        spin_desc = f"fastball spin n/a (pitchers in this cluster typically don't throw a primary four-seamer/sinker)"
    elif abs(spin_z) >= 1.5:
        spin_desc = (
            f"{'notably higher' if spin_z > 0 else 'notably lower'} {spin_src} spin than average "
            f"({avg_spin:.0f} rpm, {spin_diff:+.0f} vs. league) — "
            f"{'suggesting plus ride / break' if spin_z > 0 else 'suggesting flat, sink-heavy movement'}"
        )
    elif spin_z >= 0.75:
        spin_desc = f"above-average {spin_src} spin ({avg_spin:.0f} rpm, {spin_diff:+.0f} vs. league)"
    elif spin_z <= -0.75:
        spin_desc = f"below-average {spin_src} spin ({avg_spin:.0f} rpm, {spin_diff:+.0f} vs. league)"
    else:
        spin_desc = f"league-average {spin_src} spin ({avg_spin:.0f} rpm, {spin_diff:+.0f})"

    # ---- Pitch mix narrative ----
    if above:
        above_str = ", ".join(
            f"{pitch_names.get(pt, pt)} ({pct:.0%}, {diffs[pt]:+.0%})"
            for pt, pct, _ in above[:3]
        )
        pitch_mix_text = f"leans heavily on the {above_str}"
    else:
        top_two = sorted(cluster_pcts.items(), key=lambda x: -x[1])[:2]
        pitch_mix_text = (
            f"shows a balanced mix led by the "
            f"{pitch_names.get(top_two[0][0], top_two[0][0])} ({top_two[0][1]:.0%}) "
            f"and {pitch_names.get(top_two[1][0], top_two[1][0])} ({top_two[1][1]:.0%})"
        )

    if below:
        below_str = " / ".join(
            f"{pitch_names.get(pt, pt)} ({diffs[pt]:+.0%})"
            for pt, _, _ in below[:2]
        )
        pitch_mix_text += f", and avoids the {below_str}"

    # ---- Extension: only if > 1 SD from league mean ----
    ext_text = ""
    if avg_ext and ls["avg_extension_std"] > 0:
        ext_diff = avg_ext - ls["avg_extension_mean"]
        ext_z = ext_diff / ls["avg_extension_std"]
        if abs(ext_z) >= 1.0:
            direction = "longer" if ext_z > 0 else "shorter"
            ext_text = (f" Extension: {avg_ext:.1f} ft "
                        f"({ext_diff:+.2f} vs. league — {direction} release than average).")

    # ---- Hard-hit rate: only if > 1 SD from league mean ----
    hh_text = ""
    if hard_hit is not None and ls["hard_hit_pct_std"] > 0:
        hh_diff = hard_hit - ls["hard_hit_pct_mean"]
        hh_z = hh_diff / ls["hard_hit_pct_std"]
        if abs(hh_z) >= 1.0:
            direction = "above" if hh_z > 0 else "below"
            hh_text = (f" Hard-hit rate: {hard_hit:.1%} "
                       f"({hh_diff:+.1%} {direction} league avg).")

    archetype = _archetype_name(velo_z, spin_z, above, pitch_names)
    bullets = [
        f"• {velo_desc}",
        f"• {spin_desc}",
        f"• Arsenal: {pitch_mix_text}",
    ]
    if ext_text:
        bullets.append(f"• {ext_text.strip()}")
    if hh_text:
        bullets.append(f"• {hh_text.strip()}")
    return (
        f"Cluster {cluster_id} ({n} pitchers) — {archetype}:\n" + "\n".join(bullets)
    )


def generate_all_summaries(
    profiles: pd.DataFrame,
    labels: np.ndarray,
    features: list[str],
    use_llm: bool = True,
    provider: str = "openai",
    api_key: Optional[str] = None,
    out_filename: str = "cluster_summaries.json",
) -> dict[int, str]:
    """Generate summaries for all clusters and persist to disk."""
    unique_clusters = sorted(set(int(x) for x in labels) - {-1})
    summaries = {}

    for cid in unique_clusters:
        print(f"Generating summary for cluster {cid}...")
        if use_llm:
            summaries[str(cid)] = generate_cluster_summary(
                cid, profiles, labels, features, provider=provider, api_key=api_key
            )
        else:
            summaries[str(cid)] = _fallback_summary(cid, profiles, labels, features)

    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUMMARIES_DIR / out_filename
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved summaries to {out_path}")

    return summaries


if __name__ == "__main__":
    from preprocess_data import get_clustering_features
    profiles = pd.read_parquet(PROCESSED_DIR / "pitcher_profiles_clustered.parquet")
    features = get_clustering_features(profiles)

    provider = "groq" if os.environ.get("GROQ_API_KEY") else "openai"
    use_llm = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY"))

    for cluster_col, fname in [("cluster_hdbscan", "cluster_summaries.json"),
                                ("cluster_kmeans",  "cluster_summaries_kmeans.json")]:
        if cluster_col not in profiles.columns:
            continue
        labels = profiles[cluster_col].values
        print(f"\n=== {cluster_col} ===")
        summaries = generate_all_summaries(
            profiles, labels, features,
            use_llm=use_llm, provider=provider, out_filename=fname,
        )
        for cid, text in summaries.items():
            print(f"\nCluster {cid}: {text[:180]}…")
