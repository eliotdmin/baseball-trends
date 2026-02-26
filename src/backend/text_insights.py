"""
Human-Readable Stat Interpretation
====================================
Converts raw baseball numbers into plain-English narrative labels and summaries.
Uses league benchmarks and percentile thresholds from Baseball Savant / BBREF
to contextualize velocity, spin, ERA, WHIP, SO9, xERA, and more.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# League benchmark thresholds (based on 2021-2024 MLB averages)
# ---------------------------------------------------------------------------

VELO_TIERS = [
    (97.0, "elite (97+ mph)"),
    (95.0, "plus (95-97 mph)"),
    (93.0, "above average (93-95 mph)"),
    (91.0, "average (91-93 mph)"),
    (89.0, "below average (89-91 mph)"),
    (0.0,  "well below average (<89 mph)"),
]

SPIN_TIERS = [
    (2500, "elite spin (2500+ rpm)"),
    (2350, "above average spin (2350-2500 rpm)"),
    (2150, "league average spin (2150-2350 rpm)"),
    (0,    "below average spin (<2150 rpm)"),
]

ERA_TIERS = [
    (2.50, "ace-caliber (sub-2.50)"),
    (3.25, "frontline starter (2.50-3.25)"),
    (4.00, "solid mid-rotation (3.25-4.00)"),
    (4.75, "league average (4.00-4.75)"),
    (5.50, "below average (4.75-5.50)"),
    (99.0, "replacement level (5.50+)"),
]

WHIP_TIERS = [
    (1.00, "elite (sub-1.00)"),
    (1.15, "plus (1.00-1.15)"),
    (1.30, "above average (1.15-1.30)"),
    (1.45, "average (1.30-1.45)"),
    (1.60, "below average (1.45-1.60)"),
    (99.0, "poor (1.60+)"),
]

SO9_TIERS = [
    (11.0, "elite strikeout pitcher (11+ K/9)"),
    (9.5,  "plus strikeout rate (9.5-11 K/9)"),
    (8.0,  "above average (8-9.5 K/9)"),
    (6.5,  "average (6.5-8 K/9)"),
    (0.0,  "below average (<6.5 K/9)"),
]

HARD_HIT_TIERS = [
    (47.0, "very hard contact (47%+ hard hit against)"),
    (42.0, "above average contact (42-47%)"),
    (38.0, "average contact (38-42%)"),
    (0.0,  "soft contact allowed (<38%)"),
]


def _classify(value: float, tiers: list[tuple]) -> str:
    """Return the label for the first tier where value < threshold."""
    for threshold, label in tiers:
        if value < threshold:
            return label
    return tiers[-1][1]


def _classify_ge(value: float, tiers: list[tuple]) -> str:
    """Return the label for the first tier where value >= threshold (descending thresholds)."""
    for threshold, label in tiers:
        if value >= threshold:
            return label
    return tiers[-1][1]


# ---------------------------------------------------------------------------
# Individual stat interpretations
# ---------------------------------------------------------------------------

def describe_velo(mph: float) -> str:
    return _classify_ge(mph, VELO_TIERS)


def describe_spin(rpm: float) -> str:
    return _classify_ge(rpm, SPIN_TIERS)


def describe_era(era: float) -> str:
    return _classify(era, ERA_TIERS)


def describe_whip(whip: float) -> str:
    return _classify(whip, WHIP_TIERS)


def describe_so9(so9: float) -> str:
    return _classify_ge(so9, SO9_TIERS)


def describe_hard_hit(pct: float) -> str:
    """pct should be 0-100 or 0-1; normalised automatically."""
    if pct <= 1.0:
        pct *= 100
    return _classify_ge(pct, HARD_HIT_TIERS)


def describe_xera_gap(era: float, xera: float) -> str:
    """Interpret the gap between ERA and xERA (expected ERA)."""
    diff = era - xera
    if diff < -0.75:
        return f"ERA is {abs(diff):.2f} runs below xERA — likely been lucky; regression risk"
    elif diff < -0.30:
        return f"ERA slightly outpacing underlying skill by {abs(diff):.2f} runs — minor luck component"
    elif diff <= 0.30:
        return f"ERA and xERA closely aligned ({diff:+.2f}) — performance matches quality of contact"
    elif diff <= 0.75:
        return f"xERA better than ERA by {diff:.2f} runs — some bad luck; could improve going forward"
    else:
        return f"xERA {diff:.2f} runs better than ERA — significant bad luck; strong regression candidate"


def describe_percentile(pct: float, stat_name: str) -> str:
    """Convert a 0-100 percentile rank into a plain label."""
    if pd.isna(pct):
        return "N/A"
    pct = int(round(pct))
    if pct >= 95:
        return f"{pct}th percentile (elite)"
    elif pct >= 80:
        return f"{pct}th percentile (plus)"
    elif pct >= 60:
        return f"{pct}th percentile (above average)"
    elif pct >= 40:
        return f"{pct}th percentile (average)"
    elif pct >= 20:
        return f"{pct}th percentile (below average)"
    else:
        return f"{pct}th percentile (poor)"


# ---------------------------------------------------------------------------
# Pitch-type-aware stat extraction (avoids blending across pitch types)
# ---------------------------------------------------------------------------

FASTBALL_VELO_COLS = ["velo_FF", "velo_SI", "velo_FC"]


def get_fastball_velo(row: pd.Series) -> Optional[float]:
    """Best available fastball velocity (FF → SI → FC). Treats 0 as invalid (not thrown)."""
    for col in FASTBALL_VELO_COLS:
        v = row.get(col)
        if pd.notna(v) and v > 50:  # 0 or tiny = invalid/sentinel
            return float(v)
    avg = row.get("avg_velo")
    if pd.notna(avg) and avg > 50:
        return float(avg)
    return None


def get_primary_spin(row: pd.Series) -> tuple[Optional[float], Optional[str]]:
    """Spin of the most-used pitch with spin data. Returns (rpm, pitch_type)."""
    best_rpm, best_pt = None, None
    best_pct = 0.0
    for pt in ["FF", "SI", "SL", "CH", "CU", "FC", "ST", "KC", "FS"]:
        pct = row.get(f"pct_{pt}") or 0
        spin = row.get(f"spin_{pt}")
        if pd.notna(spin) and spin > 0 and pct > 0.03:
            if pct > best_pct:
                best_pct, best_rpm, best_pt = pct, float(spin), pt
    return (best_rpm, best_pt)


# ---------------------------------------------------------------------------
# Pitcher profile narrative
# ---------------------------------------------------------------------------

PITCH_NAMES = {
    "FF": "Four-seam fastball",
    "SI": "Sinker",
    "SL": "Slider",
    "CH": "Changeup",
    "CU": "Curveball",
    "FC": "Cutter",
    "ST": "Sweeper",
    "KC": "Knuckle-curve",
    "FS": "Splitter",
}


def build_pitcher_narrative(
    row: pd.Series,
    league_avg_pct: Optional[dict] = None,
) -> str:
    """
    Given a row from the pitcher profiles DataFrame, return a 2-4 sentence
    plain-English narrative describing the pitcher's profile.

    league_avg_pct: dict of {pitch_type: proportion (0-1)} for league averages.
    When provided, pitch mix sentences say "throws the slider 8pp above average"
    rather than just "slider is the secondary pitch".
    """
    lines = []
    name = row.get("player_name", "This pitcher")

    # Velocity — fastball velo (FF/SI/FC), not blended avg
    velo = get_fastball_velo(row)
    if pd.notna(velo):
        lines.append(f"{name}'s fastball sits at **{velo:.1f} mph** — {describe_velo(velo)}.")

    # Pitch mix — compare to league average when available
    pct_cols = {k: row.get(f"pct_{k}", 0) for k in PITCH_NAMES}
    pct_cols = {k: v for k, v in pct_cols.items() if pd.notna(v) and v > 0.04}
    if pct_cols:
        pitch_parts = []
        for pt, pct in sorted(pct_cols.items(), key=lambda x: -x[1]):
            pct_pct = pct * 100
            if league_avg_pct and pt in league_avg_pct:
                lg_avg = league_avg_pct[pt] * 100
                diff = pct_pct - lg_avg
                diff_str = f"{diff:+.0f}pp vs. avg"
                pitch_parts.append(f"{PITCH_NAMES[pt]} {pct_pct:.0f}% ({diff_str})")
            else:
                pitch_parts.append(f"{PITCH_NAMES[pt]} {pct_pct:.0f}%")

        if pitch_parts:
            lines.append("Pitch mix: " + ", ".join(pitch_parts[:4]) + ".")

    # Spin — per-pitch-type for top 2–3 pitches by usage (no blended avg)
    spin_parts = []
    pct_for_spin = {k: row.get(f"pct_{k}", 0) or 0 for k in PITCH_NAMES}
    pct_for_spin = {k: v for k, v in pct_for_spin.items() if pd.notna(v) and v > 0.03}
    for pt, pct in sorted(pct_for_spin.items(), key=lambda x: -x[1]):
        spin = row.get(f"spin_{pt}")
        if pd.notna(spin) and spin > 0 and len(spin_parts) < 3:
            spin_parts.append(f"{PITCH_NAMES.get(pt, pt)} {spin:.0f} rpm")
    if spin_parts:
        lines.append("Spin by pitch: " + ", ".join(spin_parts) + ".")

    # Hard hit
    hard_hit = row.get("hard_hit_pct")
    if pd.notna(hard_hit):
        pct_display = hard_hit * 100 if hard_hit <= 1 else hard_hit
        lines.append(f"Opponents' hard-hit rate: **{pct_display:.1f}%** — {describe_hard_hit(pct_display)}.")

    return " ".join(lines) if lines else f"Insufficient data to generate narrative for {name}."


def build_traditional_stats_narrative(row: pd.Series) -> str:
    """
    Narrative for a pitcher's traditional/expected stats row.
    row should include ERA, WHIP, SO9, xera, era_minus_xera_diff (optional).
    """
    parts = []
    name = row.get("Name") or row.get("player_name") or "This pitcher"

    era = row.get("ERA") or row.get("era")
    whip = row.get("WHIP") or row.get("whip")
    so9 = row.get("SO9") or row.get("so9")
    xera = row.get("xera") or row.get("xERA")

    if pd.notna(era):
        parts.append(f"**ERA {era:.2f}** ({describe_era(era)})")
    if pd.notna(whip):
        parts.append(f"**WHIP {whip:.3f}** ({describe_whip(whip)})")
    if pd.notna(so9):
        parts.append(f"**{so9:.1f} K/9** ({describe_so9(so9)})")

    summary = ", ".join(parts) if parts else "No traditional stats available."

    xera_note = ""
    if pd.notna(era) and pd.notna(xera):
        xera_note = f" {describe_xera_gap(era, xera)}"

    return f"{name}: {summary}.{xera_note}"


def build_trend_annotation(values: list[float], years: list[int], stat: str) -> str:
    """Return a brief trend description for a multi-year stat series."""
    clean = [(y, v) for y, v in zip(years, values) if pd.notna(v)]
    if len(clean) < 2:
        return ""
    ys, vs = zip(*clean)
    delta = vs[-1] - vs[0]
    n_years = ys[-1] - ys[0]

    direction = "improved" if delta < 0 else "worsened"  # lower ERA/WHIP is better
    if stat in ("SO9",):
        direction = "improved" if delta > 0 else "declined"

    magnitude = abs(delta)
    if magnitude < 0.2:
        return f"Stable {stat} over {n_years} years."
    elif magnitude < 0.7:
        return f"{stat} has {direction} modestly ({delta:+.2f}) since {ys[0]}."
    else:
        return f"Significant {stat} movement: {direction} by {abs(delta):.2f} since {ys[0]}."
