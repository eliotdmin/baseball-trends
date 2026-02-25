"""
Pitcher Analytics Dashboard
============================
Tabs:
  1. Classic Clustering (KMeans)
  2. Modern Clustering (UMAP+HDBSCAN)
  3. Pitcher Similarity Search
  4. Traditional & Expected Stats (ERA/WHIP/SO9 + xERA, all years)
  5. Regression Predictor (ERA actual-vs-expected modelling)
  6. Raw Data Explorer

Run: streamlit run src/frontend/app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "backend"))

import unicodedata
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import json

# Light chart backgrounds — avoids weird colors when Streamlit is in dark mode
# Color schemes tuned for dark-theme app (plots on white, saturated colors for visibility)
pio.templates.default = "plotly_white"
CLUSTER_COLORS = list(px.colors.qualitative.Set1) + list(px.colors.qualitative.Vivid)  # Saturated, distinct
# Semantic colors
COLOR_LUCK_HIST = "#0284c7"       # Deep blue
COLOR_K9_SCATTER = "#10b981"      # Emerald
COLOR_QUALITY_DIST = "#8b5cf6"    # Violet

# Transparent plot backgrounds (blend with dark dashboard) + white text
_PLOT_TRANSPARENT = dict(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    xaxis=dict(tickfont=dict(color="white"), title_font=dict(color="white"), gridcolor="rgba(128,128,128,0.3)"),
    yaxis=dict(tickfont=dict(color="white"), title_font=dict(color="white"), gridcolor="rgba(128,128,128,0.3)"),
    legend=dict(font=dict(color="white")),
)


def _last_first_to_first_last(name: str) -> str:
    """Convert 'Last, First' to 'First Last' if comma present."""
    if not name or not isinstance(name, str):
        return name or ""
    if "," in name:
        parts = name.split(",", 1)
        return f"{parts[1].strip()} {parts[0].strip()}" if len(parts) == 2 else name
    return name


def _normalize_for_search(s) -> str:
    """Strip accents and lowercase for fuzzy search (e.g. José → jose)."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    nfd = unicodedata.normalize("NFD", str(s))
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn").lower()

st.set_page_config(
    page_title="Pitcher Analytics",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# UI polish: spacing, typography. Theme-agnostic — uses Streamlit default (light or dark).
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; }
    .stTabs [data-baseweb="tab"] { padding: 0.6rem 1.2rem; font-size: 0.95rem; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }
    h1, h2, h3 { font-weight: 600; letter-spacing: -0.02em; }
    .main-title { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.25rem; }
    .main-subtitle { font-size: 0.9rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = ROOT / "data"
PROCESSED_DIR = ROOT / "data" / "processed"
# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_cluster_data(_mtime: float = 0):
    """_mtime: file mtime as cache key — re-run pipeline updates file, invalidates cache."""
    path = PROCESSED_DIR / "pitcher_profiles_clustered.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


# Load with mtime so cache invalidates when pipeline regenerates files
def _cluster_data_mtime() -> float:
    p = PROCESSED_DIR / "pitcher_profiles_clustered.parquet"
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data
def load_kmeans_summaries(_mtime: float = 0):
    p = PROCESSED_DIR / "summaries" / "cluster_summaries_kmeans.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _kmeans_summaries_mtime() -> float:
    p = PROCESSED_DIR / "summaries" / "cluster_summaries_kmeans.json"
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data
def load_arsenal_importances():
    p = PROCESSED_DIR / "arsenal_feature_importances.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


@st.cache_data
def load_pitching_profiles(year: int):
    path = PROCESSED_DIR / f"pitching_profiles_{year}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


@st.cache_data
def load_pitching_trends():
    path = PROCESSED_DIR / "pitching_trends.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data
def load_arsenal_stats(years: tuple = tuple(range(2015, 2026))) -> pd.DataFrame:
    """Load per-pitch-type arsenal stats (run value, whiff%, hard-hit%, etc.)."""
    frames = []
    for yr in years:
        p = DATA_DIR / "statcast_pitcher_arsenal_stats" / f"data_{yr}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["year"] = yr
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


TARGETS_META = [
    dict(key="ERA",        label="ERA",             group="A", baseline="xera",     lower_better=True),
    dict(key="wOBA",       label="wOBA against",    group="A", baseline="est_woba", lower_better=True),
    dict(key="BA",         label="BA against",      group="A", baseline="est_ba",   lower_better=True),
    dict(key="ERA_resid",  label="ERA − xERA",      group="B", baseline=None,       lower_better=True),
    dict(key="wOBA_resid", label="wOBA − est.wOBA", group="B", baseline=None,       lower_better=True),
    dict(key="BA_resid",   label="BA − est.BA",     group="B", baseline=None,       lower_better=True),
]


@st.cache_data
def load_regression_outputs():
    out = {}
    # Comparison overview
    p = PROCESSED_DIR / "regression_comparison.parquet"
    if p.exists():
        out["comparison"] = pd.read_parquet(p)
    # Per-target files
    for t in TARGETS_META:
        k = t["key"]
        for prefix in ["proj_", "fi_", "cand_{}_unlucky", "cand_{}_lucky", "pred_vs_actual_"]:
            fname = prefix.format(k) if "{}" in prefix else f"{prefix}{k}"
            fp = PROCESSED_DIR / f"{fname}.parquet"
            if fp.exists():
                out[fname] = pd.read_parquet(fp)
    # Quality leaderboard
    ql = PROCESSED_DIR / "quality_leaderboard.parquet"
    if ql.exists():
        out["quality_leaderboard"] = pd.read_parquet(ql)
    # This-year vs next-year luck (ERA − xERA)
    luck_p = PROCESSED_DIR / "luck_this_vs_next.parquet"
    if luck_p.exists():
        out["luck_this_vs_next"] = pd.read_parquet(luck_p)
    # Regression metadata (e.g. latest_season for quality leaderboard)
    meta_p = PROCESSED_DIR / "regression_metadata.json"
    if meta_p.exists():
        try:
            out["regression_metadata"] = json.load(meta_p.open())
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Load data (before sidebar so cluster selector works)
# ---------------------------------------------------------------------------

df = load_cluster_data(_cluster_data_mtime())
kmeans_summaries = load_kmeans_summaries(_kmeans_summaries_mtime())
arsenal_importances = load_arsenal_importances()
trends_df = load_pitching_trends()
regression_outputs = load_regression_outputs()

_clustering_available = df is not None

PITCH_SYMBOL_KEY = {
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

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    with st.expander("How data updates work", expanded=False):
        st.markdown(
            "Ordinarily, this dashboard would allow you to **rerun the full pipeline** (clustering, regression, quality scores) with the push of a button. "
            "However, Streamlit Cloud's stateless execution and resource limits make it infeasible to run compute-heavy pipelines on demand—each rerun would rebuild from scratch with no persistent disk."
        )
        st.markdown(
            "Instead, the app **reads from pre-computed outputs** (parquet files, JSON summaries) generated when you run `python run_pipeline.py` locally. "
            "When deployed, the **sidebar has no flexibility** (year range and re-run clustering are hidden); "
            "however, some **individual tabs** offer filters, year selectors, and other controls that work on the pre-loaded data."
        )

    # Check if we have local pitch cache (Streamlit Cloud typically does not)
    _sc_root = ROOT / "data" / "statcast_pitches"
    _cached_dates = sorted(
        p.parent.name.replace("game_date=", "")
        for p in _sc_root.glob("game_date=*/pitches.parquet")
    ) if _sc_root.exists() else []
    _cached_years = sorted(set(d[:4] for d in _cached_dates)) if _cached_dates else []
    _has_cache = len(_cached_dates) >= 7

    if _has_cache:
        st.subheader("Year range")
        st.caption("Controls data range for stats, similarity, and re-run clustering.")
        _avail_years = list(range(2015, 2026))
        _start_yr = st.selectbox("Start year", _avail_years, index=5, key="start_year")
        _end_yr = st.selectbox("End year", _avail_years, index=10, key="end_year")
        if _start_yr > _end_yr:
            _end_yr, _start_yr = _start_yr, _end_yr
        all_years_mode = _start_yr < _end_yr
        selected_year = _start_yr if _start_yr == _end_yr else None
        _year_range = (_start_yr, _end_yr)
        _clust_years = [y for y in range(_start_yr, _end_yr + 1) if str(y) in _cached_years]
    else:
        # No pitch cache = deployed (Streamlit Cloud) or minimal local setup
        _year_range = (2020, 2025)
        all_years_mode = True
        selected_year = None
        st.info("**Deployed view:** Sidebar is fixed (2020–2025 data only; no re-run). Some tabs have in-tab filters and controls.")

    st.divider()

    st.subheader("Pitch symbols")
    st.markdown("\n".join(f"- **{c}** = {n}" for c, n in PITCH_SYMBOL_KEY.items()))

    # Only show Re-run clustering when local pitch cache exists (not on Streamlit Cloud)
    if _has_cache:
        with st.expander("Re-run clustering", expanded=False):
            st.caption("KMeans on arsenal features. ~30–60s. Uses year range above.")
            st.caption(f"Cache: {_cached_years[0] or '?'}–{_cached_years[-1] or '?'} ({len(_cached_dates)} days)")
            _k_clust = st.slider("KMeans k", 3, 12, 5, key="sidebar_k")

            if not _clust_years:
                st.warning(f"No cached data for {_start_yr}–{_end_yr}. Run download_statcast.py for those years.")
            if st.button("Re-run Clustering", use_container_width=True, disabled=not _clust_years):
                with st.spinner("Running…"):
                    try:
                        import importlib
                        import clustering
                        importlib.reload(clustering)
                        from preprocess_data import run_preprocessing
                        from llm_summaries import generate_all_summaries

                        _profiles = run_preprocessing(years=_clust_years, min_pitches=200)
                        _results = clustering.run_clustering_pipeline(_profiles, n_clusters=_k_clust)
                        from preprocess_data import get_clustering_features
                        _feats = get_clustering_features(_results["profiles"])
                        generate_all_summaries(_results["profiles"], _results["kmeans_labels"], _feats,
                                              use_llm=False, out_filename="cluster_summaries_kmeans.json")
                        _n = len(_results["profiles"])
                        st.success(f"Done! {_n} pitchers, k={_k_clust}")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as _e:
                        st.error(f"Failed: {_e}")

# ---------------------------------------------------------------------------
# Insights helpers
# ---------------------------------------------------------------------------

try:
    import importlib, sys
    # Force re-read from disk on every Streamlit rerun so hot-reload picks up
    # changes to backend modules (Python caches modules in sys.modules across reruns).
    if "text_insights" in sys.modules:
        importlib.reload(sys.modules["text_insights"])
    from text_insights import (
        describe_velo, describe_spin, describe_era, describe_whip,
        describe_so9, describe_xera_gap, describe_percentile,
        get_fastball_velo, get_primary_spin,
        build_pitcher_narrative, build_traditional_stats_narrative,
        build_trend_annotation,
    )
    _insights_available = True
except Exception:
    _insights_available = False

    def get_fastball_velo(row):
        for c in ["velo_FF", "velo_SI", "velo_FC"]:
            v = row.get(c)
            if pd.notna(v) and v > 50:
                return float(v)
        avg = row.get("avg_velo")
        return float(avg) if pd.notna(avg) and avg > 50 else None

    def get_primary_spin(row):
        best_pct, best_rpm, best_pt = 0.0, None, None
        for pt in ["FF", "SI", "SL", "CH", "CU", "FC", "ST", "KC", "FS"]:
            pct = row.get(f"pct_{pt}") or 0
            spin = row.get(f"spin_{pt}")
            if pd.notna(spin) and spin > 0 and pct > 0.03 and pct > best_pct:
                best_pct, best_rpm, best_pt = pct, float(spin), pt
        return (best_rpm, best_pt)

    def describe_velo(_):
        return ""

    def describe_spin(_):
        return ""


def compute_league_avg_pct(df: pd.DataFrame) -> dict:
    """Compute mean pitch-type proportion across all pitchers in the dataset."""
    pitch_types = ["FF", "SI", "SL", "CH", "CU", "FC", "ST", "KC", "FS"]
    avgs = {}
    for pt in pitch_types:
        col = f"pct_{pt}"
        if col in df.columns:
            avgs[pt] = float(df[col].mean())
    return avgs


# ---------------------------------------------------------------------------
# Plot / render helpers
# ---------------------------------------------------------------------------

import re as _re


def _narrative_to_plain(text: str) -> str:
    """Format narrative as flowing paragraphs (no bullets)."""
    if not text:
        return text
    # Collapse bullet-like lines into sentences
    out = text.replace("\n\n", "\n").replace("\n• ", ". ").replace("\n", " ")
    # Single space between sentences
    out = " ".join(out.split())
    return out


def _parse_archetype(summary_text: str) -> str:
    """
    Extract the archetype label from a rule-based summary string.
    Expected format: "Cluster N (M pitchers) — ARCHETYPE: …"
    Returns the archetype portion, or an empty string if not parseable.
    """
    m = _re.search(r"—\s*(.+?):", summary_text)
    return m.group(1).strip() if m else ""


def _truncate(s: str, n: int = 18) -> str:
    """Truncate a string to n characters, adding ellipsis if needed."""
    return s if len(s) <= n else s[:n - 1] + "…"


def _build_labels_map(summaries_dict: dict) -> dict[str, str]:
    """Return {cluster_id_str: 'N · Archetype (truncated)'} from a summaries JSON dict."""
    out = {}
    for cid, text in summaries_dict.items():
        arch = _parse_archetype(text)
        out[cid] = f"{cid} · {_truncate(arch, 20)}" if arch else cid
    return out


def cluster_scatter(df, cluster_col, title, labels_map: dict | None = None):
    """PCA projection of raw features — used for KMeans (no UMAP)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    numeric_cols = [
        "velo_FF", "velo_SI", "spin_FF", "spin_SL", "avg_extension",
        "cmd_plate_x_std", "cmd_plate_z_std",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    if len(available) < 4:
        available = [c for c in ["avg_velo", "avg_spin", "avg_extension", "cmd_plate_x_std", "cmd_plate_z_std"] if c in df.columns]
    X = df[available].fillna(df[available].median())
    X_scaled = StandardScaler().fit_transform(X)
    coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    raw_cluster = df[cluster_col].astype(str)
    label_series = raw_cluster.map(labels_map) if labels_map else raw_cluster
    label_series = label_series.fillna(raw_cluster)

    fb_velo = df.apply(lambda r: get_fastball_velo(r) if hasattr(r, "get") else None, axis=1)
    primary_spin = df.apply(lambda r: get_primary_spin(r)[0] if hasattr(r, "get") else None, axis=1)
    plot_df = pd.DataFrame({
        "PC1": coords[:, 0], "PC2": coords[:, 1],
        "Cluster": label_series,
        "Pitcher": df["player_name"],
        "FB velo": fb_velo,
        "Primary spin": primary_spin,
    })
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster",
                     hover_data=["Pitcher", "FB velo", "Primary spin"], title=title, opacity=0.85,
                     color_discrete_sequence=CLUSTER_COLORS)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(height=550, legend_title_text="Cluster · Archetype",
                      **_PLOT_TRANSPARENT)
    return fig


def umap_scatter(df, cluster_col, labels_map: dict | None = None,
                 x_axis: str = "umap_0", y_axis: str = "umap_1", algo_label: str = ""):
    """Scatter using saved UMAP dimensions. Pick any 2 axes when clustering used 4D+."""
    umap_cols = [c for c in df.columns if c.startswith("umap_")]
    if not umap_cols:
        return cluster_scatter(df, cluster_col, f"{algo_label or 'Clusters'} (PCA fallback)", labels_map)

    if x_axis not in df.columns:
        x_axis = umap_cols[0]
    if y_axis not in df.columns:
        y_axis = umap_cols[1] if len(umap_cols) > 1 else umap_cols[0]

    raw_cluster = df[cluster_col].astype(str)
    label_series = raw_cluster.map(labels_map) if labels_map else raw_cluster
    label_series = label_series.fillna(raw_cluster)

    fb_velo = df.apply(lambda r: get_fastball_velo(r) if hasattr(r, "get") else None, axis=1)
    primary_spin = df.apply(lambda r: get_primary_spin(r)[0] if hasattr(r, "get") else None, axis=1)
    n_dims = len(umap_cols)
    plot_df = pd.DataFrame({
        x_axis: df[x_axis], y_axis: df[y_axis],
        "Cluster": label_series,
        "Pitcher": df["player_name"],
        "FB velo": fb_velo,
        "Primary spin": primary_spin,
    })
    title = f"{algo_label} ({n_dims}D UMAP: {x_axis} vs {y_axis})" if algo_label else f"Clusters ({n_dims}D UMAP)"
    fig = px.scatter(plot_df, x=x_axis, y=y_axis, color="Cluster",
                     hover_data=["Pitcher", "FB velo", "Primary spin"],
                     title=title, opacity=0.7)
    fig.update_layout(height=550, legend_title_text="Cluster · Archetype",
                      **_PLOT_TRANSPARENT)
    return fig


def render_pitcher_card(row: pd.Series, league_avg_pct: dict = None):
    st.markdown(f"### {_last_first_to_first_last(row.get('player_name', 'Unknown'))}")
    cols = st.columns(4)
    velo = get_fastball_velo(row)
    spin_rpm, spin_pt = get_primary_spin(row)
    ext = row.get("avg_extension")
    hard_hit = row.get("hard_hit_pct")

    if pd.notna(velo) and velo > 50:  # 0 = invalid (pitch not thrown)
        cols[0].metric("Fastball velo", f"{velo:.1f} mph",
                        help=describe_velo(velo) if _insights_available else "")
    if pd.notna(spin_rpm):
        spin_label = f"{spin_pt} spin" if spin_pt else "Primary spin"
        cols[1].metric(spin_label, f"{spin_rpm:.0f} rpm",
                        help=describe_spin(spin_rpm) if _insights_available else "")
    if pd.notna(ext):
        cols[2].metric("Extension", f"{ext:.1f} ft")
    if pd.notna(hard_hit):
        pct = hard_hit * 100 if hard_hit <= 1 else hard_hit
        cols[3].metric("Hard Hit% Against", f"{pct:.1f}%")

    if _insights_available:
        try:
            narrative = build_pitcher_narrative(row, league_avg_pct=league_avg_pct)
        except TypeError:
            narrative = build_pitcher_narrative(row)
        st.info(narrative)


def render_traditional_stats_card(row: pd.Series):
    era = row.get("ERA") or row.get("era")
    whip = row.get("WHIP") or row.get("whip")
    so9 = row.get("SO9") or row.get("so9")
    xera = row.get("xera")
    ip = row.get("IP")

    cols = st.columns(5)
    if pd.notna(era):
        cols[0].metric("ERA", f"{era:.2f}",
                        help=describe_era(era) if _insights_available else "")
    if pd.notna(xera):
        delta = f"{era - xera:+.2f} vs xERA" if pd.notna(era) else None
        cols[1].metric("xERA", f"{xera:.2f}", delta=delta, delta_color="inverse",
                        help="Expected ERA from quality of contact. ERA < xERA = lucky; ERA > xERA = unlucky.")
    if pd.notna(whip):
        cols[2].metric("WHIP", f"{whip:.3f}",
                        help=describe_whip(whip) if _insights_available else "")
    if pd.notna(so9):
        cols[3].metric("K/9", f"{so9:.1f}",
                        help=describe_so9(so9) if _insights_available else "")
    if pd.notna(ip):
        cols[4].metric("IP", f"{ip:.1f}")

    if _insights_available and pd.notna(era):
        narrative = build_traditional_stats_narrative(row)
        st.info(narrative)


# ---------------------------------------------------------------------------
# Title & Hero image
# ---------------------------------------------------------------------------

st.title("⚾ Pitcher Analytics Dashboard")

_n_pitchers = len(df) if df is not None else 0
_n_pitches = len(PITCH_SYMBOL_KEY)
_intro = (
    f"There are {_n_pitchers:,} pitchers, {_n_pitches} distinct pitch types, and any possible combination of "
    "velocity, spin, and arm slots to throw from—plus a bevy of performance metrics to analyze. "
    "Simply scanning the data and having a sense of the different types of pitchers in MLB is messy and overwhelming. "
    "This dashboard intends to make sense of the madness and provide a tool to compare pitchers "
    f"or find trends in performance. Data spans **{_year_range[0]}–{_year_range[1]}**."
)

# Max Scherzer pitching (Wikimedia Commons, CC0). Use smaller thumb (400px) for faster load.
HERO_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/"
    "Max_Scherzer_pitching%2C_March_30%2C_2023_%281%29_%28cropped%29.jpg/"
    "400px-Max_Scherzer_pitching%2C_March_30%2C_2023_%281%29_%28cropped%29.jpg"
)
col_intro, col_photo = st.columns([3, 2])
with col_intro:
    st.markdown(_intro)
with col_photo:
    st.markdown(
        f'<div style="text-align:center;">'
        f'<div style="display:inline-block; max-width:100%;">'
        f'<img src="{HERO_IMAGE_URL}" alt="Max Scherzer pitching" style="max-width:100%; max-height:220px; object-fit:contain; display:block;" loading="eager" decoding="async" />'
        f'</div></div>',
        unsafe_allow_html=True,
    )
with st.expander("📖 Methodology & how this dashboard works", expanded=False):
    st.markdown("""
    **Why arsenal-based?**  
    Pitchers are defined by *what they throw*, not just box scores. Two guys with 3.50 ERA can be totally different—one survives on weak contact, another on pure strikeouts. I wanted to group by arsenal (velo, spin, movement, pitch mix) so comps actually reflect *how* they pitch, not random variance.

    **Clustering**  
    KMeans on scaled arsenal features: usage, velo, spin, movement, extension, arm angle per pitch type. **Why KMeans?** Simple, interpretable, and the rule-based narratives give each cluster a human-readable label. Used for roster grouping and similarity.

    **Similarity search**  
    Euclidean distance in the same feature space. **Why arsenal and not stats?** Stats are noisy; arsenals are more stable. "Pitchers who throw like X" is useful for comps and role projections.

    **Feature importance** (Regression tab)  
    Random Forest predicts **xERA** from arsenal features. **Why exclude hard-hit%, whiff%, etc.?** Those are *outcomes*. I wanted to know which *traits* (velo, spin, mix) drive expected performance—so we know what to prioritise when evaluating pitchers.

    **Null values & rare pitches**  
    Pitches thrown &lt;1% get zeroed out. Median imputation for missing extension, command. Keeps things stable without blowing up noise from one-off offerings.

    **Data sources**  
    Statcast (pitch-level), Baseball Reference, plus pipeline outputs.
    """)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_cluster, tab_search, tab_rosters, tab_trad, tab_quality, tab_regression, tab_data, tab_planned = st.tabs([
    "Overview",
    "Clustering",
    "Similarity",
    "Performance by cluster",
    "Stats",
    "Pitcher ranking tool",
    "Regression",
    "Raw + intermediate Data",
    "Planned follow-ups",
])


# ===========================================================================
# Tab: Overview
# ===========================================================================

with tab_overview:
    st.header("Overview")
    st.markdown(
        "Each tab addresses a distinct question: *What types of pitchers exist?* *How similar is pitcher X to pitcher Y?* "
        "*How do they perform—and how much is luck vs skill?*"
    )
    st.markdown(
        "Fantasy rankings often treat pitchers as interchangeable after ERA and K adjustments. "
        "Arsenals vary widely; these tools support exploration and valuation."
    )
    st.markdown("### What each tab does")
    st.markdown(
        "| Tab | Focus |\n"
        "|-----|-------|\n"
        "| **Clustering** | KMeans partition of pitchers by arsenal (velo, spin, movement per pitch type). PCA projection and cluster narratives. |\n"
        "| **Similarity** | Find pitchers with similar arsenals via weighted Euclidean distance. Compare two pitchers side by side. |\n"
        "| **Performance by cluster** | ERA/WHIP/SO9 by cluster for a given year. Rosters of who's in each cluster. |\n"
        "| **Stats** | Traditional stats (ERA, WHIP, K/9) + xERA. Scatter of ERA vs xERA. |\n"
        "| **Pitcher ranking tool** | Custom quality score (xERA, K%, whiff%, etc.). Grade distribution and leaderboard. |\n"
        "| **Regression** | Can we predict luck? ERA−xERA residual forecasting. Next-season projections. Arsenal → xERA feature importance. |\n"
        "| **Raw + intermediate Data** | Pitcher profiles, clustering feature matrix, exported parquet tables. |\n"
        "| **Planned follow-ups** | Proposed analyses: first-half vs second-half predictiveness, pull rate vs xBA, aging curves, platoon splits by archetype, injury signals. |\n"
    )
    st.caption("Pitch symbols (FF, SI, SL, etc.) are in the sidebar — available on every tab.")


# ===========================================================================
# Tab: Clustering
# ===========================================================================

with tab_cluster:
    st.header("Clustering")
    if not _clustering_available:
        st.info("Run `python run_pipeline.py` to generate clustering.")
    else:
        st.markdown(
            "KMeans clustering partitions pitchers by arsenal: pitch mix, velocity, spin, and movement per pitch type. "
            "Arsenal-based grouping isolates stable archetypes (e.g. power fastball/slider vs sinker/changeup specialists) without conflating skill and luck. "
            "For trait–performance relationships, see **Regression**."
        )
        st.caption(
            "We've added the 3D PCA view to give an idea of the distribution of different pitcher types."
        )
        sel_col = "cluster_kmeans" if "cluster_kmeans" in df.columns else None
        if sel_col:
            summ = kmeans_summaries
            labels_map = _build_labels_map(summ) if summ else None
            col_viz, col_info = st.columns([3, 1])
            with col_viz:
                x_ax = "pca_0" if "pca_0" in df.columns else None
                y_ax = "pca_1" if "pca_1" in df.columns else None
                z_ax = "pca_2" if "pca_2" in df.columns else None
                if x_ax and y_ax and z_ax:
                    fig = px.scatter_3d(
                        df, x=x_ax, y=y_ax, z=z_ax, color=df[sel_col].astype(str),
                        hover_data=["player_name"], title="KMeans clusters (3D PCA projection)",
                        color_discrete_sequence=CLUSTER_COLORS,
                    )
                    fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color="white")))
                    fig.update_layout(height=550, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        scene=dict(
                            bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(tickfont=dict(color="white"), title_font=dict(color="white"), gridcolor="rgba(128,128,128,0.3)"),
                            yaxis=dict(tickfont=dict(color="white"), title_font=dict(color="white"), gridcolor="rgba(128,128,128,0.3)"),
                            zaxis=dict(tickfont=dict(color="white"), title_font=dict(color="white"), gridcolor="rgba(128,128,128,0.3)"),
                        ), font=dict(color="white"), legend=dict(font=dict(color="white")))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = cluster_scatter(df, sel_col, "KMeans (PCA)", labels_map=labels_map)
                    st.plotly_chart(fig, use_container_width=True)

            with col_info:
                sizes = df[sel_col].value_counts().sort_index()
                st.metric("Clusters", len(sizes))
                st.bar_chart(sizes)

            if summ:
                with st.expander("Cluster narratives", expanded=True):
                    for cid, text in sorted(summ.items(), key=lambda x: int(x[0])):
                        n = int((df[sel_col] == int(cid)).sum())
                        arch = _parse_archetype(text)
                        if arch:
                            st.markdown(f"### Cluster {cid} · **{arch}**")
                            st.caption(f"{n} pitchers")
                            # Skip redundant "Cluster N — ARCHETYPE:" line when we have header
                            display_text = _narrative_to_plain(text)
                            if "\n" in display_text and "—" in display_text.split("\n")[0]:
                                display_text = "\n".join(display_text.split("\n")[1:]).strip()
                        else:
                            st.markdown(f"**Cluster {cid}** ({n})")
                            display_text = _narrative_to_plain(text)
                        st.markdown(display_text[:600] + "…" if len(display_text) > 600 else display_text)


# ===========================================================================
# Tab: Similarity Search
# ===========================================================================

with tab_search:
    st.header("Similarity Search")
    if not _clustering_available:
        st.info("Run the clustering pipeline first to use similarity search.")
    else:
        league_avg_pct = compute_league_avg_pct(df)

        st.markdown(
            "This tab is a more granular focus on pitcher-by-pitcher similarity instead of large-scale clustering. Pitcher-by-pitcher similarity is computed from **Statcast arsenal profiles**—velocity, spin, movement per pitch type, and pitch mix. "
            "Arsenal-based matching yields stylistic comps (useful for role projections and buy-low identification) rather than surface-stat similarity."
        )
        st.markdown(
            "Each pitch type is compared apples-to-apples (e.g., four-seamer vs four-seamer). "
        )
        pitcher_list = sorted(df["player_name"].dropna().unique())
        selected = st.selectbox("Select a pitcher", pitcher_list,
                                format_func=lambda x: _last_first_to_first_last(x))

        if selected:
            from clustering import find_similar_pitchers
            from preprocess_data import prepare_clustering_matrix, get_similarity_features

            features = get_similarity_features(df)
            X_scaled, features, _ = prepare_clustering_matrix(df, features)
            similar = find_similar_pitchers(df, X_scaled, selected, n=10, feature_names=features)

            pitcher_row = df[df["player_name"] == selected].iloc[0]
            cluster_col = "cluster_kmeans" if "cluster_kmeans" in df.columns else None
            cluster_id = int(pitcher_row.get(cluster_col, -1))

            st.divider()
            render_pitcher_card(pitcher_row, league_avg_pct=league_avg_pct)

            # Surface stats for selected pitcher
            if trends_df is not None and len(trends_df) > 0 and "mlbID" in trends_df.columns:
                pid = pitcher_row.get("pitcher")
                if pd.notna(pid):
                    pid = pd.to_numeric(pid, errors="coerce")
                    _yr_start, _yr_end = _year_range
                    pt = trends_df[(trends_df["mlbID"] == pid) & (trends_df["year"] >= _yr_start) & (trends_df["year"] <= _yr_end)]
                    if not pt.empty:
                        pt = pt.sort_values("year", ascending=False).iloc[0]
                        surf = [f"**ERA** {float(pt['ERA']):.2f}" if pd.notna(pt.get('ERA')) else "",
                                f"**WHIP** {float(pt['WHIP']):.3f}" if pd.notna(pt.get('WHIP')) else "",
                                f"**K/9** {float(pt['SO9']):.1f}" if pd.notna(pt.get('SO9')) else "",
                                f"**IP** {float(pt['IP']):.0f}" if pd.notna(pt.get('IP')) else ""]
                        surf = [s for s in surf if s]
                        if surf:
                            st.caption(f"Surface stats ({int(pt['year'])}): " + " · ".join(surf))

            if cluster_col and cluster_id >= 0:
                arch = _parse_archetype(kmeans_summaries.get(str(cluster_id), "")) if (kmeans_summaries and str(cluster_id) in kmeans_summaries) else ""
                cluster_label = f"Cluster {cluster_id}" + (f" · {arch}" if arch else "")
            else:
                cluster_label = "—"
            st.markdown(
                f'<div style="font-size: 1.4rem; font-weight: 600; margin: 0.5rem 0;">'
                f'Cluster Assignment: {cluster_label}</div>',
                unsafe_allow_html=True,
            )
            if kmeans_summaries and cluster_col and str(cluster_id) in kmeans_summaries:
                st.markdown("**Cluster narrative**")
                st.markdown(_narrative_to_plain(kmeans_summaries[str(cluster_id)]))

            st.divider()
            st.subheader(f"10 Most Similar Pitchers to {_last_first_to_first_last(selected)}")
            st.caption(
                "Ranked by Euclidean distance over per-pitch-type features (usage, velo, spin, movement). "
                "Lower = more similar. Euclidean penalizes velocity differences (e.g. 92 vs 95 mph)."
            )

            # Enrich similar with surface stats (ERA, WHIP, SO9, IP) from trends_df
            similar_display = similar.copy()
            similar_display = similar_display.rename(columns={"player_name": "Pitcher", "distance": "Arsenal Distance"})
            similar_display["Pitcher"] = similar_display["Pitcher"].apply(_last_first_to_first_last)
            similar_display["Arsenal Distance"] = similar_display["Arsenal Distance"].round(4)
            if trends_df is not None and len(trends_df) > 0 and "mlbID" in trends_df.columns and "year" in trends_df.columns:
                _yr_start, _yr_end = _year_range
                trends_in_range = trends_df[(trends_df["year"] >= _yr_start) & (trends_df["year"] <= _yr_end)]
                if len(trends_in_range) > 0:
                    # Most recent year per pitcher in range
                    latest = trends_in_range.sort_values("year", ascending=False).drop_duplicates("mlbID", keep="first")
                    stat_cols = [c for c in ["ERA", "WHIP", "SO9", "IP"] if c in latest.columns]
                    if stat_cols:
                        stats_sub = latest[["mlbID", "year"] + stat_cols].copy()
                        stats_sub["mlbID"] = pd.to_numeric(stats_sub["mlbID"], errors="coerce")
                        similar_display["pitcher"] = pd.to_numeric(similar_display["pitcher"], errors="coerce")
                        similar_display = similar_display.merge(
                            stats_sub.rename(columns={"year": "Year"}),
                            left_on="pitcher", right_on="mlbID", how="left"
                        ).drop(columns=["mlbID"], errors="ignore")
            disp_cols = ["Pitcher", "Arsenal Distance"] + [c for c in ["ERA", "WHIP", "SO9", "IP", "Year"] if c in similar_display.columns]
            if "Year" in disp_cols:
                st.caption(f"Surface stats from most recent year in selected range ({_year_range[0]}–{_year_range[1]}).")
            st.dataframe(similar_display[disp_cols], use_container_width=True, column_config={
                "ERA": st.column_config.NumberColumn(format="%.2f"),
                "WHIP": st.column_config.NumberColumn(format="%.3f"),
                "SO9": st.column_config.NumberColumn("K/9", format="%.1f"),
                "IP": st.column_config.NumberColumn(format="%.1f"),
            })

            st.markdown("**Top 3 similar pitchers — quick profile:**")
            cols = st.columns(3)
            # Build surface stats lookup for similar pitchers
            surf_lookup = {}
            if trends_df is not None and len(trends_df) > 0 and "mlbID" in trends_df.columns:
                _yr_start, _yr_end = _year_range
                trends_in_range = trends_df[(trends_df["year"] >= _yr_start) & (trends_df["year"] <= _yr_end)]
                if len(trends_in_range) > 0:
                    latest = trends_in_range.sort_values("year", ascending=False).drop_duplicates("mlbID", keep="first")
                    for _, row in latest.iterrows():
                        surf_lookup[int(row["mlbID"])] = row
            for i, (_, sim_row) in enumerate(similar.head(3).iterrows()):
                sim_name = sim_row["player_name"]
                sim_data = df[df["player_name"] == sim_name]
                if not sim_data.empty:
                    with cols[i]:
                        sr = sim_data.iloc[0]
                        st.markdown(f"**{_last_first_to_first_last(sim_name)}**")
                        pid = pd.to_numeric(sr.get("pitcher"), errors="coerce")
                        if pd.notna(pid) and int(pid) in surf_lookup:
                            pt = surf_lookup[int(pid)]
                            st.write(f"ERA {pt.get('ERA', 0):.2f} · WHIP {pt.get('WHIP', 0):.3f} · K/9 {pt.get('SO9', 0):.1f} · {pt.get('IP', 0):.0f} IP")
                        velo = get_fastball_velo(sr)
                        spin_rpm, spin_pt = get_primary_spin(sr)
                        if pd.notna(velo) and velo > 50:
                            tier = describe_velo(velo) if _insights_available else ""
                            st.write(f"🔥 {velo:.1f} mph FB — {tier}")
                        if pd.notna(spin_rpm):
                            tier = describe_spin(spin_rpm) if _insights_available else ""
                            lbl = f"{spin_pt} " if spin_pt else ""
                            st.write(f"🌀 {lbl}{spin_rpm:.0f} rpm — {tier}")
                        pitch_mix = {k: sr.get(f"pct_{k}", 0)
                                     for k in ["FF","SI","SL","CH","CU","FC","ST"]}
                        top_p = max(pitch_mix, key=pitch_mix.get)
                        lg = league_avg_pct.get(top_p, 0)
                        diff = pitch_mix[top_p] - lg
                        st.write(f"🎯 {top_p} {pitch_mix[top_p]:.0%} ({diff:+.0%} vs avg)")
                        st.caption(f"Distance: {sim_row.get('distance', 0):.4f}")

        st.divider()
        st.subheader("Compare Two Pitchers")
        comp_cols = st.columns(2)
        with comp_cols[0]:
            p1 = st.selectbox("Pitcher A", pitcher_list, key="comp_a",
                              format_func=lambda x: _last_first_to_first_last(x))
        with comp_cols[1]:
            p2 = st.selectbox("Pitcher B", pitcher_list, key="comp_b",
                             index=min(1, len(pitcher_list) - 1) if pitcher_list else 0,
                             format_func=lambda x: _last_first_to_first_last(x))
        if p1 and p2 and p1 != p2:
            from preprocess_data import prepare_clustering_matrix, get_similarity_features, compute_weighted_distances_from_row
            _feat = get_similarity_features(df)
            _X, _feat, _ = prepare_clustering_matrix(df, _feat)
            _i1 = int(np.where(df["player_name"].values == p1)[0][0])
            _i2 = int(np.where(df["player_name"].values == p2)[0][0])
            _dists = compute_weighted_distances_from_row(df, _X, _feat, _i1)
            _d = float(_dists[_i2])
            sim_pct = max(0, min(100, 100 * (1 - _d / 8)))  # 0–8 dist maps to 100–0% similar
            st.markdown(
                f'<div style="text-align: center; padding: 1.25rem; background: linear-gradient(135deg, #0d9488 0%, #06b6d4 50%, #0ea5e9 100%); '
                f'border-radius: 12px; margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(6,182,212,0.3); border: 2px solid rgba(255,255,255,0.3);">'
                f'<span style="font-size: 2.5rem; font-weight: bold; color: white; text-shadow: 0 1px 2px rgba(0,0,0,0.2);">'
                f'{sim_pct:.0f}% similar</span><br><span style="color: rgba(255,255,255,0.95); font-size: 0.9rem;">'
                f'Arsenal distance: {_d:.3f} (lower = more similar)</span></div>',
                unsafe_allow_html=True,
            )
            r1 = df[df["player_name"] == p1].iloc[0]
            r2 = df[df["player_name"] == p2].iloc[0]
            compare_rows = []
            surf_rows = []
            if trends_df is not None and len(trends_df) > 0 and "mlbID" in trends_df.columns:
                _yr_start, _yr_end = _year_range
                for name, row in [(p1, r1), (p2, r2)]:
                    pid = pd.to_numeric(row.get("pitcher"), errors="coerce")
                    pt = trends_df[(trends_df["mlbID"] == pid) & (trends_df["year"] >= _yr_start) & (trends_df["year"] <= _yr_end)]
                    if not pt.empty:
                        pt = pt.sort_values("year", ascending=False).iloc[0]
                        surf_rows.append((name, pt))
                if len(surf_rows) == 2:
                    pt1, pt2 = surf_rows[0][1], surf_rows[1][1]
                    for lbl, col in [("IP", "IP"), ("K/9", "SO9"), ("WHIP", "WHIP"), ("ERA", "ERA")]:
                        if col in pt1.index and col in pt2.index:
                            v1, v2 = pt1[col], pt2[col]
                            diff = float(v1) - float(v2) if pd.notna(v1) and pd.notna(v2) else None
                            fmt = "{:.0f}" if col == "IP" else ("{:.1f}" if col == "SO9" else "{:.2f}" if col == "ERA" else "{:.3f}")
                            diff_fmt = {"IP": "+.0f", "SO9": "+.1f", "WHIP": "+.3f", "ERA": "+.2f"}
                            diff_str = f"{diff:{diff_fmt.get(col, '+.2f')}}" if diff is not None else "—"
                            compare_rows.insert(0, {
                                "Attribute": f"Surface · {lbl}",
                                p1: fmt.format(float(v1)) if pd.notna(v1) else "—",
                                p2: fmt.format(float(v2)) if pd.notna(v2) else "—",
                                "Δ (A−B)": diff_str,
                            })
            velo_cols = [c for c in df.columns if c.startswith("velo_")]
            spin_cols = [c for c in df.columns if c.startswith("spin_")]
            break_cols = [c for c in df.columns if c.startswith("break_")]
            axis_cols = [c for c in df.columns if c.startswith("spin_axis_")]
            pct_cols = [c for c in df.columns if c.startswith("pct_")]
            for c in velo_cols + spin_cols + break_cols + axis_cols:
                v1, v2 = r1.get(c), r2.get(c)
                if pd.notna(v1) or pd.notna(v2):
                    diff = float(v1) - float(v2) if pd.notna(v1) and pd.notna(v2) else None
                    if "velo" in c:
                        unit = " mph"
                    elif "spin_axis" in c:
                        unit = "°"
                    elif "break" in c:
                        unit = " in"
                    else:
                        unit = " rpm"
                    label = (c.replace("spin_axis_", "axis ")
                             .replace("break_z_", "IVB ").replace("break_x_", "HB ")
                             .replace("velo_", "velo ").replace("spin_", "spin "))
                    compare_rows.append({
                        "Attribute": label,
                        p1: f"{float(v1):.1f}{unit}" if pd.notna(v1) else "—",
                        p2: f"{float(v2):.1f}{unit}" if pd.notna(v2) else "—",
                        "Δ (A−B)": f"{diff:+.1f}" if diff is not None else "—",
                    })
            for c in pct_cols:
                v1, v2 = r1.get(c), r2.get(c)
                if (pd.notna(v1) and v1 > 0.01) or (pd.notna(v2) and v2 > 0.01):
                    pct1 = f"{float(v1)*100:.0f}%" if pd.notna(v1) else "—"
                    pct2 = f"{float(v2)*100:.0f}%" if pd.notna(v2) else "—"
                    diff = (float(v1 or 0) - float(v2 or 0)) * 100 if pd.notna(v1) and pd.notna(v2) else None
                    compare_rows.append({
                        "Attribute": c.replace("pct_", "usage "),
                        p1: pct1,
                        p2: pct2,
                        "Δ (A−B)": f"{diff:+.0f}pp" if diff is not None else "—",
                    })
            fb1, fb2 = get_fastball_velo(r1), get_fastball_velo(r2)
            compare_rows.append({
                "Attribute": "Fastball velo (best)",
                p1: f"{fb1:.1f} mph" if pd.notna(fb1) else "—",
                p2: f"{fb2:.1f} mph" if pd.notna(fb2) else "—",
                "Δ (A−B)": f"{fb1 - fb2:+.1f} mph" if pd.notna(fb1) and pd.notna(fb2) else "—",
            })
            comp_df = pd.DataFrame(compare_rows)
            comp_df = comp_df.rename(columns={
                p1: _last_first_to_first_last(p1),
                p2: _last_first_to_first_last(p2),
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ===========================================================================
# Tab: Cluster Rosters & Performance
# ===========================================================================

with tab_rosters:
    st.header("Performance by Cluster")
    if not _clustering_available:
        st.info("Run the clustering pipeline first to see cluster rosters and performance.")
    elif trends_df is None or (hasattr(trends_df, "__len__") and len(trends_df) == 0):
        st.info("Run the pitching stats pipeline to see cluster performance by year.")
    elif "year" not in trends_df.columns or len(trends_df) == 0:
        st.info("Run the pitching stats pipeline to see cluster performance by year.")
    else:
        st.markdown(
            "Cluster membership and performance (ERA, WHIP, SO9) by year. "
            "Identifies which arsenal archetypes outperform and supports drafting and valuation decisions."
        )
        _roster_year = st.selectbox(
            "Performance year",
            options=sorted(trends_df["year"].dropna().unique().astype(int), reverse=True),
            key="roster_year",
            help="ERA/WHIP/SO9 are from this season.",
        )
        cluster_col = "cluster_kmeans" if "cluster_kmeans" in df.columns else None
        if not cluster_col:
            st.warning("No cluster labels. Re-run clustering pipeline.")
            st.stop()

        cluster_map = df[["pitcher", "player_name", cluster_col]].copy()
        cluster_map = cluster_map.rename(columns={"pitcher": "mlbID"})
        cluster_map["mlbID"] = pd.to_numeric(cluster_map["mlbID"], errors="coerce")
        trends_yr = trends_df[trends_df["year"] == _roster_year].copy()
        if "mlbID" not in trends_yr.columns:
            st.warning("Pitching trends missing mlbID — re-run pipeline.")
        trends_yr["mlbID"] = pd.to_numeric(trends_yr["mlbID"], errors="coerce")
        merged = trends_yr.merge(cluster_map, on="mlbID", how="inner")
        merged = merged[merged["IP"].fillna(0) >= 20]  # min 20 IP

        if merged.empty:
            st.warning(f"No pitchers with ≥20 IP in {_roster_year} matched to clusters.")
        else:
            # Cluster performance summary
            perf = merged.groupby(cluster_col).agg(
                n=("Name", "count"),
                ERA=("ERA", "mean"),
                WHIP=("WHIP", "mean"),
                SO9=("SO9", "mean"),
            ).round(3)
            perf = perf[perf.index >= 0]  # exclude noise -1
            perf = perf.sort_values("ERA")
            perf.index = perf.index.astype(int)
            summaries_roster = kmeans_summaries or {}
            perf["Cluster"] = [f"{cid} · {_parse_archetype(summaries_roster.get(str(cid), '')) or '—'}"
                               for cid in perf.index]
            # Keep Cluster as column so names show in table
            perf = perf.reset_index(drop=True)
            perf = perf[["Cluster", "n", "ERA", "WHIP", "SO9"]]
            st.subheader(f"Cluster Performance ({_roster_year})")
            st.caption("Sorted by ERA (best first). Cluster names from arsenal archetypes.")
            st.dataframe(perf.rename(columns={"n": "# pitchers"}), use_container_width=True, hide_index=True)

            # Roster per cluster (summaries_roster already set above)
            st.subheader("Roster by Cluster")
            for cid in sorted(merged[cluster_col].unique()):
                if cid < 0:
                    continue
                subset = merged[merged[cluster_col] == cid].sort_values("ERA")
                arch = _parse_archetype(summaries_roster.get(str(int(cid)), "")) if summaries_roster else ""
                header = f"Cluster {cid}" + (f" · {arch}" if arch else "") + f" — {len(subset)} pitchers"
                with st.expander(header):
                    cols = ["Name", "ERA", "WHIP", "SO9", "IP"]
                    cols = [c for c in cols if c in subset.columns]
                    st.dataframe(subset[cols].reset_index(drop=True), use_container_width=True, hide_index=True)


# ===========================================================================
# Tab: Traditional & Expected Stats
# ===========================================================================

with tab_trad:
    st.header("Traditional & Expected Stats")
    st.markdown(
        "Traditional stats (ERA, WHIP, K/9) from Baseball Reference, enriched with Statcast **expected** metrics (xERA, est.wOBA). "
        "Comparing actual to expected helps separate contact quality and skill from luck, defense, sequencing, and park effects."
    )
    st.markdown(
        "ERA is inherently noisy—it reflects defense, batted-ball sequencing, and park factors. xERA and est.wOBA strip these out; "
        "they represent what *should* have happened given the contact allowed. Pitchers with ERA &gt; xERA were often unlucky; "
        "ERA &lt; xERA suggests luck or execution beyond expectations. The **Regression** tab models whether luck is predictable."
    )

    # --- Load data for selected season(s) ---
    if all_years_mode:
        _yr_start, _yr_end = _year_range
        pitching_df = trends_df[(trends_df["year"] >= _yr_start) & (trends_df["year"] <= _yr_end)] if "year" in trends_df.columns else trends_df
        season_label = f"{_yr_start}–{_yr_end}"
    else:
        pitching_df = load_pitching_profiles(selected_year)
        season_label = str(selected_year)

    if pitching_df is None or (hasattr(pitching_df, "__len__") and len(pitching_df) == 0):
        st.warning(
            f"No pitching data found for {season_label}. Run:\n"
            "```\npython src/backend/preprocess_pitching_bref.py\n```"
        )
        st.stop()

    year_col = "year" if "year" in pitching_df.columns else None

    n_rows = len(pitching_df)
    year_range = f"{pitching_df[year_col].min()}–{pitching_df[year_col].max()}" if year_col else season_label
    st.caption(f"{n_rows} pitcher-seasons · {year_range} · min 20 IP")

    display_df = pitching_df.copy()

    # --- Visuals (scatter, luck dist, etc.) ---
    st.subheader(f"Visuals — {season_label}")

    # ERA vs xERA scatter
    scatter_src = display_df.dropna(subset=["ERA", "xera"]).copy()
    if len(scatter_src) > 10:
        scatter_src["era_vs_xera"] = scatter_src["ERA"] - scatter_src["xera"]
        hover = ["Name", "Tm", "IP"] + (["year"] if all_years_mode else [])
        hover = [c for c in hover if c in scatter_src.columns]
        fig_s = px.scatter(
            scatter_src, x="xera", y="ERA", color="era_vs_xera",
            hover_data=hover,
            color_continuous_scale="RdBu_r",   # Red=unlucky, blue=lucky (diverging)
            labels={"xera": "xERA (expected)", "ERA": "Actual ERA"},
            title=f"ERA vs xERA — {season_label}",
        )
        fig_s.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
        mn = min(scatter_src[["ERA", "xera"]].min())
        mx = max(scatter_src[["ERA", "xera"]].max())
        fig_s.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                        line=dict(dash="dash", color="rgba(200,200,200,0.7)"))
        fig_s.add_annotation(x=mx * 0.92, y=mx * 0.92, text="ERA = xERA",
                              showarrow=False, font=dict(color="white", size=11))
        fig_s.update_layout(height=500, coloraxis_showscale=False, **_PLOT_TRANSPARENT)
        st.plotly_chart(fig_s, use_container_width=True)
        st.caption(
            "**Above the diagonal:** ERA exceeded xERA—unlucky outcomes or poor defense. "
            "**Below the diagonal:** ERA beat xERA—lucky outcomes or elite execution. "
            "Pitchers far from the line are strong regression candidates; the scatter helps identify buy-low or sell-high opportunities."
        )

    # Luck distribution (ERA − xERA)
    if "era_minus_xera" in display_df.columns or ("ERA" in display_df.columns and "xera" in display_df.columns):
        luck_src = display_df.copy()
        if "era_minus_xera" not in luck_src.columns and "ERA" in luck_src.columns and "xera" in luck_src.columns:
            luck_src["era_minus_xera"] = luck_src["ERA"] - luck_src["xera"]
        luck_src = luck_src.dropna(subset=["era_minus_xera"])
        if len(luck_src) > 5:
            fig_luck = px.histogram(
                luck_src, x="era_minus_xera", nbins=40,
                labels={"era_minus_xera": "ERA − xERA (positive = unlucky)"},
                title="Luck distribution (ERA − xERA)",
                color_discrete_sequence=[COLOR_LUCK_HIST],
            )
            fig_luck.update_traces(marker_line_color="white", marker_line_width=1)
            fig_luck.add_vline(x=0, line_dash="dash", line_color="#e11d48", line_width=2)
            fig_luck.update_layout(height=350, **_PLOT_TRANSPARENT)
            st.plotly_chart(fig_luck, use_container_width=True)
            st.caption(
                "Values above zero indicate unlucky pitcher-seasons (actual ERA exceeded expected); values below zero indicate outperformance. "
                "Most seasons cluster near zero; the tails represent unusually lucky or unlucky outcomes that may regress."
            )

    # K/9 vs ERA
    k9_era = display_df.dropna(subset=["SO9", "ERA"])
    if len(k9_era) > 10:
        fig_k9 = px.scatter(
            k9_era, x="SO9", y="ERA", hover_data=["Name", "IP"] + (["year"] if all_years_mode else []),
            title="Strikeout rate (K/9) vs ERA",
            labels={"SO9": "K/9", "ERA": "ERA"},
        )
        fig_k9.update_traces(marker=dict(size=8, symbol="diamond", color=COLOR_K9_SCATTER, line=dict(width=0.5, color="white")))
        fig_k9.update_layout(height=400, **_PLOT_TRANSPARENT)
        st.plotly_chart(fig_k9, use_container_width=True)
        r, p = pearsonr(k9_era["SO9"], k9_era["ERA"])
        sig = "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.05 else f"p = {p:.2f} (n.s.)"
        st.caption(
            f"Pearson r = {r:.3f} ({sig}). Higher strikeout rates tend to correlate with lower ERA, "
            "since strikeouts eliminate the role of defense and batted-ball outcomes."
        )

    st.divider()


# ===========================================================================
# Tab 5: Pitcher Quality Scores
# ===========================================================================

with tab_quality:
    st.header("Pitcher Quality Ranking")
    ql_raw = regression_outputs.get("quality_leaderboard")
    if ql_raw is None:
        st.warning(
            "Quality scores not generated yet. Run:\n"
            "```\npython run_pipeline.py\n```"
        )
    else:
        st.markdown(
            "The **quality score** (0–100) aggregates several underlying stats: xERA, K/9, est.wOBA, BB%, Barrel%, IP. "
            "Each stat is z-scored across the league; lower-is-better stats (xERA, BB%, etc.) are inverted so a higher score indicates better performance. "
            "The weighted sum is then scaled to 0–100. **You can adjust the component weights below to reflect your own priorities.**"
        )
        _lb_year = regression_outputs.get("regression_metadata", {}).get("latest_season")
        if _lb_year is not None:
            st.info(f"**Only pitchers who pitched in {_lb_year}** are included. One row per pitcher, using that season's stats. Run `python run_pipeline.py` to refresh.")
        else:
            st.caption(
                "**Single-season snapshot:** Only pitchers from the most recent season in the pipeline. "
                "One row per pitcher; multi-year pitchers appear once with their latest season's stats."
            )
        ZSCORE_COLS_UI = ["xera", "SO9", "est_woba", "bb_rate", "brl_percent", "IP"]
        ZSCORE_LABELS = {"xera": "xERA", "SO9": "SO9", "est_woba": "est.wOBA", "bb_rate": "BB%", "brl_percent": "Barrel%", "IP": "IP"}
        DEFAULT_W = {"xera": 25, "SO9": 22, "est_woba": 18, "bb_rate": 12, "brl_percent": 8, "IP": 15}

        st.markdown("**Adjust quality score weights** (auto-normalised)")
        w_cols = st.columns(3)
        raw_w = {}
        for i, col in enumerate(ZSCORE_COLS_UI):
            raw_w[col] = w_cols[i % 3].slider(
                ZSCORE_LABELS[col], 0, 50, DEFAULT_W.get(col, 10), 5, key=f"qw_{col}")
        total_w = sum(raw_w.values()) or 1
        norm_w = {k: v / total_w for k, v in raw_w.items()}
        st.caption("Normalised: " + "  ·  ".join(f"{ZSCORE_LABELS[k]} {v:.0%}" for k, v in norm_w.items()))

        def compute_custom_score_z(row):
            lower_better = {"xera", "est_woba", "bb_rate", "brl_percent"}
            total, tw = 0.0, 0.0
            for col in ZSCORE_COLS_UI:
                v = row.get(col)
                w = norm_w.get(col, 0)
                if pd.isna(v) or w <= 0 or col not in z_mean:
                    continue
                mu, s = z_mean[col], z_std[col]
                z = (float(v) - mu) / s if s and s != 0 else 0
                if col in lower_better:
                    z = -z
                total += z * w
                tw += w
            if tw < 0.2:
                return None
            return round(total / tw, 2)

        ql = ql_raw.copy()
        # Fallback: compute bb_rate if missing (for older pipeline output)
        if "bb_rate" not in ql.columns and "BB" in ql.columns and "BF" in ql.columns:
            ql["bb_rate"] = ql["BB"] / ql["BF"].replace(0, np.nan)
        z_mean, z_std = {}, {}
        for col in ZSCORE_COLS_UI:
            if col in ql.columns:
                s = ql[col].dropna()
                if len(s) >= 5:
                    z_mean[col] = float(s.mean())
                    z_std[col] = float(s.std()) if s.std() and not pd.isna(s.std()) else 1.0
        if any(c in ql.columns for c in ["xera", "SO9", "IP"]):
            ql["quality_score"] = ql.apply(compute_custom_score_z, axis=1)
            lo, hi = ql["quality_score"].quantile(0.02), ql["quality_score"].quantile(0.98)
            if hi > lo:
                ql["quality_score"] = ((ql["quality_score"] - lo) / (hi - lo) * 100).clip(0, 100).round(1)
        # Re-grade
        GRADE_THRESHOLDS_UI = [(90,"A+"),(80,"A"),(70,"B+"),(60,"B"),(50,"C+"),(40,"C"),(30,"D"),(0,"F")]
        def _grade(s):
            if s is None or (isinstance(s, float) and pd.isna(s)):
                return "N/A"
            for thr, g in GRADE_THRESHOLDS_UI:
                if s >= thr: return g
            return "F"
        ql["grade"] = ql["quality_score"].apply(_grade)

        # ---- Grade distribution ----
        grade_order = ["A+", "A", "B+", "B", "C+", "C", "D", "F"]
        grade_dist = ql["grade"].value_counts().reindex(grade_order, fill_value=0)
        grade_df = pd.DataFrame({"Grade": grade_order, "Count": [int(grade_dist.get(g, 0)) for g in grade_order]})
        col_dist, col_hist = st.columns(2)
        with col_dist:
            st.dataframe(grade_df, use_container_width=True, hide_index=True)
        with col_hist:
            fig_gd = px.bar(grade_df, x="Grade", y="Count", category_orders={"Grade": grade_order},
                             color="Count", color_continuous_scale="Viridis", labels={"Count": "Pitchers"})
            fig_gd.update_traces(marker_line_color="white", marker_line_width=1)
            fig_gd.update_layout(height=300, margin=dict(t=20, b=40), coloraxis_showscale=False,
                                xaxis_tickangle=0, **_PLOT_TRANSPARENT)
            st.plotly_chart(fig_gd, use_container_width=True)

        st.divider()

        # ---- Cluster surface stats ----
        if _clustering_available and df is not None:
            _qual_col = "cluster_kmeans" if "cluster_kmeans" in df.columns else None
            if _qual_col in df.columns:
                with st.expander("Which cluster has the best surface-level stats?", expanded=False):
                    cluster_col = _qual_col
                    # Cluster profiles use 'pitcher' as the MLBAM ID; leaderboard uses 'mlbID'
                    cluster_map = df[["pitcher", cluster_col]].copy().rename(
                        columns={"pitcher": "mlbID"})
                    cluster_map["mlbID"] = pd.to_numeric(cluster_map["mlbID"], errors="coerce")

                    if "mlbID" not in ql.columns:
                        st.info("Quality leaderboard missing mlbID — re-run `python run_pipeline.py` to rebuild.")
                    else:
                        ql_numeric = ql.copy()
                        ql_numeric["mlbID"] = pd.to_numeric(ql_numeric["mlbID"], errors="coerce")
                        ql_with_cluster = ql_numeric.merge(cluster_map, on="mlbID", how="left")
                        stat_cols = [c for c in ["ERA", "WHIP", "SO9", "quality_score"]
                                     if c in ql_with_cluster.columns]
                        grp = (ql_with_cluster.dropna(subset=[cluster_col])
                               .groupby(cluster_col)[stat_cols].mean().round(3))
                        grp.index = grp.index.astype(int)
                        grp = grp.sort_index()
                        # Add Cluster label: number · archetype name
                        _summaries = kmeans_summaries or {}
                        grp["Cluster"] = [
                            f"{cid} · {_parse_archetype(_summaries.get(str(cid), '')) or '—'}"
                            for cid in grp.index
                        ]
                        grp = grp[["Cluster"] + [c for c in grp.columns if c != "Cluster"]]
                        num_cols_grp = [c for c in grp.columns if c != "Cluster" and grp[c].dtype in ("float64", "float32", "int64", "int32")]
                        st.dataframe(
                            grp.style
                            .background_gradient(
                                subset=["ERA", "WHIP"] if "ERA" in grp.columns else [],
                                cmap="RdYlGn_r")
                            .background_gradient(
                                subset=[c for c in ["SO9", "quality_score"] if c in grp.columns],
                                cmap="RdYlGn")
                            .format({c: "{:.2f}" for c in num_cols_grp}, na_rep="—"),
                            use_container_width=True,
                        )
                        st.caption("Mean surface stats per cluster. Identifies which archetype tends to perform best.")

        # ---- Leaderboard ----
        st.subheader("Leaderboard")
        display_ql = ql.copy()
        min_score = st.slider("Minimum quality score", 0, 100, 0, 5)
        display_ql = display_ql[display_ql["quality_score"].fillna(0) >= min_score]

        show_cols = [c for c in ["Name", "Tm", "IP", "ERA", "WHIP", "SO9", "quality_score", "grade"]
                     if c in display_ql.columns]
        fmt = {"ERA": "{:.2f}", "WHIP": "{:.3f}", "SO9": "{:.1f}",
               "IP": "{:.0f}", "quality_score": "{:.1f}"}
        st.dataframe(
            display_ql[show_cols].style
            .background_gradient(subset=["quality_score"], cmap="RdYlGn", vmin=20, vmax=80)
            .format({k: v for k, v in fmt.items() if k in show_cols}),
            use_container_width=True, height=500,
        )

        fig_dist = px.histogram(
            ql.dropna(subset=["quality_score"]), x="quality_score", nbins=20,
            color_discrete_sequence=[COLOR_QUALITY_DIST],
            title="Distribution of Pitcher Quality Scores",
            labels={"quality_score": "Quality Score (0–100)"},
        )
        fig_dist.update_traces(marker_line_color="white", marker_line_width=1)
        fig_dist.add_vline(x=50, line_dash="dash", line_color="#e11d48", line_width=2,
                           annotation_text="League avg (50)", annotation_font=dict(color="white"))
        fig_dist.update_layout(height=300, **_PLOT_TRANSPARENT)
        st.plotly_chart(fig_dist, use_container_width=True)


# ===========================================================================
# Tab 6: Regression Predictor
# ===========================================================================

with tab_regression:
    st.header("Regression Analysis")
    st.markdown("#### Can we predict luck?")
    st.markdown(
        "**Luck** = ERA − xERA (deviation from expectations). Unlucky &gt; 0; lucky &lt; 0. "
        "Predicting next-season luck would inform buy-low and sell-high decisions; this analysis tests whether such prediction is feasible."
    )
    st.markdown(
        "**What is xERA?** Expected ERA from Statcast: it estimates what a pitcher's ERA *should* be based on contact quality (exit velocity, launch angle, sprint speed). "
        "ERA &gt; xERA means the pitcher was *unlucky* (worse results than contact quality implied); ERA &lt; xERA means *lucky* (better results than expected). "
        "Deviations often regress toward zero year over year."
    )
    st.caption(
        "**Regression setup:** We use stats from **year t** (xERA, ERA, K/9, WHIP, Savant percentiles, etc.) to predict **year t+1** (next season). "
        "Walk-forward CV: train on seasons 1..k, test on k+1; expand window each fold."
    )

    col_ga, col_gb = st.columns(2)
    col_ga.markdown("**Group A:** Predict year t+1 ERA, wOBA, BA from year t features. Sanity check—these should be predictable.")
    col_gb.markdown("**Group B:** Predict year t+1 luck (ERA−xERA, etc.) from year t. The real question—can we beat \"regress to mean\"?")
    st.caption("Green = best RMSE per row. Caveat: only 2 folds, small RMSE gaps—treat as suggestive.")
    st.divider()

    reg = regression_outputs

    if not reg or "comparison" not in reg:
        st.warning(
            "Regression models have not been run yet. Run:\n"
            "```\npython src/backend/predict_actual_expected_residual.py\n```"
        )
        # Still show arsenal FI if available
        if arsenal_importances is not None:
            st.divider()
            with st.expander("**Arsenal → xERA:** Which pitch traits predict expected performance?", expanded=True):
                st.caption(
                    "Random Forest trained on Statcast arsenal features (velocity, spin, pitch mix) to predict xERA. "
                    "Feature importances indicate which traits drive expected run prevention."
                )
                top_n = st.slider("Show top N features", 10, 50, 25, key="arsenal_fi_top")
                imp = arsenal_importances.head(top_n).copy()
                imp["importance"] = imp["importance"].round(3)
                fig_arsenal = px.bar(
                    imp, x="importance", y="feature",
                    orientation="h",
                    labels={"importance": "Importance", "feature": "Feature"},
                    color="importance", color_continuous_scale="Plasma",
                )
                _layout = dict(_PLOT_TRANSPARENT)
                _layout["yaxis"] = dict(_layout["yaxis"], categoryorder="total ascending")
                fig_arsenal.update_layout(height=min(500, 200 + top_n * 12), coloraxis_showscale=False, **_layout)
                st.plotly_chart(fig_arsenal, use_container_width=True)
    else:
        # --- Overview RMSE table ---
        st.subheader("Model RMSE Overview (walk-forward CV)")
        cmp = reg["comparison"].copy()
        model_cols = [c for c in cmp.columns
                      if c not in ("Target", "Group", "Best Model", "Folds")
                      and not c.endswith("_std")]
        fmt = {c: "{:.4f}" for c in model_cols if c in cmp.columns}
        # Optionally show mean ± std if _std columns exist
        display_cmp = cmp[[c for c in cmp.columns if not c.endswith("_std")]].copy()
        n_folds = cmp["Folds"].iloc[0] if "Folds" in cmp.columns else ""
        st.dataframe(
            display_cmp.style
            .highlight_min(subset=model_cols, axis=1, color="#1a9850")
            .format(fmt),
            use_container_width=True,
            height=260,
        )
        st.caption("Green = best per row. Group A: models beat naive. Group B: naive wins or ties → models don't help.")
        try:
            _nf = int(n_folds) if n_folds != "" else 0
            if _nf <= 2:
                st.caption("Only 2 folds: walk-forward needs 3+ train seasons before first test; 2020–2025 gives 2 test years.")
        except (ValueError, TypeError):
            pass

        st.divider()

        # This-year vs next-year luck (ERA − xERA): does luck persist within pitcher?
        luck_key = "luck_this_vs_next"
        if luck_key in reg and reg[luck_key] is not None and len(reg[luck_key]) > 10:
            st.subheader("Does luck persist? This year vs next year (ERA − xERA)")
            luck_df = reg[luck_key].copy()
            luck_df = luck_df.dropna(subset=["luck_this_year", "luck_next_year"])
            if len(luck_df) > 10:
                fig_luck = px.scatter(
                    luck_df, x="luck_this_year", y="luck_next_year",
                    hover_data=["Name", "season"] if "Name" in luck_df.columns else ["season"],
                    labels={"luck_this_year": "This year (ERA − xERA)", "luck_next_year": "Next year (ERA − xERA)"},
                    title="Within pitcher: does this year's luck predict next year's?",
                )
                # Diagonal = luck persists (next year = this year)
                mn = min(luck_df["luck_this_year"].min(), luck_df["luck_next_year"].min())
                mx = max(luck_df["luck_this_year"].max(), luck_df["luck_next_year"].max())
                fig_luck.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                                  line=dict(dash="dash", color="rgba(200,200,200,0.7)"))
                fig_luck.add_hline(y=0, line_dash="dot", line_color="rgba(150,150,150,0.5)")
                fig_luck.add_vline(x=0, line_dash="dot", line_color="rgba(150,150,150,0.5)")
                fig_luck.update_traces(marker=dict(size=6, opacity=0.7))
                fig_luck.update_layout(height=420, **_PLOT_TRANSPARENT)
                st.plotly_chart(fig_luck, use_container_width=True)
                from scipy.stats import pearsonr
                corr, p_val = pearsonr(luck_df["luck_this_year"], luck_df["luck_next_year"])
                sig = "significant" if p_val < 0.05 else "not significant"
                st.info(
                    f"**Takeaway:** Each point = one pitcher-season pair. "
                    f"If luck persisted, points would follow the diagonal (unlucky this year → unlucky next). "
                    f"A flat cloud = luck resets. Correlation: **r = {corr:.3f}** (p = {p_val:.4f}, {sig}). "
                    f"{'Weak' if abs(corr) < 0.2 else 'Moderate'} year-over-year relationship. "
                    f"\"Buy unlucky, sell lucky\" gets little support."
                )
            else:
                st.caption("Luck pairs data has too few points. Re-run the pipeline.")
        else:
            st.caption("*Luck this-year vs next-year* scatter will appear here after you re-run the pipeline.")

        st.divider()
        if arsenal_importances is not None:
            with st.expander("**Arsenal → xERA:** Which pitch traits predict expected performance?", expanded=False):
                st.caption("RF on arsenal features → xERA. Importances show what actually drives run prevention.")
                top_n = st.slider("Show top N features", 10, 50, 25, key="arsenal_fi_top")
                imp = arsenal_importances.head(top_n).copy()
                imp["importance"] = imp["importance"].round(3)
                fig_arsenal = px.bar(
                    imp, x="importance", y="feature",
                    orientation="h",
                    labels={"importance": "Importance", "feature": "Feature"},
                    color="importance", color_continuous_scale="Plasma",
                )
                _layout = dict(_PLOT_TRANSPARENT)
                _layout["yaxis"] = dict(_layout["yaxis"], categoryorder="total ascending")
                fig_arsenal.update_layout(height=min(500, 200 + top_n * 12), coloraxis_showscale=False, **_layout)
                st.plotly_chart(fig_arsenal, use_container_width=True)

        st.divider()

        # --- Target selector ---
        target_options = {t["key"]: f"{'[A] ' if t['group']=='A' else '[B] '}{t['label']}"
                          for t in TARGETS_META}
        selected_key = st.selectbox(
            "Drill into a target",
            list(target_options.keys()),
            format_func=lambda k: target_options[k],
        )
        sel = next(t for t in TARGETS_META if t["key"] == selected_key)

        col_proj, col_fi = st.columns([1, 1])

        # --- Projections ---
        with col_proj:
            proj_file = f"proj_{selected_key}"
            if proj_file in reg:
                proj = reg[proj_file].copy()
                proj_col = [c for c in proj.columns if c.startswith("proj_")]
                if proj_col:
                    pc = proj_col[0]
                    proj = proj.rename(columns={pc: f"Proj. {sel['label']}"})
                    lbl = f"Proj. {sel['label']}"
                    vmin = proj[lbl].quantile(0.05)
                    vmax = proj[lbl].quantile(0.95)
                    num_fmt = ".2f" if sel["key"] == "ERA" else ".4f"
                    display_proj = proj[[c for c in proj.columns if c != pc]].copy()
                    st.markdown(f"**Next-season projections — {sel['label']}**")
                    st.caption("Sorted best → worst projected value. RF model trained on all historical data.")
                    num_cols = display_proj.select_dtypes("number").columns.tolist()
                    st.dataframe(
                        display_proj.style
                        .background_gradient(subset=[lbl],
                                             cmap="RdYlGn_r" if sel["lower_better"] else "RdYlGn",
                                             vmin=vmin, vmax=vmax)
                        .format({c: f"{{:{num_fmt}}}" for c in num_cols}, na_rep="—"),
                        use_container_width=True, height=450,
                    )
            else:
                st.info("No projection data for this target.")

        # --- Feature importances ---
        with col_fi:
            fi_file = f"fi_{selected_key}"
            if fi_file in reg:
                fi = reg[fi_file].copy()
                fi["importance"] = fi["importance"].round(3)
                feat_labels = {
                    "xera": "xERA", "est_woba": "est.wOBA", "est_ba": "est.BA",
                    "era_vs_xera": "ERA−xERA (luck)", "woba_vs_est": "wOBA−est (luck)",
                    "ba_vs_est": "BA−est (luck)",
                    "era": "ERA", "woba": "wOBA", "ba": "BA",
                    "SO9": "K/9", "WHIP": "WHIP", "IP": "IP (sample size)",
                    "pct_k_percent": "K% percentile", "pct_bb_percent": "BB% percentile",
                    "pct_whiff_percent": "Whiff% percentile",
                    "pct_hard_hit_percent": "Hard Hit% percentile",
                    "pct_xera": "xERA percentile",
                }
                fi["label"] = fi["feature"].map(feat_labels).fillna(fi["feature"])
                fig_fi = px.bar(
                    fi.sort_values("importance"),
                    x="importance", y="label", orientation="h",
                    title=f"RF Feature Importances — {sel['label']}",
                    color="importance", color_continuous_scale="Plasma",
                )
                fig_fi.update_layout(coloraxis_showscale=False, height=400, yaxis_title="", xaxis_title="Importance", **_PLOT_TRANSPARENT)
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("No feature importance data for this target.")

        # --- Candidates (Group B residuals only) ---
        if sel["group"] == "B":
            st.divider()
            unlucky_key = f"cand_{selected_key}_unlucky"
            lucky_key   = f"cand_{selected_key}_lucky"
            has_cands = unlucky_key in reg or lucky_key in reg

            if has_cands:
                res_col = sel["label"]
                unlucky_desc = {
                    "ERA_resid":  ("ERA much worse than xERA", "ERA exceeded expectations — ball found holes or defence was poor. Expect improvement."),
                    "wOBA_resid": ("wOBA much worse than est.wOBA", "Allowed higher wOBA than contact quality suggests. Expect normalisation."),
                    "BA_resid":   ("BA much worse than est.BA", "Batters hit above their expected BA. Regression likely."),
                }
                lucky_desc = {
                    "ERA_resid":  ("ERA much better than xERA", "ERA well below expectation — strand rate or defence-aided. Possible regression."),
                    "wOBA_resid": ("wOBA much better than est.wOBA", "Allowed lower wOBA than contact quality suggests. Likely to regress."),
                    "BA_resid":   ("BA much better than est.BA", "Batters hit below their expected BA. Regression likely."),
                }

                col_u, col_l = st.columns(2)
                with col_u:
                    short, long = unlucky_desc.get(selected_key, ("Unlucky", ""))
                    st.markdown(f"**🟢 Buy Low — {short}**")
                    st.caption(long)
                    if unlucky_key in reg:
                        df_u = reg[unlucky_key].copy()
                        num_c = df_u.select_dtypes("number").columns.tolist()
                        fmt_u = {c: "{:.4f}" if df_u[c].abs().median() < 1 else "{:.2f}" for c in num_c}
                        st.dataframe(df_u.style.background_gradient(
                            subset=[c for c in num_c if "vs" in c or "resid" in c.lower()],
                            cmap="RdYlGn_r").format(fmt_u),
                            use_container_width=True, height=450)

                with col_l:
                    short, long = lucky_desc.get(selected_key, ("Lucky", ""))
                    st.markdown(f"**🔴 Sell High — {short}**")
                    st.caption(long)
                    if lucky_key in reg:
                        df_l = reg[lucky_key].copy()
                        num_c = df_l.select_dtypes("number").columns.tolist()
                        fmt_l = {c: "{:.4f}" if df_l[c].abs().median() < 1 else "{:.2f}" for c in num_c}
                        st.dataframe(df_l.style.background_gradient(
                            subset=[c for c in num_c if "vs" in c or "resid" in c.lower()],
                            cmap="RdYlGn").format(fmt_l),
                            use_container_width=True, height=450)

        st.divider()
        with st.expander("ℹ️ Methodology & Limitations"):
            st.markdown("""
**Dataset:** Year-over-year pairs from 2020–2025 where a pitcher had Savant expected stats
in year N and ≥20 IP in year N+1. ~2,400 paired observations.

**Features (for all targets):** xERA, est.wOBA, est.BA, current-season residuals
(ERA−xERA, wOBA−est, BA−est), ERA, wOBA, BA, K/9, WHIP, IP, Savant percentile
ranks (K%, BB%, whiff%, hard hit%, xERA).

**Models:** Naïve baseline, Linear OLS (interpretable), Random Forest (best performer).
Linear uses StandardScaler; tree models use raw features. Median imputation for missing values.

**Cross-validation:** Walk-forward (expanding window). Fold 1: train 2015, test 2016; fold 2: train 2015–2016, test 2017; … Final fold trains on all prior years. Mean RMSE reported over folds.

**Feature importance:** From the full-data model (best type per target), not from fold models. This gives a single stable ranking of features and avoids fold-to-fold variation.

**Group A findings:** Expected stats (xERA, est.wOBA, est.BA) are the best predictors
of next-year performance. Linear models match or beat tree models, confirming the
relationship is mostly additive.

**Group B findings:** Next-year residuals (luck components) are nearly unpredictable
from the current season — all models perform essentially at the naïve baseline.
This is the analytically important result: luck resets almost entirely each year.
The exception is a small persistent skill component captured in K% (pitchers who
strike batters out consistently outperform their contact-based expectations).
            """)


# ===========================================================================
# Tab: Raw + intermediate Data
# ===========================================================================

with tab_data:
    st.header("Raw & Intermediate Data")
    if not _clustering_available:
        st.info("No Statcast clustering data available yet.")
    else:
        st.markdown(
            "Raw pitcher profiles and the feature matrix used for clustering. "
            "Exposed for replication, extension, and validation."
        )

        with st.expander("Data Dictionary", expanded=True):
            st.markdown("""
            | Term | Definition |
            |------|------------|
            | **ERA** | Earned Run Average (earned runs × 9 / IP) |
            | **xERA** | Expected ERA from Statcast contact quality (exit velo, launch angle, sprint speed) |
            | **est.wOBA** | Expected weighted on-base average (contact-quality-based) |
            | **WHIP** | Walks + hits per inning |
            | **SO9** | Strikeouts per 9 innings |
            | **bb_rate** | Walk rate (BB / BF) |
            | **brl_percent** | Barrel rate (% of batted balls with optimal exit velo + launch angle) |
            | **quality_score** | 0–100 composite from z-scored xERA, SO9, est.wOBA, BB%, Barrel% |
            | **pct_X** | Share of pitches that are type X (e.g. pct_FF = four-seam fastball %) |
            | **velo_X** | Average release velocity (mph) for pitch type X |
            | **spin_X** | Average spin rate (rpm) for pitch type X |
            | **pfx_x**, **pfx_z** | Horizontal and vertical movement (inches) from release to plate |
            | **break_x**, **break_z** | Break components (inches) |
            | **spin_axis_X** | Tilt of spin axis for pitch type X |
            | **avg_extension** | Average release extension (feet) |
            | **avg_arm_angle** | Average arm slot (degrees) |
            | **cmd_plate_x_std**, **cmd_plate_z_std** | Command variability (std of plate-crossing location) |
            | **n_pitch_types** | Number of distinct pitch types thrown (≥1% usage) |
            | **hard_hit_pct** | % of batted balls ≥95 mph exit velocity |
            """)

        st.subheader("Pitcher Profiles Dataset (Statcast)")
        st.write(f"{len(df)} pitchers · {len(df.columns)} features")

        # Clustering Feature Matrix — exact features used for clustering
        with st.expander("Clustering Feature Matrix", expanded=True):
            from preprocess_data import get_clustering_features
            feat_cols = get_clustering_features(df)
            _dc = "cluster_kmeans" if "cluster_kmeans" in df.columns else "cluster_kmeans"
            id_cols = ["player_name", _dc] if _dc in df.columns else ["player_name"]
            feat_cols = [c for c in feat_cols if c in df.columns]
            matrix_df = df[[c for c in id_cols if c in df.columns] + feat_cols].copy()
            matrix_df = matrix_df.round(3)
            st.caption(
                f"Features used for clustering ({len(feat_cols)} columns). "
                "I standardize values (StandardScaler) before clustering and similarity search, and use median imputation for missing values."
            )
            cluster_filter = st.selectbox(
                "Filter by cluster",
                options=["All"] + sorted(df[_dc].unique().astype(int).tolist()) if _dc in df.columns else ["All"],
                format_func=lambda x: f"Noise (–1)" if x == -1 else f"Cluster {x}" if x != "All" else "All",
                key="feat_matrix_cluster",
            )
            if cluster_filter != "All" and _dc in matrix_df.columns:
                matrix_df = matrix_df[matrix_df[_dc] == cluster_filter]
            feat_search = st.text_input("Filter features by name", key="feat_matrix_search", placeholder="e.g. velo_FF")
            if feat_search:
                keep_cols = [c for c in matrix_df.columns if feat_search.lower() in c.lower() or c in id_cols]
                matrix_df = matrix_df[[c for c in matrix_df.columns if c in keep_cols]]
            st.dataframe(matrix_df, use_container_width=True, height=350)
            csv_feat = matrix_df.to_csv(index=False)
            st.download_button("Download feature matrix", csv_feat, "clustering_features.csv", "text/csv", key="dl_feat")

        search_name = st.text_input("Filter by name", key="raw_search")
        display_df = df.copy()
        if search_name:
            _nq = _normalize_for_search(search_name)
            display_df = display_df[display_df["player_name"].apply(lambda x: _nq in _normalize_for_search(x))]

        st.dataframe(display_df, use_container_width=True, height=500)
        csv = display_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "pitcher_profiles.csv", "text/csv")


# ===========================================================================
# Tab: Planned follow-ups
# ===========================================================================

with tab_planned:
    st.header("Planned Follow-Up Analyses")
    st.markdown(
        "I intend to continue working on this dashboard as I think of new questions I have about baseball. "
        "Here is the short list of questions I have right now:"
    )
    st.markdown(
        "- **First-half vs second-half predictiveness:** Much is made of how pitchers start vs finish a season — "
        "when differentiating between players with similar yearlong statlines, starting poorly and ending strong "
        "is viewed more favorably than the reverse. Is the second half of a previous season, then, genuinely more "
        "predictive of the upcoming season than the first half?"
    )
    st.markdown(
        "- **Pull rate and expected BA:** Does pull percentage add predictive power beyond exit velo and launch angle "
        "when modeling expected batting average?"
    )
    st.markdown(
        "- **Arsenal aging curves:** How do velocity, spin, and pitch mix evolve with pitcher age? Can we model "
        "career trajectory to flag early decline or late bloomers?"
    )
    st.markdown(
        "- **Platoon splits by archetype:** Do certain cluster archetypes (e.g. power fastball/slider) show larger "
        "platoon splits than others? Useful for roster construction and matchup planning."
    )
    st.markdown(
        "- **Injury/durability signals:** Do specific arsenal traits (e.g. spin rate, extension, arm slot) correlate "
        "with IL stints or innings capacity? Could inform risk adjustment in valuation."
    )
