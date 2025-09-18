import streamlit as st
import os
from dotenv import load_dotenv
import traceback

# Load environment variables at startup
# load_dotenv()

# Validate required environment variables
try:
    db_password = st.secrets["DB_PASSWORD"]
except KeyError:
    raise ValueError("DB_PASSWORD secret is required")

# Helper function for database connections
def get_db_connection():
    """Create a database connection using Streamlit secrets."""
    import mysql.connector
    return mysql.connector.connect(
        host=st.secrets.get("DB_HOST", "34.230.115.21"),
        user=st.secrets.get("DB_USER", "standard_user"),
        password=st.secrets["DB_PASSWORD"],
        database=st.secrets.get("DB_NAME", "tread_database_ec2")
    )

# Configure page to completely hide sidebar - MUST be first Streamlit command
st.set_page_config(
    page_title="UNLV Pitcher Role Simulator",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={}  # This hides the menu
)

# Hide sidebar completely with CSS
st.markdown("""
    <style>
        [data-testid="collapsedControl"] {display: none;}
        section[data-testid="stSidebar"] {display: none;}
    </style>
""", unsafe_allow_html=True)

import polars as pl
import numpy as np
import mysql.connector
import pandas as pd
from typing import Dict
from stopper_core import (
    scenario_rank,
    build_pa_table,
    attach_running_score,
    add_state_features,
    label_home_win,
    compute_wpa_per_pa,
    attach_pa_pitcher,
    compute_stopper_leaderboard,
    leverage_from_state,
    train_wp_model,
)
from stopper_predictor import load_or_train_pitch_delta_model, build_pitcher_profiles, simulate_expected_delta
from simulate_situations import (
    simulate_all_situations,
    analyze_simulation_results,
    DO_NOT_PITCH,
    PHASES,
    SCORE_CONTEXTS,

)

# --- Simple Login ---
USERNAME = "ope"
PASSWORD = "UNLV"

def login_gate() -> bool:
    """Render a simple login form and gate access."""
    if st.session_state.get("auth_ok"):
        return True

    st.markdown("#### Please log in to continue")
    with st.form("login_form", clear_on_submit=False):
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")
        submit = st.form_submit_button("Log In")
    if submit:
        if u == USERNAME and p == PASSWORD:
            st.session_state.auth_ok = True
            st.success("Login successful. Redirecting…")
            st.rerun()
        else:
            st.error("Invalid credentials. Try again.")
    return False

# Execution+ module (reloadable to pick up mapping changes)
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import subprocess


# ------------------ Basic App Setup ------------------

ACCENT = "#CC0000"        # Rebel Scarlet
ACCENT_2 = "#C0C0C0"      # Rebel Silver
SURFACE = "#0B0B0B"       # near-black background
SURFACE_SOFT = "#141414"  # soft panel surface
TEXT = "#F3F4F6"          # near-white text

# UNLV Pitchers list
UNLV_PITCHERS = [
    'Albright, Cody', 'Barna, Cal', 'Bland, Yates', 'Bowen, Gavyn',
    'Dilhoff, Parker', 'Donegan, Josh', 'Evangelista, Jase', 'Foxson, Tate',
    'Gomberg, Jacob', 'Jones, Jaylen', 'Kubasky, Noah', 'Lane, Carson',
    'Lueck, Reese', 'Manning, LJ', 'Marton, Ryan', 'Ong, Felix',
    'Rogers, Dylan', 'Sundloff, Colton'
]

# Mapping of situation buckets to recommended bullpen roles
ROLE_SITUATIONS = {
    "Starter": [
        "Early Up Little",
        "Early Tight",
        "Early Down Little",
        "Middle Up Little",
        "Middle Tight",
        "Middle Down Little"
    ],
    "Bridge": [
        "Middle Up Little",
        "Middle Tight",
        "Middle Down Little",
        "Late Up Little",
        "Late Tight",
        "Late Down Little"
    ],
    "High Leverage": [
        "Late Up Little",
        "Late Tight",
        "Late Down Little",
        "Early Tight",
        "Middle Tight"
    ],
    "Inning Eater": [
        "Early Down Lots",
        "Early Up Lots",
        "Middle Down Lots",
        "Middle Up Lots",
        "Late Down Lots",
        "Late Up Lots"
    ],
    "Long Relief": [
        "Early Up Little",
        "Early Up Lots",
        "Early Down Lots"
    ]
}

LEVERAGE_WEIGHTED_ROLES = {"Bridge", "High Leverage"}


def leverage_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Compute a leverage-weighted average with safe fallbacks."""
    if values.empty:
        return float("nan")

    values_numeric = pd.to_numeric(values, errors="coerce")
    weights_numeric = pd.to_numeric(weights, errors="coerce")
    mask = values_numeric.notna() & weights_numeric.notna()
    if not mask.any():
        return float("nan")

    v = values_numeric[mask].to_numpy(dtype=float)
    w = weights_numeric[mask].to_numpy(dtype=float)
    # Ensure weights are positive to avoid zero-division
    w = np.where(w > 0, w, 1e-6)
    total = w.sum()
    if total <= 0:
        return float(np.mean(v)) if v.size else float("nan")
    return float(np.dot(v, w) / total)

# Cache practice rows by pitcher to reuse in Execution+ without re-querying
_PRACTICE_BY_PITCHER: Dict[str, pd.DataFrame] = {}


def _grid_from_profiles(profiles: Dict, pitcher_name: str, *, side_n: int = 21, height_n: int = 26) -> pd.DataFrame:
    """Fallback: construct a prediction-ready grid using the already-computed
    pitch averages from get_pitcher_profiles(profiles). This bypasses any issues
    in raw practice rows and guarantees expansion.
    """
    rows = []
    for (pname, pt), prof in profiles.items():
        if pname != pitcher_name:
            continue
        if prof.get("freq", 0.0) < 0.03:   # <— enforce 3% here
            continue
        means = prof.get("means", {})
        rows.append({
            "Pitcher": pitcher_name,
            "TaggedPitchType": pt,
            "PitcherThrows": prof.get("pitcher_hand", "Right"),
            "HorzBreak": means.get("HorzBreak"),
            "InducedVertBreak": means.get("InducedVertBreak"),
            "RelSpeed": means.get("start_speed"),
            "RelSide": means.get("release_side"),
            "RelHeight": means.get("release_height"),
            "Extension": means.get("extension"),
            "SpinRate": means.get("spin_rate"),
            "VertApprAngle": None,
            "HorzApprAngle": None,
            "ax0": means.get("ax"),
            "az0": means.get("az"),
        })
    if not rows:
        return pd.DataFrame()

    base = pd.DataFrame(rows)
    # Build plate lattice and counts/bats facets exactly like make_prediction_ready_plate_grid
    x_vals = np.linspace(-2.0, 2.0, num=side_n)
    z_vals = np.linspace(0.0, 5.0, num=height_n)
    X, Z = np.meshgrid(x_vals, z_vals)
    plate = pd.DataFrame({"PlateLocSide": X.ravel(), "PlateLocHeight": Z.ravel()})

    COUNT_CLASS_NUM = {"Even": 0, "Pitcher": 1, "Hitter": 2}
    REP_COUNTS = {"Even": (1, 1), "Hitter": (2, 0), "Pitcher": (0, 2)}
    cnt = pd.DataFrame({"count_class": ["Even", "Hitter", "Pitcher"]})
    cnt["count_class_num"] = cnt["count_class"].map(COUNT_CLASS_NUM)
    cnt[["Balls", "Strikes"]] = cnt["count_class"].map(REP_COUNTS).apply(pd.Series)

    bats = pd.DataFrame({"BatterSide": ["Right", "Left"], "BatterSideFacet": ["RHH", "LHH"]})

    left = base.merge(cnt, how="cross").merge(bats, how="cross")
    grid = left.merge(plate, how="cross")

    # Fill remaining required fields
    grid["Top.Bottom"] = "Undefined"
    grid["Level"] = "College"
    grid["GameID"] = ""
    grid["Batter"] = ""
    grid["AwayTeam"] = ""
    grid["HomeTeam"] = ""
    grid["PitchCall"] = ""
    grid["SpinAxis"] = np.nan
    grid["Outs"] = 0
    grid["Inning"] = 1
    grid["PAofInning"] = 1
    grid["PitchNo"] = 1
    grid["PitcherId"] = ""
    grid["Date"] = pd.NaT
    grid["TaggedHitType"] = "Undefined"
    grid["PlayResult"] = "Undefined"
    grid["OutsOnPlay"] = 0.0
    grid["RunsScored"] = 0.0
    grid["KorBB"] = "Undefined"

    grid["Balls"] = grid["Balls"].astype(int)
    grid["Strikes"] = grid["Strikes"].astype(int)
    return grid

def inject_styles():
    css = f"""
    <style>
      html, body, [data-testid="stAppViewContainer"] {{
        background: radial-gradient(1200px 600px at 10% 10%, rgba(204,0,0,0.08), transparent),
                    radial-gradient(1000px 500px at 90% 20%, rgba(192,192,192,0.06), transparent),
                    {SURFACE};
        color: {TEXT};
      }}
      /* Remove all default Streamlit centering */
      .main .block-container {{ 
          max-width: none !important;
          padding-left: 16px !important;
          padding-right: 16px !important;
      }}
      h1, h2, h3 {{ 
          text-align: left !important;
          margin: 0 !important;
          padding: 0 !important;
      }}
      .stButton>button {{
          background: linear-gradient(135deg, {ACCENT} 0%, #990000 100%);
          color: #fff;
          font-weight: 800;
          border: 0;
          border-radius: 10px;
      }}
      /* Force left alignment */
      [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {{
          padding-left: 0 !important;
          padding-right: 0 !important;
      }}
      .stTabs [role="tab"] {{ font-weight: 700; }}
      .rank-row {{
        display: grid;
        grid-template-columns: 38px 1fr 64px;
        align-items: center;
        gap: 10px;
        padding: 6px 8px;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.06);
        background: rgba(255,255,255,0.02);
        margin-bottom: 6px;
      }}
      .rank-num {{
        width: 34px; height: 34px; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 900; color: #0b0c10;
        background: linear-gradient(135deg, #ffd700, #ffb400);
      }}
      .rank-num.silver {{ background: linear-gradient(135deg, #d9dbe0, #bfc4cc); }}
      .rank-num.bronze {{ background: linear-gradient(135deg, #cd7f32, #b26a2a); }}
      .rank-num.other {{ background: linear-gradient(135deg, #9aa5b1, #7b8794); color: #0b0c10; }}
      .rank-pitcher {{ font-weight: 800; color: #eef2f7; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
      .rank-score {{ text-align: right; font-weight: 900; color: #e5e7eb; }}
      .rank-sd {{ color: #93a1b1; font-size: 0.85rem; margin-left: 6px; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


@st.cache_resource(show_spinner=True)
def load_full_pipeline():
    """Load data from the database and build models for recommendations."""
    needed_columns = (
        """
        TaggedPitchType, HorzBreak, InducedVertBreak, RelSpeed,
        RelSide, RelHeight, Extension, SpinRate, PitcherThrows,
        Pitcher, PitchCall, SpinAxis, PlateLocHeight,
        PlateLocSide, VertApprAngle, HorzApprAngle,
        ax0, az0, `Date`, Balls, Strikes, Outs, Inning, `Top.Bottom`, Level,
        PitcherId, GameID, Batter, PAofInning, AwayTeam, HomeTeam, PitchNo,
        BatterSide, KorBB, TaggedHitType, PlayResult, OutsOnPlay, RunsScored
        """
    ).strip()

    placeholders = ", ".join(["%s"] * len(UNLV_PITCHERS))
    query_unlv = f"""
        SELECT {needed_columns}
        FROM College_TM_Data
        WHERE Pitcher IN ({placeholders}) AND `Date` > '2025-01-01'
        UNION ALL
        SELECT {needed_columns}
        FROM College_Consulting_v3
        WHERE Pitcher IN ({placeholders}) AND `Date` > '2025-01-01'
    """

    conn = get_db_connection()
    try:
        df_unlv = pd.read_sql(query_unlv, conn, params=UNLV_PITCHERS * 2)
    finally:
        conn.close()
    df_unlv = pl.from_pandas(df_unlv)
    # ensure at_bat_id exists for pitcher attachment
    df_unlv = df_unlv.with_columns(
        (
            pl.col("GameID").cast(pl.Utf8)
            + "_"
            + pl.col("Top.Bottom").cast(pl.Utf8)
            + "_"
            + pl.col("Inning").cast(pl.Utf8)
            + "_"
            + pl.col("Batter").cast(pl.Utf8)
            + "_"
            + pl.col("PAofInning").cast(pl.Utf8)
        ).alias("at_bat_id")
    )

    # For now, use UNLV data as the full dataset
    df_all = df_unlv

    pa_all = build_pa_table(df_all)
    pa_all = attach_running_score(pa_all)
    pa_all = add_state_features(pa_all)
    pa_all = label_home_win(pa_all)
    wp_full = train_wp_model(pa_all)

    wpa_pa_unlv = compute_wpa_per_pa(pa_all, wp_full)
    wpa_pa_unlv = attach_pa_pitcher(wpa_pa_unlv, df_unlv)
    leaderboard_unlv = compute_stopper_leaderboard(wpa_pa_unlv)

    return wp_full, wpa_pa_unlv, leaderboard_unlv, df_all, df_unlv


def get_pitcher_profiles(pitcher_name=None):
    """Get pitch profiles for one or all UNLV pitchers"""
    # Build query
    if pitcher_name:
        where_clause = f"WHERE Pitcher = '{pitcher_name}'"
    else:
        # Create the IN clause with proper string formatting
        pitcher_list = [f"'{p}'" for p in UNLV_PITCHERS]
        pitcher_string = ", ".join(pitcher_list)
        where_clause = f"WHERE Pitcher IN ({pitcher_string})"
        
    query = f"""
    SELECT 
        Pitcher,
        TaggedPitchType as pitch_type,
        RelSpeed as start_speed,
        SpinRate as spin_rate,
        ax0 as ax,
        az0 as az,
        PlateLocSide as px,
        PlateLocHeight as pz,
        'R' as batter_hand,  # Default to RHH if not available
        PitcherThrows as pitcher_hand,
        HorzBreak,
        InducedVertBreak,
        VertBreak,
        RelHeight as release_height,
        RelSide as release_side,
        Extension as extension
    FROM tread_database_ec2.College_Consulting_practice
    {where_clause}
    """
    
    # Connect and query
    conn = get_db_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Populate cache so downstream views can reuse rows without another query
    try:
        if pitcher_name:
            _PRACTICE_BY_PITCHER[pitcher_name] = df.copy()
        else:
            for p, g in df.groupby('Pitcher', dropna=False):
                _PRACTICE_BY_PITCHER[p] = g.copy()
    except Exception:
        pass

    if len(df) == 0:
        return {}

    # Build profiles
    profiles = {}
    for pitcher in df['Pitcher'].unique():
        pitcher_data = df[df['Pitcher'] == pitcher]
        pitcher_hand = pitcher_data['pitcher_hand'].mode().iloc[0] if not pitcher_data['pitcher_hand'].empty else 'Right'
        
        for pitch_type, group in pitcher_data.groupby('pitch_type'):
            profiles[(pitcher, pitch_type)] = {
                'freq': len(group) / len(pitcher_data),
                'means': {
                    'start_speed': group['start_speed'].mean(),
                    'spin_rate': group['spin_rate'].mean(),
                    'ax': group['ax'].mean(),
                    'az': group['az'].mean(),
                    'px': group['px'].mean(),
                    'pz': group['pz'].mean(),
                    'HorzBreak': group['HorzBreak'].mean(),
                    'InducedVertBreak': group['InducedVertBreak'].mean(),
                    'release_height': group['release_height'].mean(),
                    'release_side': group['release_side'].mean(),
                    'extension': group['extension'].mean()
                },
                'stds': {
                    'start_speed': group['start_speed'].std(),
                    'spin_rate': group['spin_rate'].std(),
                    'ax': group['ax'].std(),
                    'az': group['az'].std(),
                    'px': group['px'].std(),
                    'pz': group['pz'].std(),
                    'HorzBreak': abs(group['HorzBreak'].std()),
                    'InducedVertBreak': group['InducedVertBreak'].std(),
                    'release_height': group['release_height'].std(),
                    'release_side': abs(group['release_side'].std()),
                    'extension': group['extension'].std()
                },
                'count': len(group),
                'pitcher_hand': pitcher_hand
            }
    
    return profiles


# ------------------ Execution+ helpers ------------------
def _load_pitcher_tm_raw(pitcher_name: str) -> pd.DataFrame:
    """
    Load TM rows for a pitcher from College_TM_Data (D1).
    Adds 'freq' = percentage (0–100) of each TaggedPitchType for this pitcher
    and filters out pitch types thrown < 3%.
    """
    sql = (
        """
        SELECT
          Pitcher            AS player_name,
          PitcherId          AS pitcher,
          HorzBreak          AS pfx_x_pv_adj,
          InducedVertBreak   AS pfx_z,
          RelSpeed           AS release_speed,
          RelSide            AS release_pos_x_pv_adj,
          RelHeight          AS release_pos_z,
          Extension          AS release_extension,
          PlateLocSide       AS plate_x_pv_adj,
          PlateLocHeight     AS plate_z,
          VertApprAngle      AS VAA_pred,
          HorzApprAngle      AS adj_HAA_pred,
          -- canonical pitch-type column for downstream logic
          TaggedPitchType,
          Balls,
          Strikes,
          PitcherThrows,
          BatterSide,
          -- keep raw fields EP may expect
          HorzBreak,
          InducedVertBreak,
          RelSpeed,
          RelSide,
          RelHeight,
          Extension,
          PlateLocSide,
          PlateLocHeight,
          VertApprAngle,
          HorzApprAngle,
          ax0,
          az0,
          `Date`,
          CASE
            WHEN TaggedPitchType IN ('Fastball','FourSeamFastBall','TwoSeamFastBall','Sinker','Cutter') THEN 'Fastball'
            WHEN TaggedPitchType IN ('ChangeUp','Splitter','Split-Finger') THEN 'Offspeed'
            WHEN TaggedPitchType IN ('Curveball','Slider','Sweeper','Knuckle Curve') THEN 'Breaking Ball'
            ELSE NULL
          END AS pitch_class
        FROM tread_database_ec2.College_TM_Data
        WHERE `Date` > '2025-01-14' AND Level = 'D1' AND Pitcher = %s
        """
    )
    conn = get_db_connection()
    try:
        df = pd.read_sql(sql, conn, params=[pitcher_name])
    finally:
        conn.close()

    if df.empty or "TaggedPitchType" not in df.columns:
        return df

    # compute % share and filter < 3%
    counts = df["TaggedPitchType"].value_counts(dropna=True)
    total = float(counts.sum())
    freq_map = (counts / total * 100.0).to_dict() if total > 0 else {}
    df["freq"] = df["TaggedPitchType"].map(freq_map).astype(float).fillna(0.0)

    keep = [pt for pt, pct in freq_map.items() if pct >= 3.0]
    if keep:
        df = df[df["TaggedPitchType"].isin(keep)].reset_index(drop=True)

    return df


def _execution_plus_hexbin_figure(
    grid_preds: pd.DataFrame,
    *,
    bins: int = 20,                 # kept for interface parity (imshow doesn't use it)
    cmin: float = -150.0,
    cmax: float = 150.0,
    pitch_types: list[str] | None = None,
    max_cols: int = 2               # <=— wrap pitch types into 2-wide columns
):
    """
    Create a DARK-THEME matplotlib raster heatmap figure of Execution+.
    Rows = count_class (Even/Hitter/Pitcher), and within each count_class,
    pitch types are laid out in a 2-wide grid (wrapping as needed).

    Returns: matplotlib.figure.Figure
    """
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle, Polygon

    df = grid_preds.copy()

    # Coordinate + value prep
    df["plate_x_pv_adj"] = df.get("plate_x_pv_adj", df.get("PlateLocSide")).astype(float)
    df["plate_z"] = df.get("PlateLocHeight").astype(float)
    df["execution_plus"] = pd.to_numeric(df["w.exec_plus"], errors="coerce").astype(float)

    # Pitch label column preference
    col_pitch = "pitch_type" if "pitch_type" in df.columns else ("TaggedPitchType" if "TaggedPitchType" in df.columns else None)
    if col_pitch is None:
        return None

    # Canonical label mapping for visuals (doesn't merge types—just renames)
    _name_map = {
        "FF": "Fastball", "FA": "Fastball", "4-Seam": "Fastball", "4S": "Fastball",
        "SI": "Sinker", "SNK": "Sinker", "TwoSeam": "Sinker", "2-Seam": "Sinker",
        "SL": "Slider", "CU": "Curveball", "KC": "Curveball", "KN": "Knuckleball",
        "CH": "ChangeUp", "Change-Up": "ChangeUp",
        "FS": "Splitter", "SF": "Split-Finger",
        "FC": "Cutter",
    }
    df["pitch_name_vis"] = (
        df[col_pitch].astype("string").str.strip().map(_name_map).fillna(df[col_pitch])
    )

    # Pitch order (optionally constrained by pitch_types)
    present = sorted(df["pitch_name_vis"].dropna().unique().tolist())
    if pitch_types:
        allowed = set(pitch_types)
        present = [p for p in present if p in allowed]
    if not present:
        return None

    # Count-class rows
    available_cc = set(df.get("count_class", pd.Series(dtype=object)).dropna().unique().tolist())
    count_order = [c for c in ["Even", "Hitter", "Pitcher"] if c in available_cc] or ["Even", "Hitter", "Pitcher"]

    # Color tags per pitch for titles
    pitch_levels = [
        "ChangeUp", "Changeup", "Curveball", "Knuckle Curve", "Slurve",
        "Sweeper", "Cutter", "Fastball", "4-Seam Fastball", "Knuckleball",
        "Sinker", "Slider", "Splitter", "Split-Finger", "Forkball",
        "Undefined", "Eephus", "Other", "Slow Curve", "First Pitch",
    ]
    color_vector = [
        "limegreen", "limegreen", "orange", "orange", "mediumaquamarine",
        "mediumaquamarine", "magenta", "red", "red", "green",
        "dodgerblue", "blueviolet", "cyan", "cyan", "cyan",
        "black", "gray", "gray", "gray", "tan",
    ]
    pitch_color_map = dict(zip(pitch_levels, color_vector))

    # Wrap pitch types into rows of max_cols (2-wide by default)
    max_cols = max(1, int(max_cols))
    pt_groups = [present[i:i + max_cols] for i in range(0, len(present), max_cols)]
    n_pt_rows = len(pt_groups)

    # Final subplot grid: for each count_class we stack all pt_rows
    fig_rows = len(count_order) * n_pt_rows
    fig_cols = max_cols

    # Sizing: bigger & taller per facet
    facet_w = 5.6   # width per facet (inches)
    facet_h = 5.0   # height per facet (inches)
    figsize = (facet_w * fig_cols, facet_h * fig_rows)

    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=figsize, sharex=True, sharey=True)

    # Normalize axes to 2D array
    if isinstance(axes, np.ndarray) and axes.ndim == 1:
        axes = axes.reshape(fig_rows, fig_cols)

    # Dark theme
    dark_bg = "#0b0b0b"
    dark_ax = "#141414"
    fig.patch.set_facecolor(dark_bg)

    norm = TwoSlopeNorm(vmin=float(cmin), vcenter=100.0, vmax=float(cmax))
    cmap = "RdBu_r"

    def add_strike_zone(ax, x_half=0.83, z_bot=1.5, z_top=3.5, lw=1.0, color="black"):
        ax.add_patch(Rectangle((-x_half, z_bot), 2 * x_half, z_top - z_bot, fill=False, lw=lw, edgecolor=color, zorder=10))

    def add_home_plate(ax, z0=0.0, height=0.17, width=17 / 12.0, lw=1.0, color="black"):
        w, h = width, height
        pts = np.array([(-w / 2, z0), (w / 2, z0), (w / 2, z0 + h * 0.62), (0, z0 + h), (-w / 2, z0 + h * 0.62)])
        ax.add_patch(Polygon(pts, closed=True, fill=False, lw=lw, edgecolor=color, zorder=10))

    last_img = None

    # Plot loop: for each count_class block through all pt_groups (rows) and columns
    for r_cc, cnt in enumerate(count_order):
        for r_pg, group in enumerate(pt_groups):
            grid_row_base = r_cc * n_pt_rows + r_pg
            for c_idx in range(fig_cols):
                ax = axes[grid_row_base, c_idx]

                if c_idx >= len(group):
                    # Empty cell in the final group row; hide axis
                    ax.set_axis_off()
                    continue

                pt = group[c_idx]
                sub = df[(df["count_class"] == cnt) & (df["pitch_name_vis"] == pt)].dropna(subset=["plate_x_pv_adj", "plate_z"])
                if sub.empty:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#e5e7eb")
                    ax.set_axis_off()
                    continue

                # Make a dense raster heatmap (mean EP by (z, x))
                heat = sub.pivot_table(index="plate_z", columns="plate_x_pv_adj", values="execution_plus", aggfunc="mean")
                heat = heat.sort_index().sort_index(axis=1)

                x_vals = heat.columns.to_numpy()
                z_vals = heat.index.to_numpy()
                img = ax.imshow(
                    heat.to_numpy()[::-1, :],
                    extent=[x_vals.min(), x_vals.max(), z_vals.min(), z_vals.max()],
                    origin="lower",
                    aspect="equal",    # square-ish cells, true strike-zone geometry
                    cmap=cmap,
                    norm=norm,
                    interpolation="nearest",
                )
                last_img = img

                # Overlays + aesthetics
                add_strike_zone(ax)
                add_home_plate(ax)
                ax.set_facecolor(dark_ax)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#888888")
                ax.tick_params(colors="#e5e7eb", labelsize=9)
                ax.xaxis.label.set_color("#e5e7eb")
                ax.yaxis.label.set_color("#e5e7eb")

                # Labels only on outer edges to save space
                is_bottom_band = (grid_row_base == fig_rows - 1)
                is_left_col = (c_idx == 0)
                if is_bottom_band:
                    ax.set_xlabel("Plate X (ft)", color="#e5e7eb")
                if is_left_col:
                    ax.set_ylabel("Plate Z (ft)", color="#e5e7eb")

                # Titles and count-class tag on the left-most column
                color = pitch_color_map.get(pt, "gray")
                ax.set_title(pt, fontsize=12, color="white",
                             bbox=dict(facecolor=color, edgecolor="none", pad=3.0))
                if is_left_col:
                    ax.text(
                        -0.24, 0.5, cnt,
                        transform=ax.transAxes, rotation=90,
                        va="center", ha="right",
                        fontsize=14, fontweight="bold", color="#ffffff",
                    )

    # Tight spacing between panels
    plt.subplots_adjust(
        left=0.035, right=0.99, top=0.93, bottom=0.12,
        wspace=0.06,   # closer columns
        hspace=0.08    # closer rows
    )

    # Wide, low colorbar close to the plots
    if last_img is not None:
        cax = fig.add_axes([0.16, 0.055, 0.68, 0.028])
        cbar = fig.colorbar(last_img, cax=cax, orientation="horizontal")
        cbar.outline.set_edgecolor("#888888")
        cbar.outline.set_linewidth(0.8)
        cbar.ax.tick_params(colors="#e5e7eb", labelsize=10)
        cbar.set_label("Execution+ (zero-centered)", color="#e5e7eb")

    fig.suptitle(
        "Pitch Effectiveness Heat Maps\nBlue = Positive Execution+ (Advantageous for the pitcher)",
        y=0.965, color="#e5e7eb",
    )
    return fig


@st.cache_resource(show_spinner=False)
def _ensure_xgboost() -> bool:
    """Ensure xgboost is importable in the current Streamlit Python.
    Attempts on-demand install with pip if missing.
    Returns True on success, False otherwise.
    """
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        pass
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost>=2.0"], stdout=subprocess.DEVNULL)
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False






def analyze_team_scenarios(mdl, leverage_from_state):
    """Run simulations for all pitchers and summarize overall performance."""
    with st.spinner("Getting pitcher profiles..."):
        all_profiles = get_pitcher_profiles()
    if not all_profiles:
        st.error("No pitcher profiles found")
        return
    all_results = []
    progress_container = st.empty()
    for i, pitcher in enumerate(UNLV_PITCHERS, 1):
        progress_container.markdown(
            f"<div style='text-align: center'>Simulating {pitcher} ({i}/{len(UNLV_PITCHERS)})...</div>",
            unsafe_allow_html=True,
        )
        pitcher_profiles = {k: v for k, v in all_profiles.items() if k[0] == pitcher}
        if not pitcher_profiles:
            continue
        results = simulate_all_situations(
            selected_pitcher=pitcher,
            profiles=pitcher_profiles,
            mdl=mdl,
            leverage_from_state=leverage_from_state,
        )
        if not results.empty:
            all_results.append(results)
    progress_container.empty()
    if not all_results:
        st.error("No simulation results available")
        return
    team_results = pd.concat(all_results, ignore_index=True)


    st.write("### Role Leaderboards")
    st.caption("Situations can contribute to multiple roles (e.g., Middle Tight counts for both Starter and Bridge).")
    role_summaries = {}
    for role, situation_list in ROLE_SITUATIONS.items():
        subset = team_results[team_results['Situation'].isin(situation_list)]
        if subset.empty:
            continue
        def _aggregate(group: pd.DataFrame) -> pd.Series:
            weights = group['Leverage']
            return pd.Series({
                'Avg ΔWP': group['Expected_Impact'].mean(),
                'Avg Stopper+': group['Stopper+'].mean(),
                'Weighted ΔWP': leverage_weighted_mean(group['Expected_Impact'], weights),
                'Weighted Stopper+': leverage_weighted_mean(group['Stopper+'], weights),
                'Avg Leverage': weights.mean(),
                'Samples': len(group),
            })

        summary = subset.groupby('Pitcher').apply(_aggregate).reset_index()
        summary['Samples'] = summary['Samples'].astype(int)

        leverage_weighted = role in LEVERAGE_WEIGHTED_ROLES
        sort_col = 'Weighted ΔWP' if leverage_weighted else 'Avg ΔWP'
        summary = summary.sort_values(sort_col, ascending=False)

        if leverage_weighted:
            display_cols = ['Pitcher', 'Weighted ΔWP', 'Weighted Stopper+', 'Avg Leverage', 'Samples']
        else:
            display_cols = ['Pitcher', 'Avg ΔWP', 'Avg Stopper+', 'Avg Leverage', 'Samples']

        role_summaries[role] = {
            'data': summary,
            'columns': display_cols,
            'caption': "Ranked by leverage-weighted change in win probability." if leverage_weighted else None,
        }

    role_tabs = st.tabs(list(ROLE_SITUATIONS.keys()))
    for tab, role in zip(role_tabs, ROLE_SITUATIONS.keys()):
        with tab:
            config = role_summaries.get(role)
            if config is None or config['data'].empty:
                st.info("No simulations available for this role.")
            else:
                if config.get('caption'):
                    st.caption(config['caption'])
                st.dataframe(config['data'][config['columns']], use_container_width=True)


    st.write("\n### ⚠️ Do Not Pitch List")
    st.write("The following pitchers should not be used:")
    for pitcher in DO_NOT_PITCH:
        st.write(f"- {pitcher}")


def main():
    if not login_gate():
        return

    inject_styles()
    st.title("UNLV Pitcher Role Simulator")

    with st.spinner("Building models from All-College data and UNLV subset (cached)…"):
        wp, pa_wpa_unlv, leaderboard_unlv, df_all, df_unlv = load_full_pipeline()

    # Create tabs
    tab_names = ["Individual Breakdown", "Team Breakdown", "Practice Planning"]  # Added Practice Planning tab
    tabs = st.tabs(tab_names)
    
    # Clear any sidebar content
    st.sidebar.empty()

    # --- Individual Breakdown ---
    with tabs[0]:
        st.subheader("Individual Pitcher Analysis")
        
        # Left-aligned dropdown
        selected_pitcher = st.selectbox(
            "Select Pitcher",
            options=UNLV_PITCHERS,
            index=0,
            key="individual_pitcher"
        )
        generate_info = st.button("Generate Info", key="generate_info_btn")
        if selected_pitcher and generate_info:
            spinner_placeholder = st.empty()
            spinner_placeholder.markdown('''
                <style>
                .ep-spinner-overlay {
                    position: fixed;
                    top: 0; left: 0; width: 100vw; height: 100vh;
                    background: rgba(0,0,0,0.82); z-index: 9999;
                    display: flex; align-items: center; justify-content: center;
                }
                .ep-spinner {
                    border: 8px solid #f3f3f3; /* Light grey */
                    border-top: 8px solid #CC0000; /* Red */
                    border-radius: 50%; width: 80px; height: 80px;
                    animation: ep-spin 1.1s linear infinite;
                }
                @keyframes ep-spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                .ep-spinner-text {
                    color: #fff; font-size: 1.3rem; margin-top: 24px; text-align: center;
                    font-weight: 700; letter-spacing: 0.02em;
                }
                </style>
                <div class="ep-spinner-overlay">
                    <div>
                        <div class="ep-spinner"></div>
                        <div class="ep-spinner-text">Generating pitcher info...</div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
            # --- Heavy computation and data loading here ---
            try:
                profiles = get_pitcher_profiles(selected_pitcher)
                
                if not profiles:
                    spinner_placeholder.empty()
                    st.warning("No pitch data found for this pitcher.")
                    return

                # Display pitch mix
                st.subheader("Pitch Mix")
                
                # Define pitch colors to match R color scheme
                pitch_colors = {
                    'ChangeUp': '#32CD32',        # limegreen
                    'Changeup': '#32CD32',        # limegreen
                    'Curveball': '#FFA500',       # orange
                    'Knuckle Curve': '#FFA500',   # orange
                    'Slurve': '#66CDAA',          # mediumaquamarine
                    'Sweeper': '#66CDAA',         # mediumaquamarine
                    'Cutter': '#FF00FF',          # magenta
                    'Fastball': '#FF0000',        # red
                    '4-Seam Fastball': '#FF0000', # red
                    'Knuckleball': '#008000',     # green
                    'Sinker': '#1E90FF',          # dodgerblue
                    'Slider': '#8A2BE2',          # blueviolet
                    'Splitter': '#00FFFF',        # cyan
                    'Split-Finger': '#00FFFF',    # cyan
                    'Forkball': '#00FFFF',        # cyan
                    'Undefined': '#000000'         # black
                }
                
                # Create pitch mix table with clean formatting
                pitch_data = []
                for (_, pt), profile in profiles.items():
                    # Skip undefined/NA pitch types
                    if pd.isna(pt) or pt == '0' or not pt:
                        continue
                        
                    # Map raw pitch types to standard names
                    pitch_type_map = {
                        'Fastball': 'Fastball',
                        '4-Seam Fastball': 'Fastball',
                        'FF0000[Fastball]': 'Fastball',
                        'Slider': 'Slider',
                        'SL': 'Slider',
                        'ChangeUp': 'ChangeUp',
                        'CH': 'ChangeUp',
                        'Curveball': 'Curveball',
                        'CU': 'Curveball',
                        'Sinker': 'Sinker',
                        'SI': 'Sinker',
                        'Cutter': 'Cutter',
                        'FC': 'Cutter'
                    }
                    
                    # Clean and standardize pitch type
                    clean_pt = pitch_type_map.get(pt.replace('[', '').replace(']', ''), pt)
                    
                    # Only include pitches thrown at least 3% of the time
                    if profile.get('freq', 0) >= 0.03:
                        pitch_data.append({
                            'FREQ': float(f"{profile['freq']*100:.1f}"),
                            'VELO': f"{profile['means']['start_speed']:.1f}",
                            'VERT': f"{profile['means']['InducedVertBreak']:.1f}",
                            'HORZ': f"{profile['means']['HorzBreak']:.1f}",
                            'HEIGHT': f"{profile['means']['release_height']:.1f}",
                            'SIDE': f"{profile['means']['release_side']:.1f}",
                            'PITCH': clean_pt
                        })
                
                # Convert to DataFrame and sort by frequency
                pitch_mix = pd.DataFrame(pitch_data)
                if not pitch_mix.empty:
                    pitch_mix = pitch_mix.sort_values('FREQ', ascending=False)
                    # Format frequency with percent sign for display
                    pitch_mix['FREQ'] = pitch_mix['FREQ'].map(lambda x: f"{x:.1f}%")
                    
                    # Reorder columns to match image (without '#' column)
                    pitch_mix = pitch_mix[['PITCH', 'FREQ', 'VELO', 'VERT', 'HORZ', 'HEIGHT', 'SIDE']]
                    
                # Create a styled dataframe with colored pitch type column only
                def style_pitch_type(row):
                    styles = [''] * len(row)  # Default style for all cells
                    pitch_idx = row.index.get_loc('PITCH')
                    pitch_type = row['PITCH']
                    
                    # Get color for this pitch type
                    color = pitch_colors.get(pitch_type, '#FFFFFF')
                    
                    # Only color the pitch type cell
                    styles[pitch_idx] = f'background-color: {color}; color: white; font-weight: 800'
                    
                    return styles

                # Apply styling to PITCH column and set numeric columns larger/bolder
                numeric_cols = ['FREQ','VELO','VERT','HORZ','HEIGHT','SIDE']
                styled_df = (
                    pitch_mix
                    .style
                    .apply(style_pitch_type, axis=1)
                    .set_properties(subset=numeric_cols, **{
                        'font-weight': '800',
                        'font-size': '1.05rem',
                        'color': '#F3F4F6',
                        'text-align': 'right'
                    })
                )
                
                # Display with custom CSS
                st.markdown("""
                    <style>
                    /* Simple table width control */
                    [data-testid="stDataFrame"] {
                        width: 45vw !important;
                        max-width: 45vw !important;
                    }
                    
                    /* Ensure table stays in its container */
                    [data-testid="column"] {
                        width: 45vw !important;
                        min-width: 45vw !important;
                    }
                    </style>
                """, unsafe_allow_html=True)
                
                # Force horizontal layout with flexbox
                st.markdown("""
                    <style>
                        /* Container styles */
                        [data-testid="stHorizontalBlock"] {
                            display: flex !important;
                            flex-direction: row !important;
                            flex-wrap: nowrap !important;
                            justify-content: flex-start !important;
                            gap: 1rem !important;
                            max-width: 100% !important;
                        }
                        
                        /* Both columns exactly 50% */
                        [data-testid="column"] {
                            width: 50% !important;
                            flex: 0 0 50% !important;
                            min-width: 0 !important;
                        }
                        
                        /* Ensure content doesn't wrap */
                        [data-testid="stVerticalBlock"] {
                            flex-wrap: nowrap !important;
                        }
                    </style>
                """, unsafe_allow_html=True)

                # Create equal-width columns
                cols = st.columns([1, 1])
                
                # Left column - Table (45% viewport width)
                with cols[0]:
                    st.dataframe(
                        styled_df,
                        hide_index=True,
                        use_container_width=True
                    )
                    try:
                        if not _ensure_xgboost():
                            raise ModuleNotFoundError("xgboost is not installed and auto-install failed")
                        raw_tm = _PRACTICE_BY_PITCHER.get(selected_pitcher)
                        if raw_tm is None or raw_tm.empty:
                            raw_tm = _load_pitcher_tm_raw(selected_pitcher)
                        if raw_tm.empty:
                            st.warning("No TrackMan rows found for this pitcher since 2025-01-01.")
                        else:
                            out_dir = Path(__file__).parent / "outputs" / "streamlit_ep_plots"
                            out_dir.mkdir(parents=True, exist_ok=True)
                            cli_args = [
                                "--pitcher", selected_pitcher,
                                "--query-since", "2025-01-01",
                                "--save-dir", str(out_dir.resolve()),
                                "--models-root", ".",
                                "--bins", "20",
                                "--cmin", "0",
                                "--cmax", "200"
                            ]
                            import subprocess
                            import sys
                            result = subprocess.run(
                                [sys.executable, "execution_plus_cli2.py", *cli_args],
                                capture_output=True,
                                text=True
                            )
                            print("STDOUT:", result.stdout)
                            print("STDERR:", result.stderr)
                            if result.returncode != 0:
                                st.error(f"Execution+ subprocess failed: {result.stderr}")
                            combined_img = out_dir / f"execution_plus_{selected_pitcher.replace(', ', '_')}_platoon.png"
                            if not combined_img.exists():
                                st.error(f"Expected output file not found: {combined_img}")
                            else:
                                st.image(str(combined_img), use_container_width=True)
                                spinner_placeholder.empty()  # Remove overlay as soon as PNG is shown
                    except Exception as e:
                        st.error(f"Error generating Execution+ plots: {str(e)}")

                # Right column - Plot (will be added later)
                with cols[1]:
                    pass  # Plot will be added later
                
                # Filter out undefined pitch types
                movement_data = pd.DataFrame([
                    {
                        'pitch_type': pt,
                        'HorzBreak': profile['means']['HorzBreak'],
                        'InducedVertBreak': profile['means']['InducedVertBreak'],
                        'start_speed': profile['means']['start_speed']
                    }
                    for (_, pt), profile in profiles.items()
                    if (not pd.isna(pt)) and pt != '0' and pt and (profile.get('freq', 0) >= 0.03)
                ])
                
                # Create plot using Plotly
                import plotly.express as px
                import plotly.graph_objects as go
                
                # Create scatter plot with bigger markers
                fig = px.scatter(
                    movement_data,
                    x='HorzBreak',
                    y='InducedVertBreak',
                    color='pitch_type',
                    color_discrete_map=pitch_colors,  # Use same colors as table
                    labels={
                        'HorzBreak': 'Horizontal Break (in.)',
                        'InducedVertBreak': 'Induced Vertical Break (in.)'
                    }
                )
                
                # Update marker size and opacity
                fig.update_traces(
                    marker=dict(
                        size=20,          # Bigger markers
                        line=dict(        # Add white border
                            width=2,
                            color='white'
                        ),
                        opacity=0.8       # Slight transparency
                    )
                )
                
                # Update layout for static, fixed-ratio plot
# Fixed plot layout
                fig.update_layout(
                    width=550,              # Adjusted width for 50% column
                    height=600,             # Match table height
                    showlegend=True,
                    legend=dict(
                        orientation="h",     # Horizontal legend
                        yanchor="bottom",
                        y=-0.15,
                        xanchor="left",
                        x=0
                    ),
                    margin=dict(l=10, r=10, t=10, b=30),  # Minimal margins
                    plot_bgcolor='black',
                    paper_bgcolor='black',
                    font=dict(color='white', size=16),   # slightly larger text
                    xaxis=dict(
                        range=[-20, 20],
                        gridcolor='rgba(128, 128, 128, 0.2)',
                        zerolinecolor='rgba(255, 255, 255, 0.5)',
                        zerolinewidth=2,
                        showgrid=True,
                        zeroline=True
                    ),
                    yaxis=dict(
                        range=[-20, 20],
                        gridcolor='rgba(128, 128, 128, 0.2)',
                        zerolinecolor='rgba(255, 255, 255, 0.5)',
                        zerolinewidth=2,
                        showgrid=True,
                        zeroline=True
                    ),
                )

                
                # Disable zoom and pan
                fig.update_xaxes(fixedrange=True)
                fig.update_yaxes(fixedrange=True)
                
                # Display plot in the right column
                with cols[1]:
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={'displayModeBar': False}
                    )


                # # First, run Execution+ analysis
                # cmin, cmax, bins = 0.0, 200.0, 20
                # execution_plus_results = None
                # with st.spinner("Building Execution+ predictions and heat map…"):
                #     try:
                #         # Ensure xgboost available for model inference (auto-install if needed)
                #         if not _ensure_xgboost():
                #             raise ModuleNotFoundError("xgboost is not installed and auto-install failed")
                #         raw_tm = _PRACTICE_BY_PITCHER.get(selected_pitcher)
                #         if raw_tm is None or raw_tm.empty:
                #             raw_tm = _load_pitcher_tm_raw(selected_pitcher)
                #         if raw_tm.empty:
                #             st.warning("No TrackMan rows found for this pitcher since 2025-01-01.")
                #         else:
                #             # Only keep the CLI call for Execution+
                #             out_dir = Path(__file__).parent / "outputs" / "streamlit_ep_plots"
                #             out_dir.mkdir(parents=True, exist_ok=True)
                #             cli_args = [
                #                 "--pitcher", selected_pitcher,
                #                 "--query-since", "2025-01-01",
                #                 "--save-dir", str(out_dir.resolve()),
                #                 "--models-root", ".",
                #                 "--bins", "20",
                #                 "--cmin", "0",
                #                 "--cmax", "200"
                #             ]
                #             subprocess.run([
                #                 sys.executable, "-m", "execution_plus_cli2", *cli_args
                #             ], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            
                #             # Display the generated plot
                #             combined_img = out_dir / f"execution_plus_{selected_pitcher.replace(', ', '_')}_platoon.png"
                #             st.image(str(combined_img), use_container_width=True)
                #             spinner_placeholder.empty()  # Remove overlay as soon as PNG is shown
                #     except Exception as e:
                #         st.error(f"Error generating Execution+ plots: {str(e)}")

                with cols[1]:  # Right column - continue after plot
                    st.subheader("Situation Analysis")
                    if selected_pitcher in DO_NOT_PITCH:
                        st.error(f"⚠️ WARNING: {selected_pitcher} is on the Do Not Pitch list. Scores will be penalized by -15 points.")
                    
                    # Run simulation automatically
                    with st.spinner("Analyzing situations..."):
                        try:
                            mdl = load_or_train_pitch_delta_model(df_all)
                            results_df = simulate_all_situations(
                                selected_pitcher=selected_pitcher,
                                profiles=profiles,
                                mdl=mdl,
                                leverage_from_state=leverage_from_state
                            )
                            analyze_simulation_results(results_df)
                        except Exception as e:
                            spinner_placeholder.empty()
                            st.error(f"Error processing data: {str(e)}")


            except Exception as e:
                spinner_placeholder.empty()
                st.error(f"Error processing data: {str(e)}")

    # --- Team Breakdown ---
    # with tabs[1]:
    #     st.subheader("Team Scenario Analysis")
    #     # Center the team analysis button
    #     _tb1, _tb2, _tb3 = st.columns([1, 3, 1])
    #     with _tb2:
    #         team_btn = st.button("Run Team Analysis", key="team_analysis")
        
    #     if team_btn:
    #         # Get the model
    #         mdl = load_or_train_pitch_delta_model(df_all)
    #         
    #         # Run team analysis
    #         analyze_team_scenarios(mdl, leverage_from_state)

    # --- Practice Planning ---
    with tabs[2]:
        main_col, sidebar_col = st.columns([2, 1], gap="large")
        with main_col:
            st.subheader("Bullpen Planner")
            st.write("Plan bullpen usage for an upcoming outing.")
            bullpen_pitcher = st.selectbox(
                "1) What pitcher are we planning for?",
                options=UNLV_PITCHERS,
                index=0,
                key="bullpen_planner_pitcher"
            )
            bullpen_roles = ["Starter", "Bridge", "High Leverage", "Long Relief", "Inning Eater"]
            bullpen_role = st.selectbox(
                "2) What role are we planning for?",
                options=bullpen_roles,
                index=0,
                key="bullpen_planner_role"
            )
            days_options = list(range(2, 8))  # Remove 1 day option
            bullpen_days = st.selectbox(
                "3) How many days til projected game outing?",
                options=days_options,
                index=0,
                key="bullpen_planner_days"
            )
            # --- Plotly plot for selected pitch type ---
            import importlib.util
            import sys
            import plotly.graph_objs as go
            # Extract pitch type from selected_pitch (set below in sidebar_col)
            selected_pitch_type = st.session_state.get('bullpen_selected_pitch_type', None)
            if not selected_pitch_type:
                selected_pitch_type = None
            if selected_pitch_type:
                plotly_path = 'execution_plus_cli2_plotly.py'
                spec = importlib.util.spec_from_file_location("execution_plus_cli2_plotly", plotly_path)
                ep_plotly = importlib.util.module_from_spec(spec)
                sys.modules["execution_plus_cli2_plotly"] = ep_plotly
                spec.loader.exec_module(ep_plotly)
                fig = ep_plotly.get_pitch_type_figure(bullpen_pitcher, selected_pitch_type, batter_side='RHH')
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with sidebar_col:
            st.markdown("### Bullpen Plan")
            # Max pitches by role
            if bullpen_role in ["Starter", "Inning Eater", "Long Relief"]:
                max_pitches = 40
            elif bullpen_role == "Bridge":
                max_pitches = 25
            elif bullpen_role == "High Leverage":
                max_pitches = 15
            else:
                max_pitches = 25
            # Diminishing workload by days
            if bullpen_days in [5, 6, 7]:
                default_pitches = max_pitches
            else:
                default_pitches = max_pitches - 5
            st.write(f"**Planning for:** {bullpen_pitcher}")
            st.write(f"**Role:** {bullpen_role}")
            st.write(f"**Days until outing:** {bullpen_days}")
            st.write(f"**Default bullpen length:** {default_pitches} pitches")
            shrink_pct = {
                "Starter": 0.0,
                "Inning Eater": 0.05,
                "Long Relief": 0.10,
                "Bridge": 0.15,
                "High Leverage": 0.20
            }.get(bullpen_role, 0.0)
            shrunk_pitches = int(default_pitches * (1 - shrink_pct))
            if shrink_pct > 0:
                st.write(f"**Adjusted bullpen length:** {shrunk_pitches} pitches (shrunk by {int(shrink_pct*100)}%)")
            # Get pitch type frequencies for this pitcher
            try:
                df_tm = _load_pitcher_tm_raw(bullpen_pitcher)
                pitch_freq = df_tm['TaggedPitchType'].value_counts(normalize=True).to_dict()
                pitch_types = list(pitch_freq.keys())
                pitch_counts = [int(round(freq * shrunk_pitches)) for freq in pitch_freq.values()]
                # Adjust to match shrunk_pitches exactly
                while sum(pitch_counts) < shrunk_pitches:
                    for i in range(len(pitch_counts)):
                        if sum(pitch_counts) < shrunk_pitches:
                            pitch_counts[i] += 1
                while sum(pitch_counts) > shrunk_pitches:
                    for i in range(len(pitch_counts)):
                        if pitch_counts[i] > 0 and sum(pitch_counts) > shrunk_pitches:
                            pitch_counts[i] -= 1
                pitch_sequence = []
                for pt, count in zip(pitch_types, pitch_counts):
                    pitch_sequence.extend([pt] * count)
            except Exception:
                pitch_sequence = ["Unknown"] * shrunk_pitches
            # Interactive pitch selection
            st.markdown("---")
            st.markdown("#### Pitch Sequence")
            pitch_labels = [f"Pitch {i+1}: {pitch_sequence[i]}" for i in range(shrunk_pitches)]
            selected_pitch = st.radio(
                "Select a pitch to plan:",
                pitch_labels,
                index=0,
                key="bullpen_selected_pitch"
            )
            # Save selected pitch type to session state for use in main_col
            if selected_pitch:
                st.session_state['bullpen_selected_pitch_type'] = selected_pitch.split(':', 1)[1].strip()
            st.markdown(f"<div style='background-color:#222;border-radius:6px;padding:8px 12px;margin-top:8px;'><b>Selected:</b> <span style='color:#CC0000;font-weight:bold;'>{selected_pitch}</span></div>", unsafe_allow_html=True)


    # --- Leaderboard (Predictive only) --- [COMMENTED OUT]
    # with tabs[2]:
    #     st.subheader("Top Recommendations")
    #
    #     # Game situation filters in compact rows
    #     col1, col2, col3, col4 = st.columns(4)
    #     with col1:
    #         inn = st.number_input("Inning", min_value=1, max_value=20, value=1, step=1)
    #         on1 = st.checkbox("1B", value=False)
    #     with col2:
    #         half = st.selectbox("Half", options=["Top", "Bottom"], index=1)
    #         on2 = st.checkbox("2B", value=False)
    #     with col3:
    #         outs = st.number_input("Outs", min_value=0, max_value=2, value=0, step=1)
    #         on3 = st.checkbox("3B", value=False)
    #     with col4:
    #         batter_side = st.selectbox("Batter", ["All", "L", "R"], index=0)
    #         n_pitches = st.number_input("Pitches", min_value=5, max_value=40, value=20, step=1)
    #
    #     # Settings in a single row
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         sd = st.slider("Run Differential", -10, 10, 0, step=1)
    #     with col2:
    #         min_samp = st.slider("Min Samples", 0, 50, 10, step=5)
    #
    #     # Store filter values
    #     args = {
    #         "inning": int(inn),
    #         "half": half,
    #         "outs": int(outs),
    #         "on1": 1 if on1 else 0,
    #         "on2": 1 if on2 else 0,
    #         "on3": 1 if on3 else 0,
    #         "score_diff": float(sd),
    #         "batter_side": batter_side,
    #         "min_samples": int(min_samp),
    #     }
    #
    #     # Load models
    #     mdl = load_or_train_pitch_delta_model(df_all)
    #     profiles = build_pitcher_profiles(df_unlv)
    #     # Compute LI from the selected scenario (auto)
    #     leverage = leverage_from_state(
    #         inning=args["inning"],
    #         half=args["half"],
    #         outs=args["outs"],
    #         on1=args["on1"], on2=args["on2"], on3=args["on3"],
    #         score_diff=args["score_diff"],
    #     )
    #     st.caption(f"Leverage Index (auto): {leverage:.2f}")
    #     batter_hand = args["batter_side"]  # reuse sidebar selection
    #
    #     pitchers = sorted({k[0] for k in profiles.keys()})
    #     # LHP subset for standardization (used here only to tag hand)
    #     LHP_STANDARDIZE_NAMES = {
    #         'Dilhoff, Parker', 'Manning, LJ', 'Gomberg, Jacob',
    #         'Bowen, Gavyn', 'Kubasky, Noah'
    #     }
    #
    #     rows = []
    #     for p in pitchers:
    #         try:
    #             exp = simulate_expected_delta(
    #                 mdl=mdl,
    #                 profiles=profiles,
    #                 df_all=df_all,
    #                 pitcher_name=p,
    #                 n_pitches=n_pitches,
    #                 batter_hand=batter_hand,
    #                 half=args["half"],
    #                 inning=args["inning"],
    #                 leverage_index=leverage,
    #                 balls=args["outs"] if False else 0,  # keep counts simple for now
    #                 strikes=0,
    #                 paths=200,
    #                 pitcher_hand=("Left" if p in LHP_STANDARDIZE_NAMES else "Right"),
    #             )
    #         except Exception:
    #             exp = 0.0
    #         rows.append({"Pitcher": p, "Predicted_Delta_WP": exp})
    #
    #     pdf = pl.DataFrame(rows)
    #     # Scale predictions to 100-base, sd=10
    #     if pdf.height > 0:
    #         mu = pdf.select(pl.col("Predicted_Delta_WP").mean()).item()
    #         sd = pdf.select(pl.col("Predicted_Delta_WP").std()).item() or 0.0
    #         if sd == 0 or sd is None:
    #             pdf = pdf.with_columns(pl.lit(100.0).alias("Stopper+ (Pred)"))
    #         else:
    #             pdf = pdf.with_columns((100.0 + 10.0 * ((pl.col("Predicted_Delta_WP") - float(mu)) / float(sd))).alias("Stopper+ (Pred)"))
    #         
    #         # Apply penalty to do-not-pitch pitchers
    #         pdf = pdf.with_columns(
    #             pl.when(pl.col("Pitcher").is_in(DO_NOT_PITCH))
    #             .then(pl.col("Stopper+ (Pred)") - 15.0)
    #             .otherwise(pl.col("Stopper+ (Pred)"))
    #             .alias("Stopper+ (Pred)")
    #         )
    #         
    #         pdf = pdf.select(["Pitcher", "Stopper+ (Pred)", "Predicted_Delta_WP"]).sort("Stopper+ (Pred)", descending=True)
    #     # Center the leaderboard table
    #     _lb1, _lb2, _lb3 = st.columns([1, 3, 1])
    #     with _lb2:
    #         st.dataframe(pdf.to_pandas(), use_container_width=True)


if __name__ == "__main__":
    main()
