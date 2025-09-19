import base64
import io
import pandas as pd
import polars as pl
import streamlit as st
import numpy as np
from typing import Dict, Any
from stopper_predictor import simulate_expected_delta
import sklearn
import hashlib

_LEVERAGE_TABLE = None

def _load_leverage_table() -> pl.DataFrame:
    global _LEVERAGE_TABLE
    if _LEVERAGE_TABLE is not None:
        return _LEVERAGE_TABLE
    path = 'game_states_with_roles.parquet'
    try:
        tbl = pl.read_parquet(path)
        if 'game_state' in tbl.columns and 'leverage_index' in tbl.columns:
            _LEVERAGE_TABLE = tbl.select(['game_state', 'leverage_index']).unique()
            return _LEVERAGE_TABLE
    except Exception as e:
        st.error(f"Could not load leverage table: {e}")
    return None

# List of pitchers that should not be recommended
DO_NOT_PITCH = ['Barna, Cal', 'Rogers, Dylan']

# Generic baseline for Stopper+ calculation
GLOBAL_BASELINE = {'mean': -0.1, 'sd': 0.1}

# Game context categories
PHASES = ['Early', 'Middle', 'Late']
SCORE_CONTEXTS = ['Down Lots', 'Down Little', 'Tight', 'Up Little', 'Up Lots']

BASE_PITCH_COUNT = 16
PHASE_WORKLOAD_MULTIPLIER = {
    'Early': 1.15,
    'Middle': 1.0,
    'Late': 0.75,
}
SCORE_WORKLOAD_MULTIPLIER = {
    'Down Lots': 1.25,
    'Down Little': 1.1,
    'Tight': 1.0,
    'Up Little': 0.9,
    'Up Lots': 0.8,
}
SCORE_COUNT_TENDENCIES = {
    'Down Lots': (2, 1),
    'Down Little': (2, 1),
    'Tight': (1, 1),
    'Up Little': (1, 0),
    'Up Lots': (0, 0),
}

PRECOMPUTED_BASE64 = """$(cat precomputed_base64.txt)"""
_PRECOMPUTED_CACHE: pd.DataFrame | None = None


def _get_precomputed_df() -> pd.DataFrame:
    """Return the dataframe of precomputed situation metrics."""
    global _PRECOMPUTED_CACHE
    if _PRECOMPUTED_CACHE is None:
        decoded = base64.b64decode(PRECOMPUTED_BASE64)
        _PRECOMPUTED_CACHE = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return _PRECOMPUTED_CACHE.copy()


def standardize_score(score: float, baseline: dict | None = None) -> float:
    """Standardize a score to a 100-based scale with SD=10."""
    base = baseline or GLOBAL_BASELINE
    sd = base.get('sd', 1.0) or 0.0
    mu = base.get('mean', 0.0)
    if sd == 0:
        return 100.0
    return 100.0 + 10.0 * ((score - mu) / sd)

def leverage_from_state(inning, half, outs, on1, on2, on3, score_diff):
    tbl = _load_leverage_table()
    if tbl is None:
        return 1.0
    half_token = "Top" if str(half).lower().startswith("t") else "Bot"
    runner_state = f"{int(on1)}{int(on2)}{int(on3)}"
    gs = f"{round(float(inning),1)}_{half_token}_{int(outs)}_{runner_state}_{round(float(score_diff),1)}"
    try:
        val = tbl.filter(pl.col("game_state") == gs).select("leverage_index").to_series()
        if len(val) > 0 and val[0] is not None:
            return float(val[0])
    except Exception:
        pass
    return 1.0

def simulate_all_situations(selected_pitcher: str, profiles: Dict, mdl: Any, leverage_from_state: callable = leverage_from_state):
    """Return hardcoded situation metrics for the selected pitcher."""
    # Comment out any simulation logic
    # Instead, use the hardcoded table below
    import pandas as pd
    hardcoded_data = [
        # Pitcher, Phase, Score_Context, MeanImpact, StdImpact, MeanStopper, StdStopper, Samples
        ["Albright, Cody", "Early", "Down Lots", 0.0640052, 0.24080527, 116.40052, 24.080527, 25],
        ["Albright, Cody", "Early", "Tight", 0.06249557, 0.40193184, 116.24956, 40.193184, 25],
        ["Albright, Cody", "Middle", "Tight", 0.03129765, 0.24043233, 113.12977, 24.043233, 25],
        ["Albright, Cody", "Middle", "Down Lots", 0.01246947, 0.10690855, 111.24695, 10.690855, 25],
        ["Albright, Cody", "Late", "Up Little", -0.01060847, 0.07289174, 108.93915, 7.289174, 25],
        ["Albright, Cody", "Early", "Up Lots", -0.01854034, 0.06174234, 108.14597, 6.174234, 25],
        ["Albright, Cody", "Middle", "Up Little", -0.02186254, 0.19882148, 107.81375, 19.882148, 25],
        ["Albright, Cody", "Late", "Up Lots", -0.02508872, 0.05803313, 107.49113, 5.803313, 25],
        ["Albright, Cody", "Middle", "Up Lots", -0.02556165, 0.03016765, 107.44383, 3.016765, 25],
        ["Albright, Cody", "Early", "Down Little", -0.02671709, 0.25879542, 107.32829, 25.879542, 25],
        ["Albright, Cody", "Late", "Down Lots", -0.03259994, 0.05778683, 106.74001, 5.778683, 25],
        ["Albright, Cody", "Middle", "Down Little", -0.0530737, 0.29881899, 104.69263, 29.881899, 25],
        ["Albright, Cody", "Early", "Up Little", -0.05602953, 0.21465283, 104.39705, 21.465283, 25],
        ["Albright, Cody", "Late", "Tight", -0.07316393, 0.14939128, 102.68361, 14.939128, 25],
        ["Albright, Cody", "Late", "Down Little", -0.1272504, 0.16472398, 97.27496, 16.472398, 25],
        ["Barna, Cal", "Middle", "Tight", -0.009834482, 0.21404578, 94.01655, 21.404578, 25],
        ["Barna, Cal", "Early", "Tight", -0.01267873, 0.42708624, 93.73213, 42.708624, 25],
        ["Barna, Cal", "Early", "Down Lots", -0.02215836, 0.22144631, 92.78416, 22.144631, 25],
        ["Barna, Cal", "Late", "Up Little", -0.0415542, 0.06341703, 90.84458, 6.341703, 25],
        ["Barna, Cal", "Late", "Up Lots", -0.06112477, 0.04716243, 88.88752, 4.716243, 25],
        ["Barna, Cal", "Middle", "Down Lots", -0.06373863, 0.09704728, 88.62614, 9.704728, 25],
        ["Barna, Cal", "Middle", "Up Little", -0.06528345, 0.18250925, 88.47165, 18.250925, 25],
        ["Barna, Cal", "Early", "Up Lots", -0.07204081, 0.05661045, 87.79592, 5.661045, 25],
        ["Barna, Cal", "Middle", "Up Lots", -0.07289111, 0.02716787, 87.71089, 2.716787, 25],
        ["Barna, Cal", "Early", "Down Little", -0.08452919, 0.23707583, 86.54708, 23.707583, 25],
        ["Barna, Cal", "Late", "Down Lots", -0.08473282, 0.05145415, 86.52672, 5.145415, 25],
        ["Barna, Cal", "Middle", "Down Little", -0.09229935, 0.26968317, 85.77006, 26.968317, 25],
        ["Barna, Cal", "Late", "Tight", -0.1046097, 0.13633001, 84.53903, 13.633001, 25],
        ["Barna, Cal", "Early", "Up Little", -0.1118917, 0.18787657, 83.81083, 18.787657, 25],
        ["Barna, Cal", "Late", "Down Little", -0.1566953, 0.14373849, 79.33047, 14.373849, 25],
        # ... (add all other rows from your provided table here) ...
    ]
    columns = ["Pitcher", "Phase", "Score_Context", "MeanImpact", "StdImpact", "MeanStopper", "StdStopper", "Samples"]
    df = pd.DataFrame(hardcoded_data, columns=columns)
    filtered = df[df['Pitcher'] == selected_pitcher].copy()
    if filtered.empty:
        st.warning(f"No hardcoded situations for {selected_pitcher}.")
        return pd.DataFrame()
    filtered['Phase'] = pd.Categorical(filtered['Phase'], categories=PHASES, ordered=True)
    filtered['Score_Context'] = pd.Categorical(filtered['Score_Context'], categories=SCORE_CONTEXTS, ordered=True)
    filtered = filtered.sort_values(['Phase', 'Score_Context']).reset_index(drop=True)
    return filtered

def analyze_simulation_results(results_df: pd.DataFrame):
    import pandas as pd
    import numpy as np
    import streamlit as st

    if results_df.empty:
        st.error("No results to analyze")
        return

    pitcher_name = results_df['Pitcher'].iloc[0] if not results_df.empty else "Unknown"

    # Use MeanStopper for coloring and display
    phase_analysis = results_df.groupby(['Phase', 'Score_Context']).agg(
        MeanStopper=('MeanStopper', 'mean'),
        Samples=('MeanStopper', 'count')
    ).round(4)

    matrix_df = phase_analysis.reset_index().rename(columns={'Score_Context': 'Score'})

    PHASES = ['Early', 'Middle', 'Late']
    SCORE_CONTEXTS = ['Down Lots', 'Down Little', 'Tight', 'Up Little', 'Up Lots']

    pivot = matrix_df.pivot(index='Phase', columns='Score', values='MeanStopper').reindex(index=PHASES, columns=SCORE_CONTEXTS)

    RED_RGB = (139, 0, 0)
    GREEN_RGB = (0, 100, 0)
    WHITE_RGB = (255, 255, 255)

    def blend_rgb(color_from: tuple[int, int, int], color_to: tuple[int, int, int], weight: float) -> tuple[int, int, int]:
        weight = max(0.0, min(1.0, weight))
        return tuple(int(round(cf + (ct - cf) * weight)) for cf, ct in zip(color_from, color_to))

    def style_from_rgb(rgb: tuple[int, int, int]) -> str:
        r, g, b = rgb
        luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
        text_color = '#111111' if luminance > 0.6 else '#F5F5F5'
        return f'background-color: rgb({r},{g},{b}); color: {text_color}; font-weight: 700'

    def apply_self_gradient(dataframe: pd.DataFrame) -> pd.DataFrame:
        values = dataframe.to_numpy(dtype=float)
        mask = np.isfinite(values)
        style = pd.DataFrame('background-color: #2b2b2b; color: #bbbbbb', index=dataframe.index, columns=dataframe.columns)
        if not mask.any():
            return style
        min_val = np.nanmin(values)
        max_val = np.nanmax(values)
        mean_val = np.nanmean(values)
        span_neg = mean_val - min_val
        span_pos = max_val - mean_val
        for row_label in dataframe.index:
            for col_label in dataframe.columns:
                val = dataframe.loc[row_label, col_label]
                if pd.isna(val):
                    continue
                if span_neg <= 1e-9 and span_pos <= 1e-9:
                    rgb = WHITE_RGB
                elif val <= mean_val:
                    weight = 0.0 if span_neg <= 1e-9 else (mean_val - val) / span_neg
                    rgb = blend_rgb(WHITE_RGB, RED_RGB, weight)
                else:
                    weight = 0.0 if span_pos <= 1e-9 else (val - mean_val) / span_pos
                    rgb = blend_rgb(WHITE_RGB, GREEN_RGB, weight)
                style.loc[row_label, col_label] = style_from_rgb(rgb)
        return style

    styled_pivot = pivot.style.apply(apply_self_gradient, axis=None).format(precision=2)
    st.dataframe(styled_pivot, use_container_width=True)

    # Optionally, show the raw numbers below
    st.markdown("#### Raw MeanStopper Table")
    st.dataframe(pivot, use_container_width=True)

    # --- Rest of function unchanged (role recommendations, etc) ---
    def get_role_recommendations(pivot_table: pd.DataFrame) -> list:
        if not isinstance(pivot_table, pd.DataFrame) or pivot_table.empty:
            return []
        pivot = pivot_table.reindex(index=PHASES, columns=SCORE_CONTEXTS)
        mean_val = pivot.mean().mean()
        max_val = pivot.max().max()
        def is_good(val):
            return val > mean_val if pd.notna(val) else False
        def is_best(val):
            return val == max_val if pd.notna(val) else False
        starter_score = sum([
            is_good(pivot.loc['Early', 'Up Little']),
            is_good(pivot.loc['Early', 'Tight']),
            is_good(pivot.loc['Early', 'Down Little']),
            is_good(pivot.loc['Middle', 'Up Little']),
            is_good(pivot.loc['Middle', 'Tight']),
            is_good(pivot.loc['Middle', 'Down Little'])
        ])
        bridge_score = sum([
            is_good(pivot.loc['Middle', 'Up Little']),
            is_good(pivot.loc['Middle', 'Tight']),
            is_good(pivot.loc['Middle', 'Down Little']),
            is_good(pivot.loc['Late', 'Up Little']),
            is_good(pivot.loc['Late', 'Tight']),
            is_good(pivot.loc['Late', 'Down Little'])
        ])
        high_leverage_score = sum([
            is_good(pivot.loc['Late', 'Up Little']),
            is_good(pivot.loc['Late', 'Tight']),
            is_good(pivot.loc['Late', 'Down Little']),
            is_good(pivot.loc['Early', 'Tight']),
            is_good(pivot.loc['Middle', 'Tight'])
        ])
        inning_eater_score = sum([
            is_good(pivot.loc['Early', 'Down Lots']),
            is_good(pivot.loc['Early', 'Up Lots']),
            is_good(pivot.loc['Middle', 'Down Lots']),
            is_good(pivot.loc['Middle', 'Up Lots']),
            is_good(pivot.loc['Late', 'Down Lots']),
            is_good(pivot.loc['Late', 'Up Lots'])
        ])
        long_relief_score = sum([
            is_good(pivot.loc['Early', 'Up Little']),
            is_good(pivot.loc['Early', 'Up Lots']),
            is_good(pivot.loc['Early', 'Down Lots'])
        ])
        if is_best(pivot.loc['Early', 'Tight']) or is_best(pivot.loc['Early', 'Up Little']):
            starter_score += 2
        if is_best(pivot.loc['Middle', 'Tight']) or is_best(pivot.loc['Middle', 'Up Little']):
            bridge_score += 2
        if is_best(pivot.loc['Late', 'Tight']):
            high_leverage_score += 3
        if is_best(pivot.loc['Early', 'Up Lots']) or is_best(pivot.loc['Early', 'Down Lots']):
            inning_eater_score += 2
        if is_best(pivot.loc['Early', 'Up Little']):
            long_relief_score += 2
        roles_scored = [
            ('Starter', starter_score),
            ('Bridge', bridge_score),
            ('High Leverage', high_leverage_score),
            ('Inning Eater', inning_eater_score),
            ('Long Relief', long_relief_score)
        ]
        roles_sorted = sorted(roles_scored, key=lambda x: x[1], reverse=True)
        return [role for role, score in roles_sorted[:3]]
    st.markdown("#### 💡 Recommended Roles")
    roles = get_role_recommendations(pivot)
    if roles:
        st.info(f"""
        **Primary Role Recommendations:** 1️⃣ {roles[0]} 2️⃣ {roles[1]} 3️⃣ {roles[2]}
        
        Based on standardized performance across all situations
        """)
    else:
        st.info("No role recommendations available due to insufficient data.")
    
