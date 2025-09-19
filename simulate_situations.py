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
    """Simulate pitcher performance across evenly sampled game contexts."""

    # Restrict profiles to the selected pitcher only to avoid cross-contamination
    profiles = {k: v for k, v in profiles.items() if k[0] == selected_pitcher}
    if not profiles:
        st.error(f"No pitch data available for {selected_pitcher}.")
        return pd.DataFrame()

    try:
        # Load game states
        gs = pl.read_parquet('game_states_with_roles.parquet')
        valid_states = (
            gs
            .filter(pl.col('game_state').is_not_null())
            .to_pandas()
        )
        if valid_states.empty:
            st.error("No game states available after filtering nulls.")
            return pd.DataFrame()

        parts = valid_states['game_state'].str.split('_')
        valid_states['inning'] = parts.str[0].astype(float).astype(int)
        valid_states['half'] = parts.str[1]
        valid_states['outs'] = parts.str[2].astype(float).astype(int)
        valid_states['runners'] = parts.str[3]
        valid_states['score_diff'] = parts.str[4].astype(float).astype(int)

        def phase_from_inning(inning: int) -> str:
            if inning <= 3:
                return 'Early'
            elif inning <= 6:
                return 'Middle'
            return 'Late'

        def score_context(diff: int) -> str:
            if diff <= -4:
                return 'Down Lots'
            if diff <= -1:
                return 'Down Little'
            if diff == 0:
                return 'Tight'
            if diff <= 3:
                return 'Up Little'
            return 'Up Lots'

        valid_states['phase'] = valid_states['inning'].apply(phase_from_inning)
        valid_states['score_ctx'] = valid_states['score_diff'].apply(score_context)

        situations = []
        for phase in PHASES:
            for score in SCORE_CONTEXTS:
                subset = valid_states[(valid_states['phase'] == phase) & (valid_states['score_ctx'] == score)]
                if subset.empty:
                    continue
                sample_n = 25
                sampled = subset.sample(n=sample_n, replace=len(subset) < sample_n, random_state=42)
                for _, row in sampled.iterrows():
                    situations.append({
                        'inning': float(row['inning']),
                        'half': row['half'],
                        'outs': int(row['outs']),
                        'runners': row['runners'],
                        'score_diff': float(row['score_diff']),
                        'phase': phase,
                        'score_ctx': score,
                        'game_state': row['game_state']
                    })
    except Exception as e:
        st.error(f"Error loading game states: {str(e)}")
        return pd.DataFrame()
    
    # Run simulations
    results = []
    progress_bar = st.progress(0)
    total_sims = len(situations)

    # Determine pitcher hand if available
    try:
        first_profile = next(iter(profiles.values()))
        pitcher_hand = first_profile.get('pitcher_hand', 'Right')
    except Exception:
        pitcher_hand = 'Right'

    for i, situation in enumerate(situations):
        try:
            # Calculate leverage index for the situation
            leverage = leverage_from_state(
                inning=situation['inning'],
                half=situation['half'],
                outs=situation['outs'],
                on1=int(situation['runners'][0]),
                on2=int(situation['runners'][1]),
                on3=int(situation['runners'][2]),
                score_diff=situation['score_diff']
            )

            phase = situation['phase']
            score_ctx = situation['score_ctx']
            phase_mult = PHASE_WORKLOAD_MULTIPLIER.get(phase, 1.0)
            score_mult = SCORE_WORKLOAD_MULTIPLIER.get(score_ctx, 1.0)
            base_workload = BASE_PITCH_COUNT * phase_mult * score_mult
            seed_material = f"{selected_pitcher}|{situation['game_state']}|{i}"
            seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
            workload_jitter = (seed % 3) - 1  # -1, 0, or 1 pitches of noise
            n_pitches = int(round(base_workload)) + workload_jitter
            n_pitches = max(8, min(28, n_pitches))

            balls, strikes = SCORE_COUNT_TENDENCIES.get(score_ctx, (1, 1))
            if phase == 'Late':
                strikes = min(strikes + 1, 2)
            exp_delta = simulate_expected_delta(
                mdl=mdl,
                profiles={k: v for k, v in profiles.items() if k[0] == selected_pitcher},
                df_all=None,  # Not needed for pure stuff-based simulation
                pitcher_name=selected_pitcher,
                n_pitches=n_pitches,
                batter_hand='All',  # Test against both L/R
                half=situation['half'],
                inning=float(situation['inning']),
                leverage_index=float(leverage),
                balls=balls,
                strikes=strikes,
                paths=500,  # 500 Monte Carlo paths
                pitcher_hand=pitcher_hand,
                rng=seed,
            )
            # Calculate standardized score
            standardized_score = standardize_score(exp_delta)
            # Apply penalty for do-not-pitch pitchers
            if selected_pitcher in DO_NOT_PITCH:
                standardized_score -= 15  # Subtract 15 points from their scores
            results.append({
                'Pitcher': selected_pitcher,
                'Game_State': situation['game_state'],
                'Situation': f"{situation['phase']} {situation['score_ctx']}",
                'Phase': situation['phase'],
                'Score_Context': situation['score_ctx'],
                'Leverage': leverage,
                'Expected_Impact': exp_delta,
                'Stopper+': standardized_score,
                'Pitcher_Hand': pitcher_hand,
                'Pitch_Workload': n_pitches,
                'Count_Balls': balls,
                'Count_Strikes': strikes,
            })
        except Exception as e:
            continue
        progress_bar.progress((i + 1) / total_sims)
    return pd.DataFrame(results)

def analyze_simulation_results(results_df: pd.DataFrame):
    import pandas as pd
    import numpy as np
    import streamlit as st
    import sklearn

    if results_df.empty:
        st.error("No results to analyze")
        return

    pitcher_name = results_df['Pitcher'].iloc[0] if not results_df.empty else "Unknown"

    phase_analysis = results_df.groupby(['Phase', 'Score_Context']).agg(
        Impact=('Expected_Impact', 'mean'),
        Samples=('Expected_Impact', 'count'),
        Avg_Leverage=('Leverage', 'mean'),
        Avg_Workload=('Pitch_Workload', 'mean')
    ).round(4)

    matrix_df = phase_analysis.reset_index().rename(columns={'Score_Context': 'Score'})

    PHASES = ['Early', 'Middle', 'Late']
    SCORE_CONTEXTS = ['Down Lots', 'Down Little', 'Tight', 'Up Little', 'Up Lots']

    pivot = matrix_df.pivot(index='Phase', columns='Score', values='Impact').reindex(index=PHASES, columns=SCORE_CONTEXTS)

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

    styled_pivot = pivot.style.apply(apply_self_gradient, axis=None).format(precision=3)
    st.dataframe(styled_pivot, use_container_width=True)

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
    
