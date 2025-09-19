import pandas as pd
import polars as pl
import streamlit as st
import numpy as np
from typing import Dict, Any
from stopper_predictor import simulate_expected_delta
import os
import sklearn

# List of pitchers that should not be recommended
DO_NOT_PITCH = ['Barna, Cal', 'Rogers, Dylan']

# Generic baseline for Stopper+ calculation
GLOBAL_BASELINE = {'mean': -0.1, 'sd': 0.1}

# Game context categories
PHASES = ['Early', 'Middle', 'Late']
SCORE_CONTEXTS = ['Down Lots', 'Down Little', 'Tight', 'Up Little', 'Up Lots']


def standardize_score(score: float, baseline: dict | None = None) -> float:
    """Standardize a score to a 100-based scale with SD=10."""
    base = baseline or GLOBAL_BASELINE
    sd = base.get('sd', 1.0) or 0.0
    mu = base.get('mean', 0.0)
    if sd == 0:
        return 100.0
    return 100.0 + 10.0 * ((score - mu) / sd)

def simulate_all_situations(selected_pitcher: str, profiles: Dict, mdl: Any, leverage_from_state: callable):
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
    predictions_debug = []
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
            # Simulate performance
            exp_delta = simulate_expected_delta(
                mdl=mdl,
                profiles={k: v for k, v in profiles.items() if k[0] == selected_pitcher},
                df_all=None,  # Not needed for pure stuff-based simulation
                pitcher_name=selected_pitcher,
                n_pitches=15,
                batter_hand='All',  # Test against both L/R
                half=situation['half'],
                inning=float(situation['inning']),
                leverage_index=float(leverage),
                balls=0,  # Starting fresh count
                strikes=0,
                paths=500,  # 500 Monte Carlo paths
                pitcher_hand=pitcher_hand
            )
            if i < 10:
                predictions_debug.append(exp_delta)
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
            })
        except Exception as e:
            continue
        progress_bar.progress((i + 1) / total_sims)
    st.info(f"First 10 predictions: {np.array(predictions_debug)}")
    return pd.DataFrame(results)

def analyze_simulation_results(results_df: pd.DataFrame):
    import pandas as pd
    import numpy as np
    import streamlit as st
    import os
    import sklearn
    st.info(f'scikit-learn version: {sklearn.__version__}')

    if results_df.empty:
        st.error("No results to analyze")
        return

    pitcher_name = results_df['Pitcher'].iloc[0] if not results_df.empty else "Unknown"

    phase_analysis = results_df.groupby(['Phase', 'Score_Context']).agg(
        Impact=('Expected_Impact', 'mean'),
        Samples=('Expected_Impact', 'count'),
        Avg_Leverage=('Leverage', 'mean')
    ).round(4)

    baseline_path = os.path.join(os.path.dirname(__file__), 'team_role_baselines.csv')
    baselines = pd.read_csv(baseline_path)
    sit_baselines = baselines[baselines['Category'] == 'Situation']
    baseline_lookup = {}
    for _, row in sit_baselines.iterrows():
        phase, score_ctx = row['Subcategory'].split('_', 1)
        baseline_lookup[(phase, score_ctx)] = (row['Mean'], row['Std'])

    PHASES = ['Early', 'Middle', 'Late']
    SCORE_CONTEXTS = ['Down Lots', 'Down Little', 'Tight', 'Up Little', 'Up Lots']
    matrix_data = []
    for phase in PHASES:
        for score in SCORE_CONTEXTS:
            try:
                data = phase_analysis.loc[(phase, score)]
                impact = data['Impact']
                mean, std = baseline_lookup.get((phase, score), (0, 1))
                z = (impact - mean) / std if std > 0 else 0
                matrix_data.append({
                    'Phase': phase,
                    'Score': score,
                    'Impact': impact,
                    'Z': z,
                    'LI': data['Avg_Leverage'],
                    'Samples': data['Samples']
                })
            except KeyError:
                matrix_data.append({
                    'Phase': phase,
                    'Score': score,
                    'Impact': np.nan,
                    'Z': 0,
                    'LI': np.nan,
                    'Samples': 0
                })
    matrix_df = pd.DataFrame(matrix_data)
    pivot = matrix_df.pivot(index='Phase', columns='Score', values='Impact').reindex(index=PHASES, columns=SCORE_CONTEXTS)
    z_pivot = matrix_df.pivot(index='Phase', columns='Score', values='Z').reindex(index=PHASES, columns=SCORE_CONTEXTS)

    def z_to_color(z):
        z = np.clip(z, -2.5, 2.5)
        if np.isnan(z):
            return 'background-color: #ffffff'
        if z < 0:
            r1, g1, b1 = (220, 53, 69)
            r2, g2, b2 = (255, 255, 255)
            f = (z + 2.5) / 2.5
        else:
            r1, g1, b1 = (255, 255, 255)
            r2, g2, b2 = (40, 167, 69)
            f = z / 2.5
        r = int(r1 + (r2 - r1) * f)
        g = int(g1 + (g2 - g1) * f)
        b = int(b1 + (b2 - b1) * f)
        return f'background-color: rgb({r},{g},{b})'

    def color_df(df, z_pivot):
        color_matrix = pd.DataFrame(index=df.index, columns=df.columns)
        for r in df.index:
            for c in df.columns:
                z = z_pivot.loc[r, c]
                color_matrix.loc[r, c] = z_to_color(z)
        return color_matrix

    styled_pivot = pivot.style.apply(color_df, z_pivot=z_pivot, axis=None).format(precision=3)
    st.dataframe(styled_pivot, use_container_width=True)

    # --- Rest of function unchanged (role recommendations, etc) ---
    def get_role_recommendations(matrix_df: pd.DataFrame) -> list:
        if not isinstance(matrix_df, pd.DataFrame) or matrix_df.empty:
            return []
        pivot = matrix_df.pivot(index='Phase', columns='Score', values='Impact').reindex(index=PHASES, columns=SCORE_CONTEXTS)
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
    roles = get_role_recommendations(matrix_df)
    if roles:
        st.info(f"""
        **Primary Role Recommendations:** 1️⃣ {roles[0]} 2️⃣ {roles[1]} 3️⃣ {roles[2]}
        
        Based on standardized performance across all situations
        """)
    else:
        st.info("No role recommendations available due to insufficient data.")
    
