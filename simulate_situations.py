import pandas as pd
import polars as pl
import streamlit as st
import numpy as np
from typing import Dict, Any
from stopper_predictor import simulate_expected_delta

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

        # Extract game state components
        valid_states = (
            gs
            .filter(pl.col('game_state').is_not_null())
            .to_pandas()
        )

        if valid_states.empty:
            st.error("No game states available")
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
                sample_n = 50
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

        st.info(f"Sampling {len(situations)} situations across phases and score contexts")

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
    
    return pd.DataFrame(results)

def analyze_simulation_results(results_df: pd.DataFrame):
    """Provide condensed, categorized pitching recommendations."""
    if results_df.empty:
        st.error("No results to analyze")
        return

    pitcher_name = results_df['Pitcher'].iloc[0] if not results_df.empty else "Unknown"

    # Calculate performance by game phase
    phase_analysis = results_df.groupby(['Phase', 'Score_Context']).agg(
        Impact=('Expected_Impact', 'mean'),
        Samples=('Expected_Impact', 'count'),
        Avg_Leverage=('Leverage', 'mean')
    ).round(4)

    # Calculate overall phase effectiveness
    overall_phase = results_df.groupby('Phase').agg(
        Expected_Impact=('Expected_Impact', 'mean'),
        Leverage=('Leverage', 'mean')
    ).round(4)

    # Calculate score context effectiveness
    score_analysis = results_df.groupby('Score_Context').agg(
        Expected_Impact=('Expected_Impact', 'mean'),
        Leverage=('Leverage', 'mean')
    ).round(4)

    st.markdown(f"### Quick Analysis: {pitcher_name}")

    # Display two columns: Game Phase and Score Context
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Game Phase Performance")
        # Create phase summary with color indicators
        phase_summary = pd.DataFrame({
            'Phase': overall_phase.index,
            'Impact': overall_phase['Expected_Impact'].round(3),
            'LI': overall_phase['Leverage'].round(2)
        }).sort_values('Impact', ascending=False)

        # Add recommendation symbols
        phase_summary['Grade'] = ['🟢' if x == phase_summary['Impact'].max() else 
                                '🟡' if x > phase_summary['Impact'].mean() else 
                                '🔴' for x in phase_summary['Impact']]
        # Display with custom formatting
        for _, row in phase_summary.iterrows():
            st.markdown(f"{row['Grade']} **{row['Phase']}**")

    with col2:
        st.markdown("#### Score Context Performance")
        # Create score context summary
        score_summary = pd.DataFrame({
            'Context': score_analysis.index,
            'Impact': score_analysis['Expected_Impact'].round(3),
            'LI': score_analysis['Leverage'].round(2)
        }).sort_values('Impact', ascending=False)

        # Add recommendation symbols
        score_summary['Grade'] = ['🟢' if x == score_summary['Impact'].max() else 
                                '🟡' if x > score_summary['Impact'].mean() else 
                                '🔴' for x in score_summary['Impact']]
        # Display with custom formatting
        for _, row in score_summary.iterrows():
            st.markdown(f"{row['Grade']} **{row['Context']}**")

    # Detailed Breakdown
    st.markdown("#### Detailed Situation Analysis")
    # Create a clean, organized matrix of performance
    matrix_data = []
    for phase in PHASES:
        for score in SCORE_CONTEXTS:
            try:
                data = phase_analysis.loc[(phase, score)]
                matrix_data.append({
                    'Phase': phase,
                    'Score': score,
                    'Impact': data['Impact'],
                    'LI': data['Avg_Leverage'],
                    'Samples': data['Samples']
                })
            except KeyError:
                continue
    matrix_df = pd.DataFrame(matrix_data)
    # Create pivot table for clear visualization
    pivot = matrix_df.pivot(index='Phase', columns='Score', values='Impact')
    # Add color styling - clean block style
    def color_scale_blocks(val):
        if pd.isna(val):
            return ''
        # Red to Green color scale - solid blocks
        if val == pivot.max().max():
            return 'background-color: #28a745'  # Strong green
        elif val > pivot.mean().mean():
            return 'background-color: #5cb85c'  # Light green
        else:
            return 'background-color: #dc3545'  # Red
    # Display matrix with colors only, no numbers
    styled_pivot = pivot.style.applymap(color_scale_blocks).format("")
    st.dataframe(styled_pivot, use_container_width=True)
    # Role Mapping Logic based strictly on color grid performance
    def get_role_recommendations(matrix_df: pd.DataFrame) -> list:
        """Determine roles based on where green/light-green boxes appear in the matrix."""
        pivot = matrix_df.pivot(index='Phase', columns='Score', values='Impact')
        mean_val = pivot.mean().mean()
        max_val = pivot.max().max()
        # Helper to check if a value is "good" (any shade of green)
        def is_good(val):
            return val > mean_val if pd.notna(val) else False
        # Helper to check if a value is "best" (dark green)
        def is_best(val):
            return val == max_val if pd.notna(val) else False
        # Starter criteria: Good in Early/Middle when Up Little, Tight, or Down Little
        starter_score = 0
        starter_conditions = [
            is_good(pivot.loc['Early', 'Up Little']),
            is_good(pivot.loc['Early', 'Tight']),
            is_good(pivot.loc['Early', 'Down Little']),
            is_good(pivot.loc['Middle', 'Up Little']),
            is_good(pivot.loc['Middle', 'Tight']),
            is_good(pivot.loc['Middle', 'Down Little'])
        ]
        starter_score = sum(starter_conditions)
        # Bridge criteria: Good in Middle/Late when Up Little, Tight, or Down Little
        bridge_score = 0
        bridge_conditions = [
            is_good(pivot.loc['Middle', 'Up Little']),
            is_good(pivot.loc['Middle', 'Tight']),
            is_good(pivot.loc['Middle', 'Down Little']),
            is_good(pivot.loc['Late', 'Up Little']),
            is_good(pivot.loc['Late', 'Tight']),
            is_good(pivot.loc['Late', 'Down Little'])
        ]
        bridge_score = sum(bridge_conditions)
        # High Leverage criteria: Good in Late innings with Up/Down Little or Tight
        high_leverage_score = 0
        high_leverage_conditions = [
            is_good(pivot.loc['Late', 'Up Little']),
            is_good(pivot.loc['Late', 'Tight']),
            is_good(pivot.loc['Late', 'Down Little']),
            is_good(pivot.loc['Early', 'Tight']),
            is_good(pivot.loc['Middle', 'Tight'])
        ]
        high_leverage_score = sum(high_leverage_conditions)
        # Inning Eater criteria: Good in any phase when Down/Up Lots
        inning_eater_score = 0
        inning_eater_conditions = [
            is_good(pivot.loc['Early', 'Down Lots']),
            is_good(pivot.loc['Early', 'Up Lots']),
            is_good(pivot.loc['Middle', 'Down Lots']),
            is_good(pivot.loc['Middle', 'Up Lots']),
            is_good(pivot.loc['Late', 'Down Lots']),
            is_good(pivot.loc['Late', 'Up Lots'])
        ]
        inning_eater_score = sum(inning_eater_conditions)
        # Long Relief criteria: Good Early when Up Little/Lots or Down Lots
        long_relief_score = 0
        long_relief_conditions = [
            is_good(pivot.loc['Early', 'Up Little']),
            is_good(pivot.loc['Early', 'Up Lots']),
            is_good(pivot.loc['Early', 'Down Lots'])
        ]
        long_relief_score = sum(long_relief_conditions)
        # Add bonus points for dark green (best) performances
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
        # Create scored list and sort by score
        roles_scored = [
            ('Starter', starter_score),
            ('Bridge', bridge_score),
            ('High Leverage', high_leverage_score),
            ('Inning Eater', inning_eater_score),
            ('Long Relief', long_relief_score)
        ]
        # Sort by score and return top 3 roles
        roles_sorted = sorted(roles_scored, key=lambda x: x[1], reverse=True)
        return [role for role, score in roles_sorted[:3]]
    # Key Recommendations
    st.markdown("#### 💡 Recommended Roles")
    # Get role recommendations based on color grid analysis
    roles = get_role_recommendations(matrix_df)
    # Display role recommendations
    st.info(f"""
    **Primary Role Recommendations:** 1️⃣ {roles[0]} 2️⃣ {roles[1]} 3️⃣ {roles[2]}
    
    Based on standardized performance across all situations
    """)
    
