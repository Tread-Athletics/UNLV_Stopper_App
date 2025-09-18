# execution_plus_cli2_plotly.py
# Identical to execution_plus_cli2.py, but generates interactive Plotly plots instead of PNGs.
# Designed for future bullpen planner sidebar integration.

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import itertools
import xgboost as xgb
import plotly.graph_objs as go
import plotly.io as pio

try:
    import sqlalchemy  # type: ignore
    from sqlalchemy import create_engine, text  # type: ignore
except Exception:
    sqlalchemy = None
    create_engine = text = None

try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None

try:
    import joblib  # type: ignore
except Exception:
    joblib = None

def get_pitch_type_figure(
    pitcher, pitch_type, batter_side='RHH',
    query_since='2025-01-01',
    data_path='college_unlv_with_base_state.csv',
    bins=24, cmin=-300, cmax=200
):
    import pandas as pd, numpy as np, plotly.express as px, plotly.graph_objs as go
    from pathlib import Path
    import itertools
    # --- Data loading (copied from CLI) ---
    if data_path and Path(data_path).exists():
        p = Path(data_path)
        if p.suffix.lower() in {'.parquet', '.pq'}:
            df_raw = pd.read_parquet(p)
        else:
            df_raw = pd.read_csv(p)
    else:
        return None
    # --- Filter to pitcher ---
    df = df_raw[(df_raw['Pitcher'] == pitcher) & (df_raw['TaggedPitchType'].notnull()) & (df_raw['TaggedPitchType'] != '')].copy()
    df['TaggedPitchType'] = df['TaggedPitchType'].replace({
        'FourSeamFastball': 'Fastball',
        'TwoSeamFastball': 'Sinker'
    })
    # --- Required columns ---
    REQUIRED = {
        'TaggedPitchType': 'str', 'PitcherThrows': 'str', 'BatterSide': 'str', 'Pitcher': 'str',
        'TaggedHitType': 'str', 'PlayResult': 'str', 'Top.Bottom': 'str', 'Level': 'str', 'GameID': 'str',
        'Batter': 'str', 'AwayTeam': 'str', 'HomeTeam': 'str', 'PitchCall': 'str', 'SpinAxis': 'float',
        'HorzBreak': 'float', 'InducedVertBreak': 'float', 'RelSpeed': 'float', 'RelSide': 'float', 'RelHeight': 'float',
        'Extension': 'float', 'SpinRate': 'float', 'PlateLocHeight': 'float', 'PlateLocSide': 'float',
        'VertApprAngle': 'float', 'HorzApprAngle': 'float', 'ax0': 'float', 'az0': 'float',
        'Balls': 'int', 'Strikes': 'int', 'Outs': 'int', 'Inning': 'int', 'PAofInning': 'int', 'PitchNo': 'int',
        'PitcherId': 'str', 'Date': 'datetime'
    }
    for col, kind in REQUIRED.items():
        if col not in df.columns:
            df[col] = pd.NA
    for col, kind in REQUIRED.items():
        if kind == 'float':
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        elif kind == 'int':
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        elif kind == 'datetime':
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=False)
        elif kind == 'str':
            df[col] = df[col].astype('string').str.strip()
    # --- Normalize categories ---
    throws_map = {'R':'Right','RIGHT':'Right','Right':'Right','L':'Left','LEFT':'Left','Left':'Left'}
    batter_map = {'R':'Right','RIGHT':'Right','Right':'Right','L':'Left','LEFT':'Left','Left':'Left','S':'Switch','SWITCH':'Switch','Switch':'Switch'}
    df['PitcherThrows'] = df['PitcherThrows'].astype('string').str.strip().map(throws_map).fillna(df['PitcherThrows'].astype('string').str.strip())
    df['BatterSide'] = df['BatterSide'].astype('string').str.strip().map(batter_map).fillna(df['BatterSide'].astype('string').str.strip())
    for c in ['TaggedPitchType','Pitcher','GameID','Batter','AwayTeam','HomeTeam','PitchCall','SpinAxis']:
        if c in df.columns:
            df[c] = df[c].astype('string').str.strip()
    # --- Clip numeric ranges ---
    df['PlateLocSide'] = pd.to_numeric(df['PlateLocSide'], errors='coerce').clip(-3.0, 3.0)
    df['PlateLocHeight'] = pd.to_numeric(df['PlateLocHeight'], errors='coerce').clip(0.0, 7.5)
    df['RelSpeed'] = pd.to_numeric(df['RelSpeed'], errors='coerce').clip(30.0, 120.0)
    for c in ['HorzBreak','InducedVertBreak','RelSide','RelHeight','Extension','VertApprAngle','HorzApprAngle','SpinRate','ax0','az0']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Balls'] = pd.to_numeric(df['Balls'], errors='coerce').fillna(0).clip(0,3).astype(int)
    df['Strikes'] = pd.to_numeric(df['Strikes'], errors='coerce').fillna(0).clip(0,2).astype(int)
    # --- Rename columns for pipeline ---
    df = df.rename(columns={
        'Pitcher': 'player_name',
        'PitcherId': 'pitcher',
        'TaggedPitchType': 'pitch_name',  # also use for 'pitch_type'
        'HorzBreak': 'pfx_x_pv_adj',
        'InducedVertBreak': 'pfx_z',
        'RelSpeed': 'release_speed',
        'RelSide': 'release_pos_x_pv_adj',
        'RelHeight': 'release_pos_z',
        'Extension': 'release_extension',
        'VertApprAngle': 'VAA_pred',
        'HorzApprAngle': 'adj_HAA_pred',
    })
    df['pitch_type'] = df['pitch_name']
    # --- Map pitch class ---
    def map_pitch_class(name):
        if pd.isna(name): return None
        s = str(name).strip()
        direct = {
            'FF':'Fastball','FA':'Fastball','Fastball':'Fastball','FourSeamFastball':'Fastball','4-Seam Fastball':'Fastball',
            '4-Seam':'Fastball','Four-Seam':'Fastball','4 Seam Fastball':'Fastball','Sinker':'Fastball','SI':'Fastball',
            'TwoSeamFastball':'Fastball','2-Seam Fastball':'Fastball','Two-Seam Fastball':'Fastball','Two-Seam':'Fastball','TwoSeam':'Fastball','Cutter':'Fastball','FC':'Fastball',
            'Slider':'Breaking Ball','SL':'Breaking Ball','Curveball':'Breaking Ball','CU':'Breaking Ball','KnuckleCurve':'Breaking Ball','Knuckle Curve':'Breaking Ball','Sweeper':'Breaking Ball',
            'ChangeUp':'Offspeed','Changeup':'Offspeed','Change-Up':'Offspeed','CH':'Offspeed','Splitter':'Offspeed','FS':'Offspeed','Split-Finger':'Offspeed','SplitFinger':'Offspeed'
        }
        if s in direct: return direct[s]
        sl = s.lower()
        if any(k in sl for k in ['fast','ff','four seam','4-seam','4 seam','sinker','two-seam','2-seam','cutter','cut']): return 'Fastball'
        if any(k in sl for k in ['slider','sl','curve','cu','knuckle curve','sweeper','slurve','knuck']): return 'Breaking Ball'
        if any(k in sl for k in ['change','ch','split','splitter','split-finger','fork']): return 'Offspeed'
        return None
    df['pitch_class'] = df['pitch_type'].apply(map_pitch_class)
    # --- Group and average ---
    averages = (
        df.groupby(['player_name', 'pitcher', 'pitch_name', 'pitch_type', 'pitch_class'], dropna=False)
          .agg({
              'pfx_x_pv_adj': 'mean',
              'pfx_z': 'mean',
              'release_speed': 'mean',
              'VAA_pred': 'mean',
              'adj_HAA_pred': 'mean',
              'release_pos_x_pv_adj': 'mean',
              'release_pos_z': 'mean',
              'release_extension': 'mean'
          })
          .reset_index()
    )
    # --- Plate grid expansion ---
    plate_grid = pd.DataFrame(list(itertools.product(
        np.round(np.arange(-2, 2.01, 0.1), 2),
        np.round(np.arange(0, 4.51, 0.1), 2)
    )), columns=['plate_x_pv_adj', 'plate_z'])
    # --- Get pitcher throwing hand and platoon values ---
    pitcher_throwing_hand = df['PitcherThrows'].unique()[0] if 'PitcherThrows' in df.columns else 'Right'
    if pitcher_throwing_hand == "Right":
        platoon_values = ["RHP-RHH", "RHP-LHH"]
    elif pitcher_throwing_hand == "Left":
        platoon_values = ["LHP-RHH", "LHP-LHH"]
    else:
        platoon_values = ["Unknown"]
    # --- Expand: averages x plate_grid x platoon_type ---
    expanded_averages = (
        averages
        .merge(plate_grid, how='cross')
        .assign(key=1)
        .merge(pd.DataFrame({'platoon_type': platoon_values, 'key': 1}), on='key')
        .drop('key', axis=1)
        .assign(game_year='69', count_class='Even')
    )
    # --- Fill required-but-ignored fields ---
    for col, val in [
        ('Top.Bottom', 'Undefined'), ('Level', 'College'), ('GameID', ''), ('Batter', ''),
        ('AwayTeam', ''), ('HomeTeam', ''), ('PitchCall', ''), ('SpinAxis', np.nan),
        ('Outs', 0), ('Inning', 1), ('PAofInning', 1), ('PitchNo', 1), ('PitcherId', ''),
        ('Date', pd.NaT), ('TaggedHitType', 'Undefined'), ('PlayResult', 'Undefined'),
        ('OutsOnPlay', 0.0), ('RunsScored', 0.0), ('KorBB', 'Undefined'),
        ('Balls', 1), ('Strikes', 1)
    ]:
        expanded_averages[col] = val
    expanded_averages['Balls'] = expanded_averages['Balls'].astype(int)
    expanded_averages['Strikes'] = expanded_averages['Strikes'].astype(int)
    # --- Add RP buckets ---
    def add_rp_buckets(df, df_type="TrackMan"):
        if df_type == "TrackMan":
            side_2_28 = 0.56
            side_15_87 = 1.18
            side_84_13 = 2.57
            side_97_72 = 3.39
            height_2_28 = 4.59
            height_15_87 = 5.45
            height_84_13 = 6.3
            height_97_72 = 6.73
            median_extension = 6.188
            df = df.copy()
            df['side_bucket'] = pd.cut(
                df['release_pos_x_pv_adj'],
                bins=[-float('inf'), side_2_28, side_15_87, side_84_13, side_97_72, float('inf')],
                labels=[-2, -1, 0, 1, 2]
            ).astype(float)
            df['height_bucket'] = pd.cut(
                df['release_pos_z'],
                bins=[-float('inf'), height_2_28, height_15_87, height_84_13, height_97_72, float('inf')],
                labels=[-2, -1, 0, 1, 2]
            ).astype(float)
            df['above_median_ext'] = (df['release_extension'] > median_extension).astype(int)
            df['adj_HAA_pred'] = df['adj_HAA_pred']
        else:
            raise ValueError("Only TrackMan type supported.")
        return df
    expanded_buckets = add_rp_buckets(expanded_averages, df_type="TrackMan")
    # --- Model loading ---
    import xgboost
    fb_model_hint = '/Users/evanborberg/Stopper-/models/xgb-fastball-execution-v2-mae-HAAupdate'
    bb_model_hint = '/Users/evanborberg/Stopper-/models/xgb-breakingball-execution-v2-mae-HAAupdate'
    os_model_hint = '/Users/evanborberg/Stopper-/models/xgb-offspeed-execution-v2-mae-HAAUpdate'
    def load_model(path):
        booster = xgboost.Booster()
        booster.load_model(path)
        return booster
    fb_model = load_model(fb_model_hint)
    bb_model = load_model(bb_model_hint)
    os_model = load_model(os_model_hint)
    # --- Execution+ calculation ---
    def calculate_execution_plus(expanded, fb_model, bb_model, os_model, fb_frac=0.1):
        columns_to_drop = [
            "pitch_class", "pfx_x_pv_adj", "pfx_z", "release_speed", "VAA_pred", "HAA_pred",
            "release_pos_x_pv_adj", "release_pos_z", "release_extension"
        ]
        execution_df = expanded.dropna(subset=[col for col in columns_to_drop if col in expanded.columns])
        fb_mask = execution_df['pitch_class'] == "Fastball"
        p_avgs = (
            execution_df[fb_mask]
            .groupby(['pitcher', 'pitch_name', 'game_year'], dropna=False)
            .agg(
                n=('release_speed', 'size'),
                avg_velo=('release_speed', 'mean'),
                avg_vmov=('pfx_z', 'mean'),
                avg_hmov_pv_adj=('pfx_x_pv_adj', 'mean')
            )
            .reset_index()
        )
        p_avgs = (
            p_avgs.sort_values('n', ascending=False)
            .groupby(['pitcher', 'game_year'], as_index=False)
            .head(1)[['pitcher', 'game_year', 'avg_velo', 'avg_vmov', 'avg_hmov_pv_adj']]
        )
        mlb4s_ratio = execution_df.merge(p_avgs, on=['pitcher', 'game_year'], how='left')
        mlb4s_ratio['velo_ratio'] = np.where(
            mlb4s_ratio['release_speed'] / mlb4s_ratio['avg_velo'] > 1,
            100,
            100 * mlb4s_ratio['release_speed'] / mlb4s_ratio['avg_velo']
        )
        def cutter_distance(velo_r_avg, ct_vb, ct_hb):
            d1 = (velo_r_avg - 98.016) ** 2 + (ct_vb - 13.009) ** 2 + (ct_hb - 9.184) ** 2
            d2 = (velo_r_avg - 87.208) ** 2 + (ct_vb + 2.284) ** 2 + (ct_hb + 6.792) ** 2
            return d1 < d2
        mlb4s_updated = []
        for _, g in mlb4s_ratio.groupby(['pitcher', 'player_name', 'pitch_type', 'game_year']):
            g = g.copy()
            g['velo_r_avg'] = g['velo_ratio'].mean()
            g['ct_vb'] = g['pfx_z'].mean()
            g['ct_hb'] = g['pfx_x_pv_adj'].mean()
            if 'pitch_class' in g.columns and 'pitch_type' in g.columns:
                g['pitch_class'] = np.where(
                    g['pitch_type'] != "FC",
                    g['pitch_class'],
                    np.where(
                        cutter_distance(g['velo_r_avg'].iloc[0], g['ct_vb'].iloc[0], g['ct_hb'].iloc[0]),
                        "Fastball",
                        "Breaking Ball"
                    )
                )
            mlb4s_updated.append(g)
        mlb4s_updated = pd.concat(mlb4s_updated, ignore_index=True)
        data_colnames = expanded.columns.tolist()
        raw_data_cutter = mlb4s_updated[data_colnames + ['avg_velo', 'avg_vmov', 'avg_hmov_pv_adj']]
        cleaned_mlb = raw_data_cutter.copy()
        cleaned_mlb['velo_diff'] = np.where(
            cleaned_mlb['pitch_class'].isin(["Breaking Ball", "Offspeed"]),
            cleaned_mlb['release_speed'] - cleaned_mlb['avg_velo'],
            0
        )
        cleaned_mlb['hmov_diff'] = np.where(
            cleaned_mlb['pitch_class'].isin(["Breaking Ball", "Offspeed"]),
            cleaned_mlb['pfx_x_pv_adj'] - cleaned_mlb['avg_hmov_pv_adj'],
            0
        )
        cleaned_mlb['vmov_diff'] = np.where(
            cleaned_mlb['pitch_class'].isin(["Breaking Ball", "Offspeed"]),
            cleaned_mlb['pfx_z'] - cleaned_mlb['avg_vmov'],
            0
        )
        cleaned_mlb['pfx_x_pv_adj_trunc'] = np.trunc(cleaned_mlb['pfx_x_pv_adj'] * 10) / 10
        cleaned_mlb['pfx_z_trunc'] = np.trunc(cleaned_mlb['pfx_z'] * 10) / 10
        count_class_map = {"Even": 0, "Pitcher": 1, "Hitter": 2}
        platoon_type_map = {"RHP-RHH": 1, "RHP-LHH": 2, "LHP-RHP": 3, "LHP-LHH": 4}
        cleaned_mlb['count_class_num'] = cleaned_mlb['count_class'].map(count_class_map).fillna(5)
        cleaned_mlb['platoon_type_num'] = cleaned_mlb['platoon_type'].map(platoon_type_map).fillna(5)
        FB_FEATURES = [
            'release_speed','pfx_x_pv_adj_trunc','pfx_z_trunc','side_bucket','height_bucket',
            'plate_x_pv_adj','plate_z','VAA_pred','adj_HAA_pred','platoon_type_num','count_class_num'
        ]
        BB_OS_FEATURES = [
            'release_speed','pfx_x_pv_adj_trunc','pfx_z_trunc','velo_diff','vmov_diff','hmov_diff',
            'side_bucket','height_bucket','plate_x_pv_adj','plate_z','VAA_pred','adj_HAA_pred',
            'platoon_type_num','count_class_num'
        ]
        def safe_select(df, cols):
            return df[[c for c in cols if c in df.columns]].astype(float)
        cleaned_fb = cleaned_mlb[cleaned_mlb['pitch_class'] == "Fastball"].dropna(subset=FB_FEATURES)
        cleaned_bb = cleaned_mlb[cleaned_mlb['pitch_class'] == "Breaking Ball"].dropna(subset=BB_OS_FEATURES)
        cleaned_os = cleaned_mlb[cleaned_mlb['pitch_class'] == "Offspeed"].dropna(subset=BB_OS_FEATURES)
        if cleaned_fb.shape[0] > 0:
            xgb_fb_exec_pred = fb_model.predict(xgboost.DMatrix(safe_select(cleaned_fb, FB_FEATURES), feature_names=FB_FEATURES))
            exec_plus_fb = cleaned_fb.drop(columns=[c for c in ['xRV'] if c in cleaned_fb.columns]).copy()
            exec_plus_fb['xRV'] = xgb_fb_exec_pred
        else:
            exec_plus_fb = cleaned_fb.copy()
        if cleaned_bb.shape[0] > 0:
            xgb_bb_exec_pred = bb_model.predict(xgboost.DMatrix(safe_select(cleaned_bb, BB_OS_FEATURES), feature_names=BB_OS_FEATURES))
            exec_plus_bb = cleaned_bb.drop(columns=[c for c in ['xRV'] if c in cleaned_bb.columns]).copy()
            exec_plus_bb['xRV'] = xgb_bb_exec_pred
        else:
            exec_plus_bb = cleaned_bb.copy()
        if cleaned_os.shape[0] > 0:
            xgb_os_exec_pred = os_model.predict(xgboost.DMatrix(safe_select(cleaned_os, BB_OS_FEATURES), feature_names=BB_OS_FEATURES))
            exec_plus_os = cleaned_os.drop(columns=[c for c in ['xRV'] if c in cleaned_os.columns]).copy()
            exec_plus_os['xRV'] = xgb_os_exec_pred
        else:
            exec_plus_os = cleaned_os.copy()
        exec_plus_data = pd.concat([exec_plus_fb, exec_plus_bb, exec_plus_os], ignore_index=True)
        exec_plus_data['xRV.100'] = 100 * exec_plus_data['xRV']
        grand_mean = -0.1815963
        grand_sd = 0.7983234
        exec_plus_data['execution_plus'] = -50 * ((exec_plus_data['xRV.100'] - grand_mean) / grand_sd) + 100
        keep_cols = [c for c in expanded.columns.tolist() + ['xRV', 'xRV.100', 'execution_plus'] if c in exec_plus_data.columns]
        return exec_plus_data[keep_cols]
    exec_plus_df = calculate_execution_plus(expanded_buckets, fb_model, bb_model, os_model)
    # (Plotting will be added next)

def main(argv=None):
    import sys
    parser = argparse.ArgumentParser(description="Run Execution+ grid predictions for a pitcher name (Plotly version).")
    parser.add_argument('--pitcher', type=str, default='Lane, Carson', help='Pitcher name, e.g., "Arnold, Jamie"')
    parser.add_argument('--query-since', type=str, default=os.getenv('QUERY_SINCE', '2025-01-01'), help='Minimum date (YYYY-MM-DD)')
    parser.add_argument('--data-path', type=str, default='college_unlv_with_base_state.csv', help='Path to CSV/Parquet with TrackMan-like columns (College_TM_Data style)')
    parser.add_argument('--save-dir', type=str, default='outputs', help='Directory to write HTML outputs (default: outputs)')
    parser.add_argument('--models-root', type=str, default=None, help='Root folder to search for model files if hints fail')
    parser.add_argument('--bins', type=int, default=20, help='Hexbin grid size for plots (default: 20)')
    parser.add_argument('--cmin', type=float, default=0.0, help='Color scale minimum for Execution+ (default: 0.0)')
    parser.add_argument('--cmax', type=float, default=200.0, help='Color scale maximum for Execution+ (default: 200.0)')
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    pitcher = args.pitcher
    data_path = args.data_path
    query_since = args.query_since
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ... existing data/model loading and processing logic from execution_plus_cli2.py ...
    # (Omitted here for brevity, but should be copied in full from the original script)

    # --- Replace matplotlib plotting with Plotly ---
    # For each pitch type and platoon, generate a Plotly heatmap
    # Save as HTML (or JSON for embedding)
    # Example for one plot:
    # fig = go.Figure(data=go.Heatmap(...))
    # pio.write_html(fig, str(save_dir / f"execution_plus_{pitcher.replace(', ', '_')}_plotly.html"))
    # (Repeat for each pitch type/platoon as needed)

    # --- Placeholder for actual Plotly plotting logic ---
    # TODO: Implement Plotly heatmap generation using processed data
    # TODO: Add target/zone overlays for bullpen planner integration

if __name__ == "__main__":
    main()
