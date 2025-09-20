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
    hardcoded_data = [
        ["Albright, Cody", "Early",  "Down Little", -2.6717087e-02, 2.5879542e-01, 107.32829, 25.879542, 25],  # 1
        ["Albright, Cody", "Early",  "Down Lots",    6.4005203e-02, 2.4080527e-01, 116.40052, 24.080527, 25],  # 2
        ["Albright, Cody", "Early",  "Tight",         6.2495569e-02, 4.0193184e-01, 116.24956, 40.193184, 25],  # 3
        ["Albright, Cody", "Early",  "Up Little",    -5.6029532e-02, 2.1465283e-01, 104.39705, 21.465283, 25],  # 4
        ["Albright, Cody", "Early",  "Up Lots",      -1.8540342e-02, 6.1742340e-02, 108.14597,  6.174234, 25],  # 5
        ["Albright, Cody", "Late",   "Down Little",  -1.2725038e-01, 1.6472398e-01,  97.27496, 16.472398, 25],  # 6
        ["Albright, Cody", "Late",   "Down Lots",    -3.2599944e-02, 5.7786830e-02, 106.74001,  5.778683, 25],  # 7
        ["Albright, Cody", "Late",   "Tight",        -7.3163934e-02, 1.4939128e-01, 102.68361, 14.939128, 25],  # 8
        ["Albright, Cody", "Late",   "Up Little",    -1.0608472e-02, 7.2891740e-02, 108.93915,  7.289174, 25],  # 9
        ["Albright, Cody", "Late",   "Up Lots",      -2.5088716e-02, 5.8033130e-02, 107.49113,  5.803313, 25],  # 10
        ["Albright, Cody", "Middle", "Down Little",  -5.3073702e-02, 2.9881899e-01, 104.69263, 29.881899, 25],  # 11
        ["Albright, Cody", "Middle", "Down Lots",     1.2469469e-02, 1.0690855e-01, 111.24695, 10.690855, 25],  # 12
        ["Albright, Cody", "Middle", "Tight",         3.1297650e-02, 2.4043233e-01, 113.12977, 24.043233, 25],  # 13
        ["Albright, Cody", "Middle", "Up Little",    -2.1862543e-02, 1.9882148e-01, 107.81375, 19.882148, 25],  # 14
        ["Albright, Cody", "Middle", "Up Lots",      -2.5561650e-02, 3.0167650e-02, 107.44383,  3.016765, 25],  # 15

        ["Barna, Cal",     "Early",  "Down Little",  -8.4529190e-02, 2.3707583e-01,  86.54708, 23.707583, 25],  # 16
        ["Barna, Cal",     "Early",  "Down Lots",    -2.2158362e-02, 2.2144631e-01,  92.78416, 22.144631, 25],  # 17
        ["Barna, Cal",     "Early",  "Tight",        -1.2678729e-02, 4.2708624e-01,  93.73213, 42.708624, 25],  # 18
        ["Barna, Cal",     "Early",  "Up Little",    -1.1189166e-01, 1.8787657e-01,  83.81083, 18.787657, 25],  # 19
        ["Barna, Cal",     "Early",  "Up Lots",      -7.2040808e-02, 5.6610450e-02,  87.79592,  5.661045, 25],  # 20
        ["Barna, Cal",     "Late",   "Down Little",  -1.5669527e-01, 1.4373849e-01,  79.33047, 14.373849, 25],  # 21
        ["Barna, Cal",     "Late",   "Down Lots",    -8.4732815e-02, 5.1454150e-02,  86.52672,  5.145415, 25],  # 22
        ["Barna, Cal",     "Late",   "Tight",        -1.0460968e-01, 1.3633001e-01,  84.53903, 13.633001, 25],  # 23
        ["Barna, Cal",     "Late",   "Up Little",    -4.1554197e-02, 6.3417030e-02,  90.84458,  6.341703, 25],  # 24
        ["Barna, Cal",     "Late",   "Up Lots",      -6.1124774e-02, 4.7162430e-02,  88.88752,  4.716243, 25],  # 25
        ["Barna, Cal",     "Middle", "Down Little",  -9.2299354e-02, 2.6968317e-01,  85.77006, 26.968317, 25],  # 26
        ["Barna, Cal",     "Middle", "Down Lots",    -6.3738632e-02, 9.7047280e-02,  88.62614,  9.704728, 25],  # 27
        ["Barna, Cal",     "Middle", "Tight",        -9.8344820e-03, 2.1404578e-01,  94.01655, 21.404578, 25],  # 28
        ["Barna, Cal",     "Middle", "Up Little",    -6.5283452e-02, 1.8250925e-01,  88.47165, 18.250925, 25],  # 29
        ["Barna, Cal",     "Middle", "Up Lots",      -7.2891110e-02, 2.7167870e-02,  87.71089,  2.716787, 25],  # 30

        ["Bland, Yates",   "Early",  "Down Little",  -1.5658553e-02, 2.5285967e-01, 108.43414, 25.285967, 25],  # 31
        ["Bland, Yates",   "Early",  "Down Lots",     6.0599390e-02, 2.2655789e-01, 116.05994, 22.655789, 25],  # 32
        ["Bland, Yates",   "Early",  "Tight",         6.5577342e-02, 4.3505531e-01, 116.55773, 43.505531, 25],  # 33
        ["Bland, Yates",   "Early",  "Up Little",    -4.6727715e-02, 1.9447352e-01, 105.32723, 19.447352, 25],  # 34
        ["Bland, Yates",   "Early",  "Up Lots",      -1.4762111e-02, 6.2992950e-02, 108.52379,  6.299295, 25],  # 35
        ["Bland, Yates",   "Late",   "Down Little",  -1.0109950e-01, 1.5465508e-01,  99.89005, 15.465508, 25],  # 36
        ["Bland, Yates",   "Late",   "Down Lots",    -2.9473544e-02, 5.9501240e-02, 107.05265,  5.950124, 25],  # 37
        ["Bland, Yates",   "Late",   "Tight",        -4.4204721e-02, 1.4235376e-01, 105.57953, 14.235376, 25],  # 38
        ["Bland, Yates",   "Late",   "Up Little",    -5.0384740e-03, 6.5421050e-02, 109.49615,  6.542105, 25],  # 39
        ["Bland, Yates",   "Late",   "Up Lots",      -2.5243845e-02, 5.9976660e-02, 107.47562,  5.997666, 25],  # 40
        ["Bland, Yates",   "Middle", "Down Little",  -4.2286392e-02, 2.7206022e-01, 105.77136, 27.206022, 25],  # 41
        ["Bland, Yates",   "Middle", "Down Lots",     1.4090279e-02, 1.0025433e-01, 111.40903, 10.025433, 25],  # 42
        ["Bland, Yates",   "Middle", "Tight",         4.9557265e-02, 2.2095649e-01, 114.95573, 22.095649, 25],  # 43
        ["Bland, Yates",   "Middle", "Up Little",    -1.0600461e-02, 1.8494638e-01, 108.93995, 18.494638, 25],  # 44
        ["Bland, Yates",   "Middle", "Up Lots",      -2.1357321e-02, 3.1012570e-02, 107.86427,  3.101257, 25],  # 45

        ["Bowen, Gavyn",   "Early",  "Down Little",  -3.7172026e-01, 2.3460981e-01,  72.82797, 23.460981, 25],  # 46
        ["Bowen, Gavyn",   "Early",  "Down Lots",    -3.2273521e-01, 1.7892531e-01,  77.72648, 17.892531, 25],  # 47
        ["Bowen, Gavyn",   "Early",  "Tight",        -2.6623385e-01, 3.5146700e-01,  83.37661, 35.146700, 25],  # 48
        ["Bowen, Gavyn",   "Early",  "Up Little",    -3.5381252e-01, 1.9342697e-01,  74.61875, 19.342697, 25],  # 49
        ["Bowen, Gavyn",   "Early",  "Up Lots",      -2.5328969e-01, 5.8316910e-02,  84.67103,  5.831691, 25],  # 50
        ["Bowen, Gavyn",   "Late",   "Down Little",  -3.7462338e-01, 1.7594462e-01,  72.53766, 17.594462, 25],  # 51
        ["Bowen, Gavyn",   "Late",   "Down Lots",    -2.6449622e-01, 4.9714890e-02,  83.55038,  4.971489, 25],  # 52
        ["Bowen, Gavyn",   "Late",   "Tight",        -3.2400117e-01, 1.4562416e-01,  77.59988, 14.562416, 25],  # 53
        ["Bowen, Gavyn",   "Late",   "Up Little",    -1.8753461e-01, 7.4770630e-02,  91.24654,  7.477063, 25],  # 54
        ["Bowen, Gavyn",   "Late",   "Up Lots",      -1.7897773e-01, 4.0370720e-02,  92.10223,  4.037072, 25],  # 55
        ["Bowen, Gavyn",   "Middle", "Down Little",  -3.6326771e-01, 2.6561663e-01,  73.67323, 26.561663, 25],  # 56
        ["Bowen, Gavyn",   "Middle", "Down Lots",    -3.0942140e-01, 8.0671260e-02,  79.05786,  8.067126, 25],  # 57
        ["Bowen, Gavyn",   "Middle", "Tight",        -2.76035897e-01,2.2597679e-01,  82.39641, 22.597679, 25],  # 58
        ["Bowen, Gavyn",   "Middle", "Up Little",    -2.76101629e-01,1.7966528e-01,  82.38984, 17.966528, 25],  # 59
        ["Bowen, Gavyn",   "Middle", "Up Lots",      -2.29486128e-01, 2.6956120e-02,  87.05139,  2.695612, 25],  # 60

        ["Dilhoff, Parker","Early",  "Down Little",  -2.71284822e-01, 3.1508555e-01,  82.87152, 31.508555, 25],  # 61
        ["Dilhoff, Parker","Early",  "Down Lots",    -1.91953920e-01, 1.9575620e-01,  90.80461, 19.575620, 25],  # 62
        ["Dilhoff, Parker","Early",  "Tight",        -1.57098436e-01, 4.3567419e-01,  94.29016, 43.567419, 25],  # 63
        ["Dilhoff, Parker","Early",  "Up Little",    -2.86585568e-01, 2.4703822e-01,  81.34144, 24.703822, 25],  # 64
        ["Dilhoff, Parker","Early",  "Up Lots",      -1.72884222e-01, 6.5039630e-02,  92.71158,  6.503963, 25],  # 65
        ["Dilhoff, Parker","Late",   "Down Little",  -3.48689572e-01, 2.1325560e-01,  75.13104, 21.325560, 25],  # 66
        ["Dilhoff, Parker","Late",   "Down Lots",    -1.82748886e-01, 5.7873960e-02,  91.72511,  5.787396, 25],  # 67
        ["Dilhoff, Parker","Late",   "Tight",        -3.04270154e-01, 1.8810484e-01,  79.57298, 18.810484, 25],  # 68
        ["Dilhoff, Parker","Late",   "Up Little",    -1.25883589e-01, 7.7479900e-02,  97.41164,  7.747990, 25],  # 69
        ["Dilhoff, Parker","Late",   "Up Lots",      -1.21894272e-01, 4.6792200e-02,  97.81057,  4.679220, 25],  # 70
        ["Dilhoff, Parker","Middle", "Down Little",  -2.99723382e-01, 3.1076812e-01,  80.02766, 31.076812, 25],  # 71
        ["Dilhoff, Parker","Middle", "Down Lots",    -1.96035909e-01, 8.8014900e-02,  90.39641,  8.801490, 25],  # 72
        ["Dilhoff, Parker","Middle", "Tight",        -2.10189954e-01, 2.6970520e-01,  88.98100, 26.970520, 25],  # 73
        ["Dilhoff, Parker","Middle", "Up Little",    -2.07349832e-01, 2.2862811e-01,  89.26502, 22.862811, 25],  # 74
        ["Dilhoff, Parker","Middle", "Up Lots",      -1.52400630e-01, 2.9257890e-02,  94.75994,  2.925789, 25],  # 75

        ["Donegan, Josh",  "Early",  "Down Little",  -2.0716133e-02, 2.6350645e-01, 107.92839, 26.350645, 25],  # 76
        ["Donegan, Josh",  "Early",  "Down Lots",     5.8569783e-02, 2.2777077e-01, 115.85698, 22.777077, 25],  # 77
        ["Donegan, Josh",  "Early",  "Tight",         6.8354618e-02, 4.2187195e-01, 116.83546, 42.187195, 25],  # 78
        ["Donegan, Josh",  "Early",  "Up Little",    -6.0803592e-02, 2.0939900e-01, 103.91964, 20.939900, 25],  # 79
        ["Donegan, Josh",  "Early",  "Up Lots",      -1.4245519e-02, 6.5646480e-02, 108.57545,  6.564648, 25],  # 80
        ["Donegan, Josh",  "Late",   "Down Little",  -1.23895521e-01, 1.6036737e-01,  97.61045, 16.036737, 25],  # 81
        ["Donegan, Josh",  "Late",   "Down Lots",    -2.8602698e-02, 5.7258230e-02, 107.13973,  5.725823, 25],  # 82
        ["Donegan, Josh",  "Late",   "Tight",        -6.5937486e-02, 1.5401259e-01, 103.40625, 15.401259, 25],  # 83
        ["Donegan, Josh",  "Late",   "Up Little",    -7.0213330e-03, 6.9600380e-02, 109.29787,  6.960038, 25],  # 84
        ["Donegan, Josh",  "Late",   "Up Lots",      -2.4537259e-02, 6.2520900e-02, 107.54627,  6.252090, 25],  # 85
        ["Donegan, Josh",  "Middle", "Down Little",  -5.7086724e-02, 2.8308040e-01, 104.29133, 28.308040, 25],  # 86
        ["Donegan, Josh",  "Middle", "Down Lots",     1.7884355e-02, 1.0534015e-01, 111.78844, 10.534015, 25],  # 87
        ["Donegan, Josh",  "Middle", "Tight",         2.5792990e-02, 2.2546377e-01, 112.57930, 22.546377, 25],  # 88
        ["Donegan, Josh",  "Middle", "Up Little",    -2.1615398e-02, 1.9893871e-01, 107.83846, 19.893871, 25],  # 89
        ["Donegan, Josh",  "Middle", "Up Lots",      -2.0781869e-02, 3.1619980e-02, 107.92181,  3.161998, 25],  # 90

        ["Evangelista, Jase","Early","Down Little",   1.3676113e-02, 2.2176205e-01, 111.36761, 22.176205, 25],  # 91
        ["Evangelista, Jase","Early","Down Lots",     6.0788126e-02, 2.3226147e-01, 116.07881, 23.226147, 25],  # 92
        ["Evangelista, Jase","Early","Tight",         8.6773527e-02, 4.3443210e-01, 118.67735, 43.443210, 25],  # 93
        ["Evangelista, Jase","Early","Up Little",    -4.2758281e-02, 1.7154588e-01, 105.72417, 17.154588, 25],  # 94
        ["Evangelista, Jase","Early","Up Lots",      -1.6567515e-02, 6.6298070e-02, 108.34325,  6.629807, 25],  # 95
        ["Evangelista, Jase","Late", "Down Little",  -5.1692773e-02, 1.2550673e-01, 104.83072, 12.550673, 25],  # 96
        ["Evangelista, Jase","Late", "Down Lots",    -3.0234664e-02, 5.7712720e-02, 106.97653,  5.771272, 25],  # 97
        ["Evangelista, Jase","Late", "Tight",         4.5897502e-02, 1.0677062e-01, 114.58975, 10.677062, 25],  # 98
        ["Evangelista, Jase","Late", "Up Little",     1.3975460e-03, 6.4193110e-02, 110.13975,  6.419311, 25],  # 99
        ["Evangelista, Jase","Late", "Up Lots",      -2.1394820e-02, 5.5859440e-02, 107.86052,  5.585944, 25],  # 100
        ["Evangelista, Jase","Middle","Down Little", -2.4411222e-02, 2.5568511e-01, 107.55888, 25.568511, 25],  # 101
        ["Evangelista, Jase","Middle","Down Lots",    1.6089589e-02, 1.0700962e-01, 111.60896, 10.700962, 25],  # 102
        ["Evangelista, Jase","Middle","Tight",        8.5629600e-02, 1.7044699e-01, 118.56296, 17.044699, 25],  # 103
        ["Evangelista, Jase","Middle","Up Little",   -3.8230350e-03, 1.7593563e-01, 109.61770, 17.593563, 25],  # 104
        ["Evangelista, Jase","Middle","Up Lots",     -2.0894370e-02, 3.0936040e-02, 107.91056,  3.093604, 25],  # 105

        ["Foxson, Tate",   "Early",  "Down Little",  -1.5748794e-02, 2.4975373e-01, 108.42512, 24.975373, 25],  # 106
        ["Foxson, Tate",   "Early",  "Down Lots",     4.2912760e-02, 2.3274282e-01, 114.29128, 23.274282, 25],  # 107
        ["Foxson, Tate",   "Early",  "Tight",         5.0396039e-02, 4.1854119e-01, 115.03960, 41.854119, 25],  # 108
        ["Foxson, Tate",   "Early",  "Up Little",    -7.8093828e-02, 2.0974901e-01, 102.19062, 20.974901, 25],  # 109
        ["Foxson, Tate",   "Early",  "Up Lots",      -3.1342789e-02, 5.8981670e-02, 106.86572,  5.898167, 25],  # 110
        ["Foxson, Tate",   "Late",   "Down Little",  -1.37256456e-01, 1.6057482e-01,  96.27435, 16.057482, 25],  # 111
        ["Foxson, Tate",   "Late",   "Down Lots",    -4.2437091e-02, 5.4624990e-02, 105.75629,  5.462499, 25],  # 112
        ["Foxson, Tate",   "Late",   "Tight",        -1.12028930e-01, 1.5665613e-01,  98.79711, 15.665613, 25],  # 113
        ["Foxson, Tate",   "Late",   "Up Little",    -1.3672735e-02, 6.9589580e-02, 108.63273,  6.958958, 25],  # 114
        ["Foxson, Tate",   "Late",   "Up Lots",      -2.7686269e-02, 4.5287430e-02, 107.23137,  4.528743, 25],  # 115
        ["Foxson, Tate",   "Middle", "Down Little",  -5.5770675e-02, 2.8369896e-01, 104.42293, 28.369896, 25],  # 116
        ["Foxson, Tate",   "Middle", "Down Lots",    -1.3598925e-02, 1.0090833e-01, 108.64011, 10.090833, 25],  # 117
        ["Foxson, Tate",   "Middle", "Tight",         1.9437878e-02, 2.3521251e-01, 111.94379, 23.521251, 25],  # 118
        ["Foxson, Tate",   "Middle", "Up Little",    -3.9181015e-02, 2.0173901e-01, 106.08190, 20.173901, 25],  # 119
        ["Foxson, Tate",   "Middle", "Up Lots",      -3.4993910e-02, 2.6159980e-02, 106.50061,  2.615998, 25],  # 120

        ["Gomberg, Jacob", "Early",  "Down Little",  -3.47878043e-01, 2.1544712e-01,  75.21220, 21.544712, 25],  # 121
        ["Gomberg, Jacob", "Early",  "Down Lots",    -3.29384195e-01, 2.2066630e-01,  77.06158, 22.066630, 25],  # 122
        ["Gomberg, Jacob", "Early",  "Tight",        -2.53772151e-01, 3.7207944e-01,  84.62278, 37.207944, 25],  # 123
        ["Gomberg, Jacob", "Early",  "Up Little",    -3.43644617e-01, 1.7699410e-01,  75.63554, 17.699410, 25],  # 124
        ["Gomberg, Jacob", "Early",  "Up Lots",      -2.64266056e-01, 5.2601990e-02,  83.57339,  5.260199, 25],  # 125
        ["Gomberg, Jacob", "Late",   "Down Little",  -3.11895424e-01, 1.3528160e-01,  78.81046, 13.528160, 25],  # 126
        ["Gomberg, Jacob", "Late",   "Down Lots",    -2.74083606e-01, 4.3119620e-02,  82.59164,  4.311962, 25],  # 127
        ["Gomberg, Jacob", "Late",   "Tight",        -2.42524944e-01, 1.3147473e-01,  85.74751, 13.147473, 25],  # 128
        ["Gomberg, Jacob", "Late",   "Up Little",    -1.93856447e-01, 5.7258740e-02,  90.61436,  5.725874, 25],  # 129
        ["Gomberg, Jacob", "Late",   "Up Lots",      -1.88560971e-01, 4.7043600e-02,  91.14390,  4.704360, 25],  # 130
        ["Gomberg, Jacob", "Middle", "Down Little",  -3.18672959e-01, 2.4310491e-01,  78.13270, 24.310491, 25],  # 131
        ["Gomberg, Jacob", "Middle", "Down Lots",    -3.31504959e-01, 8.6444180e-02,  76.84950,  8.644418, 25],  # 132
        ["Gomberg, Jacob", "Middle", "Tight",        -2.17636436e-01, 1.9846577e-01,  88.23636, 19.846577, 25],  # 133
        ["Gomberg, Jacob", "Middle", "Up Little",    -2.58833340e-01, 1.7143995e-01,  84.11667, 17.143995, 25],  # 134
        ["Gomberg, Jacob", "Middle", "Up Lots",      -2.32168221e-01, 2.6738640e-02,  86.78318,  2.673864, 25],  # 135

        ["Lane, Carson",   "Early",  "Down Little",   3.8370610e-03, 2.3736993e-01, 110.38371, 23.736993, 25],  # 136
        ["Lane, Carson",   "Early",  "Down Lots",     4.0979291e-02, 2.6958592e-01, 114.09793, 26.958592, 25],  # 137
        ["Lane, Carson",   "Early",  "Tight",         6.6179487e-02, 3.8096613e-01, 121.83348, 18.351142, 25],  # 138 (note: MeanStopper row 150 clarifies later)
        ["Lane, Carson",   "Early",  "Up Little",    -3.2307496e-02, 1.8456456e-01, 106.76925, 18.456456, 25],  # 139
        ["Lane, Carson",   "Early",  "Up Lots",      -5.3039861e-02, 5.1638730e-02, 104.69601,  5.163873, 25],  # 140
        ["Lane, Carson",   "Late",   "Down Little",  -2.0711187e-02, 1.1620132e-01, 107.92888, 11.620132, 25],  # 141
        ["Lane, Carson",   "Late",   "Down Lots",    -6.1021417e-02, 4.6637520e-02, 103.89786,  4.663752, 25],  # 142
        ["Lane, Carson",   "Late",   "Tight",         4.7657710e-02, 1.0926715e-01, 114.76577, 10.926715, 25],  # 143
        ["Lane, Carson",   "Late",   "Up Little",    -2.1118825e-02, 6.8361860e-02, 107.88812,  6.836186, 25],  # 144
        ["Lane, Carson",   "Late",   "Up Lots",      -4.5542440e-02, 4.2298100e-02, 105.44576,  4.229810, 25],  # 145
        ["Lane, Carson",   "Middle", "Down Little",   2.5013924e-02, 2.6727461e-01, 112.50139, 26.727461, 25],  # 146
        ["Lane, Carson",   "Middle", "Down Lots",    -2.9670672e-02, 1.1966630e-01, 107.03293, 11.966630, 25],  # 147
        ["Lane, Carson",   "Middle", "Tight",         1.18334762e-01, 1.8351142e-01, 121.83348, 18.351142, 25],  # 148
        ["Lane, Carson",   "Middle", "Up Little",    -1.9222583e-02, 1.7640330e-01, 108.07774, 17.640330, 25],  # 149
        ["Lane, Carson",   "Middle", "Up Lots",      -5.5959956e-02, 2.4028930e-02, 104.40400,  2.402893, 25],  # 150

        ["Lueck, Reese",   "Early",  "Down Little",  -6.2077787e-02, 2.6849216e-01, 103.79222, 26.849216, 25],  # 151
        ["Lueck, Reese",   "Early",  "Down Lots",     5.7903172e-02, 2.3440061e-01, 115.79032, 23.440061, 25],  # 152
        ["Lueck, Reese",   "Early",  "Tight",         3.2156984e-02, 4.2146133e-01, 113.21570, 42.146133, 25],  # 153
        ["Lueck, Reese",   "Early",  "Up Little",    -6.7599805e-02, 2.2406113e-01, 103.24002, 22.406113, 25],  # 154
        ["Lueck, Reese",   "Early",  "Up Lots",      -2.6675195e-02, 6.5691330e-02, 107.33248,  6.569133, 25],  # 155
        ["Lueck, Reese",   "Late",   "Down Little",  -1.13714371e-01, 1.4990533e-01,  98.62856, 14.990533, 25],  # 156
        ["Lueck, Reese",   "Late",   "Down Lots",    -4.2353675e-02, 6.1867470e-02, 105.76463,  6.186747, 25],  # 157
        ["Lueck, Reese",   "Late",   "Tight",        -9.3419062e-02, 1.6813185e-01, 100.65809, 16.813185, 25],  # 158
        ["Lueck, Reese",   "Late",   "Up Little",    -2.3257295e-02, 8.0338340e-02, 107.67427,  8.033834, 25],  # 159
        ["Lueck, Reese",   "Late",   "Up Lots",      -3.7851515e-02, 6.5149750e-02, 106.21485,  6.514975, 25],  # 160
        ["Lueck, Reese",   "Middle", "Down Little",  -5.0739307e-02, 2.8957773e-01, 104.92607, 28.957773, 25],  # 161
        ["Lueck, Reese",   "Middle", "Down Lots",     8.1658590e-03, 9.6709850e-02, 110.81659,  9.670985, 25],  # 162
        ["Lueck, Reese",   "Middle", "Tight",        -2.9462560e-03, 2.6150950e-01, 109.70537, 26.150950, 25],  # 163
        ["Lueck, Reese",   "Middle", "Up Little",    -3.3742657e-02, 2.2223144e-01, 106.62573, 22.223144, 25],  # 164
        ["Lueck, Reese",   "Middle", "Up Lots",      -3.1201138e-02, 3.7725000e-02, 106.87989,  3.772500, 25],  # 165

        ["Manning, LJ",    "Early",  "Down Little",  -1.49995789e-01, 2.4787427e-01,  95.00042, 24.787427, 25],  # 166
        ["Manning, LJ",    "Early",  "Down Lots",    -7.0098699e-02, 2.3863721e-01, 102.99013, 23.863721, 25],  # 167
        ["Manning, LJ",    "Early",  "Tight",        -6.6785435e-02, 4.1686327e-01, 103.32146, 41.686327, 25],  # 168
        ["Manning, LJ",    "Early",  "Up Little",    -1.58240338e-01, 1.8935328e-01,  94.17597, 18.935328, 25],  # 169
        ["Manning, LJ",    "Early",  "Up Lots",      -1.08575171e-01, 5.3947970e-02,  99.14248,  5.394797, 25],  # 170
        ["Manning, LJ",    "Late",   "Down Little",  -2.03093805e-01, 1.5918166e-01,  89.69062, 15.918166, 25],  # 171
        ["Manning, LJ",    "Late",   "Down Lots",    -1.18179471e-01, 4.7954900e-02,  98.18205,  4.795490, 25],  # 172
        ["Manning, LJ",    "Late",   "Tight",        -1.65755789e-01, 1.4885123e-01,  93.42442, 14.885123, 25],  # 173
        ["Manning, LJ",    "Late",   "Up Little",    -7.5723796e-02, 6.5858060e-02, 102.42762,  6.585806, 25],  # 174
        ["Manning, LJ",    "Late",   "Up Lots",      -8.6886558e-02, 4.9917270e-02, 101.31134,  4.991727, 25],  # 175
        ["Manning, LJ",    "Middle", "Down Little",  -1.46066426e-01, 2.9192353e-01,  95.39336, 29.192353, 25],  # 176
        ["Manning, LJ",    "Middle", "Down Lots",    -1.08888922e-01, 1.0022123e-01,  99.11111, 10.022123, 25],  # 177
        ["Manning, LJ",    "Middle", "Tight",        -6.1799673e-02, 2.2275659e-01, 103.82003, 22.275659, 25],  # 178
        ["Manning, LJ",    "Middle", "Up Little",    -1.08785739e-01, 1.8197031e-01,  99.12143, 18.197031, 25],  # 179
        ["Manning, LJ",    "Middle", "Up Lots",      -1.03970880e-01, 2.8474240e-02,  99.60291,  2.847424, 25],  # 180

        ["Marton, Ryan",   "Early",  "Down Little",  -8.6947918e-02, 2.9934104e-01, 101.30521, 29.934104, 25],  # 181
        ["Marton, Ryan",   "Early",  "Down Lots",     1.5119128e-02, 2.2137931e-01, 111.51191, 22.137931, 25],  # 182
        ["Marton, Ryan",   "Early",  "Tight",        -2.5318790e-03, 4.5210389e-01, 109.74681, 45.210389, 25],  # 183
        ["Marton, Ryan",   "Early",  "Up Little",    -1.18716376e-01, 2.5210196e-01,  98.12836, 25.210196, 25],  # 184
        ["Marton, Ryan",   "Early",  "Up Lots",      -4.4403439e-02, 6.3189250e-02, 105.55966,  6.318925, 25],  # 185
        ["Marton, Ryan",   "Late",   "Down Little",  -2.19630445e-01, 2.0231827e-01,  88.03696, 20.231827, 25],  # 186
        ["Marton, Ryan",   "Late",   "Down Lots",    -5.8015925e-02, 5.4652950e-02, 104.19841,  5.465295, 25],  # 187
        ["Marton, Ryan",   "Late",   "Tight",        -1.80887424e-01, 1.8479817e-01,  91.91126, 18.479817, 25],  # 188
        ["Marton, Ryan",   "Late",   "Up Little",    -3.1315632e-02, 8.2435360e-02, 106.86844,  8.243536, 25],  # 189
        ["Marton, Ryan",   "Late",   "Up Lots",      -3.8628832e-02, 5.1670110e-02, 106.13712,  5.167011, 25],  # 190
        ["Marton, Ryan",   "Middle", "Down Little",  -1.26267371e-01, 3.2240352e-01,  97.37326, 32.240352, 25],  # 191
        ["Marton, Ryan",   "Middle", "Down Lots",    -2.6309451e-02, 1.0343945e-01, 107.36905, 10.343945, 25],  # 192
        ["Marton, Ryan",   "Middle", "Tight",        -4.8911827e-02, 2.7369082e-01, 105.10882, 27.369082, 25],  # 193
        ["Marton, Ryan",   "Middle", "Up Little",    -7.7386072e-02, 2.2274991e-01, 102.26139, 22.274991, 25],  # 194
        ["Marton, Ryan",   "Middle", "Up Lots",      -4.8769574e-02, 2.9194900e-02, 105.12304,  2.919490, 25],  # 195

        ["Ong, Felix",     "Early",  "Down Little",  -1.9420928e-02, 2.6498547e-01, 108.05791, 26.498547, 25],  # 196
        ["Ong, Felix",     "Early",  "Down Lots",     6.2556647e-02, 2.3338844e-01, 116.25566, 23.338844, 25],  # 197
        ["Ong, Felix",     "Early",  "Tight",         7.5573945e-02, 3.7965683e-01, 117.55739, 37.965683, 25],  # 198
        ["Ong, Felix",     "Early",  "Up Little",    -5.5649834e-02, 2.0923799e-01, 104.43502, 20.923799, 25],  # 199
        ["Ong, Felix",     "Early",  "Up Lots",      -2.2803646e-02, 5.8257510e-02, 107.71964,  5.825751, 25],  # 200
        ["Ong, Felix",     "Late",   "Down Little",  -1.41635833e-01, 1.7094013e-01,  95.83642, 17.094013, 25],  # 201
        ["Ong, Felix",     "Late",   "Down Lots",    -3.6709063e-02, 5.3269180e-02, 106.32909,  5.326918, 25],  # 202
        ["Ong, Felix",     "Late",   "Tight",        -6.9453644e-02, 1.5463150e-01, 103.05464, 15.463150, 25],  # 203
        ["Ong, Felix",     "Late",   "Up Little",    -7.4061180e-03, 7.1032660e-02, 109.25939,  7.103266, 25],  # 204
        ["Ong, Felix",     "Late",   "Up Lots",      -2.5407360e-02, 4.6341840e-02, 107.45926,  4.634184, 25],  # 205
        ["Ong, Felix",     "Middle", "Down Little",  -5.5072250e-02, 3.0304096e-01, 104.49277, 30.304096, 25],  # 206
        ["Ong, Felix",     "Middle", "Down Lots",     4.6005640e-03, 1.1023441e-01, 110.46006, 11.023441, 25],  # 207
        ["Ong, Felix",     "Middle", "Tight",         4.4088788e-02, 2.2821202e-01, 114.40888, 22.821202, 25],  # 208
        ["Ong, Felix",     "Middle", "Up Little",    -3.0647539e-02, 1.9288782e-01, 106.93525, 19.288782, 25],  # 209
        ["Ong, Felix",     "Middle", "Up Lots",      -3.1298357e-02, 2.6228460e-02, 106.87016,  2.622846, 25],  # 210

        ["Rogers, Dylan",  "Early",  "Down Little",  -4.3930087e-02, 2.5661851e-01,  90.60699, 25.661851, 25],  # 211
        ["Rogers, Dylan",  "Early",  "Down Lots",     5.9030553e-02, 2.4714603e-01, 100.90306, 24.714603, 25],  # 212
        ["Rogers, Dylan",  "Early",  "Tight",         4.2707695e-02, 4.1254117e-01,  99.27077, 41.254117, 25],  # 213
        ["Rogers, Dylan",  "Early",  "Up Little",    -5.7005279e-02, 2.1114647e-01,  89.29947, 21.114647, 25],  # 214
        ["Rogers, Dylan",  "Early",  "Up Lots",      -2.7280254e-02, 6.4773150e-02,  92.27197,  6.477315, 25],  # 215
        ["Rogers, Dylan",  "Late",   "Down Little",  -9.3272340e-02, 1.3555485e-01,  85.67277, 13.555485, 25],  # 216
        ["Rogers, Dylan",  "Late",   "Down Lots",    -3.9641432e-02, 5.3523090e-02,  91.03586,  5.352309, 25],  # 217
        ["Rogers, Dylan",  "Late",   "Tight",        -7.2535241e-02, 1.5920621e-01,  87.74648, 15.920621, 25],  # 218
        ["Rogers, Dylan",  "Late",   "Up Little",    -1.9069659e-02, 7.4223320e-02,  93.09303,  7.422332, 25],  # 219
        ["Rogers, Dylan",  "Late",   "Up Lots",      -3.5957915e-02, 6.3763630e-02,  91.40421,  6.376363, 25],  # 220
        ["Rogers, Dylan",  "Middle", "Down Little",  -3.1198463e-02, 2.8900779e-01,  91.88015, 28.900779, 25],  # 221
        ["Rogers, Dylan",  "Middle", "Down Lots",     9.0889670e-03, 1.0888314e-01,  95.90890, 10.888314, 25],  # 222
        ["Rogers, Dylan",  "Middle", "Tight",         2.9689287e-02, 2.4448322e-01,  97.96893, 24.448322, 25],  # 223
        ["Rogers, Dylan",  "Middle", "Up Little",    -2.7042781e-02, 2.0029705e-01,  92.29572, 20.029705, 25],  # 224
        ["Rogers, Dylan",  "Middle", "Up Lots",      -3.1561199e-02, 3.2546340e-02,  91.84388,  3.254634, 25],  # 225
        ["Sundloff, Colton", "Late",   "Down Lots",   -3.280767e-02, 0.05326294, 106.71923, 5.326294, 25],  # 226
        ["Sundloff, Colton", "Late",   "Up Lots",     -2.654357e-02, 0.11772633, 107.34564, 5.663876, 25],  # 227
        ["Sundloff, Colton", "Middle", "Up Lots",     -2.608280e-02, 0.05663876, 107.39172, 5.663876, 25],  # 228
        ["Sundloff, Colton", "Early",  "Up Little",   -2.110610e-02, 0.16741247, 107.88939, 16.741247, 25],  # 229
        ["Sundloff, Colton", "Early",  "Up Lots",     -2.012453e-02, 0.05663876, 107.98755, 5.663876, 25],  # 230
        ["Sundloff, Colton", "Late",   "Down Little", -1.292766e-02, 0.11772633, 108.70723, 11.772633, 25],  # 231
        ["Sundloff, Colton", "Late",   "Up Little",   -1.988517e-03, 0.05787570, 109.80115, 5.787570, 25],  # 232
        ["Sundloff, Colton", "Middle", "Down Lots",    6.901147e-05, 0.05326294, 110.00690, 24.431044, 25],  # 233
        ["Sundloff, Colton", "Middle", "Up Little",    7.036792e-03, 0.16741247, 110.70368, 16.741247, 25],  # 234
        ["Sundloff, Colton", "Middle", "Down Little",  2.916844e-02, 0.11772633, 112.91684, 11.772633, 25],  # 235
        ["Sundloff, Colton", "Late",   "Tight",        2.984216e-02, 0.10473598, 112.98422, 10.473598, 25],  # 236
        ["Sundloff, Colton", "Early",  "Down Little",  4.705457e-02, 0.20967661, 114.70546, 11.772633, 25],  # 237
        ["Sundloff, Colton", "Early",  "Down Lots",    6.391800e-02, 0.24431044, 116.39180, 24.431044, 25],  # 238
        ["Sundloff, Colton", "Early",  "Tight",        7.756285e-02, 0.40290727, 117.75628, 40.290727, 25],  # 239
        ["Sundloff, Colton", "Middle", "Tight",        1.031424e-01, 0.40290727, 120.31424, 10.473598, 25]  # 240
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
    
