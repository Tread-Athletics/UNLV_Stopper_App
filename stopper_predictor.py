import polars as pl
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
import joblib


NUM_FEATURES = [
    "start_speed", "spin_rate", "ax", "az", "px", "pz", "balls", "strikes", "leverage_index", "inning",
]
CAT_FEATURES = [
    "pitch_type", "batter_hand", "pitcher_hand", "half",  # Added pitcher_hand
]


@dataclass
class PitchDeltaModel:
    pipeline: Any
    feature_cols: list
    num_cols: list
    cat_cols: list


def _clean_for_training(df: pl.DataFrame) -> pl.DataFrame:
    cols = set(df.columns)
    # Normalize key identifiers if necessary
    if "batter_hand" not in cols and "BatterSide" in cols:
        df = df.with_columns(
            pl.when(pl.col("BatterSide").cast(pl.Utf8).str.to_lowercase().str.contains("left")).then(pl.lit("L"))
             .when(pl.col("BatterSide").cast(pl.Utf8).str.to_lowercase().str.contains("right")).then(pl.lit("R"))
             .otherwise(pl.lit("UNK")).alias("batter_hand")
        )
        cols.add("batter_hand")
    if "pitcher_name" not in cols and "Pitcher" in cols:
        df = df.with_columns(pl.col("Pitcher").cast(pl.Utf8).alias("pitcher_name"))
        cols.add("pitcher_name")
    # Normalize inning and half
    if "inning" not in cols and "Inning" in cols:
        df = df.with_columns(pl.col("Inning").cast(pl.Float64).alias("inning"))
        cols.add("inning")
    if "half" not in cols:
        source = "Top.Bottom" if "Top.Bottom" in cols else ("Top_Bottom" if "Top_Bottom" in cols else None)
        if source:
            df = df.with_columns(
                pl.when(pl.col(source).cast(pl.Utf8).str.to_lowercase().str.contains("top")).then(pl.lit("Top"))
                 .otherwise(pl.lit("Bottom")).alias("half")
            )
            cols.add("half")
    need = set(NUM_FEATURES + CAT_FEATURES + ["delta_field_win_exp", "pitcher_name"])
    missing = need - cols
    if missing:
        # create minimal placeholders to avoid failures
        fill_exprs = []
        for c in missing:
            if c in NUM_FEATURES:
                fill_exprs.append(pl.lit(None).cast(pl.Float64).alias(c))
            elif c in CAT_FEATURES or c == "pitcher_name":
                fill_exprs.append(pl.lit(None).cast(pl.Utf8).alias(c))
            elif c == "delta_field_win_exp":
                fill_exprs.append(pl.lit(None).cast(pl.Float64).alias(c))
        if fill_exprs:
            df = df.with_columns(fill_exprs)

    # Cast types and sanitize values
    num_exprs = [pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in NUM_FEATURES]
    cat_exprs = [pl.col(c).cast(pl.Utf8, strict=False).fill_null("UNK").alias(c) for c in CAT_FEATURES + ["pitcher_name"]]
    df = df.with_columns(num_exprs + cat_exprs)

    # Limit leverage_index to reasonable range
    if "leverage_index" in df.columns:
        df = df.with_columns(pl.col("leverage_index").clip(0.0, 10.0))
    return df


def train_pitch_delta_model(df: pl.DataFrame, cache_path: str = "model/pitch_delta_model.joblib") -> PitchDeltaModel:
    df = _clean_for_training(df)
    train = df.select(NUM_FEATURES + CAT_FEATURES + ["delta_field_win_exp"]).drop_nulls(subset=["delta_field_win_exp"])  # allow feature nulls (imputed)
    if train.height == 0:
        # Fallback zero model
        dummy = Pipeline([("pre", ColumnTransformer([
            ("num", Pipeline(steps=[("impute", SimpleImputer(strategy="median"))]), NUM_FEATURES),
            ("cat", Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT_FEATURES),
        ])), ("rf", DummyRegressor(strategy="constant", constant=0.0))])
        dummy.fit(np.zeros((1, len(NUM_FEATURES) + len(CAT_FEATURES))), np.array([0.0]))
        return PitchDeltaModel(pipeline=dummy, feature_cols=NUM_FEATURES + CAT_FEATURES, num_cols=NUM_FEATURES, cat_cols=CAT_FEATURES)

    X = train.select(NUM_FEATURES + CAT_FEATURES).to_pandas()
    y = train["delta_field_win_exp"].to_numpy()

    pre = ColumnTransformer([
        ("num", Pipeline(steps=[("impute", SimpleImputer(strategy="median"))]), NUM_FEATURES),
        ("cat", Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT_FEATURES),
    ])

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=3)
    pipe = Pipeline([("pre", pre), ("rf", rf)])
    pipe.fit(X, y)

    mdl = PitchDeltaModel(pipeline=pipe, feature_cols=NUM_FEATURES + CAT_FEATURES, num_cols=NUM_FEATURES, cat_cols=CAT_FEATURES)
    try:
        joblib.dump(mdl, cache_path)
    except Exception:
        pass
    return mdl


def load_or_train_pitch_delta_model(df: pl.DataFrame, cache_path: str = "model/pitch_delta_model.joblib") -> PitchDeltaModel:
    try:
        mdl = joblib.load(cache_path)
        return mdl
    except Exception:
        return train_pitch_delta_model(df, cache_path)


def build_pitcher_profiles(df: pl.DataFrame) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Return per (pitcher_name, pitch_type) profile with means/stds and frequency.
    Keys: (pitcher_name, pitch_type)
    Values: { 'freq': float, 'means': {...}, 'stds': {...}, 'count': int, 'hand_rates': {'L':p,'R':p} }
    """
    cols = set(df.columns)
    # Add batter_hand if only BatterSide is available
    if "batter_hand" not in cols and "BatterSide" in cols:
        df = df.with_columns(
            pl.when(pl.col("BatterSide").cast(pl.Utf8).str.to_lowercase().str.contains("left")).then(pl.lit("L"))
             .when(pl.col("BatterSide").cast(pl.Utf8).str.to_lowercase().str.contains("right")).then(pl.lit("R"))
             .otherwise(pl.lit("UNK")).alias("batter_hand")
        )
        cols.add("batter_hand")
    if "pitcher_name" not in cols and "Pitcher" in cols:
        df = df.with_columns(pl.col("Pitcher").cast(pl.Utf8).alias("pitcher_name"))
        cols.add("pitcher_name")
    if "inning" not in cols and "Inning" in cols:
        df = df.with_columns(pl.col("Inning").cast(pl.Float64).alias("inning"))
        cols.add("inning")
    if "half" not in cols:
        source = "Top.Bottom" if "Top.Bottom" in cols else ("Top_Bottom" if "Top_Bottom" in cols else None)
        if source:
            df = df.with_columns(
                pl.when(pl.col(source).cast(pl.Utf8).str.to_lowercase().str.contains("top")).then(pl.lit("Top"))
                 .otherwise(pl.lit("Bottom")).alias("half")
            )
            cols.add("half")

    need = {"pitcher_name", "pitch_type", "batter_hand", *NUM_FEATURES}
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"Missing columns for profiles: {missing}")

    total_by_pitcher = df.group_by("pitcher_name").len().rename({"len": "n_total"})

    # Basic per-type stats
    stats = (
        df.group_by(["pitcher_name", "pitch_type"]).agg([
            pl.len().alias("n"),
            pl.col("batter_hand").is_in(["L"]).mean().alias("hand_L_rate"),
            pl.col("batter_hand").is_in(["R"]).mean().alias("hand_R_rate"),
            *[pl.col(c).mean().alias(f"mean_{c}") for c in ["start_speed", "spin_rate", "ax", "az", "px", "pz"]],
            *[pl.col(c).std(ddof=1).alias(f"std_{c}") for c in ["start_speed", "spin_rate", "ax", "az", "px", "pz"]],
        ])
        .join(total_by_pitcher, on="pitcher_name", how="left")
        .with_columns((pl.col("n") / pl.col("n_total")).alias("freq"))
    )

    profiles: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in stats.iter_rows(named=True):
        key = (row["pitcher_name"], row["pitch_type"])
        profiles[key] = {
            "freq": float(row.get("freq", 0.0) or 0.0),
            "count": int(row.get("n", 0) or 0),
            "hand_rates": {"L": float(row.get("hand_L_rate", 0.5) or 0.5), "R": float(row.get("hand_R_rate", 0.5) or 0.5)},
            "means": {k.replace("mean_", ""): (row[k] if row[k] is not None else np.nan) for k in row if isinstance(k, str) and k.startswith("mean_")},
            "stds": {k.replace("std_", ""): (row[k] if row[k] is not None else np.nan) for k in row if isinstance(k, str) and k.startswith("std_")},
        }
    return profiles


def _sample_normal(mu: float, sd: float, rng: np.random.Generator, default: float = 0.0) -> float:
    if mu is None or np.isnan(mu):
        return default
    sd = float(sd) if (sd is not None and not np.isnan(sd) and sd > 1e-6) else max(abs(mu) * 0.05, 0.1)
    return float(rng.normal(mu, sd))


def simulate_expected_delta(
    mdl: 'PitchDeltaModel',
    profiles: Dict[tuple, dict],
    df_all: 'pl.DataFrame',
    pitcher_name: str,
    n_pitches: int = 20,
    batter_hand: str = "All",
    half: str = "Bottom",
    inning: int = 7,
    leverage_index: float = 1.0,
    balls: int = 0,
    strikes: int = 0,
    paths: int = 200,
    rng: 'np.random.Generator' = None,
    pitcher_hand: str = "Right"
) -> float:
    import pandas as pd
    import polars as pl
    try:
        import streamlit as st
        use_st = True
    except ImportError:
        use_st = False
    rng = rng or np.random.default_rng(42)
    pt_rows = [(pt, prof["freq"]) for (p, pt), prof in profiles.items() if p == pitcher_name and prof.get("freq", 0) > 0]
    if not pt_rows:
        return 0.0
    pitch_types, weights = zip(*pt_rows)
    pitch_types = np.array(pitch_types, dtype=object)
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    total_rows = paths * n_pitches
    pt_idx = rng.choice(len(pitch_types), size=total_rows, p=weights)
    pt_chosen = pitch_types[pt_idx]
    def get_arrays(key: str):
        mu = np.array([profiles.get((pitcher_name, pt), {}).get("means", {}).get(key, np.nan) for pt in pitch_types])
        sd = np.array([profiles.get((pitcher_name, pt), {}).get("stds", {}).get(key, np.nan) for pt in pitch_types])
        mu_sel = mu[pt_idx]
        sd_sel = sd[pt_idx]
        sd_fallback = np.where((~np.isfinite(sd_sel)) | (sd_sel <= 1e-6), np.maximum(np.abs(mu_sel) * 0.05, 0.1), sd_sel)
        return mu_sel, sd_fallback
    ss_mu, ss_sd = get_arrays("start_speed")
    sr_mu, sr_sd = get_arrays("spin_rate")
    ax_mu, ax_sd = get_arrays("ax")
    az_mu, az_sd = get_arrays("az")
    px_mu, px_sd = get_arrays("px")
    pz_mu, pz_sd = get_arrays("pz")
    start_speed_arr = rng.normal(ss_mu, ss_sd)
    spin_rate_arr = rng.normal(sr_mu, sr_sd)
    ax_arr = rng.normal(ax_mu, ax_sd)
    az_arr = rng.normal(az_mu, az_sd)
    px_arr = rng.normal(px_mu, px_sd)
    pz_arr = rng.normal(pz_mu, pz_sd)
    if batter_hand in ("L", "R"):
        batter_hand_arr = np.full(total_rows, batter_hand, dtype=object)
    else:
        l_rates = np.array([profiles.get((pitcher_name, pt), {}).get("hand_rates", {}).get("L", 0.5) for pt in pitch_types])
        l_rates_sel = l_rates[pt_idx]
        rnd = rng.random(total_rows)
        batter_hand_arr = np.where(rnd < l_rates_sel, "L", "R")
    pitcher_hand_arr = np.full(total_rows, pitcher_hand, dtype=object)
    half_arr = np.full(total_rows, half if half in ("Top", "Bottom") else ("Top" if str(half).lower().startswith("t") else "Bottom"), dtype=object)
    balls_arr = np.full(total_rows, balls)
    strikes_arr = np.full(total_rows, strikes)
    li_arr = np.full(total_rows, leverage_index)
    inning_arr = np.full(total_rows, inning)
    X = pd.DataFrame({
        'pitch_type': pt_chosen,
        'batter_hand': batter_hand_arr,
        'pitcher_hand': pitcher_hand_arr,
        'start_speed': start_speed_arr,
        'spin_rate': spin_rate_arr,
        'ax': ax_arr,
        'az': az_arr,
        'px': px_arr,
        'pz': pz_arr,
        'balls': balls_arr,
        'strikes': strikes_arr,
        'leverage_index': li_arr,
        'inning': inning_arr,
        'half': half_arr,
    })
    # Ensure X is a pandas DataFrame
    if isinstance(X, pl.DataFrame):
        X = X.to_pandas()
    elif not isinstance(X, pd.DataFrame):
        raise ValueError("Input to model must be a pandas DataFrame.")
    # Print 5 random rows and dtypes
    sample_X = X.sample(n=min(5, len(X)), random_state=42) if len(X) > 5 else X
    if use_st:
        st.write("Model input sample (random 5):", sample_X)
        st.write("Model input dtypes:", X.dtypes)
    else:
        print("Model input sample (random 5):\n", sample_X)
        print("Model input dtypes:\n", X.dtypes)
    preds = mdl.pipeline.predict(X)
    per_path = preds.reshape(paths, n_pitches).sum(axis=1)
    return float(per_path.mean())
