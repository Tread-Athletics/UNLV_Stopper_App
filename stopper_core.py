import polars as pl
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
import joblib


@dataclass
class WPModel:
    model: Any
    feature_cols: list
    cat_cols: list
    num_cols: list


def _safe_lower(s: pl.Expr) -> pl.Expr:
    return s.cast(pl.Utf8).str.strip_chars().str.to_lowercase()


def build_pa_table(raw: pl.DataFrame) -> pl.DataFrame:
    """Return one row per PA with start state and outcomes.

    Expects columns present in your CSV (best-effort fallback):
      - GameID, Inning, Top.Bottom, Batter, PAofInning, PitchNo
      - KorBB, TaggedHitType, PlayResult, OutsOnPlay, RunsScored
    Produces:
      - at_bat_id, first_pitch_no, RunsScored_final, OutsOnPlay_final, outcome tokens
    """
    df = raw.clone()

    # Ensure numeric ordering and grouping
    for c in ["PitchNo", "Outs", "Inning", "PAofInning"]:
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Int64, strict=False))

    # Required grouping fields
    need = ["GameID", "Top.Bottom", "Inning", "Batter", "PAofInning", "PitchNo"]
    if not all(c in df.columns for c in need):
        raise ValueError(f"Missing required columns for PA build: {[c for c in need if c not in df.columns]}")

    # Create at_bat_id if not present
    if "at_bat_id" not in df.columns:
        df = df.with_columns([
            (pl.col("GameID").cast(pl.Utf8) + "_" + pl.col("Top.Bottom").cast(pl.Utf8) + "_" +
             pl.col("Inning").cast(pl.Utf8) + "_" + pl.col("Batter").cast(pl.Utf8) + "_" +
             pl.col("PAofInning").cast(pl.Utf8)).alias("at_bat_id")
        ])

    # First pitch number per PA
    gcols = ["GameID", "Inning", "Top.Bottom", "PAofInning", "Batter"]
    pa_first = (
        df.group_by(gcols)
          .agg(pl.col("PitchNo").min().alias("first_pitch_no"))
    )

    # If running scores are already present in the data, take their value at the first pitch of the PA
    have_scores = set(["away_score", "home_score"]).issubset(df.columns)
    if have_scores:
        first_rows = (
            df.sort(["GameID", "Inning", "Top.Bottom", "PAofInning", "PitchNo"])  # earliest first
              .unique(subset=gcols, keep="first")
              .select(gcols + ["at_bat_id", "away_score", "home_score", *( ["runner_state"] if "runner_state" in df.columns else [] )])
        )
        pa_seq = (
            pa_first
            .join(first_rows, on=gcols, how="left")
            .sort(["GameID", "Inning", "Top.Bottom", "first_pitch_no", "PAofInning"])  # true order
        )
    else:
        # Minimal PA table without scores; they will be computed later
        first_rows = (
            df.sort(["GameID", "Inning", "Top.Bottom", "PAofInning", "PitchNo"])  # earliest first
              .unique(subset=gcols, keep="first")
              .select(gcols + ["at_bat_id"])
        )
        pa_seq = (
            pa_first
            .join(first_rows, on=gcols, how="left")
            .sort(["GameID", "Inning", "Top.Bottom", "first_pitch_no", "PAofInning"])  # true order
        )
    # Derive final outcomes per PA
    final_outcomes = (
        df.sort("PitchNo")
          .group_by(gcols)
          .agg([
              pl.col("KorBB").drop_nulls().last().alias("KorBB_final"),
              pl.col("TaggedHitType").drop_nulls().last().alias("TaggedHitType_final"),
              pl.col("PlayResult").drop_nulls().last().alias("PlayResult_final"),
              pl.col("OutsOnPlay").drop_nulls().last().alias("OutsOnPlay_final_raw"),
              pl.col("RunsScored").drop_nulls().last().alias("RunsScored_final_raw"),
              pl.col("at_bat_id").drop_nulls().last().alias("at_bat_id"),
          ])
          .with_columns([
              pl.when(pl.col("KorBB_final") == "Strikeout")
                .then(1)
                .otherwise(pl.col("OutsOnPlay_final_raw"))
                .cast(pl.Int64)
                .fill_null(0)
                .alias("OutsOnPlay_final"),
              pl.col("RunsScored_final_raw").cast(pl.Int64).fill_null(0).alias("RunsScored_final"),
          ])
          .select([
              "at_bat_id",
              "KorBB_final",
              "TaggedHitType_final",
              "PlayResult_final",
              "OutsOnPlay_final",
              "RunsScored_final",
          ])
    )
    pa_seq = pa_seq.join(final_outcomes, on="at_bat_id", how="left")
    return pa_seq


def attach_running_score(pa_sequence: pl.DataFrame) -> pl.DataFrame:
    """Compute cumulative away/home scores per PA and join as columns.
    Expects columns: GameID, Top.Bottom, RunsScored_final, first_pitch_no, at_bat_id.
    """
    # If away/home already exist, return as-is
    if set(["away_score", "home_score"]).issubset(pa_sequence.columns):
        return pa_sequence

    ps = pa_sequence.select([
        "GameID", "Top.Bottom", "RunsScored_final", "first_pitch_no", "at_bat_id"
    ])

    scoring = (
        ps.with_columns([
            pl.col("RunsScored_final").cast(pl.Int64).fill_null(0).alias("runs"),
            _safe_lower(pl.col("Top.Bottom")).alias("tb"),
            pl.col("Top.Bottom").cast(pl.Utf8).alias("Top.Bottom"),
        ])
        .with_columns([
            pl.when(pl.col("tb") == "top").then(pl.col("runs")).otherwise(0).alias("runs_away"),
            pl.when(pl.col("tb") == "bottom").then(pl.col("runs")).otherwise(0).alias("runs_home"),
        ])
        .sort(["GameID", "first_pitch_no", "Top.Bottom", "at_bat_id"])
        .with_columns([
            pl.col("runs_away").cum_sum().over("GameID").alias("away_score"),
            pl.col("runs_home").cum_sum().over("GameID").alias("home_score"),
        ])
        .with_columns([
            (pl.col("away_score").cast(pl.Int64).cast(pl.Utf8) + pl.lit("-") + pl.col("home_score").cast(pl.Int64).cast(pl.Utf8)).alias("score"),
        ])
        .select(["at_bat_id", "away_score", "home_score", "score"])
    )
    return pa_sequence.join(scoring, on="at_bat_id", how="left")


def add_state_features(pa_sequence: pl.DataFrame) -> pl.DataFrame:
    """Add state features used for WP model.
    - inning (float), Top.Bottom standardized to 'Top'/'Bot'
    - score_diff relative to batting team (Top: away-home, Bot: home-away)
    - outs: best-effort (if missing, fill 0)
    - runner_state is unknown without base machine; set as 'xxx' placeholder for now
    """
    df = pa_sequence.clone()
    df = df.with_columns([
        (pl.col("Inning") if "Inning" in df.columns else pl.col("inning")).cast(pl.Float64).alias("inning"),
        (pl.when(_safe_lower(pl.col("Top.Bottom")) == "top").then(pl.lit("Top")).otherwise(pl.lit("Bot")) if "Top.Bottom" in df.columns else pl.when(_safe_lower(pl.col("half")) == "top").then(pl.lit("Top")).otherwise(pl.lit("Bot"))).alias("half"),
    ])
    # Fill missing scores to 0.0 to avoid NaNs in score_diff
    away = (pl.col("away_score").cast(pl.Float64).fill_null(0.0) if "away_score" in df.columns else pl.lit(0.0))
    home = (pl.col("home_score").cast(pl.Float64).fill_null(0.0) if "home_score" in df.columns else pl.lit(0.0))
    df = df.with_columns([
        away.alias("away_score"),
        home.alias("home_score"),
        pl.when(pl.col("half") == "Top").then(away - home).otherwise(home - away).alias("score_diff"),
    ])
    if "Outs" in df.columns:
        df = df.with_columns(pl.col("Outs").cast(pl.Int64))
    else:
        df = df.with_columns(pl.lit(0).alias("Outs"))

    # Placeholder runner_state if base-state not computed; use '000'
    if "runner_state" in df.columns:
        # normalize to 3-char 0/1 string; coerce invalid/empty to '000'
        df = df.with_columns([
            pl.when(
                pl.col("runner_state").is_not_null()
            ).then(
                pl.col("runner_state").cast(pl.Utf8)
                    .str.replace_all(r"[^01]", "0")
                    .str.pad_start(3, "0")
                    .str.slice(0, 3)
            ).otherwise(pl.lit("000")).alias("runner_state")
        ])
    elif set(["on_1b", "on_2b", "on_3b"]).issubset(df.columns):
        df = df.with_columns([
            pl.col("on_1b").cast(pl.Int64).fill_null(0),
            pl.col("on_2b").cast(pl.Int64).fill_null(0),
            pl.col("on_3b").cast(pl.Int64).fill_null(0),
        ]).with_columns([
            (pl.col("on_1b").cast(pl.Int64).cast(pl.Utf8) + pl.col("on_2b").cast(pl.Int64).cast(pl.Utf8) + pl.col("on_3b").cast(pl.Int64).cast(pl.Utf8)).alias("runner_state")
        ])
    else:
        df = df.with_columns(pl.lit("000").alias("runner_state"))
    return df


def label_home_win(pa_with_scores: pl.DataFrame) -> pl.DataFrame:
    """Compute game-level final home win and attach to each PA as label."""
    final_score = (
        pa_with_scores
        .group_by("GameID")
        .agg([
            pl.col("away_score").max().alias("away_final"),
            pl.col("home_score").max().alias("home_final"),
        ])
        .with_columns((pl.col("home_final") > pl.col("away_final")).cast(pl.Int8).alias("home_win"))
    )
    return pa_with_scores.join(final_score.select(["GameID", "home_win"]), on="GameID", how="left")


def train_wp_model(pa_df: pl.DataFrame) -> WPModel:
    """Train a simple calibrated logistic regression P(home win | state).
    Features: inning, half, outs, runner_state, score_diff
    """
    use_cols = ["inning", "half", "Outs", "runner_state", "score_diff"]
    df = pa_df.select(["GameID", "at_bat_id", *use_cols, "home_win"]).drop_nulls(subset=["home_win"]).to_pandas()

    cat_cols = ["half", "runner_state"]
    num_cols = ["inning", "Outs", "score_diff"]

    pre = ColumnTransformer([
        (
            "cat",
            Pipeline(steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            cat_cols,
        ),
        (
            "num",
            Pipeline(steps=[
                ("impute", SimpleImputer(strategy="median")),
            ]),
            num_cols,
        ),
    ])
    base = LogisticRegression(max_iter=200, n_jobs=None)
    clf = Pipeline([
        ("pre", pre),
        ("lr", base),
    ])
    # Calibrate for better probabilities
    cal = CalibratedClassifierCV(estimator=clf, method="isotonic", cv=3)

    X = df[cat_cols + num_cols]
    y = df["home_win"].astype(int)
    cal.fit(X, y)

    return WPModel(model=cal, feature_cols=cat_cols + num_cols, cat_cols=cat_cols, num_cols=num_cols)


def predict_wp(wp: WPModel, states: pl.DataFrame) -> pl.Series:
    X = states.select(wp.feature_cols).to_pandas()
    p = wp.model.predict_proba(X)[:, 1]
    return pl.Series(name="wp_home", values=p)


def compute_wpa_per_pa(pa_df: pl.DataFrame, wp: WPModel) -> pl.DataFrame:
    """Compute ΔWP between consecutive PA starts within the same game timeline.
    Approximates event WPA by next_start_wp - current_start_wp. The last PA in a game gets 0.
    """
    # Predict WP at PA start
    states = pa_df.select(["GameID", "Inning", "Top.Bottom", "PAofInning", "at_bat_id", "first_pitch_no", "inning", "half", "Outs", "runner_state", "score_diff"])
    wp_start = predict_wp(wp, states)
    tmp = pa_df.with_columns([wp_start])

    # Next start wp within same game
    tmp = tmp.sort(["GameID", "Inning", "Top.Bottom", "first_pitch_no", "PAofInning"]).with_columns([
        pl.col("wp_home").shift(-1).over("GameID").alias("wp_next")
    ])
    tmp = tmp.with_columns([
        (pl.col("wp_next") - pl.col("wp_home")).fill_null(0.0).alias("wpa_pa")
    ])
    return tmp


def attach_pa_pitcher(pa_df: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    """Assign a pitcher to each PA: last pitch's pitcher in the PA."""
    # choose pitcher identity column
    pcol = "Pitcher" if "Pitcher" in raw.columns else ("pitcher_name" if "pitcher_name" in raw.columns else None)
    if pcol is None:
        return pa_df.with_columns(pl.lit(None).cast(pl.Utf8).alias("pa_pitcher"))
    # generate at_bat_id if missing in raw pitch data
    if "at_bat_id" not in raw.columns:
        raw = raw.with_columns(
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
    last_pitcher = (
        raw.sort(["GameID", "Inning", "Top.Bottom", "PAofInning", "PitchNo"])  # ensure order across PA
           .group_by(["at_bat_id"])
           .agg([
               pl.col(pcol).drop_nulls().last().alias("pa_pitcher")
           ])
    )
    return pa_df.join(last_pitcher, on="at_bat_id", how="left")


# ──────────────────────────────────────────────────────────────────────────────
# Leverage index lookup from precomputed summary (as in dawg_plus_v1)
# ──────────────────────────────────────────────────────────────────────────────
_LEVERAGE_TABLE: Optional[pl.DataFrame] = None


def _load_leverage_table() -> Optional[pl.DataFrame]:
    global _LEVERAGE_TABLE
    if _LEVERAGE_TABLE is not None:
        return _LEVERAGE_TABLE
    paths = [
        "/Users/evanborberg/STUFF-/game_states_summary.parquet",
        "game_states_summary.parquet",
    ]
    for p in paths:
        try:
            tbl = pl.read_parquet(p)
            if "game_state" in tbl.columns and "leverage_index" in tbl.columns:
                _LEVERAGE_TABLE = tbl.select(["game_state", "leverage_index"]).unique()
                return _LEVERAGE_TABLE
        except Exception:
            continue
    return None


def leverage_from_state(
    inning: int,
    half: str,
    outs: int,
    on1: int,
    on2: int,
    on3: int,
    score_diff: float,
) -> float:
    """Return leverage_index for the selected situation using precomputed summary.
    Falls back to 1.0 if lookup not found.
    game_state format mirrors tReadCollegeDawg+: "inning_Half_Outs_runnerstate_scorediff"
    where Half is 'Top' or 'Bot'.
    """
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


def compute_stopper_leaderboard(wpa_pa: pl.DataFrame) -> pl.DataFrame:
    """Aggregate WPA by appearance (contiguous PAs with same pitcher within a game) and build a leaderboard.
    Output columns: pitcher, games, appearances, total_wpa, wpa_per_app, mean_wpa
    """
    df = wpa_pa
    # Define appearances: when pa_pitcher changes within a game/half timeline
    df = df.sort(["GameID", "Inning", "Top.Bottom", "first_pitch_no", "PAofInning"]).with_columns([
        (pl.col("pa_pitcher") != pl.col("pa_pitcher").shift(1).over("GameID")).alias("new_app")
    ])
    df = df.with_columns([
        (pl.when(pl.col("new_app")).then(1).otherwise(0)).cum_sum().over("GameID").alias("appearance_id")
    ])

    app = df.group_by(["GameID", "appearance_id", "pa_pitcher"]).agg([
        pl.col("wpa_pa").sum().alias("wpa"),
        pl.len().alias("n_pa"),
    ])

    lb = app.group_by("pa_pitcher").agg([
        pl.len().alias("appearances"),
        pl.col("wpa").sum().alias("total_wpa"),
        pl.col("wpa").mean().alias("avg_wpa"),
        pl.col("n_pa").mean().alias("pa_per_app"),
        pl.col("GameID").n_unique().alias("games"),
    ]).sort("avg_wpa", descending=True)

    # Add Stopper+ as 100 + 10*z(avg_wpa)
    mu = lb.select(pl.col("avg_wpa").mean()).item()
    sd = lb.select(pl.col("avg_wpa").std()).item() or 1.0
    lb = lb.with_columns((100.0 + 10.0 * ((pl.col("avg_wpa") - mu) / sd)).alias("Stopper+"))
    return lb.rename({"pa_pitcher": "Pitcher"})


def scenario_rank(
    wp: WPModel,
    pa_wpa: pl.DataFrame,
    inning: int,
    half: str,
    outs: int,
    on1: int,
    on2: int,
    on3: int,
    score_diff: float,
    top_k: int = 15,
    min_samples: int = 10,
) -> pl.DataFrame:
    """Rank pitchers for a given situation using historical ΔWP in similar states.
    Similarity: same half, same outs, same runner_state, score_diff within ±1 run, same inning bucket (early 1-3, mid 4-6, late 7+).
    Fallback to overall avg_wpa if insufficient samples.
    """
    runner_state = f"{int(on1)}{int(on2)}{int(on3)}"
    inning_bucket = (1 if inning <= 3 else 2 if inning <= 6 else 3)
    df = pa_wpa.with_columns([
        pl.when(pl.col("inning") <= 3).then(1).when(pl.col("inning") <= 6).then(2).otherwise(3).alias("inn_bucket"),
        # sanitized runner_state string and count of '1's
        pl.col("runner_state").cast(pl.Utf8).fill_null("000").str.replace_all(r"[^01]", "0").str.pad_start(3, "0").str.slice(0, 3).alias("runner_state"),
        pl.col("runner_state").cast(pl.Utf8).fill_null("").str.replace_all(r"[^1]", "").str.len_chars().fill_null(0).alias("runner_count"),
        pl.col("half"),
        pl.col("Outs").alias("outs"),
    ])

    # Filter to similar states
    sim = df.filter(
        (pl.col("inn_bucket") == inning_bucket) &
        (pl.col("half") == ("Top" if half.lower().startswith("t") else "Bot")) &
        (pl.col("outs") == int(outs)) &
        # match by runner_count to be less brittle without full base-state
        (pl.col("runner_count") == (int(on1) + int(on2) + int(on3))) &
        ((pl.col("score_diff") - float(score_diff)).abs() <= 1.0)
    )

    # Aggregate per pitcher
    agg = sim.group_by("pa_pitcher").agg([
        pl.col("wpa_pa").mean().alias("mean_wpa_sim"),
        pl.len().alias("n_sim")
    ])

    # Fallback overall
    overall = (
        df.group_by("pa_pitcher").agg(pl.col("wpa_pa").mean().alias("mean_wpa_overall"))
    )
    out = overall.join(agg, on="pa_pitcher", how="left").with_columns([
        pl.when((pl.col("n_sim").fill_null(0) >= min_samples) & pl.col("mean_wpa_sim").is_not_null())
          .then(pl.col("mean_wpa_sim")).otherwise(pl.col("mean_wpa_overall")).alias("score")
    ])
    out = out.with_columns(pl.col("score").fill_null(pl.col("mean_wpa_overall")))
    return out.sort("score", descending=True).rename({"pa_pitcher": "Pitcher"}).head(top_k)


def build_pipeline(raw_csv_path: str, cache_dir: str = "model") -> Tuple[WPModel, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """End-to-end: load raw CSV, build PA table, train WP model, return (wp_model, per-PA wpa table, leaderboard).
    Always trains a new model to ensure consistency.
    """
    df = pl.read_csv(
        raw_csv_path,
        infer_schema_length=20000,
        null_values=["NA", "None", "", "nan"],
        ignore_errors=True,
    )

    pa = build_pa_table(df)
    pa = attach_running_score(pa)
    pa = add_state_features(pa)
    pa = label_home_win(pa)
    # Always train a new model to ensure consistency
    wp = train_wp_model(pa)

    wpa_pa = compute_wpa_per_pa(pa, wp)
    wpa_pa = attach_pa_pitcher(wpa_pa, df)
    lb = compute_stopper_leaderboard(wpa_pa)
    return wp, wpa_pa, lb, df
