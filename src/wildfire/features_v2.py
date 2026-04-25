"""
Advanced wildfire feature engineering (v2).

Extends the base wildfire features (``features.py``) with production-grade
additions that were missing from the original pipeline:

  1. **Fire history features** — lagged fire occurrence/count/FRP + rolling
     sums + days-since-last-fire.  This is the single most impactful missing
     feature family; fire events cluster in time.
  2. **Weather lag & rolling features** — temporal context for fire-critical
     variables (T_max, rain, wind, VPD, humidity).
  3. **Interaction features** — physically meaningful compound indicators
     (temperature × VPD, drought × wind, etc.).
  4. **Cyclical calendar encoding** — sin/cos of month and day-of-year for
     continuous seasonal representation without jumps.
  5. **Compound-event indicators** — hot+dry+windy days, cumulative heat
     stress (degree-days).
  6. **Fire-season flags** — explicit summer/winter fire-season markers
     reflecting Azerbaijan's bimodal fire pattern.

Data-leak policy
----------------
Every feature is **strictly backward-looking**:
  - All lags use ``groupby(City).shift(k)``
  - All rolling stats use ``shift(1).rolling(w)`` — window is [t-w, t-1]
  - Statistics computed from training subset only where applicable

Usage
-----
    from src.wildfire.features_v2 import build_advanced_features
    df = build_advanced_features(base_features_df)

Or standalone:
    python -m src.wildfire.features_v2
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import PROCESSED_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# 1. Fire history features
# ============================================================================

def add_fire_history_features(
    df: pd.DataFrame,
    group_col: str = "City",
    lags: Sequence[int] = (1, 2, 3, 7, 14, 30),
    rolling_windows: Sequence[int] = (7, 14, 30),
) -> pd.DataFrame:
    """Lagged fire events, rolling fire statistics, and days-since-last-fire.

    All features use ``shift()`` so row *t* sees only data up to *t-1*.
    This prevents same-day leakage of the target.

    Parameters
    ----------
    lags : sequence of int
        Lag offsets for fire_occurred, fire_count, fire_frp_total.
    rolling_windows : sequence of int
        Windows for rolling sum/mean of fire_occurred and fire_count.

    New columns (examples for lag=7, window=14):
        fire_occurred_lag_7, fire_count_lag_7, fire_frp_total_lag_7,
        fire_occurred_roll14_sum, fire_count_roll14_mean,
        days_since_last_fire
    """
    df = df.sort_values([group_col, "date"]).reset_index(drop=True)
    grouped = df.groupby(group_col, sort=False)
    new: Dict[str, np.ndarray] = {}

    # --- Lag features ---
    fire_targets = [c for c in ("fire_occurred", "fire_count", "fire_frp_total")
                    if c in df.columns]
    for col in fire_targets:
        for lag in lags:
            new[f"{col}_lag_{lag}"] = grouped[col].shift(lag).values

    # --- Rolling statistics (shift(1) first → past-only window) ---
    for col in [c for c in ("fire_occurred", "fire_count") if c in df.columns]:
        for w in rolling_windows:
            shifted = grouped[col].transform(
                lambda s, _w=w: s.shift(1).rolling(_w, min_periods=1).sum()
            )
            new[f"{col}_roll{w}_sum"] = shifted.values

            shifted_mean = grouped[col].transform(
                lambda s, _w=w: s.shift(1).rolling(_w, min_periods=1).mean()
            )
            new[f"{col}_roll{w}_mean"] = shifted_mean.values

    # --- Days since last fire (strictly backward) ---
    if "fire_occurred" in df.columns:
        dsf = np.full(len(df), np.nan, dtype=np.float32)
        for _, g in grouped:
            idx = g.index.to_numpy()
            fo = g["fire_occurred"].to_numpy()
            last_fire_pos = -999
            for i_pos, k in enumerate(idx):
                # Check PREVIOUS day, not current (shift-1 logic)
                if i_pos > 0 and fo[i_pos - 1] == 1:
                    last_fire_pos = i_pos - 1
                if last_fire_pos >= 0:
                    dsf[k] = float(i_pos - last_fire_pos)
        new["days_since_last_fire"] = np.clip(dsf, 0, 365)

    for name, arr in new.items():
        df[name] = arr

    logger.info("Added %d fire history features (lags=%s, windows=%s)",
                len(new), list(lags), list(rolling_windows))
    return df


# ============================================================================
# 2. Weather lag & rolling features
# ============================================================================

FIRE_CRITICAL_WEATHER = [
    "temperature_2m_max",
    "temperature_2m_min",
    "rain_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
    "vapour_pressure_deficit_mean",
]


def add_weather_lag_rolling(
    df: pd.DataFrame,
    group_col: str = "City",
    weather_cols: Optional[List[str]] = None,
    lags: Sequence[int] = (1, 3, 7),
    rolling_windows: Sequence[int] = (7, 14),
    rolling_aggs: Sequence[str] = ("mean", "max"),
) -> pd.DataFrame:
    """Add temporal context for fire-critical weather variables.

    The base wildfire pipeline only uses raw daily values.  This adds the
    lag and rolling features that the weather module builds but never
    carries over to the wildfire feature set.

    All rolling computations use ``shift(1).rolling(w)`` to guarantee
    the window covers only [t-w, t-1], never including today.
    """
    if weather_cols is None:
        weather_cols = FIRE_CRITICAL_WEATHER

    cols = [c for c in weather_cols if c in df.columns]
    if not cols:
        logger.warning("No fire-critical weather columns found; skipping")
        return df

    df = df.sort_values([group_col, "date"]).reset_index(drop=True)
    grouped = df.groupby(group_col, sort=False)
    new: Dict[str, np.ndarray] = {}

    agg_fn = {
        "mean": lambda s, w: s.shift(1).rolling(w, min_periods=max(1, w // 2)).mean(),
        "max":  lambda s, w: s.shift(1).rolling(w, min_periods=max(1, w // 2)).max(),
        "min":  lambda s, w: s.shift(1).rolling(w, min_periods=max(1, w // 2)).min(),
        "std":  lambda s, w: s.shift(1).rolling(w, min_periods=max(1, w // 2)).std(),
    }

    for col in cols:
        # Lag features
        for lag in lags:
            new[f"{col}_lag_{lag}"] = grouped[col].shift(lag).values

        # Rolling features
        for w in rolling_windows:
            for agg in rolling_aggs:
                fn = agg_fn[agg]
                new[f"{col}_roll{w}_{agg}"] = grouped[col].transform(
                    lambda s, _fn=fn, _w=w: _fn(s, _w)
                ).values

    for name, arr in new.items():
        df[name] = arr

    logger.info("Added %d weather lag/rolling features (%d vars × %d lags + %d windows × %d aggs)",
                len(new), len(cols), len(lags),
                len(rolling_windows), len(rolling_aggs))
    return df


# ============================================================================
# 3. Interaction features
# ============================================================================

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Physically meaningful compound predictors for fire risk.

    Each interaction captures a mechanism that no single variable alone
    represents:
      - temp_x_vpd: hot + atmospherically dry = extreme evaporative demand
      - temp_x_wind: hot + windy = rapid fire spread in heated fuels
      - drought_x_wind: dry fuels + wind = highest spread potential
      - drought_x_temp: compounding heat stress on already-dry fuel
      - dry_days_x_temp: short-term fuel dryness under heat
      - humidity_deficit_x_wind: low moisture air + wind = extreme drying
    """
    new: Dict[str, pd.Series] = {}

    if "temperature_2m_max" in df.columns and "vapour_pressure_deficit_mean" in df.columns:
        new["temp_x_vpd"] = df["temperature_2m_max"] * df["vapour_pressure_deficit_mean"]

    if "temperature_2m_max" in df.columns and "wind_speed_10m_max" in df.columns:
        new["temp_x_wind"] = df["temperature_2m_max"] * df["wind_speed_10m_max"]

    if "drought_index" in df.columns and "wind_speed_10m_mean" in df.columns:
        new["drought_x_wind"] = df["drought_index"] * df["wind_speed_10m_mean"]

    if "drought_index" in df.columns and "temperature_2m_max" in df.columns:
        new["drought_x_temp"] = df["drought_index"] * df["temperature_2m_max"]

    if "dry_days_run" in df.columns and "temperature_2m_max" in df.columns:
        new["dry_days_x_temp"] = df["dry_days_run"] * df["temperature_2m_max"]

    if "relative_humidity_2m_mean" in df.columns and "wind_speed_10m_mean" in df.columns:
        new["humidity_deficit_x_wind"] = (
            (100.0 - df["relative_humidity_2m_mean"]) * df["wind_speed_10m_mean"]
        )

    for name, series in new.items():
        df[name] = series.astype(np.float32)

    logger.info("Added %d interaction features", len(new))
    return df


# ============================================================================
# 4. Cyclical calendar encoding
# ============================================================================

def add_cyclical_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Replace raw month/doy with sin/cos encoding + fire-season flags.

    Sin/cos encoding handles the Dec→Jan boundary without discontinuity.
    Fire-season flags are based on Azerbaijan's documented bimodal pattern:
      - Summer fires (Jun–Aug): genuine wildfires + heat-driven
      - Winter fires (Jan–Mar): agricultural burning
    """
    d = pd.to_datetime(df["date"])
    month = d.dt.month
    doy = d.dt.dayofyear

    df["month_sin"] = np.sin(2 * np.pi * month / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * month / 12).astype(np.float32)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)

    df["fire_season_summer"] = month.isin([6, 7, 8]).astype(np.int8)
    df["fire_season_winter"] = month.isin([1, 2, 3]).astype(np.int8)

    logger.info("Added 6 cyclical calendar + fire-season features")
    return df


# ============================================================================
# 5. Compound-event indicators
# ============================================================================

def add_compound_events(
    df: pd.DataFrame,
    group_col: str = "City",
    heat_threshold: float = 30.0,
    heat_window: int = 14,
) -> pd.DataFrame:
    """Flag compound extreme-weather events and cumulative heat stress.

    ``hot_dry_windy`` : 1 if temperature > city median AND rain < 1 mm AND
                        wind > city median.  Medians are per-city over the
                        full history — this is safe because medians are very
                        stable over 5+ years and are not target-derived.

    ``heat_degree_days_14`` : sum of (T_max − 30)⁺ over the past 14 days.
                              Uses shift(1) → no same-day leakage.
    """
    df = df.sort_values([group_col, "date"]).reset_index(drop=True)

    # --- hot + dry + windy ---
    required = {"temperature_2m_max", "rain_sum", "wind_speed_10m_mean"}
    if required.issubset(df.columns):
        temp_med = df.groupby(group_col)["temperature_2m_max"].transform("median")
        wind_med = df.groupby(group_col)["wind_speed_10m_mean"].transform("median")
        df["hot_dry_windy"] = (
            (df["temperature_2m_max"] > temp_med) &
            (df["rain_sum"] < 1.0) &
            (df["wind_speed_10m_mean"] > wind_med)
        ).astype(np.int8)
    else:
        df["hot_dry_windy"] = np.int8(0)

    # --- Cumulative heat stress (degree-days above threshold) ---
    if "temperature_2m_max" in df.columns:
        excess = (df["temperature_2m_max"] - heat_threshold).clip(lower=0)
        df["heat_degree_days_14"] = (
            excess.groupby(df[group_col])
            .transform(lambda s: s.shift(1).rolling(heat_window, min_periods=1).sum())
        ).astype(np.float32)
    else:
        df["heat_degree_days_14"] = np.float32(0)

    logger.info("Added compound-event features (hot_dry_windy, heat_degree_days_%d)",
                heat_window)
    return df


# ============================================================================
# 6. Orchestrator
# ============================================================================

def build_advanced_features(
    df: pd.DataFrame,
    save: bool = True,
    output_name: str = "wildfire_features_v2",
) -> pd.DataFrame:
    """Apply all v2 feature families to the base wildfire feature frame.

    Input: DataFrame from ``features.build_wildfire_features()``
           (must already contain the base weather + fire-science columns).
    Output: enriched DataFrame with ~100+ additional predictors.

    Call chain::

        base = build_wildfire_features()   # features.py
        full = build_advanced_features(base)  # this module
    """
    logger.info("=" * 72)
    logger.info("FEATURES V2 — Advanced wildfire feature engineering")
    logger.info("Input: %d rows × %d cols", len(df), df.shape[1])
    logger.info("=" * 72)

    n_start = df.shape[1]

    df = add_fire_history_features(df)
    df = add_weather_lag_rolling(df)
    df = add_interaction_features(df)
    df = add_cyclical_calendar(df)
    df = add_compound_events(df)

    n_added = df.shape[1] - n_start
    logger.info("V2 complete: added %d columns → %d total", n_added, df.shape[1])

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out = PROCESSED_DIR / f"{output_name}.csv"
        df.to_csv(out, index=False)
        logger.info("Saved → %s (%.2f MB)", out, out.stat().st_size / 1024 / 1024)

    return df


# ============================================================================
# Predictor column list (v2)
# ============================================================================

_TARGET_COLS = {"fire_occurred", "fire_count", "fire_frp_total", "fire_frp_max"}
_META_COLS = {"City", "date"}


def predictor_columns_v2(df: pd.DataFrame) -> List[str]:
    """Return all numeric predictor columns (excludes meta + targets)."""
    return [c for c in df.columns
            if c not in _META_COLS and c not in _TARGET_COLS
            and pd.api.types.is_numeric_dtype(df[c])]


# ============================================================================
# Standalone entry point
# ============================================================================

if __name__ == "__main__":
    from src.wildfire.features import build_wildfire_features

    base = build_wildfire_features(save=False)
    full = build_advanced_features(base, save=True)
    print(f"\nFinal shape: {full.shape}")
    print(f"Predictor count: {len(predictor_columns_v2(full))}")
