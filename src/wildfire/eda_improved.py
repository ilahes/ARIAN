"""
Comprehensive EDA for wildfire prediction (improved).

Addresses every analysis gap identified in the original notebooks:

  1. **Correlation heatmap** — Pearson + Spearman matrix of all predictors
  2. **Multicollinearity** — VIF computation + high-correlation pair report
  3. **Feature distributions** — histograms + KDE per feature, split by class
  4. **Class imbalance** — per-city, per-year, per-month breakdown
  5. **Temporal patterns** — fire autocorrelation, clustering, trends
  6. **Train/test distribution shift** — KS test per feature
  7. **FRP intensity analysis** — fire power distributions by month/year

All functions produce return values (DataFrames) and optionally save
plots to ``reports/eda/``.

Usage::

    from src.wildfire.eda_improved import run_full_eda
    results = run_full_eda()                   # uses default paths
    results = run_full_eda(features_path=...)  # custom data

Or call individual analysis functions standalone.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Safe matplotlib import (non-GUI backend for scripts)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================
# 1. Correlation analysis
# ============================================================================

def correlation_analysis(
    df: pd.DataFrame,
    method: str = "spearman",
    target_col: str = "fire_occurred",
    top_n: int = 30,
    save_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute correlation matrix and rank features by |corr| with target.

    Parameters
    ----------
    method : 'pearson' or 'spearman'
    top_n : number of top features to plot in the heatmap

    Returns
    -------
    DataFrame with columns: feature, corr_with_target, abs_corr, rank
    """
    numeric = df.select_dtypes(include=[np.number])
    if target_col not in numeric.columns:
        logger.warning("Target %s not found; skipping correlation", target_col)
        return pd.DataFrame()

    # Full correlation matrix
    corr_matrix = numeric.corr(method=method)

    # Target correlations ranked
    target_corr = corr_matrix[target_col].drop(target_col, errors="ignore")
    result = pd.DataFrame({
        "feature": target_corr.index,
        "corr_with_target": target_corr.values,
        "abs_corr": np.abs(target_corr.values),
    }).sort_values("abs_corr", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    # --- Plot: heatmap of top features ---
    if HAS_MPL and save_dir is not None:
        _ensure_dir(save_dir)
        top_features = result.head(top_n)["feature"].tolist()
        if target_col not in top_features:
            top_features.append(target_col)
        sub = corr_matrix.loc[top_features, top_features]

        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(sub.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(sub.columns)))
        ax.set_xticklabels(sub.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(sub.index)))
        ax.set_yticklabels(sub.index, fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8, label=f"{method} correlation")
        ax.set_title(f"Top-{top_n} feature correlation ({method})")
        plt.tight_layout()
        fig.savefig(save_dir / f"correlation_heatmap_{method}.png", dpi=150)
        plt.close(fig)
        logger.info("Saved correlation heatmap → %s", save_dir)

    # --- Plot: bar chart of target correlations ---
    if HAS_MPL and save_dir is not None:
        top = result.head(top_n)
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["firebrick" if v > 0 else "steelblue"
                  for v in top["corr_with_target"]]
        ax.barh(range(len(top)), top["corr_with_target"].values, color=colors)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(f"{method} correlation with {target_col}")
        ax.set_title(f"Top-{top_n} features by |correlation| with {target_col}")
        ax.axvline(0, color="black", lw=0.5)
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        fig.savefig(save_dir / f"target_correlations_{method}.png", dpi=150)
        plt.close(fig)

    logger.info("Correlation analysis (%s): %d features ranked", method, len(result))
    return result


# ============================================================================
# 2. Multicollinearity
# ============================================================================

def multicollinearity_report(
    df: pd.DataFrame,
    threshold: float = 0.90,
    exclude: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Identify feature pairs with |Pearson r| ≥ threshold.

    High multicollinearity inflates variance in linear models and wastes
    splits in tree models.  Features with r > 0.95 are near-duplicates.

    Returns a DataFrame of (feature_1, feature_2, pearson_r) sorted by |r|.
    """
    exclude = set(exclude or ["City", "date"])
    numeric = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in exclude if c in df.columns], errors="ignore")

    corr = numeric.corr()
    pairs: List[Dict] = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                pairs.append({
                    "feature_1": corr.columns[i],
                    "feature_2": corr.columns[j],
                    "pearson_r": round(r, 4),
                })

    result = (pd.DataFrame(pairs)
              .sort_values("pearson_r", key=abs, ascending=False)
              .reset_index(drop=True))

    if save_dir is not None:
        _ensure_dir(save_dir)
        result.to_csv(save_dir / "multicollinearity_pairs.csv", index=False)

    logger.info("Multicollinearity: %d pairs with |r| ≥ %.2f", len(result), threshold)
    return result


def vif_report(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    max_features: int = 40,
    save_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Variance Inflation Factor for each feature.

    VIF > 5 = moderate collinearity, VIF > 10 = severe.
    Limited to ``max_features`` most-correlated-with-target to keep
    computation tractable.
    """
    if features is None:
        features = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ("City", "date")]

    # Limit for computational sanity
    if len(features) > max_features:
        features = features[:max_features]

    X = df[features].dropna()
    if len(X) < len(features) + 2:
        logger.warning("Too few complete rows for VIF")
        return pd.DataFrame()

    vifs: List[Dict] = []
    X_arr = X.values
    for i in range(X_arr.shape[1]):
        y = X_arr[:, i]
        others = np.delete(X_arr, i, axis=1)
        others_i = np.column_stack([np.ones(len(others)), others])
        try:
            beta = np.linalg.lstsq(others_i, y, rcond=None)[0]
            y_hat = others_i @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = max(0, 1 - ss_res / ss_tot) if ss_tot > 0 else 1.0
            vif = 1 / (1 - r2) if r2 < 1.0 else float("inf")
        except Exception:
            vif = float("inf")
            r2 = float("nan")
        vifs.append({"feature": features[i], "VIF": round(vif, 2),
                     "R2_other": round(r2, 4)})

    result = pd.DataFrame(vifs).sort_values("VIF", ascending=False).reset_index(drop=True)
    severe = (result["VIF"] > 10).sum()

    if save_dir is not None:
        _ensure_dir(save_dir)
        result.to_csv(save_dir / "vif_report.csv", index=False)

    logger.info("VIF: %d/%d features with VIF > 10", severe, len(features))
    return result


# ============================================================================
# 3. Feature distributions (class-conditional)
# ============================================================================

def feature_distributions(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    target_col: str = "fire_occurred",
    max_plots: int = 20,
    save_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Summary statistics + optional class-conditional distribution plots.

    Returns a DataFrame with per-feature stats: mean, std, skew, kurtosis,
    min, max, %NaN, and class-conditional means (mean_class0, mean_class1).
    """
    if features is None:
        features = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ("City", "date", target_col)]

    labeled = df.dropna(subset=[target_col]) if target_col in df.columns else df

    rows: List[Dict] = []
    for col in features:
        s = df[col]
        row = {
            "feature": col,
            "mean": round(s.mean(), 4) if s.notna().any() else np.nan,
            "std": round(s.std(), 4) if s.notna().any() else np.nan,
            "skew": round(s.skew(), 4) if s.notna().sum() > 2 else np.nan,
            "kurtosis": round(s.kurtosis(), 4) if s.notna().sum() > 3 else np.nan,
            "min": round(s.min(), 4) if s.notna().any() else np.nan,
            "max": round(s.max(), 4) if s.notna().any() else np.nan,
            "pct_nan": round(100 * s.isna().mean(), 2),
        }
        if target_col in labeled.columns:
            c0 = labeled.loc[labeled[target_col] == 0, col]
            c1 = labeled.loc[labeled[target_col] == 1, col]
            row["mean_no_fire"] = round(c0.mean(), 4) if c0.notna().any() else np.nan
            row["mean_fire"] = round(c1.mean(), 4) if c1.notna().any() else np.nan
            row["separation"] = round(row["mean_fire"] - row["mean_no_fire"], 4) \
                if np.isfinite(row.get("mean_fire", np.nan)) else np.nan
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("separation", key=abs,
                                             ascending=False).reset_index(drop=True)

    # --- Plot class-conditional distributions ---
    if HAS_MPL and save_dir is not None and target_col in labeled.columns:
        _ensure_dir(save_dir)
        plot_features = result.head(max_plots)["feature"].tolist()
        n_cols = 4
        n_rows = (len(plot_features) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
        axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else list(axes)

        for i, col in enumerate(plot_features):
            ax = axes_flat[i]
            c0 = labeled.loc[labeled[target_col] == 0, col].dropna()
            c1 = labeled.loc[labeled[target_col] == 1, col].dropna()
            bins = min(50, max(10, int(np.sqrt(len(c0)))))
            ax.hist(c0, bins=bins, alpha=0.6, density=True, label="no fire", color="steelblue")
            ax.hist(c1, bins=bins, alpha=0.6, density=True, label="fire", color="firebrick")
            ax.set_title(col, fontsize=8)
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=6)

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.suptitle("Class-conditional feature distributions", fontsize=12)
        plt.tight_layout()
        fig.savefig(save_dir / "feature_distributions.png", dpi=150)
        plt.close(fig)
        logger.info("Saved feature distribution plots → %s", save_dir)

    if save_dir is not None:
        _ensure_dir(save_dir)
        result.to_csv(save_dir / "feature_statistics.csv", index=False)

    logger.info("Feature distributions: %d features profiled", len(result))
    return result


# ============================================================================
# 4. Class imbalance analysis
# ============================================================================

def class_imbalance_analysis(
    df: pd.DataFrame,
    target_col: str = "fire_occurred",
    group_col: str = "City",
    save_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Break down positive-class rate by city, year, month, and city×month.

    Returns dict of DataFrames: 'by_city', 'by_year', 'by_month', 'by_city_month'.
    """
    labeled = df.dropna(subset=[target_col]).copy()
    labeled["year"] = pd.to_datetime(labeled["date"]).dt.year
    labeled["month"] = pd.to_datetime(labeled["date"]).dt.month

    by_city = (labeled.groupby(group_col)[target_col]
               .agg(n_rows="count", fire_days="sum", positive_rate="mean")
               .round(4).reset_index())

    by_year = (labeled.groupby("year")[target_col]
               .agg(n_rows="count", fire_days="sum", positive_rate="mean")
               .round(4).reset_index())

    by_month = (labeled.groupby("month")[target_col]
                .agg(n_rows="count", fire_days="sum", positive_rate="mean")
                .round(4).reset_index())

    by_city_month = (labeled.groupby([group_col, "month"])[target_col]
                     .agg(n_rows="count", fire_days="sum", positive_rate="mean")
                     .round(4).reset_index())

    outputs = {
        "by_city": by_city,
        "by_year": by_year,
        "by_month": by_month,
        "by_city_month": by_city_month,
    }

    # --- Plots ---
    if HAS_MPL and save_dir is not None:
        _ensure_dir(save_dir)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # By city
        ax = axes[0, 0]
        ax.bar(by_city[group_col], by_city["positive_rate"], color="firebrick", alpha=0.8)
        ax.set_title("Fire positive rate by city")
        ax.set_ylabel("positive rate")
        ax.grid(alpha=0.3, axis="y")

        # By year
        ax = axes[0, 1]
        ax.bar(by_year["year"].astype(str), by_year["positive_rate"],
               color="darkorange", alpha=0.8)
        ax.set_title("Fire positive rate by year")
        ax.grid(alpha=0.3, axis="y")

        # By month
        ax = axes[1, 0]
        ax.bar(by_month["month"], by_month["positive_rate"],
               color="forestgreen", alpha=0.8)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J",
                            "J", "A", "S", "O", "N", "D"])
        ax.set_title("Fire positive rate by month")
        ax.grid(alpha=0.3, axis="y")

        # City × month heatmap
        ax = axes[1, 1]
        pivot = by_city_month.pivot(index=group_col, columns="month",
                                     values="positive_rate").fillna(0)
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(12))
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J",
                            "J", "A", "S", "O", "N", "D"])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_title("Fire rate: city × month")
        plt.colorbar(im, ax=ax, shrink=0.8)

        plt.suptitle("Class imbalance analysis", fontsize=13)
        plt.tight_layout()
        fig.savefig(save_dir / "class_imbalance.png", dpi=150)
        plt.close(fig)

    if save_dir is not None:
        _ensure_dir(save_dir)
        for name, frame in outputs.items():
            frame.to_csv(save_dir / f"imbalance_{name}.csv", index=False)

    logger.info("Class imbalance: overall %.1f%% positive (%d fire-days / %d total)",
                100 * labeled[target_col].mean(),
                int(labeled[target_col].sum()), len(labeled))
    return outputs


# ============================================================================
# 5. Temporal patterns
# ============================================================================

def fire_autocorrelation(
    df: pd.DataFrame,
    max_lag: int = 30,
    group_col: str = "City",
    save_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Autocorrelation of fire_occurred at lags 1..max_lag per city.

    Quantifies temporal clustering: if ACF at lag-1 is high, fire-history
    features will be very predictive.
    """
    if "fire_occurred" not in df.columns:
        return pd.DataFrame()

    work = df.dropna(subset=["fire_occurred"]).sort_values([group_col, "date"])
    rows: List[Dict] = []

    for city, g in work.groupby(group_col, sort=False):
        s = g["fire_occurred"].values.astype(float)
        n = len(s)
        if s.var() == 0 or n < max_lag + 5:
            continue
        for lag in range(1, max_lag + 1):
            acf = np.corrcoef(s[lag:], s[:-lag])[0, 1]
            rows.append({"City": city, "lag": lag,
                         "autocorrelation": round(acf, 4)})

    result = pd.DataFrame(rows)

    if HAS_MPL and save_dir is not None and not result.empty:
        _ensure_dir(save_dir)
        fig, ax = plt.subplots(figsize=(12, 5))
        for city, g in result.groupby("City"):
            ax.plot(g["lag"], g["autocorrelation"], "o-", ms=3, lw=1.2,
                    alpha=0.8, label=city)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("lag (days)")
        ax.set_ylabel("autocorrelation")
        ax.set_title("Fire occurrence autocorrelation by city")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(save_dir / "fire_autocorrelation.png", dpi=150)
        plt.close(fig)

    if save_dir is not None:
        _ensure_dir(save_dir)
        result.to_csv(save_dir / "fire_autocorrelation.csv", index=False)

    logger.info("Fire ACF: %d cities, max_lag=%d", result["City"].nunique(), max_lag)
    return result


def fire_clustering(
    df: pd.DataFrame,
    group_col: str = "City",
    save_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Analyse consecutive fire-day run lengths and inter-fire gaps."""
    if "fire_occurred" not in df.columns:
        return pd.DataFrame()

    work = df.dropna(subset=["fire_occurred"]).sort_values([group_col, "date"])
    stats: List[Dict] = []

    for city, g in work.groupby(group_col, sort=False):
        fo = g["fire_occurred"].values

        # Run lengths
        runs, current = [], 0
        for v in fo:
            if v == 1:
                current += 1
            else:
                if current > 0:
                    runs.append(current)
                current = 0
        if current > 0:
            runs.append(current)

        # Gap lengths
        gaps, current, seen_fire = [], 0, False
        for v in fo:
            if v == 0:
                current += 1
            else:
                if seen_fire and current > 0:
                    gaps.append(current)
                current = 0
                seen_fire = True

        stats.append({
            "City": city,
            "n_fire_days": int(fo.sum()),
            "n_runs": len(runs),
            "mean_run": round(np.mean(runs), 2) if runs else 0,
            "max_run": max(runs) if runs else 0,
            "pct_multiday": round(100 * sum(r > 1 for r in runs) / max(len(runs), 1), 1),
            "mean_gap": round(np.mean(gaps), 1) if gaps else 0,
            "median_gap": round(np.median(gaps), 1) if gaps else 0,
        })

    result = pd.DataFrame(stats)
    if save_dir is not None:
        _ensure_dir(save_dir)
        result.to_csv(save_dir / "fire_clustering.csv", index=False)

    logger.info("Fire clustering: %d cities analysed", len(result))
    return result


def fire_temporal_trend(
    df: pd.DataFrame,
    group_col: str = "City",
    save_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Per-city per-year fire rate and weather summary."""
    work = df.dropna(subset=["fire_occurred"]).copy()
    work["year"] = pd.to_datetime(work["date"]).dt.year

    agg_dict = {"fire_occurred": ["count", "sum", "mean"]}
    for col in ("temperature_2m_max", "rain_sum"):
        if col in work.columns:
            agg_dict[col] = "mean"

    result = work.groupby([group_col, "year"]).agg(agg_dict)
    result.columns = ["_".join(c).strip("_") for c in result.columns]
    result = result.rename(columns={
        "fire_occurred_count": "n_days",
        "fire_occurred_sum": "fire_days",
        "fire_occurred_mean": "fire_rate",
    }).reset_index()

    if HAS_MPL and save_dir is not None:
        _ensure_dir(save_dir)
        fig, ax = plt.subplots(figsize=(10, 5))
        for city, g in result.groupby(group_col):
            ax.plot(g["year"], g["fire_rate"], "o-", lw=1.5, ms=5, label=city)
        ax.set_xlabel("year")
        ax.set_ylabel("fire day rate")
        ax.set_title("Annual fire-day rate per city (declining trend?)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(save_dir / "fire_temporal_trend.png", dpi=150)
        plt.close(fig)

    if save_dir is not None:
        _ensure_dir(save_dir)
        result.to_csv(save_dir / "fire_trend.csv", index=False)

    logger.info("Fire trend: %d city-years", len(result))
    return result


# ============================================================================
# 6. Train/test distribution shift
# ============================================================================

def distribution_shift(
    df: pd.DataFrame,
    split_date: str = "2024-01-01",
    features: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """KS-test for covariate shift between train and test splits.

    A high KS statistic + low p-value means the feature distribution differs
    significantly between splits, which may degrade model performance.
    """
    from scipy.stats import ks_2samp

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    train = df[df["date"] < split_date]
    test = df[df["date"] >= split_date]

    if features is None:
        features = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ("year",)]

    rows: List[Dict] = []
    for col in features:
        tr = train[col].dropna()
        te = test[col].dropna()
        if len(tr) < 10 or len(te) < 10:
            continue
        ks_stat, ks_p = ks_2samp(tr, te)
        rows.append({
            "feature": col,
            "train_mean": round(tr.mean(), 4),
            "test_mean": round(te.mean(), 4),
            "shift_pct": round(100 * (te.mean() - tr.mean()) / (abs(tr.mean()) + 1e-8), 2),
            "train_std": round(tr.std(), 4),
            "test_std": round(te.std(), 4),
            "ks_statistic": round(ks_stat, 4),
            "ks_p_value": round(ks_p, 6),
            "significant": ks_p < 0.01,
        })

    result = (pd.DataFrame(rows)
              .sort_values("ks_statistic", ascending=False)
              .reset_index(drop=True))

    if HAS_MPL and save_dir is not None:
        _ensure_dir(save_dir)
        top = result.head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["firebrick" if s else "steelblue" for s in top["significant"]]
        ax.barh(range(len(top)), top["ks_statistic"], color=colors)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("KS statistic")
        ax.set_title(f"Distribution shift: train (< {split_date}) vs test")
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        fig.savefig(save_dir / "distribution_shift.png", dpi=150)
        plt.close(fig)

    if save_dir is not None:
        _ensure_dir(save_dir)
        result.to_csv(save_dir / "distribution_shift.csv", index=False)

    n_sig = result["significant"].sum() if len(result) else 0
    logger.info("Distribution shift: %d/%d features differ significantly (p<0.01)",
                n_sig, len(result))
    return result


# ============================================================================
# 7. FRP intensity analysis
# ============================================================================

def frp_analysis(
    firms_path: Optional[Path] = None,
    save_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """FRP distribution by month and year for vegetation fires.

    Understanding intensity patterns helps separate low-FRP agricultural
    fires from high-FRP wildfires.
    """
    firms_path = firms_path or (INTERIM_DIR / "firms.csv")
    if not firms_path.exists():
        logger.warning("FIRMS file not found: %s", firms_path)
        return pd.DataFrame()

    firms = pd.read_csv(firms_path, parse_dates=["acq_date"])
    veg = firms[(firms["type"] == 0) & (firms["confidence"] != "l")].copy()
    veg["month"] = veg["acq_date"].dt.month
    veg["year"] = veg["acq_date"].dt.year

    def _stats(g: pd.DataFrame, label: str) -> Dict:
        return {
            "group": label,
            "n": len(g),
            "frp_mean": round(g["frp"].mean(), 2),
            "frp_median": round(g["frp"].median(), 2),
            "frp_p90": round(g["frp"].quantile(0.90), 2),
            "frp_p99": round(g["frp"].quantile(0.99), 2),
            "frp_max": round(g["frp"].max(), 2),
        }

    rows = [_stats(veg, "ALL")]
    for m, g in veg.groupby("month"):
        rows.append(_stats(g, f"month_{m:02d}"))
    for y, g in veg.groupby("year"):
        rows.append(_stats(g, f"year_{y}"))

    result = pd.DataFrame(rows)

    if HAS_MPL and save_dir is not None:
        _ensure_dir(save_dir)
        monthly = result[result["group"].str.startswith("month_")]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.bar(range(1, 13), monthly["frp_mean"], alpha=0.8, color="darkorange")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J",
                            "J", "A", "S", "O", "N", "D"])
        ax.set_ylabel("mean FRP (MW)")
        ax.set_title("Mean fire intensity by month")
        ax.grid(alpha=0.3, axis="y")

        ax = axes[1]
        ax.hist(veg["frp"], bins=100, alpha=0.8, color="firebrick", log=True)
        ax.set_xlabel("FRP (MW)")
        ax.set_ylabel("count (log)")
        ax.set_title("FRP distribution (all vegetation fires)")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_dir / "frp_analysis.png", dpi=150)
        plt.close(fig)

    if save_dir is not None:
        _ensure_dir(save_dir)
        result.to_csv(save_dir / "frp_analysis.csv", index=False)

    logger.info("FRP analysis: %d vegetation hotspots, mean=%.2f MW",
                len(veg), veg["frp"].mean())
    return result


# ============================================================================
# 8. Full EDA pipeline
# ============================================================================

def run_full_eda(
    features_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    split_date: str = "2024-01-01",
) -> Dict[str, pd.DataFrame]:
    """Execute the complete EDA suite and save all reports + plots.

    Parameters
    ----------
    features_path
        Path to the wildfire feature CSV (default: processed/wildfire_features.csv).
    output_dir
        Directory for reports + plots (default: reports/eda/).
    split_date
        Train/test split boundary for distribution-shift analysis.

    Returns
    -------
    Dict mapping analysis name → result DataFrame.
    """
    features_path = features_path or (PROCESSED_DIR / "wildfire_features.csv")
    output_dir = output_dir or (REPORTS_DIR / "eda")
    output_dir = _ensure_dir(output_dir)

    logger.info("=" * 72)
    logger.info("IMPROVED EDA — comprehensive wildfire data analysis")
    logger.info("Data: %s", features_path)
    logger.info("Reports: %s", output_dir)
    logger.info("=" * 72)

    df = pd.read_csv(features_path, parse_dates=["date"])

    outputs: Dict[str, pd.DataFrame] = {}

    # 1. Correlation
    outputs["corr_spearman"] = correlation_analysis(
        df, method="spearman", save_dir=output_dir)
    outputs["corr_pearson"] = correlation_analysis(
        df, method="pearson", save_dir=output_dir)

    # 2. Multicollinearity
    outputs["multicollinearity"] = multicollinearity_report(
        df, save_dir=output_dir)
    # VIF on top correlated features
    top_feats = outputs["corr_spearman"].head(30)["feature"].tolist()
    outputs["vif"] = vif_report(df, features=top_feats, save_dir=output_dir)

    # 3. Feature distributions
    outputs["distributions"] = feature_distributions(
        df, save_dir=output_dir)

    # 4. Class imbalance
    imb = class_imbalance_analysis(df, save_dir=output_dir)
    outputs.update({f"imbalance_{k}": v for k, v in imb.items()})

    # 5. Temporal patterns
    outputs["fire_acf"] = fire_autocorrelation(df, save_dir=output_dir)
    outputs["fire_clusters"] = fire_clustering(df, save_dir=output_dir)
    outputs["fire_trend"] = fire_temporal_trend(df, save_dir=output_dir)

    # 6. Distribution shift
    outputs["shift"] = distribution_shift(
        df, split_date=split_date, save_dir=output_dir)

    # 7. FRP
    outputs["frp"] = frp_analysis(save_dir=output_dir)

    logger.info("=" * 72)
    logger.info("EDA complete: %d analyses, %d files in %s",
                len(outputs), len(list(output_dir.glob("*"))), output_dir)
    logger.info("=" * 72)

    return outputs


if __name__ == "__main__":
    run_full_eda()
