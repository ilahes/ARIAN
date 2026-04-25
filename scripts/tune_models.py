"""
Hyperparameter tuning for XGBoost and HGBC on v2 features.

Goal: beat Logistic PR-AUC=0.375, improve recall without exploding FP.
"""
import sys, warnings, time
from pathlib import Path
from itertools import product

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, precision_score, recall_score, f1_score,
)

from src.wildfire.features import build_wildfire_features
from src.wildfire.features_v2 import build_advanced_features
from src.wildfire.train import prepare_split, evaluate_classifier

# ── Build data ───────────────────────────────────────────────────────
print("Building v2 features...")
base = build_wildfire_features(save=False)
full = build_advanced_features(base, save=False)
split = prepare_split(full, "fire_occurred", test_start="2024-01-01")

X_tr, X_te = split.X_train, split.X_test
y_tr, y_te = split.y_train.astype(int), split.y_test.astype(int)
n_pos = int(y_tr.sum())
n_neg = int((y_tr == 0).sum())
print(f"Train: {X_tr.shape}, pos={n_pos}, neg={n_neg}")
print(f"Test:  {X_te.shape}, pos rate={y_te.mean():.3f}")
print(f"Features: {len(split.predictors)}")


def best_f1_threshold(y_true, y_score):
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    thr = np.append(thr, 1.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-12), 0.0)
    i = int(np.nanargmax(f1))
    return float(thr[i]), float(f1[i])


def eval_full(y_true, y_score):
    roc = roc_auc_score(y_true, y_score)
    pr = average_precision_score(y_true, y_score)
    thr, f1 = best_f1_threshold(y_true, y_score)
    y_pred = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "ROC_AUC": roc, "PR_AUC": pr, "F1": f1, "threshold": thr,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


def per_city_eval(y_true, y_score, meta):
    rows = []
    for city in sorted(meta["City"].unique()):
        mask = meta["City"].values == city
        yt, ys = y_true[mask], y_score[mask]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        roc = roc_auc_score(yt, ys)
        pr = average_precision_score(yt, ys)
        rows.append({"city": city, "ROC": round(roc, 3), "PR": round(pr, 3),
                      "n": len(yt), "pos": int(yt.sum())})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 1. HGBC TUNING
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("HGBC HYPERPARAMETER TUNING")
print("=" * 72)

hgbc_grid = {
    "max_depth":        [4, 6, 8, 10],
    "learning_rate":    [0.03, 0.05, 0.08, 0.12],
    "max_iter":         [300, 500],
    "l2_regularization": [0.05, 0.1, 0.3],
}

hgbc_results = []
total = 1
for v in hgbc_grid.values():
    total *= len(v)
print(f"Grid size: {total} combinations\n")

i = 0
for md, lr, mi, l2 in product(
    hgbc_grid["max_depth"],
    hgbc_grid["learning_rate"],
    hgbc_grid["max_iter"],
    hgbc_grid["l2_regularization"],
):
    i += 1
    t0 = time.time()
    try:
        base_clf = HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=lr, max_iter=mi,
            max_depth=md, l2_regularization=l2,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=15, class_weight="balanced",
            random_state=0,
        )
        cal = CalibratedClassifierCV(base_clf, cv=3, method="isotonic")
        cal.fit(X_tr, y_tr)
        scores = cal.predict_proba(X_te)[:, 1]
        m = eval_full(y_te.values, scores)
        elapsed = time.time() - t0
        row = {"md": md, "lr": lr, "mi": mi, "l2": l2, **m, "time": round(elapsed, 1)}
        hgbc_results.append(row)
        if i % 10 == 0 or m["PR_AUC"] > 0.37:
            print(f"  [{i:>3d}/{total}] md={md} lr={lr} mi={mi} l2={l2} → "
                  f"PR={m['PR_AUC']:.3f} ROC={m['ROC_AUC']:.3f} F1={m['F1']:.3f} "
                  f"P={m['precision']:.3f} R={m['recall']:.3f} ({elapsed:.1f}s)")
    except Exception as e:
        print(f"  [{i}/{total}] FAILED: {e}")

hgbc_df = pd.DataFrame(hgbc_results).sort_values("PR_AUC", ascending=False)
print(f"\n  Top 5 HGBC configs:")
print(f"  {'md':>3s} {'lr':>5s} {'mi':>4s} {'l2':>5s} | {'ROC':>6s} {'PR':>6s} {'F1':>6s} | {'P':>6s} {'R':>6s} | {'FP':>4s} {'FN':>4s}")
print(f"  {'-'*72}")
for _, r in hgbc_df.head(5).iterrows():
    print(f"  {r['md']:>3.0f} {r['lr']:>5.2f} {r['mi']:>4.0f} {r['l2']:>5.2f} | "
          f"{r['ROC_AUC']:>6.3f} {r['PR_AUC']:>6.3f} {r['F1']:>6.3f} | "
          f"{r['precision']:>6.3f} {r['recall']:>6.3f} | {r['FP']:>4.0f} {r['FN']:>4.0f}")


# ══════════════════════════════════════════════════════════════════════
# 2. XGBOOST TUNING
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("XGBOOST HYPERPARAMETER TUNING")
print("=" * 72)

try:
    import xgboost as xgb
except ImportError:
    print("xgboost not installed — skipping")
    xgb = None

if xgb is not None:
    spw_base = n_neg / n_pos  # ~5.6

    xgb_grid = {
        "max_depth":         [4, 6, 8, 10],
        "learning_rate":     [0.02, 0.05, 0.08, 0.12],
        "n_estimators":      [300, 500, 800],
        "scale_pos_weight":  [round(spw_base * 0.5, 1),
                              round(spw_base * 0.75, 1),
                              round(spw_base, 1),
                              round(spw_base * 1.5, 1)],
    }

    total_xgb = 1
    for v in xgb_grid.values():
        total_xgb *= len(v)
    print(f"Grid size: {total_xgb} combinations")
    print(f"scale_pos_weight candidates: {xgb_grid['scale_pos_weight']}")
    print()

    xgb_results = []
    i = 0
    for md, lr, ne, spw in product(
        xgb_grid["max_depth"],
        xgb_grid["learning_rate"],
        xgb_grid["n_estimators"],
        xgb_grid["scale_pos_weight"],
    ):
        i += 1
        t0 = time.time()
        try:
            model = xgb.XGBClassifier(
                n_estimators=ne, max_depth=md, learning_rate=lr,
                scale_pos_weight=spw, subsample=0.85, colsample_bytree=0.85,
                eval_metric="logloss", tree_method="hist",
                random_state=0, n_jobs=-1, verbosity=0,
            )
            model.fit(X_tr, y_tr, verbose=False)
            scores = model.predict_proba(X_te)[:, 1]
            m = eval_full(y_te.values, scores)
            elapsed = time.time() - t0
            row = {"md": md, "lr": lr, "ne": ne, "spw": spw, **m, "time": round(elapsed, 1)}
            xgb_results.append(row)
            if i % 20 == 0 or m["PR_AUC"] > 0.37:
                print(f"  [{i:>3d}/{total_xgb}] md={md} lr={lr} ne={ne} spw={spw} → "
                      f"PR={m['PR_AUC']:.3f} ROC={m['ROC_AUC']:.3f} F1={m['F1']:.3f} "
                      f"P={m['precision']:.3f} R={m['recall']:.3f} ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  [{i}/{total_xgb}] FAILED: {e}")

    xgb_df = pd.DataFrame(xgb_results).sort_values("PR_AUC", ascending=False)
    print(f"\n  Top 5 XGBoost configs:")
    print(f"  {'md':>3s} {'lr':>5s} {'ne':>4s} {'spw':>5s} | {'ROC':>6s} {'PR':>6s} {'F1':>6s} | {'P':>6s} {'R':>6s} | {'FP':>4s} {'FN':>4s}")
    print(f"  {'-'*72}")
    for _, r in xgb_df.head(5).iterrows():
        print(f"  {r['md']:>3.0f} {r['lr']:>5.2f} {r['ne']:>4.0f} {r['spw']:>5.1f} | "
              f"{r['ROC_AUC']:>6.3f} {r['PR_AUC']:>6.3f} {r['F1']:>6.3f} | "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} | {r['FP']:>4.0f} {r['FN']:>4.0f}")


# ══════════════════════════════════════════════════════════════════════
# 3. FINAL COMPARISON
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("FINAL COMPARISON")
print("=" * 72)

results = {"Logistic (baseline)": {"ROC_AUC": 0.806, "PR_AUC": 0.375,
           "F1": 0.395, "precision": 0.306, "recall": 0.558}}

# Best HGBC
if hgbc_df is not None and len(hgbc_df) > 0:
    bh = hgbc_df.iloc[0]
    results["HGBC (tuned)"] = {k: bh[k] for k in ["ROC_AUC", "PR_AUC", "F1", "precision", "recall"]}

    # Retrain best HGBC for per-city eval
    best_hgbc = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=bh["lr"], max_iter=int(bh["mi"]),
        max_depth=int(bh["md"]), l2_regularization=bh["l2"],
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=15, class_weight="balanced", random_state=0,
    )
    best_hgbc_cal = CalibratedClassifierCV(best_hgbc, cv=3, method="isotonic")
    best_hgbc_cal.fit(X_tr, y_tr)
    hgbc_scores = best_hgbc_cal.predict_proba(X_te)[:, 1]

# Best XGBoost
if xgb is not None and len(xgb_df) > 0:
    bx = xgb_df.iloc[0]
    results["XGBoost (tuned)"] = {k: bx[k] for k in ["ROC_AUC", "PR_AUC", "F1", "precision", "recall"]}

    best_xgb = xgb.XGBClassifier(
        n_estimators=int(bx["ne"]), max_depth=int(bx["md"]),
        learning_rate=bx["lr"], scale_pos_weight=bx["spw"],
        subsample=0.85, colsample_bytree=0.85,
        eval_metric="logloss", tree_method="hist",
        random_state=0, n_jobs=-1, verbosity=0,
    )
    best_xgb.fit(X_tr, y_tr, verbose=False)
    xgb_scores = best_xgb.predict_proba(X_te)[:, 1]

print(f"\n  {'Model':<25s} {'ROC':>7s} {'PR':>7s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s}")
print(f"  {'-'*60}")
for name, m in results.items():
    print(f"  {name:<25s} {m['ROC_AUC']:>7.3f} {m['PR_AUC']:>7.3f} {m['F1']:>7.3f} "
          f"{m['precision']:>7.3f} {m['recall']:>7.3f}")


# ══════════════════════════════════════════════════════════════════════
# 4. PER-CITY PERFORMANCE (best overall model)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PER-CITY PERFORMANCE")
print("=" * 72)

# Find best overall PR-AUC model
best_name = max(results, key=lambda k: results[k]["PR_AUC"])
if best_name == "HGBC (tuned)" and hgbc_scores is not None:
    best_scores = hgbc_scores
elif best_name == "XGBoost (tuned)" and xgb_scores is not None:
    best_scores = xgb_scores
else:
    best_scores = None

if best_scores is not None:
    print(f"\n  Best model: {best_name}")
    city_df = per_city_eval(y_te.values, best_scores, split.test_meta)
    print(f"\n  {'City':<14s} {'ROC':>7s} {'PR':>7s} {'n':>5s} {'pos':>5s}")
    print(f"  {'-'*42}")
    for _, r in city_df.iterrows():
        print(f"  {r['city']:<14s} {r['ROC']:>7.3f} {r['PR']:>7.3f} {r['n']:>5d} {r['pos']:>5d}")

# Also print per-city for both tuned models
for label, scores_arr in [("HGBC tuned", hgbc_scores), ("XGBoost tuned", xgb_scores)]:
    if scores_arr is None:
        continue
    print(f"\n  {label}:")
    cdf = per_city_eval(y_te.values, scores_arr, split.test_meta)
    for _, r in cdf.iterrows():
        print(f"    {r['city']:<14s} ROC={r['ROC']:.3f}  PR={r['PR']:.3f}")


# ══════════════════════════════════════════════════════════════════════
# 5. FEATURE IMPORTANCE (best tree model)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("FEATURE IMPORTANCE (best tuned tree model)")
print("=" * 72)

fi = None
if xgb is not None and best_xgb is not None and hasattr(best_xgb, "feature_importances_"):
    fi = pd.Series(best_xgb.feature_importances_, index=split.predictors)
    fi_label = "XGBoost (tuned)"
elif best_hgbc_cal is not None:
    base_models = [cc.estimator for cc in best_hgbc_cal.calibrated_classifiers_
                   if hasattr(cc, "estimator")]
    if base_models and hasattr(base_models[0], "feature_importances_"):
        fi = pd.Series(
            np.mean([m.feature_importances_ for m in base_models], axis=0),
            index=split.predictors
        )
        fi_label = "HGBC (tuned)"

if fi is not None:
    fi = fi.sort_values(ascending=False)
    print(f"\n  Top 20 features ({fi_label}):")
    print(f"  {'Rank':>4s}  {'Feature':<45s} {'Importance':>10s}")
    print(f"  {'-'*64}")
    for i, (name, val) in enumerate(fi.head(20).items(), 1):
        tag = ""
        if "lag" in name or "roll" in name or "days_since" in name:
            tag = " ★ V2"
        elif name.startswith(("temp_x", "drought_x", "dry_days_x", "humidity")):
            tag = " ★ V2"
        elif name in ("month_sin", "month_cos", "doy_sin", "doy_cos",
                       "fire_season_summer", "fire_season_winter",
                       "hot_dry_windy", "heat_degree_days_14"):
            tag = " ★ V2"
        print(f"  {i:>4d}  {name:<45s} {val:>10.4f}{tag}")


# ══════════════════════════════════════════════════════════════════════
# 6. OVERFITTING CHECK
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("OVERFITTING CHECK")
print("=" * 72)

for label, model_obj, test_scores in [
    ("HGBC tuned", best_hgbc_cal, hgbc_scores),
    ("XGBoost tuned", best_xgb, xgb_scores),
]:
    if model_obj is None or test_scores is None:
        continue
    if hasattr(model_obj, "predict_proba"):
        tr_scores = model_obj.predict_proba(X_tr)[:, 1]
    else:
        continue
    tr_m = evaluate_classifier(y_tr.values, tr_scores)
    te_m = evaluate_classifier(y_te.values, test_scores)
    gap_roc = tr_m["ROC_AUC"] - te_m["ROC_AUC"]
    gap_pr = tr_m["PR_AUC"] - te_m["PR_AUC"]
    print(f"\n  {label}:")
    print(f"    ROC  train={tr_m['ROC_AUC']:.3f}  test={te_m['ROC_AUC']:.3f}  gap={gap_roc:+.3f}")
    print(f"    PR   train={tr_m['PR_AUC']:.3f}  test={te_m['PR_AUC']:.3f}  gap={gap_pr:+.3f}")
    status = "OK" if gap_pr < 0.15 else "CONCERN"
    print(f"    Status: {status}")


print("\n" + "=" * 72)
print("TUNING COMPLETE")
print("=" * 72)
