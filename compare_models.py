#!/usr/bin/env python3
"""
Model comparison for KEMET1 encroachment classifier.

Runs four variants and prints a side-by-side comparison table:
  1. RF   depth=10  (v2 baseline)
  2. RF   depth=6   + min_samples_split=5   (reduce overfit further)
  3. RF   depth=6   + SMOTE                  (oversample minority class)
  4. XGBoost        + scale_pos_weight        (calibration-friendly)

Saves the winning model (best OOF AUC) to weights/encroachment_classifier_rf.pkl.

Usage:
    python compare_models.py
    python compare_models.py --no-cv    # skip leave-one-tile-out (faster)
    python compare_models.py --no-save  # don't overwrite saved model
"""

from __future__ import annotations
import sys, argparse, pickle, time, warnings
from pathlib import Path

import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Re-use data helpers from train_classifier
from train_classifier import (
    build_dataset, FEATURE_NAMES,
    DATA_DIR, WEIGHTS_DIR, MODEL_PATH,
)

BETA = 2.0   # F2: recall weighted 2× over precision


# ══════════════════════════════════════════════════════════════════════════════
#  Model factories
# ══════════════════════════════════════════════════════════════════════════════

def make_rf(max_depth=10, min_samples_split=2, min_samples_leaf=2, smote=False):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    # Imputer must come first so NaNs are filled before scaling/SMOTE
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]
    if smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        steps.append(("smote", SMOTE(random_state=42, k_neighbors=3)))
        steps.append(("rf", RandomForestClassifier(
            n_estimators=200, max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )))
        return ImbPipeline(steps)
    else:
        steps.append(("rf", RandomForestClassifier(
            n_estimators=200, max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )))
        return Pipeline(steps)


def make_xgb():
    # Placeholder — actual factory is _xgb_factory(spw) below
    pass


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(clf, X, y, thresh=0.5):
    """Returns dict of metrics at the given threshold."""
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  precision_score, recall_score, fbeta_score,
                                  confusion_matrix)
    probs = clf.predict_proba(X)[:, 1]
    preds = (probs >= thresh).astype(int)
    has_both = len(np.unique(y)) > 1

    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    return {
        "auc":  roc_auc_score(y, probs) if has_both else float("nan"),
        "ap":   average_precision_score(y, probs) if has_both else float("nan"),
        "prec": precision_score(y, preds, zero_division=0),
        "rec":  recall_score(y, preds, zero_division=0),
        "f2":   fbeta_score(y, preds, beta=BETA, zero_division=0),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "probs": probs,
    }


def best_thresh(clf, X_val, y_val):
    """Pick threshold that maximises F2 on the validation set."""
    from sklearn.metrics import fbeta_score
    probs = clf.predict_proba(X_val)[:, 1]
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f = fbeta_score(y_val, (probs >= t).astype(int), beta=BETA, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t)


def oof_auc(make_fn, X_all, y_all, tile_ids):
    """Leave-one-tile-out CV → OOF AUC over folds that contain ≥1 positive."""
    from sklearn.metrics import roc_auc_score
    unique = np.unique(tile_ids)
    all_yt, all_yp = [], []
    for tile in unique:
        mask_v = tile_ids == tile
        mask_t = ~mask_v
        if y_all[mask_v].sum() == 0:
            continue
        clf_f = make_fn()
        clf_f.fit(X_all[mask_t], y_all[mask_t])
        probs = clf_f.predict_proba(X_all[mask_v])[:, 1]
        all_yt.extend(y_all[mask_v].tolist())
        all_yp.extend(probs.tolist())
    yt = np.array(all_yt)
    yp = np.array(all_yp)
    return roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else float("nan")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cv",   action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    print("\n" + "═"*62)
    print("  KEMET1 — Model Comparison")
    print("═"*62)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n▶  Loading features ...")
    X_tr, y_tr, tid_tr = build_dataset("train", verbose=False)
    X_va, y_va, tid_va = build_dataset("val",   verbose=False)
    X_te, y_te, tid_te = build_dataset("test",  verbose=False)

    X_all = np.concatenate([X_tr, X_va, X_te])
    y_all = np.concatenate([y_tr, y_va, y_te])
    tid_all = np.concatenate([tid_tr, tid_va, tid_te])

    neg, pos = (y_tr == 0).sum(), y_tr.sum()
    spw = neg / pos   # scale_pos_weight for XGBoost
    print(f"  Train: {len(y_tr)} pairs  ({pos} pos / {neg} neg)  "
          f"scale_pos_weight={spw:.1f}")

    # ── Define variants ───────────────────────────────────────────────────────
    variants = [
        ("RF  depth=10  (v2 baseline)",
         lambda: make_rf(max_depth=10, min_samples_split=2)),
        ("RF  depth=6   min_split=5",
         lambda: make_rf(max_depth=6,  min_samples_split=5)),
        ("RF  depth=6   + SMOTE",
         lambda: make_rf(max_depth=6,  min_samples_split=5, smote=True)),
        ("XGBoost  depth=6  scale_pos_wt",
         lambda: _xgb_factory(spw)),
    ]

    results = []

    for name, factory in variants:
        print(f"\n▶  {name} ...")
        t1 = time.time()

        # Train on train split
        clf = factory()
        clf.fit(X_tr, y_tr)

        # Threshold optimised on val F2
        thr = best_thresh(clf, X_va, y_va)

        # Metrics
        tr_m = evaluate(clf, X_tr, y_tr, thresh=thr)
        va_m = evaluate(clf, X_va, y_va, thresh=thr)
        te_m = evaluate(clf, X_te, y_te, thresh=thr)

        # CV (optional, slow)
        cv_auc_val = float("nan")
        if not args.no_cv:
            print(f"    running leave-one-tile-out CV ...")
            cv_auc_val = oof_auc(factory, X_all, y_all, tid_all)

        elapsed = time.time() - t1
        results.append({
            "name": name, "clf": clf, "thresh": thr,
            "train": tr_m, "val": va_m, "test": te_m,
            "cv_auc": cv_auc_val, "time": elapsed,
        })
        print(f"    thresh={thr:.2f}  "
              f"Val AUC={va_m['auc']:.3f}  Test AUC={te_m['auc']:.3f}  "
              f"OOF AUC={cv_auc_val:.3f}  Test Recall={te_m['rec']:.2f}  "
              f"({elapsed:.1f}s)")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n\n" + "═"*62)
    print("  COMPARISON TABLE")
    print("═"*62)
    hdr = f"  {'Model':<38}  {'Train':>5}  {'Val':>5}  {'Test':>5}  {'OOF':>5}  {'Recall':>6}  {'FP':>4}  {'thresh':>6}"
    print(hdr)
    print("  " + "─"*59)

    best_oof = max(r["cv_auc"] if not np.isnan(r["cv_auc"]) else r["test"]["auc"] for r in results)

    for r in results:
        oof = r["cv_auc"] if not np.isnan(r["cv_auc"]) else r["test"]["auc"]
        star = " ★" if abs(oof - best_oof) < 1e-4 else "  "
        print(f"  {r['name']:<38}  "
              f"{r['train']['auc']:5.3f}  "
              f"{r['val']['auc']:5.3f}  "
              f"{r['test']['auc']:5.3f}  "
              f"{oof:5.3f}  "
              f"{r['test']['rec']:6.3f}  "
              f"{r['test']['fp']:4d}  "
              f"{r['thresh']:6.2f}{star}")

    print("  " + "─"*59)
    print("  Columns: Train AUC | Val AUC | Test AUC | OOF CV AUC | "
          "Test Recall | Test FP | thresh")

    # ── Feature importance (best RF) ──────────────────────────────────────────
    best_rf = next((r for r in results if "RF" in r["name"] and
                    (np.isnan(r["cv_auc"]) or r["cv_auc"] == best_oof)), results[0])
    try:
        rf_step = best_rf["clf"].named_steps.get("rf") or best_rf["clf"].named_steps.get("xgb")
        importances = rf_step.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        print(f"\n  Top 10 features  ({best_rf['name'].strip()}):")
        for rank, idx in enumerate(top_idx, 1):
            bar = "█" * int(importances[idx] * 200)
            print(f"  {rank:2d}. {FEATURE_NAMES[idx]:<28}  {importances[idx]:.4f}  {bar}")
    except Exception:
        pass

    # ── Save best model ───────────────────────────────────────────────────────
    if not args.no_save:
        # Best by OOF AUC (fallback to test AUC if CV skipped)
        best = max(results,
                   key=lambda r: r["cv_auc"] if not np.isnan(r["cv_auc"])
                                 else r["test"]["auc"])
        bundle = {
            "model":         best["clf"],
            "threshold":     best["thresh"],
            "feature_names": FEATURE_NAMES,
            "model_name":    best["name"].strip(),
            "beta":          BETA,
            "test_auc":      best["test"]["auc"],
            "oof_auc":       best["cv_auc"],
        }
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(bundle, f)
        print(f"\n  ★ Best model: {best['name'].strip()}")
        print(f"    OOF AUC={best['cv_auc']:.3f}  Test AUC={best['test']['auc']:.3f}  "
              f"thresh={best['thresh']:.2f}")
        print(f"    Saved → {MODEL_PATH}")

    print(f"\n{'═'*62}")
    print(f"  Total time: {time.time()-t0:.1f}s")
    print(f"{'═'*62}\n")

    # Return results for dashboard update
    return results


def _xgb_factory(spw):
    """XGBoost factory with scale_pos_weight baked in."""
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss", random_state=42,
        n_jobs=-1, verbosity=0,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("xgb",     clf),
    ])


if __name__ == "__main__":
    main()
