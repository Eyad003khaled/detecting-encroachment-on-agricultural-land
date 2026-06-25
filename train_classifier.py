#!/usr/bin/env python3
"""
Train a Random Forest encroachment classifier on KEMET1.

The KEMET1 TIFFs already contain 6 pre-computed spectral index bands:
    Band 0: NDVI   (vegetation health — drops when vegetation is lost)
    Band 1: NDBI   (built-up index — rises when buildings appear)
    Band 2: MNDWI  (water)
    Band 3: SAVI   (soil-adjusted vegetation)
    Band 4: BSI    (bare soil index)
    Band 5: NDWI   (water 2)

For each (T_before, T_after) pair we extract per-band statistics and
their temporal differences, then train a Random Forest binary classifier:
    label = 1  →  T_after image shows encroachment  (pos)
    label = 0  →  no encroachment                    (neg)

Usage:
    python train_classifier.py
    python train_classifier.py --no-save              # skip saving model
    python train_classifier.py --max-depth 10         # explicit depth cap
    python train_classifier.py --no-cv                # skip cross-validation
"""

from __future__ import annotations
import re, sys, argparse, pickle, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Band legend (0-based) ─────────────────────────────────────────────────────
BANDS = ["NDVI", "NDBI", "MNDWI", "SAVI", "BSI", "NDWI"]
N_BANDS = len(BANDS)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = PROJECT_ROOT / "data" / "KEMET1_split"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)
MODEL_PATH  = WEIGHTS_DIR / "encroachment_classifier_rf.pkl"


# ══════════════════════════════════════════════════════════════════════════════
#  Feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def _read(path: Path) -> np.ndarray:
    """Read GeoTIFF → float32 array (bands, H, W)."""
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def _align(t2: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Resize t2 to match t1 spatial shape if they differ (common in real tiles)."""
    if t2.shape[1:] == t1.shape[1:]:
        return t2
    import cv2
    h, w = t1.shape[1], t1.shape[2]
    return np.stack(
        [cv2.resize(t2[b], (w, h), interpolation=cv2.INTER_LINEAR)
         for b in range(t2.shape[0])],
        axis=0,
    )


def extract_features(t1_path: Path, t2_path: Path,
                     t1_is_pos: bool = False) -> np.ndarray:
    """
    Extract a fixed-length feature vector from a (T1, T2) image pair.

    Features (per band × 6 stats = 36) + 11 derived + 1 prior = 48 total:
        For each band: T1_mean, T1_std, T2_mean, T2_std, diff_mean, diff_std
        Global:        mean_abs_change, frac_changed_5pct, frac_changed_10pct,
                       ndvi_drop_mean, ndvi_drop_pct_5, ndvi_drop_pct_10,
                       ndbi_rise_mean, ndbi_rise_pct_5, ndbi_rise_pct_10,
                       bsi_rise_mean,  bsi_rise_pct_5
        Prior:         t1_is_pos  (1 if T1 was already encroached, else 0)
                       ↳ suppresses pos→pos FPs — model learns that if T1 is
                         already encroached the "change" seen is residual, not new.
    """
    t1 = _read(t1_path)
    t2 = _read(t2_path)
    t2 = _align(t2, t1)

    diff = t2 - t1   # positive = index increased, negative = decreased

    feats = []

    # Per-band stats (6 bands × 6 stats = 36 features)
    for b in range(N_BANDS):
        feats += [
            float(t1[b].mean()),
            float(t1[b].std()),
            float(t2[b].mean()),
            float(t2[b].std()),
            float(diff[b].mean()),
            float(diff[b].std()),
        ]

    # Change magnitude
    abs_diff = np.abs(diff)
    feats += [
        float(abs_diff.mean()),                         # overall change magnitude
        float((abs_diff > 0.05).mean()),                # fraction of pixels with any change
        float((abs_diff > 0.10).mean()),                # fraction with moderate change
    ]

    # NDVI drop (band 0 falls → vegetation lost)
    ndvi_drop = -diff[0]   # positive = NDVI fell (bad)
    feats += [
        float(np.clip(ndvi_drop, 0, None).mean()),      # mean vegetation loss
        float((ndvi_drop > 0.05).mean()),                # pct pixels losing veg
        float((ndvi_drop > 0.10).mean()),
    ]

    # NDBI rise (band 1 rises → more built-up)
    ndbi_rise = diff[1]   # positive = more buildings
    feats += [
        float(np.clip(ndbi_rise, 0, None).mean()),      # mean built-up increase
        float((ndbi_rise > 0.05).mean()),                # pct pixels with new built-up
        float((ndbi_rise > 0.10).mean()),
    ]

    # BSI rise (band 4 rises → more bare soil, early sign of clearing)
    bsi_rise = diff[4]
    feats += [
        float(np.clip(bsi_rise, 0, None).mean()),
        float((bsi_rise > 0.05).mean()),
    ]

    # Prior-label feature — key fix for pos→pos false positives
    feats.append(1.0 if t1_is_pos else 0.0)

    return np.array(feats, dtype=np.float32)


# Feature names (for importance display)
FEATURE_NAMES = []
for bname in BANDS:
    for stat in ["T1_mean", "T1_std", "T2_mean", "T2_std", "diff_mean", "diff_std"]:
        FEATURE_NAMES.append(f"{bname}_{stat}")
FEATURE_NAMES += [
    "abs_change_mean", "frac_changed_5pct", "frac_changed_10pct",
    "ndvi_drop_mean", "ndvi_drop_pct_5", "ndvi_drop_pct_10",
    "ndbi_rise_mean", "ndbi_rise_pct_5", "ndbi_rise_pct_10",
    "bsi_rise_mean", "bsi_rise_pct_5",
    "t1_is_pos",   # prior-label: 1 if T1 was already encroached
]


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset builder
# ══════════════════════════════════════════════════════════════════════════════

def parse_filename(fname: str):
    """T{period}_{year}_tile_{id}_{label}.tif → (period, year, tile_id, label)"""
    m = re.match(r"T(\d+)_(\d{4})_tile_(\d+)_(pos|neg)\.tif$", fname, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)


def build_pairs(split_dir: Path):
    """
    Group tiles by ID, create consecutive pairs (T1→T2, T2→T3, T3→T4).

    Label = 1 only when a TRANSITION actually occurred (neg→pos).
    pos→pos and neg→neg pairs both get label=0 because no encroachment
    change happened between those two images — their difference features
    will be near-zero, so labelling them as 1 would confuse the classifier.

    Pair labels:
        neg → pos  →  1  (encroachment appeared this period)
        neg → neg  →  0  (no change)
        pos → pos  →  0  (already encroached, no new change to detect)
        pos → neg  →  0  (recovery / mislabel edge case)
    """
    tiles: dict[int, dict[int, tuple]] = defaultdict(dict)
    for f in sorted(split_dir.glob("*.tif")):
        parsed = parse_filename(f.name)
        if parsed is None:
            continue
        period, year, tile_id, label = parsed
        tiles[tile_id][period] = (f, label)

    pairs = []
    for tile_id in sorted(tiles):
        periods = sorted(tiles[tile_id])
        for i in range(len(periods) - 1):
            p1, p2 = periods[i], periods[i + 1]
            t1_path, t1_label = tiles[tile_id][p1]
            t2_path, t2_label = tiles[tile_id][p2]
            # True positive: land that was clean and became encroached
            label = 1 if (t1_label == "neg" and t2_label == "pos") else 0
            pairs.append((t1_path, t2_path, label, tile_id))

    return pairs


def build_dataset(split: str, verbose: bool = True):
    split_dir = DATA_DIR / split
    pairs = build_pairs(split_dir)

    n_pos = sum(1 for *_, lbl, __ in pairs if lbl == 1)
    n_neg = len(pairs) - n_pos
    if verbose:
        print(f"\n  {split.upper()} — {len(pairs)} pairs  ({n_pos} pos / {n_neg} neg)")

    X, y, tile_ids = [], [], []
    ok = fail = 0

    for i, (t1_path, t2_path, label, tile_id) in enumerate(pairs):
        tag = f"tile_{tile_id:02d}  {t1_path.stem[-3:]}→{t2_path.stem[-3:]}"
        if verbose:
            print(f"    [{i+1:3d}/{len(pairs)}]  {tag}  label={label}", end="  ")
        try:
            t1_lbl = "pos" if t1_path.stem.endswith("pos") else "neg"
            feats = extract_features(t1_path, t2_path,
                                     t1_is_pos=(t1_lbl == "pos"))
            X.append(feats)
            y.append(label)
            tile_ids.append(tile_id)
            ok += 1
            if verbose:
                print("✓")
        except Exception as e:
            fail += 1
            if verbose:
                print(f"✗  {e}")

    if verbose and fail:
        print(f"  ⚠  {fail} pairs failed extraction")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), np.array(tile_ids)


# ══════════════════════════════════════════════════════════════════════════════
#  Cross-validation (leave-one-tile-out)
# ══════════════════════════════════════════════════════════════════════════════

def run_cross_validation(X_all, y_all, tile_ids_all, make_clf_fn, beta: float = 2.0):
    """
    Leave-one-tile-out CV.  Each fold holds out all pairs from one tile.
    Uses Fβ (default β=2) to weight recall over precision during threshold search.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score

    unique_tiles = np.unique(tile_ids_all)
    n_tiles = len(unique_tiles)

    all_y_true, all_y_prob = [], []
    fold_aucs, fold_aps = [], []

    print(f"\n▶  Leave-one-tile-out CV  ({n_tiles} folds) ...")

    for ti, held_tile in enumerate(unique_tiles):
        mask_val = tile_ids_all == held_tile
        mask_tr  = ~mask_val

        if mask_val.sum() == 0 or mask_tr.sum() == 0:
            continue
        # Skip folds with no positives in either split (can't compute AUC)
        if y_all[mask_val].sum() == 0:
            continue

        clf_fold = make_clf_fn()
        clf_fold.fit(X_all[mask_tr], y_all[mask_tr])
        probs = clf_fold.predict_proba(X_all[mask_val])[:, 1]

        all_y_true.extend(y_all[mask_val].tolist())
        all_y_prob.extend(probs.tolist())

        auc = roc_auc_score(y_all[mask_val], probs)
        ap  = average_precision_score(y_all[mask_val], probs)
        fold_aucs.append(auc)
        fold_aps.append(ap)
        print(f"    fold {ti+1:2d}/{n_tiles}  tile={held_tile:02d}  "
              f"n_val={mask_val.sum()}  pos={y_all[mask_val].sum()}  "
              f"AUC={auc:.3f}  AP={ap:.3f}")

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)

    # Threshold sweep on pooled OOF predictions using Fβ
    best_thresh_cv, best_fb = 0.5, 0.0
    for thresh in np.arange(0.05, 0.95, 0.01):
        preds_t = (all_y_prob >= thresh).astype(int)
        fb = fbeta_score(all_y_true, preds_t, beta=beta, zero_division=0)
        if fb > best_fb:
            best_fb, best_thresh_cv = fb, thresh

    oof_auc = roc_auc_score(all_y_true, all_y_prob) if len(np.unique(all_y_true)) > 1 else float("nan")
    oof_ap  = average_precision_score(all_y_true, all_y_prob) if len(np.unique(all_y_true)) > 1 else float("nan")

    print(f"\n  CV summary  ({len(fold_aucs)} folds with positives)")
    print(f"    Mean fold AUC : {np.mean(fold_aucs):.4f}  ± {np.std(fold_aucs):.4f}")
    print(f"    Mean fold AP  : {np.mean(fold_aps):.4f}  ± {np.std(fold_aps):.4f}")
    print(f"    OOF AUC       : {oof_auc:.4f}")
    print(f"    OOF Avg Prec  : {oof_ap:.4f}")
    print(f"    Best OOF thresh (F{beta:.0f}={best_fb:.3f}) : {best_thresh_cv:.2f}")

    return best_thresh_cv, oof_auc, oof_ap


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════════════

def print_metrics(y_true, y_pred, y_prob=None, title=""):
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_auc_score, average_precision_score,
    )
    print(f"\n{'─'*56}")
    print(f"  {title}")
    print(f"{'─'*56}")
    print(classification_report(y_true, y_pred,
                                target_names=["no encroachment", "encroachment"],
                                digits=3))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion matrix:")
    print(f"             Predicted neg    Predicted pos")
    print(f"  Actual neg     {tn:4d}  (TN)      {fp:4d}  (FP)")
    print(f"  Actual pos     {fn:4d}  (FN)      {tp:4d}  (TP)")

    if y_prob is not None and len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
        ap  = average_precision_score(y_true, y_prob)
        print(f"\n  ROC-AUC:  {auc:.4f}")
        print(f"  Avg Prec: {ap:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train KEMET1 encroachment classifier")
    parser.add_argument("--no-save",        action="store_true", help="Don't save the model")
    parser.add_argument("--no-cv",          action="store_true", help="Skip cross-validation")
    parser.add_argument("--n-estimators",   type=int,   default=200)
    parser.add_argument("--max-depth",      type=int,   default=8,
                        help="RF max_depth (default=8 to prevent overfit; None = unlimited)")
    parser.add_argument("--min-samples-leaf", type=int, default=3,
                        help="RF min_samples_leaf (default=3)")
    parser.add_argument("--beta",           type=float, default=2.0,
                        help="β for Fβ threshold optimisation (β>1 weights recall higher)")
    parser.add_argument("--top-features",   type=int,   default=15)
    args = parser.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import fbeta_score

    t0 = time.time()
    print("\n" + "═" * 56)
    print("  KEMET1 — Encroachment Classifier Training  (v2)")
    print("═" * 56)
    print(f"  Regularisation: max_depth={args.max_depth}  "
          f"min_samples_leaf={args.min_samples_leaf}")
    print(f"  Threshold opt:  F{args.beta:.0f} score  (β>{1} weights recall)")

    # ── 1. Build datasets ─────────────────────────────────────────────────────
    print("\n▶  Extracting features ...")
    X_train, y_train, tile_ids_train = build_dataset("train")
    X_val,   y_val,   tile_ids_val   = build_dataset("val")
    X_test,  y_test,  tile_ids_test  = build_dataset("test")

    print(f"\n  Feature matrix sizes:")
    print(f"    Train: {X_train.shape}   pos={y_train.sum()}  neg={(y_train==0).sum()}")
    print(f"    Val:   {X_val.shape}   pos={y_val.sum()}  neg={(y_val==0).sum()}")
    print(f"    Test:  {X_test.shape}   pos={y_test.sum()}  neg={(y_test==0).sum()}")

    # ── 2. Define model factory ───────────────────────────────────────────────
    def make_clf():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators    = args.n_estimators,
                max_depth       = args.max_depth,
                min_samples_leaf= args.min_samples_leaf,
                class_weight    = "balanced",
                random_state    = 42,
                n_jobs          = -1,
            )),
        ])

    # ── 3. Cross-validation on all tiles ─────────────────────────────────────
    cv_thresh = None
    if not args.no_cv:
        # Pool all splits for tile-level CV (gives more signal than 6-positive val)
        X_all      = np.concatenate([X_train, X_val, X_test], axis=0)
        y_all      = np.concatenate([y_train, y_val,  y_test],  axis=0)
        tile_ids_all = np.concatenate([tile_ids_train, tile_ids_val, tile_ids_test])
        cv_thresh, cv_auc, cv_ap = run_cross_validation(
            X_all, y_all, tile_ids_all, make_clf, beta=args.beta
        )

    # ── 4. Train final model on train split ───────────────────────────────────
    print(f"\n▶  Training final RF  (n_estimators={args.n_estimators}, "
          f"max_depth={args.max_depth}, min_samples_leaf={args.min_samples_leaf}) ...")
    clf = make_clf()
    clf.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── 5. Threshold optimisation ─────────────────────────────────────────────
    val_probs = clf.predict_proba(X_val)[:, 1]

    # Fβ on val set
    best_thresh_fb, best_fb = 0.5, 0.0
    for thresh in np.arange(0.05, 0.95, 0.01):
        preds_t = (val_probs >= thresh).astype(int)
        fb = fbeta_score(y_val, preds_t, beta=args.beta, zero_division=0)
        if fb > best_fb:
            best_fb, best_thresh_fb = fb, thresh

    # Plain F1 on val set (for reference)
    best_thresh_f1, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.05, 0.95, 0.01):
        preds_t = (val_probs >= thresh).astype(int)
        f1 = fbeta_score(y_val, preds_t, beta=1.0, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh_f1 = f1, thresh

    print(f"\n  Threshold comparison (val set):")
    print(f"    Default 0.50    → F{args.beta:.0f}={fbeta_score(y_val,(val_probs>=0.5).astype(int),beta=args.beta,zero_division=0):.3f}")
    print(f"    Best F1  {best_thresh_f1:.2f}   → F1={best_f1:.3f}")
    print(f"    Best F{args.beta:.0f} {best_thresh_fb:.2f}   → F{args.beta:.0f}={best_fb:.3f}  ← used for screening")
    if cv_thresh is not None:
        print(f"    CV thresh {cv_thresh:.2f}   → from leave-one-tile-out")

    # Prefer val Fβ threshold when CV threshold is more aggressive (CV folds have
    # only 1 positive each, making threshold optimisation noisy).
    # Take the more conservative (higher) of the two.
    if cv_thresh is not None:
        final_thresh = max(cv_thresh, best_thresh_fb)
        if cv_thresh != final_thresh:
            print(f"  (CV thresh {cv_thresh:.2f} is more aggressive than val F{args.beta:.0f} "
                  f"thresh {best_thresh_fb:.2f} — using val thresh to limit FP rate)")
    else:
        final_thresh = best_thresh_fb
    print(f"\n  Selected threshold: {final_thresh:.2f}")

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    print("\n  Results at default threshold (0.50):")
    for split_name, X, y in [
        ("TRAIN", X_train, y_train),
        ("VAL",   X_val,   y_val),
        ("TEST",  X_test,  y_test),
    ]:
        preds = clf.predict(X)
        probs = clf.predict_proba(X)[:, 1]
        print_metrics(y, preds, probs, title=split_name)

    print(f"\n  Results at selected threshold ({final_thresh:.2f}):")
    for split_name, X, y in [
        ("VAL",  X_val,  y_val),
        ("TEST", X_test, y_test),
    ]:
        probs = clf.predict_proba(X)[:, 1]
        preds = (probs >= final_thresh).astype(int)
        print_metrics(y, preds, probs, title=f"{split_name} @ thresh={final_thresh:.2f}")

    # ── 7. Feature importance ─────────────────────────────────────────────────
    rf = clf.named_steps["rf"]
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:args.top_features]

    print(f"\n{'─'*56}")
    print(f"  Top {args.top_features} Most Important Features")
    print(f"{'─'*56}")
    for rank, idx in enumerate(top_idx, 1):
        bar = "█" * int(importances[idx] * 200)
        print(f"  {rank:2d}. {FEATURE_NAMES[idx]:<28}  {importances[idx]:.4f}  {bar}")

    # ── 8. Save model + threshold ─────────────────────────────────────────────
    if not args.no_save:
        test_probs = clf.predict_proba(X_test)[:, 1]
        from sklearn.metrics import roc_auc_score as _auc
        _test_auc = float(_auc(y_test, test_probs)) if len(np.unique(y_test)) > 1 else float("nan")
        bundle = {
            "model":          clf,
            "threshold":      final_thresh,
            "feature_names":  FEATURE_NAMES,
            "model_name":     f"RF depth={args.max_depth} leaf={args.min_samples_leaf}",
            "max_depth":      args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "beta":           args.beta,
            "test_auc":       _test_auc,
        }
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(bundle, f)
        print(f"\n  Model + threshold ({final_thresh:.2f}) saved → {MODEL_PATH}")

    print(f"\n{'═'*56}")
    print(f"  Total time: {time.time()-t0:.1f}s")
    print(f"{'═'*56}\n")


if __name__ == "__main__":
    main()
