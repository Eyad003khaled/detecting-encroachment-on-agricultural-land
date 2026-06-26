"""
ablation_no_circular.py  -  Retrain BA RF without the circular features.

The two features pct_conv and pct_new are derived from the same spectral
threshold that generated the auto-labels, creating label-feature circularity.
This ablation removes them to get a clean generalization estimate.

Run:
    python ablation_no_circular.py

Outputs:
  models/ba_rf_ablation.pkl        - ablation model bundle
  ablation_results.json            - metrics comparison
"""
from __future__ import annotations
import json, pickle, time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

BA_DIR     = Path("data/KEMET1_BeforeAfter/KEMET1_BeforeAfter_Tiles")
LABELS     = Path("data/ba_labels.json")
ORIG_MODEL = Path("models/ba_rf_model.pkl")
ABL_MODEL  = Path("models/ba_rf_ablation.pkl")
OUT_JSON   = Path("ablation_results.json")


def extract_stats(arr):
    feats = []
    for b in range(arr.shape[0]):
        ch = arr[b].ravel(); ch = ch[np.isfinite(ch)]
        feats += [ch.mean(), ch.std(),
                  np.percentile(ch,10), np.percentile(ch,25), np.percentile(ch,50),
                  np.percentile(ch,75), np.percentile(ch,90)]
    return np.array(feats)

def pair_features_full(d1, d2):
    """46 features including circular pct_conv and pct_new."""
    fd = extract_stats(d2 - d1)
    ndvi1,ndbi1 = d1[0],d1[1]; ndvi2,ndbi2 = d2[0],d2[1]
    pct_conv = float(((ndvi1>0.25)&(ndvi2<0.25)&(ndbi2>ndbi1+0.08)).mean())
    pct_new  = float(((ndbi2>0.15)&((ndbi2-ndbi1)>0.10)).mean())
    return np.concatenate([fd, [float(np.nanmean(ndvi2-ndvi1)),
                                float(np.nanmean(ndbi2-ndbi1)),
                                pct_conv, pct_new]])

def pair_features_ablation(d1, d2):
    """44 features — pct_conv and pct_new removed."""
    fd = extract_stats(d2 - d1)
    ndvi1 = d1[0]; ndbi1 = d1[1]; ndvi2 = d2[0]; ndbi2 = d2[1]
    return np.concatenate([fd, [float(np.nanmean(ndvi2-ndvi1)),
                                float(np.nanmean(ndbi2-ndbi1 if d2.shape[0]>1 else 0))]])

def extract_all(records, feat_fn):
    X, y = [], []
    for i, r in enumerate(records):
        bp = BA_DIR/(r["site"]+"_before_2024.tif")
        ap = BA_DIR/(r["site"]+"_after_2025.tif")
        if not bp.exists(): continue
        with rasterio.open(bp) as s: d1 = s.read().astype(np.float32)
        with rasterio.open(ap) as s: d2 = s.read().astype(np.float32)
        X.append(feat_fn(d1, d2))
        y.append(1 if r["label"]=="pos" else 0)
        if (i+1) % 60 == 0: print(f"  {i+1}/{len(records)}")
    return np.array(X), np.array(y)

def train_eval(X, y, seed=42, tag=""):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30,
                                                  stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50,
                                                      stratify=y_tmp, random_state=seed)
    imp = SimpleImputer(strategy="median")
    X_tr_i  = imp.fit_transform(X_tr)
    X_val_i = imp.transform(X_val)
    X_tst_i = imp.transform(X_test)

    clf = RandomForestClassifier(n_estimators=400, max_depth=6,
                                  class_weight="balanced", random_state=seed, n_jobs=-1)
    clf.fit(X_tr_i, y_tr)

    prob_val  = clf.predict_proba(X_val_i)[:,1]
    prob_test = clf.predict_proba(X_tst_i)[:,1]
    auc_val  = roc_auc_score(y_val,  prob_val)
    auc_test = roc_auc_score(y_test, prob_test)

    p_op = (prob_test >= 0.40).astype(int)
    cm   = confusion_matrix(y_test, p_op, labels=[0,1]).tolist()
    f1   = f1_score(y_test, p_op, zero_division=0)

    print(f"\n{tag}")
    print(f"  Val AUC:  {auc_val:.4f}")
    print(f"  Test AUC: {auc_test:.4f}")
    print(f"  Test CM @ 0.40: {cm}  F1={f1:.3f}")
    return clf, imp, {"val_auc": round(auc_val,4), "test_auc": round(auc_test,4),
                      "test_cm": cm, "test_f1": round(f1,4),
                      "n_features": X.shape[1]}

def main():
    records = json.load(open(LABELS))
    print(f"Extracting FULL features (46) for {len(records)} sites...")
    t0 = time.time()
    X_full, y = extract_all(records, pair_features_full)
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\nExtracting ABLATION features (44, no circular)...")
    t0 = time.time()
    X_abl, _ = extract_all(records, pair_features_ablation)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("\n── Full model (46 features, includes pct_conv + pct_new) ──")
    clf_full, imp_full, m_full = train_eval(X_full, y, tag="FULL (46 features)")

    print("\n── Ablation model (44 features, circular features removed) ──")
    clf_abl, imp_abl, m_abl = train_eval(X_abl, y, tag="ABLATION (44 features)")

    results = {"full": m_full, "ablation": m_abl,
               "delta_val_auc":  round(m_abl["val_auc"]  - m_full["val_auc"],  4),
               "delta_test_auc": round(m_abl["test_auc"] - m_full["test_auc"], 4)}
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT_JSON}")

    pickle.dump({"model": clf_abl, "imputer": imp_abl, "results": m_abl},
                open(ABL_MODEL, "wb"))
    print(f"Ablation model: {ABL_MODEL}")

    print(f"\n{'='*50}")
    print(f"  Full    val AUC: {m_full['val_auc']:.4f}")
    print(f"  Ablation val AUC: {m_abl['val_auc']:.4f}  (Δ {results['delta_val_auc']:+.4f})")
    print(f"\n  Interpretation:")
    delta = results["delta_val_auc"]
    if abs(delta) < 0.03:
        print("  → Minimal drop: circular features contribute little;")
        print("    model generalises on spectral delta features alone.")
    elif delta < -0.05:
        print("  → Significant drop: model relied heavily on circular features.")
        print("    True generalisation AUC is ~", m_abl["val_auc"])
    else:
        print(f"  → Moderate drop ({delta:.3f}). Cite ablation AUC as conservative bound.")
    print('='*50)

if __name__ == "__main__":
    main()
