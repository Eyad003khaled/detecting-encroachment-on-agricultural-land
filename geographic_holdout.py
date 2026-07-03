"""geographic_holdout.py — Geographic holdout evaluation (split by latitude).

Splits the 300 KEMET1 sites into northern and southern halves of the Nile Delta
by median latitude, then evaluates the model trained on one half against the
other.  This tests spatial generalisation independent of the random train/val/test
split used during training.

Run from the GP folder:
    python geographic_holdout.py

Outputs:
  - data/geo_holdout.json    (AUC, confusion matrices, per-site predictions)
  Prints a summary table to stdout.
"""
from __future__ import annotations
import json, pickle
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, confusion_matrix

BA_DIR     = Path(__file__).parent / "data/KEMET1_BeforeAfter/KEMET1_BeforeAfter_Tiles"
MODEL_PATH = Path(__file__).parent / "models/ba_rf_model.pkl"
LABELS     = Path(__file__).parent / "data/ba_labels.json"
OUT_JSON   = Path(__file__).parent / "data/geo_holdout.json"


# ── Feature extraction (must match run_inference.py) ──────────────────────────
def extract_stats(arr: np.ndarray) -> np.ndarray:
    feats = []
    for b in range(arr.shape[0]):
        ch = arr[b].ravel(); ch = ch[np.isfinite(ch)]
        feats += [ch.mean(), ch.std(),
                  np.percentile(ch, 10), np.percentile(ch, 25),
                  np.percentile(ch, 50), np.percentile(ch, 75),
                  np.percentile(ch, 90)]
    return np.array(feats)


def pair_features(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """44 features — matches run_inference.py / ablation_no_circular.py."""
    fd = extract_stats(d2 - d1)
    return np.concatenate([fd, [float(np.nanmean(d2[0] - d1[0])),
                                float(np.nanmean(d2[1] - d1[1]))]])


def load_site(record: dict):
    """Return (features, label, center_lat) for one site, or None if tiles missing."""
    bp = BA_DIR / (record["site"] + "_before_2024.tif")
    ap = BA_DIR / (record["site"] + "_after_2025.tif")
    if not bp.exists() or not ap.exists():
        print(f"  Missing tiles: {record['site']}, skipping")
        return None
    with rasterio.open(bp) as src:
        d1   = src.read().astype(np.float32)
        wgs  = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        clat = (wgs[1] + wgs[3]) / 2
    with rasterio.open(ap) as src:
        d2 = src.read().astype(np.float32)
    feat  = pair_features(d1, d2)
    label = 1 if record["label"] == "pos" else 0
    return feat, label, clat


def evaluate_split(X_train, y_train, X_test, y_test, bundle):
    """Retrain RF on X_train, evaluate on X_test.  Returns AUC and CM."""
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_train)
    X_te = imp.transform(X_test)

    # Clone the stored model's hyperparameters for a fair comparison
    stored_rf = bundle["model"]
    rf = RandomForestClassifier(
        n_estimators=stored_rf.n_estimators,
        max_depth=stored_rf.max_depth,
        min_samples_leaf=stored_rf.min_samples_leaf,
        class_weight=stored_rf.class_weight,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_train)
    probs = rf.predict_proba(X_te)[:, 1]
    auc   = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else float("nan")
    preds = (probs >= 0.29).astype(int)
    cm    = confusion_matrix(y_test, preds, labels=[0, 1]).tolist()
    return auc, cm, probs.tolist()


def main():
    bundle  = pickle.load(open(MODEL_PATH, "rb"))
    records = json.load(open(LABELS))

    print(f"Loading {len(records)} sites ...")
    data = []
    for r in records:
        result = load_site(r)
        if result is None:
            continue
        feat, label, clat = result
        data.append({"site": r["site"], "feat": feat, "label": label, "lat": clat})

    print(f"Loaded {len(data)} sites.")

    lats = np.array([d["lat"] for d in data])
    median_lat = float(np.median(lats))
    print(f"Median latitude: {median_lat:.4f}°N  (splits north vs south)")

    south = [d for d in data if d["lat"] <  median_lat]
    north = [d for d in data if d["lat"] >= median_lat]
    print(f"South: {len(south)} sites ({sum(d['label'] for d in south)} pos)")
    print(f"North: {len(north)} sites ({sum(d['label'] for d in north)} pos)")

    def pack(subset):
        X = np.array([d["feat"]  for d in subset])
        y = np.array([d["label"] for d in subset])
        s = [d["site"] for d in subset]
        return X, y, s

    results = {}
    for train_name, train_data, test_name, test_data in [
        ("south", south, "north", north),
        ("north", north, "south", south),
    ]:
        X_tr, y_tr, _ = pack(train_data)
        X_te, y_te, s_te = pack(test_data)
        auc, cm, probs = evaluate_split(X_tr, y_tr, X_te, y_te, bundle)
        print(f"\nTrain={train_name}  →  Test={test_name}")
        print(f"  AUC: {auc:.4f}")
        print(f"  CM (@ thr=0.29): {cm}")
        results[f"train_{train_name}_test_{test_name}"] = {
            "train_n": len(y_tr), "train_pos": int(y_tr.sum()),
            "test_n":  len(y_te), "test_pos":  int(y_te.sum()),
            "median_lat": median_lat,
            "auc": round(auc, 4),
            "cm_at_0.29": cm,
            "per_site": [
                {"site": s, "label": int(yt), "prob": round(float(p), 4)}
                for s, yt, p in zip(s_te, y_te, probs)
            ],
        }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT_JSON}")

    print("\n── Summary ──────────────────────────────────────────")
    for k, v in results.items():
        print(f"  {k}: AUC={v['auc']:.4f}  (n_test={v['test_n']})")
    print("─────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
