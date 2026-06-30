"""
evaluate_ba.py  –  Full evaluation of the BA Random Forest on the test split.

Run from the GP folder:
    python evaluate_ba.py

Outputs:
  - data/ba_eval.json          (metrics + curve data)
  - training_results.html      (updated in-place with new evaluation section)
"""
from __future__ import annotations
import json, pickle, time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
)

BA_DIR     = Path("data/KEMET1_BeforeAfter/KEMET1_BeforeAfter_Tiles")
MODEL_PATH = Path("models/ba_rf_model.pkl")
LABELS     = Path("data/ba_labels.json")
OUT_JSON   = Path("data/ba_eval.json")
DASHBOARD  = Path("training_results.html")

# ── Feature extraction (must match run_inference.py) ─────────────────────────
def extract_stats(arr):
    feats = []
    for b in range(arr.shape[0]):
        ch = arr[b].ravel(); ch = ch[np.isfinite(ch)]
        feats += [ch.mean(), ch.std(),
                  np.percentile(ch,10), np.percentile(ch,25), np.percentile(ch,50),
                  np.percentile(ch,75), np.percentile(ch,90)]
    return np.array(feats)

def pair_features(d1, d2):
    """44 features — pct_conv and pct_new removed (circular with labels)."""
    fd = extract_stats(d2 - d1)
    ndvi2, ndbi2 = d2[0], d2[1]; ndvi1, ndbi1 = d1[0], d1[1]
    return np.concatenate([fd, [float(np.nanmean(ndvi2-ndvi1)),
                                float(np.nanmean(ndbi2-ndbi1))]])

def extract_all(records):
    X, y, sites = [], [], []
    for i, r in enumerate(records):
        bp = BA_DIR / (r["site"]+"_before_2024.tif")
        ap = BA_DIR / (r["site"]+"_after_2025.tif")
        if not bp.exists() or not ap.exists():
            print(f"  Missing: {r['site']}, skipping")
            continue
        with rasterio.open(bp) as s: d1 = s.read().astype(np.float32)
        with rasterio.open(ap) as s: d2 = s.read().astype(np.float32)
        X.append(pair_features(d1, d2))
        y.append(1 if r["label"] == "pos" else 0)
        sites.append(r["site"])
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(records)} sites processed")
    return np.array(X), np.array(y), sites

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    bundle = pickle.load(open(MODEL_PATH, "rb"))
    clf    = bundle["model"]
    imp    = bundle["imputer"]
    stored = bundle.get("results", {})

    records = json.load(open(LABELS))
    print(f"Extracting features for {len(records)} sites...")
    t0 = time.time()
    X_all, y_all, site_ids = extract_all(records)
    print(f"Done in {time.time()-t0:.1f}s. Shape: {X_all.shape}")

    # Reproduce the train/val/test split used during training
    # 70% train, 15% val, 15% test  —  stratified, seed=42
    X_tr, X_tmp, y_tr, y_tmp, s_tr, s_tmp = train_test_split(
        X_all, y_all, site_ids, test_size=0.30, stratify=y_all, random_state=42)
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_tmp, y_tmp, s_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    print(f"\nSplit: train={len(y_tr)} val={len(y_val)} test={len(y_test)}")
    print(f"  Train pos/neg: {y_tr.sum()}/{(y_tr==0).sum()}")
    print(f"  Val   pos/neg: {y_val.sum()}/{(y_val==0).sum()}")
    print(f"  Test  pos/neg: {y_test.sum()}/{(y_test==0).sum()}")

    # Refit imputer on training split (avoids sklearn version mismatch with pickled imputer)
    from sklearn.impute import SimpleImputer
    imp_new = SimpleImputer(strategy="median")
    imp_new.fit(X_tr)
    X_tr_imp   = imp_new.transform(X_tr)
    X_val_imp  = imp_new.transform(X_val)
    X_test_imp = imp_new.transform(X_test)

    # Check if this reproduces stored test CM
    stored_cm = stored.get("test", {}).get("cm")
    preds_def = clf.predict(X_test_imp)
    cm_check  = confusion_matrix(y_test, preds_def).tolist()
    print(f"\nStored test CM:     {stored_cm}")
    print(f"Reproduced test CM: {cm_check}")
    if cm_check != stored_cm:
        print("  ⚠ Split seed differs from original — results are a fresh stratified hold-out")
    else:
        print("  ✓ Exact match — same split as training")

    # Full probability predictions
    probs_val  = clf.predict_proba(X_val_imp)[:,1]
    probs_test = clf.predict_proba(X_test_imp)[:,1]

    # ROC curves
    fpr_v, tpr_v, thr_v = roc_curve(y_val,  probs_val)
    fpr_t, tpr_t, thr_t = roc_curve(y_test, probs_test)
    auc_val  = roc_auc_score(y_val,  probs_val)
    auc_test = roc_auc_score(y_test, probs_test)

    # PR curves
    prec_v, rec_v, thr_pv = precision_recall_curve(y_val,  probs_val)
    prec_t, rec_t, thr_pt = precision_recall_curve(y_test, probs_test)

    # Per-threshold metrics on test
    thresholds = np.linspace(0.05, 0.95, 91)
    thresh_rows = []
    for th in thresholds:
        p = (probs_test >= th).astype(int)
        if p.sum() == 0:
            prec, rec, f1 = 0.0, 0.0, 0.0
        else:
            prec = precision_score(y_test, p, zero_division=0)
            rec  = recall_score(y_test,    p, zero_division=0)
            f1   = f1_score(y_test,        p, zero_division=0)
        acc = accuracy_score(y_test, p)
        cm  = confusion_matrix(y_test, p, labels=[0,1]).tolist()
        thresh_rows.append({
            "threshold": round(float(th), 2),
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "accuracy":  round(acc,  4),
            "cm":        cm
        })

    # Current operating threshold (0.40)
    op_thresh = 0.40
    p_op = (probs_test >= op_thresh).astype(int)
    op_cm = confusion_matrix(y_test, p_op, labels=[0,1]).tolist()

    results = {
        "n_train": int(len(y_tr)),
        "n_val":   int(len(y_val)),
        "n_test":  int(len(y_test)),
        "val": {
            "auc":  round(auc_val, 4),
            "fpr":  [round(float(x),4) for x in fpr_v],
            "tpr":  [round(float(x),4) for x in tpr_v],
            "prec": [round(float(x),4) for x in prec_v],
            "rec":  [round(float(x),4) for x in rec_v],
        },
        "test": {
            "auc":  round(auc_test, 4),
            "fpr":  [round(float(x),4) for x in fpr_t],
            "tpr":  [round(float(x),4) for x in tpr_t],
            "prec": [round(float(x),4) for x in prec_t],
            "rec":  [round(float(x),4) for x in rec_t],
            "op_thresh": op_thresh,
            "op_cm": op_cm,
            "op_precision": round(precision_score(y_test, p_op, zero_division=0), 4),
            "op_recall":    round(recall_score(y_test,    p_op, zero_division=0), 4),
            "op_f1":        round(f1_score(y_test,        p_op, zero_division=0), 4),
            "op_accuracy":  round(accuracy_score(y_test,  p_op), 4),
        },
        "per_site": [
            {"site": s, "label": int(yt), "prob": round(float(pb), 4)}
            for s, yt, pb in zip(s_test, y_test, probs_test)
        ],
        "threshold_sweep": thresh_rows,
    }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT_JSON}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"  Val  AUC: {auc_val:.4f}")
    print(f"  Test AUC: {auc_test:.4f}")
    print(f"  Test @ thresh={op_thresh}:")
    print(f"    CM: {op_cm}")
    print(f"    Precision: {results['test']['op_precision']}")
    print(f"    Recall:    {results['test']['op_recall']}")
    print(f"    F1:        {results['test']['op_f1']}")
    print(f"    Accuracy:  {results['test']['op_accuracy']}")
    print(f"{'='*50}")

    patch_dashboard(results)
    print(f"Dashboard updated: {DASHBOARD}")


def patch_dashboard(results):
    """Inject evaluation section into training_results.html."""
    if not DASHBOARD.exists():
        print(f"  {DASHBOARD} not found, skipping dashboard patch.")
        return

    html = DASHBOARD.read_text(encoding="utf-8")

    # Build the eval section HTML
    section = build_eval_section(results)

    marker = "<!-- BA_EVAL_SECTION -->"
    if marker in html:
        # Replace between markers
        html = html.split(marker)[0] + marker + section + marker + html.split(marker)[-1]
    else:
        # Inject before </body>
        html = html.replace("</body>", marker + section + marker + "\n</body>")

    DASHBOARD.write_text(html, encoding="utf-8")


def build_eval_section(r):
    test  = r["test"]
    val   = r["val"]

    # Serialise curve data for JS
    import json as _json
    val_roc  = _json.dumps({"fpr": val["fpr"],  "tpr": val["tpr"]})
    test_roc = _json.dumps({"fpr": test["fpr"], "tpr": test["tpr"]})
    val_pr   = _json.dumps({"rec": val["rec"],  "prec": val["prec"]})
    test_pr  = _json.dumps({"rec": test["rec"], "prec": test["prec"]})
    sweep    = _json.dumps(r["threshold_sweep"])
    op_cm    = test["op_cm"]   # [[TN, FP],[FN, TP]]
    tn, fp, fn, tp = op_cm[0][0], op_cm[0][1], op_cm[1][0], op_cm[1][1]

    return f"""
<div id="ba-eval" style="background:#0d1117;color:#c9d1d9;font-family:'Segoe UI',sans-serif;padding:24px">
<h2 style="color:#8ee3ff;border-bottom:1px solid #21262d;padding-bottom:10px;margin-bottom:20px">
  Test-Split Evaluation &nbsp;<span style="font-size:0.75rem;color:#6e7681;font-weight:400">
  {r["n_train"]} train / {r["n_val"]} val / {r["n_test"]} test &nbsp;|&nbsp; stratified 70/15/15</span>
</h2>

<!-- Caveat banner -->
<div style="background:#1a1200;border:1px solid #664400;border-radius:8px;padding:12px 16px;margin-bottom:20px;font-size:12px;color:#e3a030;line-height:1.6">
  <b>⚠ Label-feature circularity note:</b>
  Auto-labels were generated from <code>pct_conv</code> and <code>max_cluster_ha</code> thresholds —
  both of which are also in the 46-feature vector. The model therefore learns to replicate the
  labelling rule, inflating test scores toward 1.0. The <b>val AUC = {val['auc']:.3f}</b> (samples
  excluded during training) is the conservative, cited metric for generalization performance.
</div>

<!-- KPI row -->
<div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:24px">
  {_kpi("Val AUC ★", f"{val['auc']:.4f}", "#44ff88")}
  {_kpi("Test AUC†", f"{test['auc']:.4f}", "#8ee3ff")}
  {_kpi("Precision", f"{test['op_precision']:.3f}", "#ff8c00")}
  {_kpi("Recall",    f"{test['op_recall']:.3f}",    "#ff8c00")}
  {_kpi("F1",        f"{test['op_f1']:.3f}",        "#c9d1d9")}
  {_kpi("Accuracy",  f"{test['op_accuracy']:.3f}",  "#c9d1d9")}
</div>
<p style="font-size:10px;color:#6e7681;margin:-16px 0 20px">★ primary metric &nbsp;|&nbsp; † inflated due to label-feature circularity</p>

<!-- Charts row -->
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:24px">
  <canvas id="rocChart"  height="260"></canvas>
  <canvas id="prChart"   height="260"></canvas>
  <canvas id="cmChart"   height="260"></canvas>
</div>

<!-- Threshold slider -->
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:24px">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px">
    <label style="font-size:12px;color:#8ee3ff;white-space:nowrap">Decision threshold</label>
    <input type="range" id="thSlider" min="0" max="90" value="24" step="1"
           style="flex:1;accent-color:#8ee3ff">
    <span id="thVal" style="font-size:14px;font-weight:700;color:#8ee3ff;min-width:40px">0.29</span>
  </div>
  <div style="display:flex;gap:24px;flex-wrap:wrap" id="thMetrics">
    <div>Precision: <b id="mPrec">–</b></div>
    <div>Recall: <b id="mRec">–</b></div>
    <div>F1: <b id="mF1">–</b></div>
    <div>Accuracy: <b id="mAcc">–</b></div>
  </div>
  <div style="margin-top:10px;font-size:11px;color:#6e7681">
    TP: <b id="mTP">–</b> &nbsp; FP: <b id="mFP">–</b> &nbsp;
    FN: <b id="mFN">–</b> &nbsp; TN: <b id="mTN">–</b>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
(function(){{
var BG='#161b22',GRID='#21262d',TXT='#8ee3ff',GRAY='#6e7681';
var valRoc={val_roc}, testRoc={test_roc};
var valPr={val_pr},   testPr={test_pr};
var sweep={sweep};

// ROC
new Chart(document.getElementById('rocChart'),{{type:'line',data:{{datasets:[
  {{label:'Test ROC (AUC={test["auc"]:.3f})',data:testRoc.fpr.map((x,i)=>{{return{{x,y:testRoc.tpr[i]}}}})
   ,borderColor:'#ff4444',borderWidth:2,pointRadius:0,fill:false,tension:0}},
  {{label:'Val  ROC (AUC={val["auc"]:.3f})', data:valRoc.fpr.map((x,i)=>{{return{{x,y:valRoc.tpr[i]}}}})
   ,borderColor:'#58a6ff',borderWidth:1.5,pointRadius:0,fill:false,tension:0,borderDash:[4,3]}},
  {{label:'Chance',data:[{{x:0,y:0}},{{x:1,y:1}}],borderColor:GRAY,borderWidth:1,pointRadius:0,fill:false,borderDash:[6,4]}}
]}},options:{{responsive:true,plugins:{{legend:{{labels:{{color:TXT,font:{{size:11}}}}}},
  title:{{display:true,text:'ROC Curve',color:TXT,font:{{size:13,weight:'bold'}}}}}},
  scales:{{x:{{title:{{display:true,text:'FPR',color:GRAY}},ticks:{{color:GRAY}},grid:{{color:GRID}}}},
           y:{{title:{{display:true,text:'TPR',color:GRAY}},ticks:{{color:GRAY}},grid:{{color:GRID}}}}}}
}}}})

// PR
new Chart(document.getElementById('prChart'),{{type:'line',data:{{datasets:[
  {{label:'Test PR',data:testPr.rec.map((x,i)=>{{return{{x,y:testPr.prec[i]}}}})
   ,borderColor:'#ff4444',borderWidth:2,pointRadius:0,fill:false,tension:0}},
  {{label:'Val  PR',data:valPr.rec.map((x,i)=>{{return{{x,y:valPr.prec[i]}}}})
   ,borderColor:'#58a6ff',borderWidth:1.5,pointRadius:0,fill:false,tension:0,borderDash:[4,3]}}
]}},options:{{responsive:true,plugins:{{legend:{{labels:{{color:TXT,font:{{size:11}}}}}},
  title:{{display:true,text:'Precision-Recall Curve',color:TXT,font:{{size:13,weight:'bold'}}}}}},
  scales:{{x:{{min:0,max:1,title:{{display:true,text:'Recall',color:GRAY}},ticks:{{color:GRAY}},grid:{{color:GRID}}}},
           y:{{min:0,max:1,title:{{display:true,text:'Precision',color:GRAY}},ticks:{{color:GRAY}},grid:{{color:GRID}}}}}}
}}}})

// Confusion matrix
var cmData=[{tn},{fp},{fn},{tp}];
var cmLabels=['TN\\n(Neg→Neg)','FP\\n(Neg→Pos)','FN\\n(Pos→Neg)','TP\\n(Pos→Pos)'];
var cmColors=['#44ff8844','#ff444444','#ff880044','#44ff8888'];
new Chart(document.getElementById('cmChart'),{{type:'bar',data:{{
  labels:cmLabels,
  datasets:[{{data:cmData,backgroundColor:cmColors,borderColor:cmColors.map(c=>c.replace('44','ff')),borderWidth:2}}]
}},options:{{responsive:true,plugins:{{legend:{{display:false}},
  title:{{display:true,text:'Confusion Matrix @ thresh={test["op_thresh"]}',color:TXT,font:{{size:13,weight:'bold'}}}}}},
  scales:{{x:{{ticks:{{color:GRAY}},grid:{{color:GRID}}}},y:{{ticks:{{color:GRAY}},grid:{{color:GRID}}}}}}
}}}})

// Threshold slider
var sl=document.getElementById('thSlider');
function updateSlider(){{
  var idx=parseInt(sl.value);
  var row=sweep[idx];
  document.getElementById('thVal').textContent=row.threshold.toFixed(2);
  document.getElementById('mPrec').textContent=row.precision.toFixed(3);
  document.getElementById('mRec').textContent=row.recall.toFixed(3);
  document.getElementById('mF1').textContent=row.f1.toFixed(3);
  document.getElementById('mAcc').textContent=row.accuracy.toFixed(3);
  document.getElementById('mTP').textContent=row.cm[1][1];
  document.getElementById('mFP').textContent=row.cm[0][1];
  document.getElementById('mFN').textContent=row.cm[1][0];
  document.getElementById('mTN').textContent=row.cm[0][0];
}}
sl.addEventListener('input',updateSlider);
updateSlider();
}})();
</script>
</div>
"""

def _kpi(label, value, color):
    return (f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;'
            f'padding:12px 18px;min-width:110px">'
            f'<div style="font-size:10px;color:#6e7681;text-transform:uppercase;'
            f'letter-spacing:.06em;margin-bottom:3px">{label}</div>'
            f'<div style="font-size:1.5rem;font-weight:700;color:{color}">{value}</div></div>')


if __name__ == "__main__":
    main()
