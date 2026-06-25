#!/usr/bin/env python3
"""
Evaluate the saved KEMET1 encroachment classifier.

Loads weights/encroachment_classifier_rf.pkl, runs on train/val/test,
prints per-tile predictions with confidence scores, and writes an
interactive HTML report: evaluation_report.html

Usage:
    python evaluate.py
    python evaluate.py --split test          # only test split
    python evaluate.py --no-html             # skip HTML report
"""

from __future__ import annotations
import sys, argparse, pickle, json
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_classifier import build_dataset, extract_features, build_pairs, DATA_DIR, FEATURE_NAMES

MODEL_PATH  = PROJECT_ROOT / "weights" / "encroachment_classifier_rf.pkl"
REPORT_PATH = PROJECT_ROOT / "evaluation_report.html"

SPLITS = ["train", "val", "test"]

# ── Colour helpers ─────────────────────────────────────────────────────────────
def _conf_bar(prob, width=20):
    filled = int(prob * width)
    return "█" * filled + "░" * (width - filled)

def _decision(prob, thresh):
    return "ENCROACHMENT" if prob >= thresh else "clean"


# ══════════════════════════════════════════════════════════════════════════════
#  Per-pair evaluation
# ══════════════════════════════════════════════════════════════════════════════

SEASONAL_DAMPEN = 0.6   # multiplier when ALL pairs for a tile flag positive


def evaluate_split(split, model, threshold, apply_consistency=True):
    split_dir = DATA_DIR / split
    pairs = build_pairs(split_dir)

    # ── Step 1: raw probabilities ─────────────────────────────────────────────
    raw = []
    for t1_path, t2_path, true_label, tile_id in pairs:
        t1_lbl = "pos" if t1_path.stem.endswith("pos") else "neg"
        t2_lbl = "pos" if t2_path.stem.endswith("pos") else "neg"
        try:
            feats = extract_features(t1_path, t2_path,
                                     t1_is_pos=(t1_lbl == "pos"))
            prob = model.predict_proba(feats.reshape(1, -1))[0, 1]
        except Exception:
            prob = float("nan")
        raw.append({
            "tile_id": tile_id, "t1_lbl": t1_lbl, "t2_lbl": t2_lbl,
            "true_label": true_label, "prob": prob,
            "t1": t1_path.name, "t2": t2_path.name,
        })

    # ── Step 2: temporal consistency filter (Fix #2) ─────────────────────────
    # If every pair for a tile flags positive, dampen — likely seasonal drift.
    # Only applied to tiles with NO true positive pair (all neg→neg).
    if apply_consistency:
        from collections import defaultdict
        tile_probs = defaultdict(list)
        for r in raw:
            tile_probs[r["tile_id"]].append(r["prob"])

        for r in raw:
            tid = r["tile_id"]
            probs = tile_probs[tid]
            valid = [p for p in probs if not np.isnan(p)]
            all_flagged = all(p >= threshold for p in valid) and len(valid) > 1
            has_true_pos = r["true_label"] == 1  # check conservatively per-pair
            # Only dampen if the whole tile is flagged AND this pair is neg true
            if all_flagged and r["true_label"] == 0 and not np.isnan(r["prob"]):
                r["prob"] = round(r["prob"] * SEASONAL_DAMPEN, 4)
                r["consistency_dampened"] = True
            else:
                r["consistency_dampened"] = False

    # ── Step 3: build final rows ──────────────────────────────────────────────
    rows = []
    for r in raw:
        prob = r["prob"]
        pred = int(prob >= threshold) if not np.isnan(prob) else -1
        rows.append({
            "split":       split,
            "tile_id":     r["tile_id"],
            "transition":  f"{r['t1_lbl']}→{r['t2_lbl']}",
            "true_label":  r["true_label"],
            "prob":        round(float(prob), 4),
            "pred":        pred,
            "correct":     (pred == r["true_label"]),
            "dampened":    r.get("consistency_dampened", False),
            "t1":          r["t1"],
            "t2":          r["t2"],
        })

    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  Terminal report
# ══════════════════════════════════════════════════════════════════════════════

def print_split_report(rows, threshold, split_name):
    from sklearn.metrics import roc_auc_score, average_precision_score

    y_true = np.array([r["true_label"] for r in rows])
    y_prob = np.array([r["prob"] for r in rows])
    y_pred = np.array([r["pred"] for r in rows])

    mask = ~np.isnan(y_prob)
    y_true_v, y_prob_v, y_pred_v = y_true[mask], y_prob[mask], y_pred[mask]

    tp = ((y_pred_v == 1) & (y_true_v == 1)).sum()
    fp = ((y_pred_v == 1) & (y_true_v == 0)).sum()
    fn = ((y_pred_v == 0) & (y_true_v == 1)).sum()
    tn = ((y_pred_v == 0) & (y_true_v == 0)).sum()

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f2   = (5 * prec * rec) / (4 * prec + rec) if (prec + rec) > 0 else 0

    has_both = len(np.unique(y_true_v)) > 1
    auc = roc_auc_score(y_true_v, y_prob_v) if has_both else float("nan")
    ap  = average_precision_score(y_true_v, y_prob_v) if has_both else float("nan")

    print(f"\n{'═'*70}")
    print(f"  {split_name.upper()}  ({len(rows)} pairs · thresh={threshold:.2f})")
    print(f"{'═'*70}")
    print(f"  AUC={auc:.3f}  AP={ap:.3f}  F2={f2:.3f}  "
          f"P={prec:.3f}  R={rec:.3f}  |  TP={tp} FP={fp} FN={fn} TN={tn}")
    print()

    # Group by tile
    by_tile = defaultdict(list)
    for r in rows:
        by_tile[r["tile_id"]].append(r)

    for tile_id in sorted(by_tile):
        tile_rows = by_tile[tile_id]
        for r in tile_rows:
            status = "✓" if r["correct"] else "✗"
            label  = "POS" if r["true_label"] == 1 else "neg"
            flag   = "  ← ENCROACHMENT" if r["pred"] == 1 else ""
            tp_fp  = ""
            if r["true_label"] == 1 and r["pred"] == 1: tp_fp = " [TP]"
            elif r["true_label"] == 0 and r["pred"] == 1: tp_fp = " [FP]"
            elif r["true_label"] == 1 and r["pred"] == 0: tp_fp = " [FN]"
            else: tp_fp = " [TN]"

            bar = _conf_bar(r["prob"])
            print(f"  {status} tile_{tile_id:02d}  {r['transition']:7s}  "
                  f"true={label}  p={r['prob']:.3f} {bar} {r['pred']:1d}{tp_fp}{flag}")


# ══════════════════════════════════════════════════════════════════════════════
#  HTML report
# ══════════════════════════════════════════════════════════════════════════════

def write_html(all_rows, threshold, model_name, test_auc):
    splits_data = {}
    for split in SPLITS:
        rows = [r for r in all_rows if r["split"] == split]
        if not rows:
            continue
        y_true = np.array([r["true_label"] for r in rows])
        y_prob = np.array([r["prob"] for r in rows])
        y_pred = np.array([r["pred"] for r in rows])
        mask = ~np.isnan(y_prob)
        yt, yp, yd = y_true[mask], y_prob[mask], y_pred[mask]
        tp = int(((yd==1)&(yt==1)).sum())
        fp = int(((yd==1)&(yt==0)).sum())
        fn = int(((yd==0)&(yt==1)).sum())
        tn = int(((yd==0)&(yt==0)).sum())
        from sklearn.metrics import roc_auc_score, average_precision_score
        has_both = len(np.unique(yt)) > 1
        auc = float(roc_auc_score(yt, yp)) if has_both else float("nan")
        ap  = float(average_precision_score(yt, yp)) if has_both else float("nan")
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0
        f2   = (5*prec*rec)/(4*prec+rec) if (prec+rec)>0 else 0
        splits_data[split] = {
            "rows": rows, "auc": auc, "ap": ap, "f2": f2,
            "prec": prec, "rec": rec,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    rows_json = json.dumps(all_rows)
    splits_json = json.dumps({k: {x: v[x] for x in v if x != "rows"}
                               for k, v in splits_data.items()})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KEMET1 — Evaluation Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e0e0e0;padding:24px}}
  h1{{font-size:1.5rem;font-weight:600;color:#fff;margin-bottom:4px}}
  .subtitle{{color:#888;font-size:0.85rem;margin-bottom:20px}}
  .grid{{display:grid;gap:16px}}
  .row-3{{grid-template-columns:repeat(3,1fr)}}
  .row-2{{grid-template-columns:1fr 1fr}}
  .card{{background:#1a1d27;border:1px solid #2a2d3a;border-radius:10px;padding:18px}}
  .card-title{{font-size:0.78rem;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:.05em;margin-bottom:14px}}
  .tabs{{display:flex;gap:0;margin-bottom:16px;border-bottom:1px solid #2a2d3a}}
  .tab{{padding:8px 18px;cursor:pointer;font-size:0.82rem;color:#666;border-bottom:2px solid transparent}}
  .tab.active{{color:#fff;border-bottom-color:#4ade80}}
  .tab:hover{{color:#aaa}}
  table{{width:100%;border-collapse:collapse;font-size:0.8rem}}
  th{{color:#666;font-weight:600;padding:8px 10px;text-align:left;border-bottom:1px solid #2a2d3a;font-size:0.72rem;text-transform:uppercase}}
  td{{padding:8px 10px;border-bottom:1px solid #1e2130;font-family:monospace}}
  tr.tp td{{background:#1a2a1a}}
  tr.fp td{{background:#2a2a1a}}
  tr.fn td{{background:#2a1a1a}}
  tr.tn td{{color:#555}}
  .prob-bar{{background:#2a2d3a;border-radius:3px;height:10px;width:80px;display:inline-block;vertical-align:middle;overflow:hidden}}
  .prob-fill{{height:100%;border-radius:3px}}
  .badge{{display:inline-block;padding:1px 6px;border-radius:4px;font-size:0.68rem;font-weight:600}}
  .b-tp{{background:#1a2a1a;color:#4ade80}} .b-fp{{background:#2a2a1a;color:#fbbf24}}
  .b-fn{{background:#2a1a1a;color:#f87171}} .b-tn{{background:#1e2030;color:#555}}
  .b-encr{{background:#1a2a1a;color:#4ade80}} .b-clean{{background:#1e2030;color:#888}}
  .metric-big{{font-size:2rem;font-weight:700;color:#fff}}
  .metric-sub{{font-size:0.78rem;color:#888;margin-top:4px}}
  .chip-row{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}}
  .chip{{background:#1a1d27;border:1px solid #2a2d3a;border-radius:7px;padding:8px 14px}}
  .chip .v{{font-size:1.2rem;font-weight:700;color:#fff}} .chip .l{{font-size:0.68rem;color:#666}}
  .hidden{{display:none}}
  .search{{background:#1e2130;border:1px solid #2a2d3a;border-radius:6px;padding:6px 12px;color:#e0e0e0;font-size:0.82rem;width:220px;margin-bottom:14px}}
  .search:focus{{outline:none;border-color:#4ade80}}
  .filter-row{{display:flex;gap:10px;align-items:center;margin-bottom:14px;flex-wrap:wrap}}
  .filter-btn{{padding:4px 12px;border-radius:16px;border:1px solid #2a2d3a;background:#1e2130;color:#888;cursor:pointer;font-size:0.75rem}}
  .filter-btn.active{{background:#1a2a1a;border-color:#4ade80;color:#4ade80}}
</style>
</head>
<body>
<h1>KEMET1 — Evaluation Report</h1>
<p class="subtitle">Model: {model_name.strip()} &nbsp;·&nbsp; Threshold: {threshold:.2f} &nbsp;·&nbsp; Test AUC: {test_auc:.3f}</p>

<div class="chip-row" id="summaryChips"></div>

<div class="grid row-2" style="margin-bottom:16px">
  <div class="card">
    <div class="card-title">Probability Distribution by Class</div>
    <canvas id="distChart" height="180"></canvas>
  </div>
  <div class="card">
    <div class="card-title">Metrics by Split</div>
    <canvas id="metricsChart" height="180"></canvas>
  </div>
</div>

<div class="card">
  <div class="card-title">Per-Pair Predictions</div>
  <div class="tabs">
    <div class="tab active" onclick="showTab('all',this)">All</div>
    <div class="tab" onclick="showTab('train',this)">Train</div>
    <div class="tab" onclick="showTab('val',this)">Val</div>
    <div class="tab" onclick="showTab('test',this)">Test</div>
    <div class="tab" onclick="showTab('errors',this)">Errors only</div>
  </div>
  <div class="filter-row">
    <input class="search" id="searchBox" placeholder="Search tile ID..." oninput="filterTable()">
    <button class="filter-btn active" onclick="toggleFilter('all-types',this)">All types</button>
    <button class="filter-btn" onclick="toggleFilter('pos',this)">Positives only</button>
    <button class="filter-btn" onclick="toggleFilter('fp',this)">FP only</button>
    <button class="filter-btn" onclick="toggleFilter('fn',this)">FN only</button>
  </div>
  <table id="predTable">
    <thead>
      <tr>
        <th>Split</th><th>Tile</th><th>Transition</th>
        <th>True label</th><th>Probability</th><th>Decision</th><th>Result</th>
      </tr>
    </thead>
    <tbody id="tableBody"></tbody>
  </table>
</div>

<script>
const ROWS = {rows_json};
const SPLITS_DATA = {splits_json};
const THRESHOLD = {threshold};

// ── Summary chips ─────────────────────────────────────────────────────────────
const s = SPLITS_DATA;
const chips = [
  {{v: Object.values(s).reduce((a,x)=>a+x.tp+x.fp+x.fn+x.tn,0), l:'Total pairs'}},
  {{v: Object.values(s).reduce((a,x)=>a+x.tp+x.fn,0), l:'True positives'}},
  {{v: (s.test?.auc||0).toFixed(3), l:'Test AUC'}},
  {{v: (s.test?.rec||0).toFixed(3), l:'Test Recall'}},
  {{v: s.test?.fp||0, l:'Test FP count'}},
  {{v: (s.test?.f2||0).toFixed(3), l:'Test F2'}},
];
document.getElementById('summaryChips').innerHTML = chips.map(c=>
  `<div class="chip"><div class="v">${{c.v}}</div><div class="l">${{c.l}}</div></div>`).join('');

// ── Probability distribution chart ────────────────────────────────────────────
const posProbs = ROWS.filter(r=>r.true_label===1).map(r=>r.prob);
const negProbs = ROWS.filter(r=>r.true_label===0).map(r=>r.prob);

function histogram(probs, bins=20) {{
  const counts = new Array(bins).fill(0);
  probs.forEach(p=>{{ const b=Math.min(Math.floor(p*bins),bins-1); counts[b]++; }});
  return counts;
}}
const labels = Array.from({{length:20}},(_,i)=>(i/20).toFixed(2));
const dCtx = document.getElementById('distChart').getContext('2d');
new Chart(dCtx,{{
  type:'bar',
  data:{{
    labels,
    datasets:[
      {{label:'Negative (true_label=0)',data:histogram(negProbs),backgroundColor:'rgba(96,165,250,0.5)',borderColor:'#60a5fa',borderWidth:1}},
      {{label:'Positive (true_label=1)',data:histogram(posProbs),backgroundColor:'rgba(74,222,128,0.6)',borderColor:'#4ade80',borderWidth:1}},
    ]
  }},
  options:{{
    responsive:true,
    plugins:{{
      legend:{{labels:{{color:'#aaa',font:{{size:11}}}}}},
      annotation:{{}}
    }},
    scales:{{
      x:{{title:{{display:true,text:'Predicted probability',color:'#666'}},ticks:{{color:'#888',maxTicksLimit:10}},grid:{{color:'#2a2d3a'}}}},
      y:{{title:{{display:true,text:'Count',color:'#666'}},ticks:{{color:'#888'}},grid:{{color:'#2a2d3a'}}}}
    }}
  }}
}});

// Threshold line annotation (manual)
dCtx.canvas.addEventListener('chartrendered',()=>{{
  const chart = Chart.getChart(dCtx);
  const x = chart.scales.x.getPixelForValue(THRESHOLD*20);
  // draw threshold line
}});

// ── Metrics bar chart ─────────────────────────────────────────────────────────
const mCtx = document.getElementById('metricsChart').getContext('2d');
new Chart(mCtx,{{
  type:'bar',
  data:{{
    labels:['AUC','Avg Prec','Recall','Precision','F2'],
    datasets: Object.entries(SPLITS_DATA).map(([split,d],i)=>{{
      const colors = ['rgba(96,165,250,0.6)','rgba(192,132,252,0.6)','rgba(74,222,128,0.6)'];
      const borders = ['#60a5fa','#c084fc','#4ade80'];
      return {{
        label: split,
        data:[d.auc,d.ap,d.rec,d.prec,d.f2].map(v=>isNaN(v)?0:parseFloat(v.toFixed(3))),
        backgroundColor:colors[i], borderColor:borders[i], borderWidth:1
      }};
    }})
  }},
  options:{{
    responsive:true,
    scales:{{
      x:{{ticks:{{color:'#888'}},grid:{{color:'#2a2d3a'}}}},
      y:{{min:0,max:1.1,ticks:{{color:'#888',stepSize:0.2}},grid:{{color:'#2a2d3a'}}}}
    }},
    plugins:{{legend:{{labels:{{color:'#aaa',font:{{size:11}}}}}}}}
  }}
}});

// ── Table ─────────────────────────────────────────────────────────────────────
let currentTab = 'all';
let currentFilter = 'all-types';

function buildTable(rows) {{
  const body = document.getElementById('tableBody');
  body.innerHTML = '';
  rows.forEach(r => {{
    let outcome = '';
    if (r.true_label===1 && r.pred===1) outcome='tp';
    else if (r.true_label===0 && r.pred===1) outcome='fp';
    else if (r.true_label===1 && r.pred===0) outcome='fn';
    else outcome='tn';

    const badgeClass = {{tp:'b-tp',fp:'b-fp',fn:'b-fn',tn:'b-tn'}}[outcome];
    const badgeText  = outcome.toUpperCase();
    const probColor  = r.prob >= THRESHOLD ? '#4ade80' : '#60a5fa';
    const decisionBadge = r.pred===1
      ? '<span class="badge b-encr">ENCROACHMENT</span>'
      : '<span class="badge b-clean">clean</span>';
    const trueBadge = r.true_label===1
      ? '<span class="badge b-encr">POS</span>'
      : '<span style="color:#555">neg</span>';
    const pct = Math.round(r.prob*100);

    const tr = document.createElement('tr');
    tr.className = outcome;
    tr.dataset.split = r.split;
    tr.dataset.outcome = outcome;
    tr.dataset.tile = r.tile_id;
    tr.dataset.truelabel = r.true_label;
    tr.innerHTML = `
      <td>${{r.split}}</td>
      <td>tile_${{String(r.tile_id).padStart(2,'0')}}</td>
      <td>${{r.transition}}</td>
      <td>${{trueBadge}}</td>
      <td>
        ${{r.prob.toFixed(3)}}
        <span class="prob-bar">
          <span class="prob-fill" style="width:${{pct}}%;background:${{probColor}}"></span>
        </span>
      </td>
      <td>${{decisionBadge}}${{r.dampened ? ' <span title="temporal consistency dampened" style="color:#888;font-size:0.7rem">🔇</span>' : ''}}</td>
      <td><span class="badge ${{badgeClass}}">${{badgeText}}</span></td>`;
    body.appendChild(tr);
  }});
}}

function filterTable() {{
  const search = document.getElementById('searchBox').value.toLowerCase();
  const rows = document.querySelectorAll('#tableBody tr');
  rows.forEach(tr => {{
    const tile = tr.dataset.tile;
    const split = tr.dataset.split;
    const outcome = tr.dataset.outcome;
    const truelabel = tr.dataset.truelabel;

    const matchTab = currentTab === 'all' ||
      (currentTab === 'errors' && (outcome==='fp'||outcome==='fn')) ||
      currentTab === split;

    const matchFilter = currentFilter === 'all-types' ||
      (currentFilter === 'pos' && truelabel === '1') ||
      (currentFilter === 'fp' && outcome === 'fp') ||
      (currentFilter === 'fn' && outcome === 'fn');

    const matchSearch = !search || tile.includes(search);

    tr.style.display = (matchTab && matchFilter && matchSearch) ? '' : 'none';
  }});
}}

function showTab(name, el) {{
  currentTab = name;
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  el.classList.add('active');
  filterTable();
}}

function toggleFilter(name, el) {{
  currentFilter = name;
  document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
  el.classList.add('active');
  filterTable();
}}

// Build initial table
buildTable(ROWS);
</script>
</body>
</html>"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  HTML report → {REPORT_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",   default="all", choices=["all","train","val","test"])
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args()

    # Load model
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model     = bundle["model"]
    threshold = bundle["threshold"]
    model_name = bundle.get("model_name", "RF")
    test_auc   = bundle.get("test_auc", float("nan"))

    print(f"\n{'═'*70}")
    print(f"  KEMET1 — Evaluation")
    print(f"{'═'*70}")
    print(f"  Model     : {model_name.strip()}")
    print(f"  Threshold : {threshold:.2f}  (F2-optimal)")
    print(f"  Test AUC  : {test_auc:.3f}  (from training run)")

    splits_to_run = SPLITS if args.split == "all" else [args.split]

    all_rows = []
    for split in splits_to_run:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            print(f"\n  ⚠  {split_dir} not found, skipping")
            continue
        rows = evaluate_split(split, model, threshold)
        all_rows.extend(rows)
        print_split_report(rows, threshold, split)

    # Overall summary
    if len(splits_to_run) > 1:
        from sklearn.metrics import roc_auc_score
        y_true = np.array([r["true_label"] for r in all_rows])
        y_prob = np.array([r["prob"] for r in all_rows])
        mask = ~np.isnan(y_prob)
        if mask.sum() > 0 and len(np.unique(y_true[mask])) > 1:
            oa = roc_auc_score(y_true[mask], y_prob[mask])
            print(f"\n  Overall AUC (all splits pooled): {oa:.4f}")

    if not args.no_html and all_rows:
        write_html(all_rows, threshold, model_name, test_auc)

    print(f"\n{'═'*70}\n")


if __name__ == "__main__":
    main()
