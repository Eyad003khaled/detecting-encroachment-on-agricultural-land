# Food Security AI System

## End-to-End Pipeline for Detecting Agricultural Land Encroachment in Egypt

An 8-step ML pipeline that compares multi-temporal Sentinel-2 satellite imagery to detect where buildings have been constructed on agricultural land — validated with the **KEMET1 BeforeAfter** Random Forest classifier trained on 300 Nile Delta sites.

---

## KEMET1 BeforeAfter Classifier — Results

Trained on 300 co-registered Sentinel-2 tile pairs (before: 2024, after: 2025) covering the Nile Delta. Detects agricultural-land encroachment using **44 spectral-change features** (7 statistics × 6 spectral bands + ΔNDVI mean + ΔNDBI mean).

**Auto-labelling:** `pct_conv > 0.02576 OR max_cluster_ha > 3.13` → 78 positive / 222 negative sites  
**Split:** 70% train / 15% val / 15% test (stratified, seed=42)

> **Circularity fix (v5 — production model):** early versions included `pct_conv` and `pct_new` in the feature vector — features derived from the same spectral threshold that generated the auto-labels. These were removed. The table below shows v5 (non-circular) metrics only.

| Metric | Value | Note |
|--------|-------|------|
| **Val AUC** | **0.861** | Primary metric (v5, non-circular) |
| **Test AUC** | **0.872** | Held-out test split |
| Decision threshold | 0.29 | RF probability → alarm |
| Alarm tiers | High ≥ 0.40 · Medium ≥ 0.23 | Fusion score (0.65×RF + 0.35×spectral) |

### KEMET1 BeforeAfter — Quick Start

```powershell
# Install dependencies (BA inference only — see requirements.txt)
pip install -r requirements.txt

# Retrain v5 model (non-circular, 44 features) → saves models/ba_rf_model.pkl
python ablation_no_circular.py

# Evaluate on held-out test split (reproduces dashboard metrics)
python evaluate_ba.py

# Single-site inference → HTML report in outputs/
python run_inference.py --site site0
# (tiles must be at data/KEMET1_BeforeAfter/KEMET1_BeforeAfter_Tiles/site0_before_2024.tif)

# Regenerate all 78 positive-site HTML reports → reports/
python batch_report.py

# Upload one site to backend API
python upload_case.py site0
python upload_case.py site0 --dry-run   # preview without submitting

# Batch-upload all 78 positive sites with geocoding + retry logic
python batch_upload.py
python batch_upload.py --dry-run
python batch_upload.py --sites site0 site3 site15   # selective

# View interactive dashboards
start results\training_results.html
start results\encroachment_summary.html
```

---

## KEMET1 Tile Classifier (legacy multi-temporal)

The original classifier operates on per-tile spectral statistics across 4 time periods (T1–T4) with temporal consistency post-processing.

| Metric | Value |
|--------|-------|
| Pooled AUC (all 201 pairs) | 0.990 |
| Val AUC (post-processing) | 0.963 |
| Test AUC (post-processing) | 0.988 |
| Recall | 1.000 |
| F2 score | 0.909 |
| False Positives (val / test) | 3 per split |

```bash
python train_classifier.py --no-cv
python evaluate.py
python predict.py --t1 T1.tif --t2 T2.tif
```

---

## Full Pipeline — 8 Steps

| # | Step | Model / Tool | Output |
|---|------|-------------|--------|
| 01 | Data Acquisition | Sentinel-2 via GEE | GeoTIFFs (T1, T2) |
| 02 | Cloud Detection | U-Net + ResNet34 (38-Cloud) | Binary cloud mask |
| 03 | Cloud Removal | OpenCV Telea inpainting | Clean GeoTIFF |
| 04 | Spectral Indices | NumPy (NDVI, NDBI, MNDWI) | Index rasters |
| 05 | Change Detection | NDBI threshold rule (NDBI_after > NDBI_before + 0.08) | Binary change map |
| 06 | Agriculture Segmentation | SegFormer-B4 (ADE20K) | Agriculture mask |
| 07 | Building Detection | SAM (SpaceNet v2) | Building footprints |
| 08 | Final Output | Spatial intersection | Colored map + JSON report |

**Color coding:** Red = buildings on farmland | Yellow = changed vegetation | Green = stable agriculture

```bash
# Run full pipeline on a tile pair
python pipeline.py full --before T1.tif --after T2.tif

# Run BeforeAfter inference only
python pipeline.py beforeafter --before site0_before_2024.tif \
                               --after  site0_after_2025.tif \
                               --site-name site0
```

---

## Repository Structure

```
models/
  ba_rf_model.pkl          # KEMET1 BeforeAfter RF v5 (44 features, non-circular)
  ba_rf_ablation.pkl       # Same model retrained via ablation_no_circular.py
  encroachment_classifier_rf.pkl  # Legacy tile classifier
data/
  KEMET1_BeforeAfter/
    KEMET1_BeforeAfter_Tiles/    # 300 site pairs (site{N}_before_2024.tif / _after_2025.tif)
  ba_labels.json           # 78 pos / 222 neg ground-truth labels
reports/                   # Pre-generated per-site HTML reports (78 positive sites)
outputs/                   # Runtime-generated reports (gitignored)
src/                       # Pipeline step modules (steps 01-08)
results/
  training_results.html    # Interactive evaluation dashboard
  encroachment_summary.html# Master Egypt map of all detected sites
  ablation_results.json    # Ablation study output
  viz/                     # Sample site imagery and change maps
docs/
  KEMET1_Final_Report.pdf  # Full GP report
  GP_Presentation.pptx     # Defence presentation
  KEMET1_Technical_Documentation.docx  # Canonical technical doc
  datasets_reference.md    # Dataset sources and notes
upload_case.py             # Upload one site to backend API
batch_upload.py            # Batch-upload all 78 positive sites
run_inference.py           # Core BA inference engine (HTML reports + JSON)
batch_report.py            # Regenerate all 78 site HTML reports
```

---

## Requirements

**BA inference (what actually runs):** `rasterio, numpy, scikit-learn, scipy, matplotlib, folium, requests, fpdf2`

**Full pipeline (steps 01–08, optional):** additionally requires `torch, transformers, ultralytics, segment-anything, segmentation-models-pytorch, earthengine-api`

See `requirements.txt` for pinned versions. The DL stack is listed under a separate comment block and can be skipped if only running BA inference.
