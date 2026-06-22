"""
Global configuration for the Food Security ML Pipeline.
Edit values here to tune thresholds, paths, and model settings.
"""

from pathlib import Path

# ── Project root ────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
OUTPUT_DIR     = ROOT_DIR / "outputs"
WEIGHTS_DIR    = ROOT_DIR / "weights"

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_CONFIG = {
    "log_file": str(OUTPUT_DIR / "pipeline.log"),
    "level": "INFO",
}

# ── Step 01 — GEE ───────────────────────────────────────────────────────────
GEE_CONFIG = {
    "project":     "your-gee-project-id",
    "start_date":  "2020-01-01",
    "end_date":    "2021-01-01",
    "region":      [30.0, 25.0, 32.0, 27.0],   # [lon_min, lat_min, lon_max, lat_max]
    "scale":       10,                           # metres per pixel
    "bands":       ["B2", "B3", "B4", "B8", "B11", "B12"],
}

# ── Step 02 — Cloud Detection ────────────────────────────────────────────────
CLOUD_DETECTION_CONFIG = {
    "model_weights":    str(WEIGHTS_DIR / "cloud_unet_resnet34.pth"),
    "threshold":        0.5,    # probability → binary mask
    # If cloud coverage (%) is BELOW this value, Step 03 is skipped entirely.
    "skip_removal_below_pct": 15.0,
    "tile_size":        512,
    "batch_size":       4,
}

# ── Step 03 — Cloud Removal ──────────────────────────────────────────────────
CLOUD_REMOVAL_CONFIG = {
    "inpaint_radius":   5,
    "inpaint_method":   "telea",   # "telea" | "ns"
    "output_suffix":    "_cloud_free",
}

# ── Step 04 — Spectral Indices ───────────────────────────────────────────────
SPECTRAL_INDICES_CONFIG = {
    # Band indices (0-based) in the GeoTIFF band order: B2,B3,B4,B8,B11,B12
    "band_blue":  0,
    "band_green": 1,
    "band_red":   2,
    "band_nir":   3,
    "band_swir1": 4,
    "band_swir2": 5,
    # NDVI drop below this → yellow alert (unhealthy vegetation signal)
    "ndvi_degradation_threshold": -0.15,
    # Set True when input TIFFs contain pre-computed indices (e.g. KEMET1):
    # Band 0=NDVI, Band 1=NDBI, Band 2=MNDWI, Band 3=SAVI, Band 4=BSI, Band 5=NDWI
    # Default False — only overridden at runtime via run.py --precomputed flag.
    "precomputed_indices": False,
}

# ── Step 05 — Change Detection ───────────────────────────────────────────────
CHANGE_DETECTION_CONFIG = {
    "model_weights": str(WEIGHTS_DIR / "ChangeFormer_LEVIR.pth"),
    "tile_size":     256,
    "overlap":       32,
    "threshold":     0.5,
    "device":        "cuda",   # "cuda" | "cpu"
}

# ── Step 06 — Agriculture Segmentation ──────────────────────────────────────
AGRICULTURE_SEGMENTATION_CONFIG = {
    "model_name":   "nvidia/segformer-b4-finetuned-ade-512-512",
    "threshold":    0.5,
    "tile_size":    512,
    "batch_size":   2,
}

# ── Step 07 — Building Detection ────────────────────────────────────────────
BUILDING_DETECTION_CONFIG = {
    "yolo_weights":  str(ROOT_DIR / "yolov8m-seg.pt"),
    "sam_weights":   str(WEIGHTS_DIR / "sam_vit_b_01ec64.pth"),
    "sam_type":      "vit_b",
    "yolo_conf":     0.35,
    "yolo_iou":      0.45,
    "device":        "cuda",
}

# ── Step 08 — Final Output ───────────────────────────────────────────────────
FINAL_OUTPUT_CONFIG = {
    "output_dir": str(OUTPUT_DIR),
    # Weighted red-alert score: change detection + spectral contribution
    "red_alert_change_weight":   0.65,
    "red_alert_spectral_weight": 0.35,
    # Minimum red-alert score to flag a region as encroachment
    "red_alert_threshold":       0.50,
    # Reverse geocoding (OpenStreetMap Nominatim)
    "geocode_user_agent": "food_security_pipeline/1.0",
    # Before/after chip size in pixels around each detected region
    "chip_padding_px": 64,
}

# ── Dataset Split ────────────────────────────────────────────────────────────
DATASET_SPLIT_CONFIG = {
    "train_ratio":   0.60,
    "val_ratio":     0.15,
    "test_ratio":    0.15,
    "reserve_ratio": 0.10,
    "random_seed":   42,
}
