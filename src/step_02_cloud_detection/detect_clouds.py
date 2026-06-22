"""
Step 02 — Cloud Detection
Uses U-Net + ResNet34 to produce a cloud probability map and binary mask.
Also computes cloud coverage % used by the pipeline to decide whether
to run Step 03 (cloud removal) or skip it.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import numpy as np

from config.settings import CLOUD_DETECTION_CONFIG, SPECTRAL_INDICES_CONFIG
from src.utils.logger import get_logger

logger = get_logger("step_02")

SKIP_THRESHOLD = CLOUD_DETECTION_CONFIG["skip_removal_below_pct"]
PRECOMPUTED   = SPECTRAL_INDICES_CONFIG.get("precomputed_indices", False)


def run(t1_path: Path, t2_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Detect clouds in T1 and T2.

    Returns:
        {
          "T1": {"prob": ndarray, "mask": ndarray, "coverage_pct": float},
          "T2": {"prob": ndarray, "mask": ndarray, "coverage_pct": float},
          "skip_removal": bool   # True when both images are mostly cloud-free
        }
    """
    from src.utils.geo_utils import read_geotiff

    t1_data, _ = read_geotiff(t1_path)
    t2_data, _ = read_geotiff(t2_path)

    if PRECOMPUTED:
        # Input bands are pre-computed indices (NDVI, NDBI, …), not raw reflectance.
        # Cloud detection is meaningless — return 0 % coverage and skip removal.
        h, w = t1_data.shape[1], t1_data.shape[2]
        zero_mask = np.zeros((h, w), dtype=np.uint8)
        logger.info("[T1] Pre-computed index bands — cloud detection skipped (0 % coverage)")
        logger.info("[T2] Pre-computed index bands — cloud detection skipped (0 % coverage)")
        t1_result = {"prob": zero_mask.astype(np.float32), "mask": zero_mask, "coverage_pct": 0.0}
        h, w = t2_data.shape[1], t2_data.shape[2]
        zero_mask2 = np.zeros((h, w), dtype=np.uint8)
        t2_result = {"prob": zero_mask2.astype(np.float32), "mask": zero_mask2, "coverage_pct": 0.0}
    else:
        t1_result = _detect_single(t1_data, label="T1")
        t2_result = _detect_single(t2_data, label="T2")

    # Skip cloud removal when BOTH images are below the threshold
    max_coverage = max(t1_result["coverage_pct"], t2_result["coverage_pct"])
    skip = max_coverage < SKIP_THRESHOLD

    if skip:
        logger.info(
            f"Cloud coverage T1={t1_result['coverage_pct']:.1f}%  "
            f"T2={t2_result['coverage_pct']:.1f}%  →  "
            f"Below {SKIP_THRESHOLD}% threshold — Step 03 will be SKIPPED"
        )
    else:
        logger.info(
            f"Cloud coverage T1={t1_result['coverage_pct']:.1f}%  "
            f"T2={t2_result['coverage_pct']:.1f}%  →  "
            f"Step 03 (cloud removal) will run"
        )

    return {"T1": t1_result, "T2": t2_result, "skip_removal": skip}


def _detect_single(image: np.ndarray, label: str) -> Dict[str, Any]:
    """Run cloud detection on a single image."""
    weights = Path(CLOUD_DETECTION_CONFIG["model_weights"])

    if weights.exists():
        prob_map = _run_model(image, weights)
    else:
        logger.warning(
            f"[{label}] Cloud detection weights not found at {weights}. "
            "Using heuristic fallback (NIR + SWIR threshold)."
        )
        prob_map = _heuristic_cloud_prob(image)

    threshold = CLOUD_DETECTION_CONFIG["threshold"]
    mask = (prob_map > threshold).astype(np.uint8)
    coverage_pct = float(mask.mean() * 100)

    logger.info(f"[{label}] Cloud coverage: {coverage_pct:.2f}%")
    return {"prob": prob_map, "mask": mask, "coverage_pct": coverage_pct}


def _run_model(image: np.ndarray, weights: Path) -> np.ndarray:
    """Run the U-Net + ResNet34 model."""
    import torch
    import segmentation_models_pytorch as smp

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = smp.Unet(encoder_name="resnet34", in_channels=image.shape[0], classes=1)
    model.load_state_dict(torch.load(str(weights), map_location=device))
    model.eval().to(device)

    tile_size = CLOUD_DETECTION_CONFIG["tile_size"]
    _, H, W = image.shape
    prob_map = np.zeros((H, W), dtype=np.float32)

    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor).squeeze().cpu().numpy()
    prob_map = 1 / (1 + np.exp(-out))   # sigmoid
    return prob_map


def _heuristic_cloud_prob(image: np.ndarray) -> np.ndarray:
    """
    Simple heuristic: bright pixels in blue + high reflectance across all bands.
    Works as a reasonable fallback when model weights are absent.
    """
    # Normalise each band to [0,1]
    normed = np.zeros_like(image)
    for i in range(image.shape[0]):
        band = image[i]
        rng = band.max() - band.min()
        normed[i] = (band - band.min()) / rng if rng > 0 else band

    # Clouds are bright in blue (band 0) AND overall high reflectance
    blue      = normed[0] if image.shape[0] > 0 else np.zeros(image.shape[1:])
    mean_refl = normed.mean(axis=0)
    prob = np.clip((blue * 0.5 + mean_refl * 0.5), 0, 1)
    return prob.astype(np.float32)
