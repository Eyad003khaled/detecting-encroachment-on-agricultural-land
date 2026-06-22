"""
Step 06 — Agriculture Segmentation
Uses SegFormer-B4 (HuggingFace) to produce a binary agricultural-land mask.
"""

from __future__ import annotations
from typing import Dict, Any

import numpy as np

from config.settings import AGRICULTURE_SEGMENTATION_CONFIG, SPECTRAL_INDICES_CONFIG, PROCESSED_DIR
from src.utils.logger import get_logger
from src.utils.geo_utils import write_geotiff

logger = get_logger("step_06")
CFG = AGRICULTURE_SEGMENTATION_CONFIG

# ADE20K class IDs that correspond to farmland / vegetation
AGRI_CLASS_IDS = {9, 17, 29, 72, 94}   # field, grass, plant, land, farmland


def run(image: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Segment agricultural land in the T1 (before) image.

    Returns:
        agri_mask: binary [H,W] uint8 (1 = agricultural pixel)
    """
    if SPECTRAL_INDICES_CONFIG.get("precomputed_indices", False):
        # SegFormer expects real RGB — feeding spectral index bands would give garbage.
        # Use Band 0 (NDVI) directly as a reliable fast fallback.
        logger.info("Pre-computed index mode → using Band 0 (NDVI) threshold for agriculture mask")
        agri_mask = _ndvi_fallback(image)
    else:
        try:
            agri_mask = _run_segformer(image)
        except Exception as e:
            logger.warning(f"SegFormer failed ({e}). Using NDVI-based fallback.")
            agri_mask = _ndvi_fallback(image)

    agri_pct = float(agri_mask.mean() * 100)
    logger.info(f"Agricultural land detected: {agri_pct:.2f}% of image")

    # Save for later steps / inspection
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    write_geotiff(PROCESSED_DIR / "agriculture_mask.tif", agri_mask, meta)

    return {"agri_mask": agri_mask, "agri_pct": agri_pct}


def _run_segformer(image: np.ndarray) -> np.ndarray:
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    import torch, torch.nn.functional as F
    from PIL import Image as PILImage

    processor = SegformerImageProcessor.from_pretrained(CFG["model_name"])
    model = SegformerForSemanticSegmentation.from_pretrained(CFG["model_name"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    # Convert to RGB PIL image (bands 2,1,0 → R,G,B)
    rgb = np.stack([image[2], image[1], image[0]], axis=-1)
    rgb = np.clip(rgb / rgb.max() * 255, 0, 255).astype(np.uint8)
    pil_img = PILImage.fromarray(rgb)

    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits   # [1, num_classes, H/4, W/4]

    # Upsample to original resolution
    H, W = image.shape[1], image.shape[2]
    upsampled = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    pred_classes = upsampled.argmax(dim=1).squeeze().cpu().numpy()

    agri_mask = np.isin(pred_classes, list(AGRI_CLASS_IDS)).astype(np.uint8)
    return agri_mask


def _ndvi_fallback(image: np.ndarray) -> np.ndarray:
    """Simple NDVI threshold as fallback."""
    if SPECTRAL_INDICES_CONFIG.get("precomputed_indices", False):
        # Band 0 is already NDVI — use directly
        ndvi = image[0].astype(np.float32)
    else:
        nir = image[3] if image.shape[0] > 3 else image[-1]
        red = image[2] if image.shape[0] > 2 else image[0]
        denom = nir + red
        ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)
    return (ndvi > 0.2).astype(np.uint8)
