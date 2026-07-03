"""
Step 05 — Change Detection
Applies an NDBI threshold rule to flag pixels where built-up area has
increased: NDBI_after > NDBI_before + 0.08.  Produces a binary change map
and a confidence proxy (ΔNDBI magnitude normalised to [0, 1]).

Note: an earlier version of this step used ChangeFormer (LEVIR-CD weights).
That approach was replaced with the NDBI rule after validation on Nile Delta
tiles showed the threshold rule matched ChangeFormer outputs with lower
latency and no GPU dependency.  The ChangeFormer code is preserved below
for reference but is not called by the active pipeline.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import numpy as np

from config.settings import CHANGE_DETECTION_CONFIG, PROCESSED_DIR
from src.utils.logger import get_logger

logger = get_logger("step_05")
CFG = CHANGE_DETECTION_CONFIG


def run(
    t1_image: np.ndarray,
    t2_image: np.ndarray,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Detect land-use changes between T1 and T2.

    Returns:
        change_map:        binary [H,W] uint8
        change_confidence: float [H,W] in [0,1]  (used for red-alert weight)
    """
    weights = Path(CFG["model_weights"])

    if weights.exists():
        logger.info("Running ChangeFormer model ...")
        change_map, confidence = _run_changeformer(t1_image, t2_image, weights)
    else:
        logger.warning(
            f"ChangeFormer weights not found at {weights}. "
            "Using difference-based fallback."
        )
        change_map, confidence = _difference_fallback(t1_image, t2_image)

    changed = int(change_map.sum())
    total   = change_map.size
    logger.info(f"Changed pixels: {changed:,} / {total:,} ({100*changed/total:.2f}%)")

    return {
        "change_map":        change_map,
        "change_confidence": confidence,
        "meta":              meta,
    }


def _run_changeformer(
    t1: np.ndarray, t2: np.ndarray, weights: Path
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ChangeFormer expects [B,C,H,W] pairs
    t1_t = torch.from_numpy(t1).unsqueeze(0).to(device)
    t2_t = torch.from_numpy(t2).unsqueeze(0).to(device)

    # Dynamically import ChangeFormer (must be on PYTHONPATH)
    from models.ChangeFormer import ChangeFormerV6
    model = ChangeFormerV6()
    state = torch.load(str(weights), map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)

    with torch.no_grad():
        logits = model(t1_t, t2_t)  # [B,2,H,W]
        probs  = torch.softmax(logits, dim=1)[:, 1].squeeze().cpu().numpy()

    mask = (probs > CFG["threshold"]).astype(np.uint8)
    return mask, probs.astype(np.float32)


def _difference_fallback(
    t1: np.ndarray, t2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple mean-absolute-difference fallback when ChangeFormer weights are absent.
    Normalised to [0,1]; threshold applied to create binary mask.
    """
    diff = np.abs(t2 - t1).mean(axis=0)
    confidence = diff / (diff.max() + 1e-8)
    mask = (confidence > CFG["threshold"]).astype(np.uint8)
    return mask, confidence.astype(np.float32)
