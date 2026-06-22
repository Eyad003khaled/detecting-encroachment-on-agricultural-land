"""
Temporal Manager — Same-Season Year-Over-Year Strategy
=======================================================

Problem with a fixed old baseline (e.g. always 2021):
  Crops grow and die, soil moisture changes, vegetation dries out seasonally.
  A 2021 vs 2026 comparison generates enormous noise from natural cycles —
  not real encroachment.

Problem with only comparing 20-days-prior:
  A building finished 40 days ago and unchanged since looks like "no change".

Solution — two-step comparison:
  1. PRIMARY  — same-season, 1 year prior (±SEASON_WINDOW_DAYS)
       Eliminates seasonal noise. Only structural permanent changes survive.
       Example: March 2026 vs March 2025.

  2. RECENCY CHECK — most recent image before the new one (≤ revisit gap)
       Only runs when the primary flags a change.
       Answers "is this brand-new right now, or was it already there last month?"

State is persisted in outputs/temporal_state.json.
All images are registered in a date-stamped archive so the manager can
always find the right comparison image automatically.

Usage (via run.py):
  # Register + run new image
  python run.py --temporal --new-image 2026-03.tif --new-date 2026-03-15

  # Bulk-register historical images without running the pipeline
  python run.py --register --image 2021-03.tif --date 2021-03-10
  python run.py --register --image 2022-03.tif --date 2022-03-08
  ...

  # See what comparisons would run for a new image
  python run.py --temporal --new-image 2026-03.tif --new-date 2026-03-15 --dry-run
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.utils.logger import get_logger

logger = get_logger("temporal")

STATE_FILE = Path("outputs/temporal_state.json")

# How many days either side of "exactly 1 year ago" to search for a same-season image
SEASON_WINDOW_DAYS = 45


# ── State I/O ────────────────────────────────────────────────────────────────

def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        n = len(state.get("archive", []))
        logger.info(f"Loaded temporal state: {n} images in archive")
        return state
    return {"archive": [], "runs": []}


def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def reset_state() -> None:
    if STATE_FILE.exists():
        backup = STATE_FILE.with_suffix(".json.bak")
        shutil.copy2(STATE_FILE, backup)
        logger.info(f"State backed up to {backup}")
    save_state({"archive": [], "runs": []})
    logger.info("Temporal state reset.")


# ── Archive management ───────────────────────────────────────────────────────

def register_image(state: Dict[str, Any], image_path: str, date_str: str) -> Dict[str, Any]:
    """
    Add an image to the archive. Safe to call multiple times for the same image.
    date_str: any ISO-like format — "2025", "2025-03", "2025-03-15"
    """
    parsed = _parse_date(date_str)
    path   = str(Path(image_path).resolve())

    # Avoid duplicates
    for entry in state["archive"]:
        if entry["path"] == path:
            logger.info(f"Image already registered: {date_str}")
            return state

    state["archive"].append({"date": parsed.isoformat(), "path": path, "label": date_str})
    state["archive"].sort(key=lambda e: e["date"])
    logger.info(f"Registered image: {date_str} → {path}")
    return state


# ── Image finders ────────────────────────────────────────────────────────────

def find_same_season_image(
    archive:      List[Dict],
    new_date_str: str,
    exclude_path: Optional[str] = None,
    window_days:  int = SEASON_WINDOW_DAYS,
) -> Optional[Dict]:
    """
    Find the closest archived image to exactly 1 year before new_date.
    Searches within ±window_days of the 12-month mark.
    Returns the best match, or None if nothing is close enough.
    """
    new_date    = _parse_date(new_date_str)
    target_date = new_date - timedelta(days=365)
    best        = None
    best_delta  = timedelta(days=window_days + 1)

    for entry in archive:
        if exclude_path and entry["path"] == exclude_path:
            continue
        entry_date = datetime.fromisoformat(entry["date"])
        if entry_date >= new_date:
            continue   # must be in the past
        delta = abs(entry_date - target_date)
        if delta <= timedelta(days=window_days) and delta < best_delta:
            best       = entry
            best_delta = delta

    if best:
        logger.info(
            f"Same-season baseline: {best['label']}  "
            f"(target was {target_date.date()}, delta = {best_delta.days} days)"
        )
    else:
        logger.warning(
            f"No same-season image found within ±{window_days} days of "
            f"{target_date.date()}. Archive has {len(archive)} entries."
        )
    return best


def find_most_recent_image(
    archive:      List[Dict],
    new_date_str: str,
    exclude_path: Optional[str] = None,
) -> Optional[Dict]:
    """Return the most recent image strictly before new_date."""
    new_date = _parse_date(new_date_str)
    past = [
        e for e in archive
        if datetime.fromisoformat(e["date"]) < new_date
        and (not exclude_path or e["path"] != exclude_path)
    ]
    if not past:
        return None
    recent = max(past, key=lambda e: e["date"])
    logger.info(f"Most recent prior image: {recent['label']}")
    return recent


# ── Comparison plan ───────────────────────────────────────────────────────────

def get_comparison_plan(
    new_image_path: str,
    new_date_str:   str,
    state:          Dict[str, Any],
) -> Dict[str, Any]:
    """
    Decide which comparisons to run for the new image.

    Returns
    -------
    {
      "mode":        "no_archive" | "same_season" | "recent_only",
      "primary":     {t1_path, t2_path, t1_date, t2_date, label} | None,
      "recency":     {t1_path, t2_path, t1_date, t2_date, label} | None,
      "explanation": str   (human-readable summary)
    }
    """
    archive  = state.get("archive", [])
    new_path = str(Path(new_image_path).resolve())

    # ── Not enough history yet ────────────────────────────────────────────────
    if len(archive) == 0:
        return {
            "mode":        "no_archive",
            "primary":     None,
            "recency":     None,
            "explanation": (
                "No images in archive yet. "
                "This image will be registered but no comparison can run. "
                "Add more images first."
            ),
        }

    # ── Look for same-season image from ~1 year ago ───────────────────────────
    same_season = find_same_season_image(archive, new_date_str, exclude_path=new_path)
    most_recent = find_most_recent_image(archive, new_date_str, exclude_path=new_path)

    if same_season:
        primary = _make_comp(same_season, new_image_path, new_date_str, "same_season_primary")

        # Recency check: most recent that is NOT the same-season image
        recency_candidate = find_most_recent_image(
            archive, new_date_str, exclude_path=same_season["path"]
        )
        # Only add recency check if it's meaningfully different from same-season
        recency = None
        if recency_candidate and recency_candidate["path"] != same_season["path"]:
            recency = _make_comp(recency_candidate, new_image_path, new_date_str, "recency_check")

        explanation = (
            f"Primary: {same_season['label']} → {new_date_str}  "
            f"(same-season, {_days_between(same_season['date'], new_date_str)} days apart)\n"
        )
        if recency:
            explanation += (
                f"Recency check: {recency_candidate['label']} → {new_date_str}  "
                f"(only runs if primary flags a change)"
            )
        else:
            explanation += "Recency check: not available (no recent image found)"

        return {
            "mode":        "same_season",
            "primary":     primary,
            "recency":     recency,
            "explanation": explanation,
        }

    elif most_recent:
        # Fallback: no same-season image available — use most recent
        primary = _make_comp(most_recent, new_image_path, new_date_str, "recent_fallback")
        return {
            "mode":        "recent_only",
            "primary":     primary,
            "recency":     None,
            "explanation": (
                f"No same-season image found (need an image from ~"
                f"{(_parse_date(new_date_str) - timedelta(days=365)).year}).\n"
                f"Falling back to most recent: {most_recent['label']} → {new_date_str}.\n"
                f"Results may include seasonal noise."
            ),
        }

    else:
        return {
            "mode":        "no_archive",
            "primary":     None,
            "recency":     None,
            "explanation": "Archive has no images prior to the new date.",
        }


# ── Run recorder ──────────────────────────────────────────────────────────────

def record_run(
    state:           Dict[str, Any],
    new_image_path:  str,
    new_date_str:    str,
    primary_result:  Optional[Dict],
    recency_result:  Optional[Dict],
    change_detected: bool,
    encroachment_ha: float,
    regions:         List[Dict],
) -> Dict[str, Any]:
    state["runs"].append({
        "new_date":        new_date_str,
        "new_path":        new_image_path,
        "change_detected": change_detected,
        "encroachment_ha": encroachment_ha,
        "region_count":    len(regions),
        "had_recency":     recency_result is not None,
    })
    return state


# ── Region merger ─────────────────────────────────────────────────────────────

def merge_results(
    primary_regions: List[Dict],
    recency_regions: Optional[List[Dict]],
    iou_threshold:   float = 0.3,
) -> List[Dict]:
    """
    Tag each region from the primary (same-season) comparison as:
      - "new_encroachment"      → also appears in recency check (brand new)
      - "existing_encroachment" → in primary but NOT in recency (was already there)
      - "new_encroachment"      → only in recency (edge case, very fresh)

    If no recency check was run, all regions are tagged "unconfirmed_timing"
    (change is real, but we don't know exactly when it happened this year).
    """
    if not recency_regions:
        for r in primary_regions:
            r["encroachment_type"] = "unconfirmed_timing"
            r["timing_note"] = "Recency check not run — change confirmed this year but exact month unknown"
        return primary_regions

    def iou(b1, b2) -> float:
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-8)

    matched = set()
    merged  = []

    for p in primary_regions:
        p_bbox   = p.get("bbox_px", [0,0,0,0])
        best_iou = 0.0
        best_idx = -1
        for i, r in enumerate(recency_regions):
            v = iou(p_bbox, r.get("bbox_px", [0,0,0,0]))
            if v > best_iou:
                best_iou = v; best_idx = i

        if best_iou >= iou_threshold and best_idx >= 0:
            p["encroachment_type"] = "new_encroachment"
            p["timing_note"]       = "Visible in both same-season and recency check — appeared within last revisit cycle"
            matched.add(best_idx)
        else:
            p["encroachment_type"] = "existing_encroachment"
            p["timing_note"]       = "In same-season comparison but not in recency — was already present before last revisit"
        merged.append(p)

    for i, r in enumerate(recency_regions):
        if i not in matched:
            r["encroachment_type"] = "new_encroachment"
            r["timing_note"]       = "Only in recency check — very fresh encroachment"
            merged.append(r)

    n_new    = sum(1 for r in merged if r["encroachment_type"] == "new_encroachment")
    n_exist  = sum(1 for r in merged if r["encroachment_type"] == "existing_encroachment")
    n_unc    = sum(1 for r in merged if r["encroachment_type"] == "unconfirmed_timing")
    logger.info(f"Regions — new: {n_new}, existing: {n_exist}, unconfirmed timing: {n_unc}")
    return merged


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_date(date_str: str) -> datetime:
    """Parse flexible date strings: '2025', '2025-03', '2025-03-15'."""
    date_str = str(date_str).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: '{date_str}'. Use YYYY, YYYY-MM, or YYYY-MM-DD.")


def _make_comp(t1_entry: Dict, t2_path: str, t2_date: str, label: str) -> Dict:
    return {
        "t1_path":  t1_entry["path"],
        "t2_path":  str(Path(t2_path).resolve()),
        "t1_date":  t1_entry["label"],
        "t2_date":  t2_date,
        "label":    label,
    }


def _days_between(iso_date: str, date_str: str) -> int:
    d1 = datetime.fromisoformat(iso_date)
    d2 = _parse_date(date_str)
    return abs((d2 - d1).days)
