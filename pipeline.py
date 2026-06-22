"""
Food Security ML Pipeline — End-to-End Orchestrator

Runs all 8 steps sequentially:
  01. Data Acquisition (GEE or offline)
  02. Cloud Detection (U-Net + ResNet34)
  03. Cloud Removal (OpenCV Telea)  ← SKIPPED automatically if coverage < 15%
  04. Spectral Indices (NDVI, NDBI, MNDWI) → yellow-alert signal
  05. Change Detection (ChangeFormer) → confidence map
  06. Agriculture Segmentation (SegFormer-B4)
  07. Building Detection (SAM + YOLOv8-seg)
  08. Final Output (colored map, before/after chips, geocoding,
                    interactive Folium map, area report)

Multi-temporal mode (run_temporal)
-----------------------------------
Tracks run history in outputs/temporal_state.json.

  No prior change detected   → rolling window: previous vs new
  Prior change detected      → dual comparison:
      1. baseline vs new   (total cumulative encroachment)
      2. previous vs new   (new encroachment this period only)
  Regions are tagged as "new_encroachment" or "existing_encroachment".
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from config.settings import (
    GEE_CONFIG, CLOUD_DETECTION_CONFIG, CLOUD_REMOVAL_CONFIG,
    SPECTRAL_INDICES_CONFIG, CHANGE_DETECTION_CONFIG,
    AGRICULTURE_SEGMENTATION_CONFIG, BUILDING_DETECTION_CONFIG,
    FINAL_OUTPUT_CONFIG, RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, LOG_CONFIG,
)
from src.utils.logger import get_logger
from src.utils.geo_utils import read_geotiff, get_rgb_from_multiband

logger = get_logger("pipeline", log_file=LOG_CONFIG["log_file"])

CLOUD_SKIP_THRESHOLD = CLOUD_DETECTION_CONFIG["skip_removal_below_pct"]


class FoodSecurityPipeline:
    """
    End-to-end pipeline orchestrator.

    Manages data flow between all 8 steps, handles intermediate
    result caching, and supports resuming from any step.

    Cloud threshold logic
    ---------------------
    After Step 02, if max(T1_coverage, T2_coverage) < CLOUD_SKIP_THRESHOLD (15%),
    Step 03 is bypassed: the raw images are passed directly to Step 04 as if
    they were already cloud-free. This avoids artefacts introduced by unnecessary
    inpainting on clean imagery.
    """

    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.timings: Dict[str, float] = {}

    def _log_step_start(self, step: int, name: str) -> float:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"  STEP {step:02d} — {name}")
        logger.info("=" * 70)
        return time.time()

    def _log_step_end(self, step: int, name: str, start_time: float) -> None:
        elapsed = time.time() - start_time
        self.timings[f"step_{step:02d}_{name}"] = elapsed
        logger.info(f"  Step {step:02d} completed in {elapsed:.1f}s")
        logger.info("")

    # ----------------------------------------------------------
    # Step 01 — Data Acquisition
    # ----------------------------------------------------------
    def step_01_data_acquisition(
        self,
        t1_path: Optional[str | Path] = None,
        t2_path: Optional[str | Path] = None,
        use_gee: bool = False,
    ) -> Dict[str, Path]:
        start = self._log_step_start(1, "DATA ACQUISITION")
        from src.step_01_data_acquisition.acquire import run, run_offline

        if use_gee:
            result = run()
        else:
            if t1_path is None or t2_path is None:
                raise ValueError(
                    "Provide t1_path and t2_path for offline mode, "
                    "or set use_gee=True"
                )
            result = run_offline(t1_path, t2_path)

        self.results["step_01"] = result
        self._log_step_end(1, "data_acquisition", start)
        return result

    # ----------------------------------------------------------
    # Step 02 — Cloud Detection
    # ----------------------------------------------------------
    def step_02_cloud_detection(self) -> Dict[str, Any]:
        """Detect clouds in T1 and T2. Sets skip_removal flag if coverage < threshold."""
        start = self._log_step_start(2, "CLOUD DETECTION")
        from src.step_02_cloud_detection.detect_clouds import run

        paths = self.results["step_01"]
        result = run(paths["T1"], paths["T2"])
        # result["skip_removal"] is set by detect_clouds.run()

        self.results["step_02"] = result
        self._log_step_end(2, "cloud_detection", start)
        return result

    # ----------------------------------------------------------
    # Step 03 — Cloud Removal  (conditional)
    # ----------------------------------------------------------
    def step_03_cloud_removal(self) -> Dict[str, Dict[str, Any]]:
        """
        Remove clouds — OR — pass raw images through if coverage is low.

        When skipped, populates self.results["step_03"] with the raw T1/T2
        images so that all downstream steps continue unchanged.
        """
        clouds = self.results["step_02"]

        if clouds.get("skip_removal", False):
            # ── SKIP: load raw images and pass them straight through ──────
            logger.info("")
            logger.info("=" * 70)
            logger.info(
                f"  STEP 03 — CLOUD REMOVAL  [SKIPPED — "
                f"coverage below {CLOUD_SKIP_THRESHOLD}%]"
            )
            logger.info("=" * 70)
            paths = self.results["step_01"]
            t1_data, t1_meta = read_geotiff(paths["T1"])
            t2_data, t2_meta = read_geotiff(paths["T2"])
            result = {
                "T1": {"image": t1_data, "meta": t1_meta},
                "T2": {"image": t2_data, "meta": t2_meta},
                "skipped": True,
            }
            logger.info("  Raw images passed directly to Step 04")
            logger.info("")
        else:
            # ── RUN cloud removal as normal ───────────────────────────────
            start = self._log_step_start(3, "CLOUD REMOVAL")
            from src.step_03_cloud_removal.remove_clouds import run

            paths = self.results["step_01"]
            result = run(
                paths["T1"], paths["T2"],
                clouds["T1"]["mask"], clouds["T2"]["mask"],
            )
            result["skipped"] = False
            self._log_step_end(3, "cloud_removal", start)

        self.results["step_03"] = result
        return result

    # ----------------------------------------------------------
    # Step 04 — Spectral Indices
    # ----------------------------------------------------------
    def step_04_spectral_indices(self) -> Dict[str, Any]:
        """
        Compute NDVI, NDBI, MNDWI and derive yellow-alert degradation signal.
        """
        start = self._log_step_start(4, "SPECTRAL INDICES")
        from src.step_04_spectral_indices.compute_indices import run

        clean = self.results["step_03"]
        result = run(
            clean["T1"]["image"], clean["T2"]["image"],
            clean["T1"]["meta"],  clean["T2"]["meta"],
        )

        self.results["step_04"] = result
        self._log_step_end(4, "spectral_indices", start)
        return result

    # ----------------------------------------------------------
    # Step 05 — Change Detection
    # ----------------------------------------------------------
    def step_05_change_detection(self) -> Dict[str, Any]:
        """Detect land-use changes; returns binary map AND confidence scores."""
        start = self._log_step_start(5, "CHANGE DETECTION")
        from src.step_05_change_detection.detect_changes import run

        clean = self.results["step_03"]
        result = run(
            clean["T1"]["image"], clean["T2"]["image"],
            clean["T1"]["meta"],
        )

        self.results["step_05"] = result
        self._log_step_end(5, "change_detection", start)
        return result

    # ----------------------------------------------------------
    # Step 06 — Agriculture Segmentation
    # ----------------------------------------------------------
    def step_06_agriculture_segmentation(self) -> Dict[str, Any]:
        """Segment agricultural land in the T1 (before) image."""
        start = self._log_step_start(6, "AGRICULTURE SEGMENTATION")
        from src.step_06_agriculture_segmentation.segment_agriculture import run

        clean = self.results["step_03"]
        result = run(clean["T1"]["image"], clean["T1"]["meta"])

        self.results["step_06"] = result
        self._log_step_end(6, "agriculture_segmentation", start)
        return result

    # ----------------------------------------------------------
    # Step 07 — Building Detection
    # ----------------------------------------------------------
    def step_07_building_detection(self) -> Dict[str, Any]:
        """Detect buildings in changed agricultural areas."""
        start = self._log_step_start(7, "BUILDING DETECTION")
        from src.step_07_building_detection.detect_buildings import run

        clean    = self.results["step_03"]
        change   = self.results["step_05"]
        agri     = self.results["step_06"]

        result = run(
            clean["T2"]["image"],
            change["change_map"],
            agri["agri_mask"],
            clean["T2"]["meta"],
        )

        self.results["step_07"] = result
        self._log_step_end(7, "building_detection", start)
        return result

    # ----------------------------------------------------------
    # Step 08 — Final Output
    # ----------------------------------------------------------
    def step_08_final_output(self) -> Dict[str, Any]:
        """
        Generate all outputs:
          • Colored encroachment map (PNG + GeoTIFF)
          • Per-region before/after chips with lat/lon bbox
          • Weighted red-alert score (0.65×change + 0.35×spectral)
          • Reverse geocoded location names
          • Total area lost
          • Interactive Folium HTML map
          • JSON summary report
        """
        start = self._log_step_start(8, "FINAL OUTPUT")
        from src.step_08_final_output.generate_output import run

        clean     = self.results["step_03"]
        spectral  = self.results["step_04"]
        change    = self.results["step_05"]
        agri      = self.results["step_06"]
        buildings = self.results["step_07"]

        t1_rgb = get_rgb_from_multiband(clean["T1"]["image"])
        t2_rgb = get_rgb_from_multiband(clean["T2"]["image"])

        result = run(
            t2_rgb                = t2_rgb,
            change_map            = change["change_map"],
            agri_mask             = agri["agri_mask"],
            building_mask         = buildings["building_mask"],
            polygons              = buildings.get("polygons", []),
            meta                  = clean["T2"]["meta"],
            # New inputs for enhanced output
            change_confidence     = change.get("change_confidence"),
            spectral_signal       = spectral.get("spectral_signal"),
            yellow_mask           = spectral.get("yellow_mask"),
            t1_rgb                = t1_rgb,
        )

        self.results["step_08"] = result
        self._log_step_end(8, "final_output", start)
        return result

    # ----------------------------------------------------------
    # Full Pipeline
    # ----------------------------------------------------------
    def run_full(
        self,
        t1_path:    Optional[str | Path] = None,
        t2_path:    Optional[str | Path] = None,
        use_gee:    bool = False,
        start_from: int = 1,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline from start to finish.

        Args:
            t1_path:    Path to T1 GeoTIFF (offline mode).
            t2_path:    Path to T2 GeoTIFF (offline mode).
            use_gee:    Download from GEE if True.
            start_from: Resume from this step number (1-8).
        """
        total_start = time.time()

        logger.info("╔" + "═" * 68 + "╗")
        logger.info("║   FOOD SECURITY ML PIPELINE — STARTING                          ║")
        logger.info("║   Detect Buildings on Agricultural Land                          ║")
        logger.info("╚" + "═" * 68 + "╝")

        steps = [
            (1, lambda: self.step_01_data_acquisition(t1_path, t2_path, use_gee)),
            (2, lambda: self.step_02_cloud_detection()),
            (3, lambda: self.step_03_cloud_removal()),    # auto-skips if coverage < 15%
            (4, lambda: self.step_04_spectral_indices()),
            (5, lambda: self.step_05_change_detection()),
            (6, lambda: self.step_06_agriculture_segmentation()),
            (7, lambda: self.step_07_building_detection()),
            (8, lambda: self.step_08_final_output()),
        ]

        for step_num, step_fn in steps:
            if step_num >= start_from:
                try:
                    step_fn()
                except Exception as e:
                    logger.error(f"Step {step_num:02d} FAILED: {e}")
                    logger.error(
                        f"Pipeline halted. Fix the issue and resume with "
                        f"--start-from {step_num}"
                    )
                    raise

        total_elapsed = time.time() - total_start
        logger.info("")
        logger.info("╔" + "═" * 68 + "╗")
        logger.info("║   PIPELINE COMPLETE                                              ║")
        logger.info(f"║   Total time: {total_elapsed:.1f}s" + " " * (53 - len(f"{total_elapsed:.1f}s")) + "║")
        logger.info("╚" + "═" * 68 + "╝")

        logger.info("\nStep Timings:")
        for name, t in self.timings.items():
            logger.info(f"  {name}: {t:.1f}s")

        # Print final summary if Step 08 completed
        if "step_08" in self.results:
            report = self.results["step_08"].get("report", {})
            logger.info(
                f"\n  Encroachment: {report.get('total_regions', 0)} regions, "
                f"{report.get('encroachment_ha', 0):.2f} ha"
            )
            logger.info(
                f"  Spectral degradation area: {report.get('yellow_alert_ha', 0):.2f} ha"
            )
            paths = self.results["step_08"].get("paths", {})
            logger.info(f"  Interactive map: {paths.get('interactive_map', '')}")

        return self.results

    # ----------------------------------------------------------
    # Multi-temporal entry point
    # ----------------------------------------------------------
    def run_temporal(
        self,
        new_image_path: str | Path,
        new_date:       str,
    ) -> Dict[str, Any]:
        """
        Register a new image and run the same-season year-over-year comparison.

        Strategy
        --------
        1. PRIMARY  — find the closest image from ~1 year ago (±45 days).
                      Eliminates seasonal noise; only structural changes survive.
        2. RECENCY CHECK — run only if primary detects change.
                      Compares against the most recent prior image to answer
                      "was this already there last month, or brand new?"

        Regions are tagged:
          new_encroachment      — visible in both primary and recency
          existing_encroachment — in primary but not recency (pre-existing)
          unconfirmed_timing    — primary flagged it but no recency image available

        Parameters
        ----------
        new_image_path : path to the new satellite image
        new_date       : date label, e.g. "2025-03-15" or "2025-03" or "2025"
        """
        from src.temporal.temporal_manager import (
            load_state, save_state, register_image,
            get_comparison_plan, record_run, merge_results,
        )

        new_image_path = str(Path(new_image_path).resolve())
        state = load_state()

        # Register this image in the archive
        state = register_image(state, new_image_path, new_date)
        save_state(state)

        plan = get_comparison_plan(new_image_path, new_date, state)
        logger.info(f"\nComparison plan: {plan['mode']}\n  {plan['explanation']}")

        if plan["primary"] is None:
            logger.warning("Cannot run pipeline — not enough images in archive yet.")
            return {"mode": plan["mode"], "results": [], "regions": [], "state": state}

        # ── Run primary comparison ───────────────────────────────────────────
        primary = plan["primary"]
        logger.info(f"\n{'─'*60}\n  PRIMARY [{primary['t1_date']} → {primary['t2_date']}]\n{'─'*60}")
        self.results = {}
        primary_pipeline = self.run_full(t1_path=primary["t1_path"], t2_path=primary["t2_path"])
        primary_regions  = primary_pipeline.get("step_08", {}).get("regions", [])

        # ── Recency check — only if primary found something ──────────────────
        recency_regions  = None
        recency_pipeline = None

        if primary_regions and plan["recency"] is not None:
            recency = plan["recency"]
            logger.info(
                f"\n{'─'*60}\n"
                f"  RECENCY CHECK [{recency['t1_date']} → {recency['t2_date']}]\n"
                f"{'─'*60}"
            )
            self.results = {}
            recency_pipeline = self.run_full(
                t1_path=recency["t1_path"], t2_path=recency["t2_path"]
            )
            recency_regions = recency_pipeline.get("step_08", {}).get("regions", [])
        elif primary_regions and plan["recency"] is None:
            logger.info("Primary flagged changes but no recency image available — skipping recency check.")

        # ── Merge and tag ────────────────────────────────────────────────────
        merged_regions  = merge_results(primary_regions, recency_regions)
        change_detected = len(merged_regions) > 0
        encroachment_ha = sum(r.get("area_ha", 0) for r in merged_regions)

        state = record_run(
            state            = state,
            new_image_path   = new_image_path,
            new_date_str     = new_date,
            primary_result   = primary_pipeline,
            recency_result   = recency_pipeline,
            change_detected  = change_detected,
            encroachment_ha  = encroachment_ha,
            regions          = merged_regions,
        )
        save_state(state)

        n_new   = sum(1 for r in merged_regions if r.get("encroachment_type") == "new_encroachment")
        n_exist = sum(1 for r in merged_regions if r.get("encroachment_type") == "existing_encroachment")
        n_unc   = sum(1 for r in merged_regions if r.get("encroachment_type") == "unconfirmed_timing")

        logger.info(
            f"\nTemporal run complete — mode: {plan['mode']}\n"
            f"  Total encroachment: {encroachment_ha:.2f} ha\n"
            f"  New:               {n_new}\n"
            f"  Existing:          {n_exist}\n"
            f"  Unconfirmed timing:{n_unc}\n"
        )

        return {
            "mode":            plan["mode"],
            "primary_result":  primary_pipeline,
            "recency_result":  recency_pipeline,
            "regions":         merged_regions,
            "state":           state,
        }
