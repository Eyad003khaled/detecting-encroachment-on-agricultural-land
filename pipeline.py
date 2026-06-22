"""
Food Security ML Pipeline — End-to-End Orchestrator

Runs all 8 steps sequentially:
  01. Data Acquisition (GEE or offline)
  02. Cloud Detection (U-Net + ResNet34)
  03. Cloud Removal (OpenCV Telea)
  04. Spectral Indices (NDVI, NDBI, MNDWI)
  05. Change Detection (ChangeFormer)
  06. Agriculture Segmentation (SegFormer-B4)
  07. Building Detection (SAM + YOLOv8-seg)
  08. Final Output (Colored map + GeoJSON + report)
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


class FoodSecurityPipeline:
    """
    End-to-end pipeline orchestrator.

    Manages data flow between all 8 steps, handles intermediate
    result caching, and supports resuming from any step.
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
        """
        Acquire satellite imagery.

        Args:
            t1_path: Path to existing T1 GeoTIFF (offline mode).
            t2_path: Path to existing T2 GeoTIFF (offline mode).
            use_gee: If True, download from GEE instead.
        """
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
    def step_02_cloud_detection(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Detect clouds in T1 and T2 images."""
        start = self._log_step_start(2, "CLOUD DETECTION")
        from src.step_02_cloud_detection.detect_clouds import run

        paths = self.results["step_01"]
        result = run(paths["T1"], paths["T2"])

        self.results["step_02"] = result
        self._log_step_end(2, "cloud_detection", start)
        return result

    # ----------------------------------------------------------
    # Step 03 — Cloud Removal
    # ----------------------------------------------------------
    def step_03_cloud_removal(self) -> Dict[str, Dict[str, Any]]:
        """Remove clouds from T1 and T2 images."""
        start = self._log_step_start(3, "CLOUD REMOVAL")
        from src.step_03_cloud_removal.remove_clouds import run

        paths = self.results["step_01"]
        clouds = self.results["step_02"]

        result = run(
            paths["T1"], paths["T2"],
            clouds["T1"]["mask"], clouds["T2"]["mask"],
        )

        self.results["step_03"] = result
        self._log_step_end(3, "cloud_removal", start)
        return result

    # ----------------------------------------------------------
    # Step 04 — Spectral Indices
    # ----------------------------------------------------------
    def step_04_spectral_indices(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute NDVI, NDBI, MNDWI for T1 and T2."""
        start = self._log_step_start(4, "SPECTRAL INDICES")
        from src.step_04_spectral_indices.compute_indices import run

        clean = self.results["step_03"]
        result = run(
            clean["T1"]["image"], clean["T2"]["image"],
            clean["T1"]["meta"], clean["T2"]["meta"],
        )

        self.results["step_04"] = result
        self._log_step_end(4, "spectral_indices", start)
        return result

    # ----------------------------------------------------------
    # Step 05 — Change Detection
    # ----------------------------------------------------------
    def step_05_change_detection(self) -> Dict[str, Any]:
        """Detect land-use changes between T1 and T2."""
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
        """Segment agricultural land in T1 image."""
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

        clean = self.results["step_03"]
        change = self.results["step_05"]
        agri = self.results["step_06"]

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
        """Generate final colored map, GeoJSON, and report."""
        start = self._log_step_start(8, "FINAL OUTPUT")
        from src.step_08_final_output.generate_output import run

        clean = self.results["step_03"]
        change = self.results["step_05"]
        agri = self.results["step_06"]
        buildings = self.results["step_07"]

        t2_rgb = get_rgb_from_multiband(clean["T2"]["image"])

        result = run(
            t2_rgb,
            change["change_map"],
            agri["agri_mask"],
            buildings["building_mask"],
            buildings.get("polygons", []),
            clean["T2"]["meta"],
        )

        self.results["step_08"] = result
        self._log_step_end(8, "final_output", start)
        return result

    # ----------------------------------------------------------
    # Full Pipeline
    # ----------------------------------------------------------
    def run_full(
        self,
        t1_path: Optional[str | Path] = None,
        t2_path: Optional[str | Path] = None,
        use_gee: bool = False,
        start_from: int = 1,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline from start to finish.

        Args:
            t1_path: Path to T1 GeoTIFF (offline mode).
            t2_path: Path to T2 GeoTIFF (offline mode).
            use_gee: Download from GEE if True.
            start_from: Resume from this step number (1-8).

        Returns:
            Dict with all step results.
        """
        total_start = time.time()

        logger.info("╔" + "═" * 68 + "╗")
        logger.info("║   FOOD SECURITY ML PIPELINE — STARTING                          ║")
        logger.info("║   Detect Buildings on Agricultural Land                          ║")
        logger.info("╚" + "═" * 68 + "╝")

        steps = [
            (1, lambda: self.step_01_data_acquisition(t1_path, t2_path, use_gee)),
            (2, lambda: self.step_02_cloud_detection()),
            (3, lambda: self.step_03_cloud_removal()),
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
                    logger.error("Pipeline halted. Fix the issue and resume with start_from={step_num}")
                    raise

        total_elapsed = time.time() - total_start
        logger.info("")
        logger.info("╔" + "═" * 68 + "╗")
        logger.info("║   PIPELINE COMPLETE                                              ║")
        logger.info(f"║   Total time: {total_elapsed:.1f}s" + " " * (53 - len(f"{total_elapsed:.1f}s")) + "║")
        logger.info("╚" + "═" * 68 + "╝")

        # Print timing summary
        logger.info("\nStep Timings:")
        for name, t in self.timings.items():
            logger.info(f"  {name}: {t:.1f}s")

        return self.results
