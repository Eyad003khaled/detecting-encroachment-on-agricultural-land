"""
Food Security ML Pipeline — CLI Entry Point

Usage:
    python run.py --t1 data/raw/T1/image.tif --t2 data/raw/T2/image.tif
    python run.py --gee                  # Download from GEE
    python run.py --test                 # Run with synthetic data
    python run.py --help                 # Show all options
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Food Security ML Pipeline — Detect Buildings on Agricultural Land",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with existing GeoTIFF images
  python run.py --t1 data/raw/T1/image.tif --t2 data/raw/T2/image.tif

  # Download from Google Earth Engine
  python run.py --gee

  # Run with synthetic test data (no model weights needed)
  python run.py --test

  # Resume from a specific step
  python run.py --t1 data/raw/T1/image.tif --t2 data/raw/T2/image.tif --start-from 5
        """,
    )

    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--t1",
        type=str,
        help="Path to T1 (before) GeoTIFF image",
    )
    input_group.add_argument(
        "--gee",
        action="store_true",
        help="Download imagery from Google Earth Engine",
    )
    input_group.add_argument(
        "--test",
        action="store_true",
        help="Run with synthetic test data (for verification)",
    )

    parser.add_argument(
        "--t2",
        type=str,
        help="Path to T2 (after) GeoTIFF image (required with --t1)",
    )

    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        choices=range(1, 9),
        help="Resume pipeline from step number (1-8)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )

    args = parser.parse_args()

    # Validate that --t2 is provided when --t1 is used
    if args.t1 and not args.t2:
        parser.error("--t2 is required when using --t1")

    return args


def run_test_mode():
    """Run pipeline with synthetic data for verification."""
    from src.utils.geo_utils import create_synthetic_geotiff
    from config.settings import RAW_DIR

    print("\n" + "=" * 60)
    print("  RUNNING IN TEST MODE (synthetic data)")
    print("=" * 60 + "\n")

    # Create synthetic T1 and T2 GeoTIFFs
    t1_path = create_synthetic_geotiff(RAW_DIR / "T1" / "T1_synthetic.tif")
    t2_path = create_synthetic_geotiff(RAW_DIR / "T2" / "T2_synthetic.tif")

    print(f"Created synthetic T1: {t1_path}")
    print(f"Created synthetic T2: {t2_path}")

    # Run pipeline
    from pipeline import FoodSecurityPipeline
    pipe = FoodSecurityPipeline()
    results = pipe.run_full(t1_path=t1_path, t2_path=t2_path)

    print("\n✅ Test mode completed successfully!")
    return results


def main():
    args = parse_args()

    if args.test:
        run_test_mode()
        return

    from pipeline import FoodSecurityPipeline

    pipe = FoodSecurityPipeline()

    if args.gee:
        results = pipe.run_full(use_gee=True, start_from=args.start_from)
    else:
        results = pipe.run_full(
            t1_path=args.t1,
            t2_path=args.t2,
            start_from=args.start_from,
        )

    # Print final summary
    if "step_08" in results:
        report = results["step_08"].get("report", {})
        areas = report.get("areas", {})

        print("\n" + "=" * 60)
        print("  FINAL RESULTS SUMMARY")
        print("=" * 60)
        print(f"  Agricultural land:   {areas.get('total_agricultural_land_ha', 'N/A')} ha")
        print(f"  Total changed area:  {areas.get('total_changed_area_ha', 'N/A')} ha")
        print(f"  Encroachment area:   {areas.get('encroachment_area_ha', 'N/A')} ha")
        print(f"  Building count:      {report.get('encroachment', {}).get('building_count', 'N/A')}")
        print("=" * 60)

        paths = results["step_08"].get("paths", {})
        print("\n  Output Files:")
        for fmt, path in paths.items():
            print(f"    {fmt}: {path}")
        print()


if __name__ == "__main__":
    main()
