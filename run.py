"""
Food Security ML Pipeline — CLI Entry Point

Single-run modes:
    python run.py --t1 before.tif --t2 after.tif
    python run.py --gee
    python run.py --test

Temporal mode (production — satellite revisits every ~20 days):
    # Step 1: Register all historical images you already have (no pipeline runs)
    python run.py --register --image 2021-03.tif --date 2021-03-10
    python run.py --register --image 2022-03.tif --date 2022-03-08
    python run.py --register --image 2023-03.tif --date 2023-03-05
    ...

    # Step 2: Each time a new image arrives, run one command
    python run.py --temporal --new-image 2026-03.tif --new-date 2026-03-15

    The pipeline automatically:
      - Finds the closest image from ~1 year ago (same season, no seasonal noise)
      - Runs primary comparison against it
      - If change is detected, runs a recency check against the most recent image
        to determine if the encroachment is brand-new or pre-existing

    # Preview what would run without executing
    python run.py --temporal --new-image 2026-03.tif --new-date 2026-03-15 --dry-run

    # See the full archive and run history
    python run.py --status

    # Wipe everything and start over
    python run.py --temporal-reset
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Food Security ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--t1",             type=str,           help="T1 (before) GeoTIFF path")
    mode.add_argument("--gee",            action="store_true", help="Download from Google Earth Engine")
    mode.add_argument("--test",           action="store_true", help="Run with synthetic test data")
    mode.add_argument("--temporal",       action="store_true", help="Temporal mode: add new image and run")
    mode.add_argument("--register",       action="store_true", help="Register an image in the archive (no pipeline run)")
    mode.add_argument("--status",         action="store_true", help="Show archive and run history")
    mode.add_argument("--temporal-reset", action="store_true", help="Clear all temporal history")

    # Single-run
    parser.add_argument("--t2",         type=str, help="T2 (after) GeoTIFF path (with --t1)")
    parser.add_argument("--start-from", type=int, default=1, choices=range(1, 9))

    # Temporal / register
    parser.add_argument("--new-image",  type=str, help="New image path (--temporal)")
    parser.add_argument("--new-date",   type=str, help="Date of new image, e.g. 2026-03-15")
    parser.add_argument("--image",      type=str, help="Image path (--register)")
    parser.add_argument("--date",       type=str, help="Image date (--register)")
    parser.add_argument("--dry-run",    action="store_true", help="Show plan without running")
    parser.add_argument(
        "--precomputed", action="store_true",
        help="Input TIFFs have pre-computed indices (e.g. KEMET1: Band0=NDVI, Band1=NDBI …). "
             "Skips cloud heuristic; uses bands directly in Steps 4 and 6. "
             "Default OFF — for raw GEE/satellite imagery leave this flag out.",
    )

    args = parser.parse_args()

    if args.precomputed:
        # Override the config flag at runtime — no permanent change to settings.py
        from config import settings as _s
        _s.SPECTRAL_INDICES_CONFIG["precomputed_indices"] = True

    if args.t1 and not args.t2:
        parser.error("--t2 required with --t1")
    if args.temporal and (not args.new_image or not args.new_date):
        parser.error("--new-image and --new-date required with --temporal")
    if args.register and (not args.image or not args.date):
        parser.error("--image and --date required with --register")

    return args


# ── Mode handlers ─────────────────────────────────────────────────────────────

def run_test_mode():
    from src.utils.geo_utils import create_synthetic_geotiff
    from config.settings import RAW_DIR

    print("\n" + "=" * 60)
    print("  RUNNING IN TEST MODE (synthetic data)")
    print("=" * 60 + "\n")
    t1 = create_synthetic_geotiff(RAW_DIR / "T1" / "T1_synthetic.tif")
    t2 = create_synthetic_geotiff(RAW_DIR / "T2" / "T2_synthetic.tif")
    print(f"Created synthetic T1: {t1}")
    print(f"Created synthetic T2: {t2}")

    from pipeline import FoodSecurityPipeline
    pipe = FoodSecurityPipeline()
    pipe.run_full(t1_path=t1, t2_path=t2)
    print("\n✅ Test mode completed successfully!")


def run_register(args):
    from src.temporal.temporal_manager import load_state, save_state, register_image
    state = load_state()
    state = register_image(state, args.image, args.date)
    save_state(state)
    print(f"✅ Registered: {args.date} → {args.image}")
    print(f"   Archive now has {len(state['archive'])} image(s).")


def run_status():
    from src.temporal.temporal_manager import load_state
    state = load_state()
    archive = state.get("archive", [])
    runs    = state.get("runs", [])

    print("\n" + "=" * 60)
    print("  TEMPORAL ARCHIVE")
    print("=" * 60)
    if not archive:
        print("  (empty — use --register to add images)")
    for e in archive:
        print(f"  {e['label']:<20} {e['path']}")

    print("\n" + "=" * 60)
    print("  RUN HISTORY")
    print("=" * 60)
    if not runs:
        print("  (no runs yet)")
    for r in runs:
        chg = "⚠ CHANGE" if r.get("change_detected") else "✓ no change"
        print(
            f"  {r['new_date']:<20} {chg:<16} "
            f"{r.get('encroachment_ha', 0):.2f} ha  "
            f"{r.get('region_count', 0)} regions"
        )
    print()


def run_temporal_mode(args):
    from src.temporal.temporal_manager import load_state, get_comparison_plan, register_image, save_state

    state = load_state()
    # Register the new image first so plan can see it
    state = register_image(state, args.new_image, args.new_date)
    plan  = get_comparison_plan(args.new_image, args.new_date, state)

    print("\n" + "=" * 60)
    print(f"  TEMPORAL MODE — {args.new_date}")
    print("=" * 60)
    print(f"  Strategy:  {plan['mode']}")
    print(f"  {plan['explanation'].replace(chr(10), chr(10) + '  ')}")
    print("=" * 60 + "\n")

    if args.dry_run:
        print("  [DRY RUN] No pipeline executed.\n")
        # Still save the registration
        save_state(state)
        return

    if plan["primary"] is None:
        print(
            "  ⚠  Not enough images in archive to run a comparison.\n"
            "  Register at least one earlier image first:\n"
            "    python run.py --register --image <path> --date <YYYY-MM-DD>\n"
        )
        save_state(state)
        return

    from pipeline import FoodSecurityPipeline
    pipe   = FoodSecurityPipeline()
    output = pipe.run_temporal(
        new_image_path = args.new_image,
        new_date       = args.new_date,
    )

    regions  = output.get("regions", [])
    n_new    = sum(1 for r in regions if r.get("encroachment_type") == "new_encroachment")
    n_exist  = sum(1 for r in regions if r.get("encroachment_type") == "existing_encroachment")
    n_unc    = sum(1 for r in regions if r.get("encroachment_type") == "unconfirmed_timing")
    total_ha = sum(r.get("area_ha", 0) for r in regions)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  New encroachment:       {n_new} regions")
    print(f"  Pre-existing:           {n_exist} regions")
    print(f"  Unconfirmed timing:     {n_unc} regions")
    print(f"  Total area:             {total_ha:.2f} ha")
    print("=" * 60)

    last = output.get("primary_result") or {}
    paths = last.get("step_08", {}).get("paths", {})
    if paths:
        print("\n  Output files:")
        for k, v in paths.items():
            print(f"    {k}: {v}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.temporal_reset:
        from src.temporal.temporal_manager import reset_state
        reset_state()
        print("✅ Temporal history cleared.")
        return

    if args.status:
        run_status()
        return

    if args.register:
        run_register(args)
        return

    if args.test:
        run_test_mode()
        return

    if args.temporal:
        run_temporal_mode(args)
        return

    # ── Standard single run ───────────────────────────────────────────────────
    from pipeline import FoodSecurityPipeline
    pipe = FoodSecurityPipeline()

    if args.gee:
        results = pipe.run_full(use_gee=True, start_from=args.start_from)
    else:
        results = pipe.run_full(
            t1_path=args.t1, t2_path=args.t2, start_from=args.start_from
        )

    if "step_08" in results:
        report = results["step_08"].get("report", {})
        print("\n" + "=" * 60)
        print("  RESULTS")
        print("=" * 60)
        print(f"  Regions:           {report.get('total_regions', 'N/A')}")
        print(f"  Encroachment area: {report.get('encroachment_ha', 'N/A')} ha")
        print(f"  Yellow alert area: {report.get('yellow_alert_ha', 'N/A')} ha")
        paths = results["step_08"].get("paths", {})
        print("\n  Output files:")
        for k, v in paths.items():
            print(f"    {k}: {v}")
        print()


if __name__ == "__main__":
    main()
