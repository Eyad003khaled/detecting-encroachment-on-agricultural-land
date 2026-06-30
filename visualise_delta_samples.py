"""
visualise_delta_samples.py
==========================
Run this AFTER downloading tiles from Google Drive.

Shows side-by-side before (2021) / after (2023/2025) comparisons
for every sample location, with:
  • NDVI false-colour composite (same scale both years)
  • NDVI difference heatmap
  • Change cluster bounding box
  • Latitude / longitude labels

Usage:
  python visualise_delta_samples.py --src data/NileDelta_samples
"""

from __future__ import annotations
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from scipy import ndimage
from rasterio.warp import transform_bounds


YEARS_BEFORE = [2021]
YEARS_AFTER  = [2025, 2024, 2023]   # prefer latest available


def load_tile(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        meta = {
            "bounds_wgs84": transform_bounds(src.crs, "EPSG:4326", *src.bounds),
            "crs": str(src.crs),
        }
    return arr, meta


def shared_rgb(d1: np.ndarray, d2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False-colour RGB: R=NDBI, G=NDVI, B=MNDWI — SHARED normalization."""
    results = []
    for ch1, ch2 in [(d1[1], d2[1]), (d1[0], d2[0]), (d1[2], d2[2])]:
        combined = np.concatenate([ch1.ravel(), ch2.ravel()])
        lo = np.nanpercentile(combined, 2)
        hi = np.nanpercentile(combined, 98)
        def n(x, lo=lo, hi=hi):
            return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
        results.append((n(ch1), n(ch2)))
    img1 = np.stack([r[0] for r in results], axis=-1)
    img2 = np.stack([r[1] for r in results], axis=-1)
    return img1, img2


def change_bbox(d1: np.ndarray, d2: np.ndarray):
    """Return (r0,r1,c0,c1, cluster_ha) of main change cluster."""
    cs     = -(d2[0] - d1[0]) + (d2[1] - d1[1])
    thresh = cs.mean() + 1.5 * cs.std()
    strong = (cs > thresh).astype(np.uint8)
    labeled, n = ndimage.label(strong)
    if n == 0:
        h, w = cs.shape
        return 0, h-1, 0, w-1, 0.0
    sizes  = sorted([((labeled == i+1).sum(), i+1) for i in range(n)], reverse=True)
    comp   = labeled == sizes[0][1]
    rows_r = np.where(comp.any(axis=1))[0]
    cols_r = np.where(comp.any(axis=0))[0]
    cluster_ha = sizes[0][0] * (10*10) / 10_000   # 10 m pixels
    return rows_r[0], rows_r[-1], cols_r[0], cols_r[-1], cluster_ha


def make_comparison(name: str, before_path: Path, after_path: Path, out_path: Path):
    d1, m1 = load_tile(before_path)
    d2, m2 = load_tile(after_path)

    year_b = before_path.stem
    year_a = after_path.stem

    img1, img2 = shared_rgb(d1, d2)
    r0, r1, c0, c1, cluster_ha = change_bbox(d1, d2)

    ndvi_diff = d2[0] - d1[0]   # NDVI change

    lon_min, lat_min, lon_max, lat_max = m1["bounds_wgs84"]
    centre_lat = (lat_min + lat_max) / 2
    centre_lon = (lon_min + lon_max) / 2

    fig = plt.figure(figsize=(18, 7), facecolor="#0d1117")
    gs  = GridSpec(1, 3, figure=fig, wspace=0.05)

    BG = "#0d1117"

    # ── Before ─────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0]); ax0.set_facecolor(BG)
    ax0.imshow(img1, interpolation="nearest")
    ax0.set_title(f"{name}\n{year_b}  (Before)", color="#8ee3ff",
                  fontsize=12, fontweight="bold", pad=6)
    ax0.set_xlabel(f"lat {centre_lat:.4f}°N  lon {centre_lon:.4f}°E",
                   color="#6e7681", fontsize=9)
    ax0.set_xticks([]); ax0.set_yticks([])

    # ── After + bbox ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1]); ax1.set_facecolor(BG)
    ax1.imshow(img2, interpolation="nearest")
    ax1.add_patch(patches.Rectangle(
        (c0, r0), c1-c0, r1-r0,
        lw=2.5, edgecolor="#ff3333", facecolor="#ff111120", ls="--"))
    ax1.annotate(f"Change\n{cluster_ha:.1f} ha",
        xy=(c0 + (c1-c0)//2, r1 + 4),
        color="#ff6060", fontsize=10, ha="center", va="top", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#1a0000", ec="#ff3333", lw=1.2))
    ax1.set_title(f"{year_a}  (After)  p = ?", color="#8ee3ff",
                  fontsize=12, fontweight="bold", pad=6)
    ax1.set_xlabel("R=NDBI  G=NDVI  B=MNDWI · 10 m/px · shared scale",
                   color="#6e7681", fontsize=9)
    ax1.set_xticks([]); ax1.set_yticks([])

    # ── NDVI diff ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2]); ax2.set_facecolor(BG)
    im = ax2.imshow(ndvi_diff, cmap="RdYlGn", interpolation="nearest",
                    vmin=-0.5, vmax=0.5)
    ax2.add_patch(patches.Rectangle(
        (c0, r0), c1-c0, r1-r0,
        lw=2.5, edgecolor="yellow", facecolor="none"))
    cb = fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)
    cb.ax.tick_params(colors="#aaa")
    cb.set_label("ΔNDVI  ±0.5 shared scale  (green=gain  red=loss)", color="#aaa", fontsize=9)
    ax2.set_title(f"ΔNDVI Raw Spectral Change ({year_b}→{year_a})\n⚠ Not RF-classified output",
                  color="#e3a030", fontsize=10, fontweight="bold", pad=6)
    ax2.set_xticks([]); ax2.set_yticks([])

    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("data/NileDelta_samples"))
    parser.add_argument("--out", type=Path, default=Path("data/NileDelta_samples/comparisons"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Group files: {tile_name: {year_str: path}}
    tiles: dict[str, dict[str, Path]] = defaultdict(dict)
    for tif in sorted(args.src.glob("*/*.tif")):
        tile_name = tif.parent.name   # e.g. tile_Tanta_fringe
        year      = tif.stem          # e.g. 2021
        tiles[tile_name][year] = tif

    if not tiles:
        print(f"[ERROR] No .tif files found under {args.src}")
        print("Expected structure:  data/NileDelta_samples/<tile_name>/<year>.tif")
        return

    print(f"Found {len(tiles)} tile locations.")
    generated = []
    for tile_name, year_files in sorted(tiles.items()):
        # Find before/after pair
        before = None
        for y in YEARS_BEFORE:
            if str(y) in year_files:
                before = year_files[str(y)]
                break

        after = None
        for y in YEARS_AFTER:
            if str(y) in year_files:
                after = year_files[str(y)]
                break

        if before is None or after is None:
            print(f"  {tile_name}: skipped (missing before/after)")
            continue

        out = args.out / f"{tile_name}_comparison.png"
        print(f"\n{tile_name}")
        make_comparison(tile_name, before, after, out)
        generated.append(out)

    print(f"\nDone — {len(generated)} comparison images in {args.out}")


if __name__ == "__main__":
    main()
