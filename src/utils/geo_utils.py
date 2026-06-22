"""
Geo I/O utilities: read/write GeoTIFFs, coordinate helpers, synthetic data.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any


# ── GeoTIFF I/O ─────────────────────────────────────────────────────────────

def read_geotiff(path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read a GeoTIFF and return (array [C,H,W], meta dict)."""
    import rasterio
    with rasterio.open(str(path)) as src:
        data = src.read().astype(np.float32)
        meta = {
            "crs":       src.crs,
            "transform": src.transform,
            "width":     src.width,
            "height":    src.height,
            "count":     src.count,
            "dtype":     str(src.dtypes[0]),
            "nodata":    src.nodata,
        }
    return data, meta


def write_geotiff(
    path: str | Path,
    data: np.ndarray,
    meta: Dict[str, Any],
) -> Path:
    """Write array [C,H,W] or [H,W] to a GeoTIFF."""
    import rasterio
    from rasterio.transform import from_bounds

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = data[np.newaxis]

    count, height, width = data.shape
    transform = meta.get("transform")
    if transform is None:
        transform = from_bounds(0, 0, width, height, width, height)

    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=data.dtype,
        crs=meta.get("crs"),
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


# ── RGB helpers ──────────────────────────────────────────────────────────────

def get_rgb_from_multiband(image: np.ndarray, rgb_indices=(2, 1, 0)) -> np.ndarray:
    """
    Extract and normalise an RGB image from a multi-band array [C,H,W].
    Returns uint8 [H,W,3].
    """
    rgb = image[list(rgb_indices), :, :]  # [3,H,W]
    rgb = np.transpose(rgb, (1, 2, 0))    # [H,W,3]
    # Normalise to 0-255
    rgb = rgb - rgb.min()
    max_val = rgb.max()
    if max_val > 0:
        rgb = rgb / max_val
    return (rgb * 255).astype(np.uint8)


# ── Coordinate helpers ───────────────────────────────────────────────────────

def pixel_to_latlon(
    row: int, col: int, transform, crs=None
) -> Tuple[float, float]:
    """
    Convert pixel (row, col) → (latitude, longitude).

    When the file CRS is projected (e.g. UTM), reprojects to WGS84 first.
    Without a CRS, assumes the transform is already in degrees.
    """
    from rasterio.transform import xy as _xy
    x, y = _xy(transform, row, col)   # native CRS coords (metres for UTM)

    if crs is not None:
        from rasterio.crs import CRS as _CRS
        from rasterio.warp import transform as _warp
        src = crs if hasattr(crs, "is_geographic") else _CRS.from_user_input(crs)
        if not src.is_geographic:
            # Reproject easting/northing → lon/lat (WGS84)
            wgs84 = _CRS.from_epsg(4326)
            lons, lats = _warp(src, wgs84, [x], [y])
            return float(lats[0]), float(lons[0])

    # Already geographic — rasterio returns (x=lon, y=lat)
    return float(y), float(x)


def bbox_to_latlon(
    row_min: int, col_min: int, row_max: int, col_max: int, transform, crs=None
) -> Dict[str, float]:
    """Return WGS84 lat/lon corners of a pixel bounding box."""
    lat1, lon1 = pixel_to_latlon(row_min, col_min, transform, crs)
    lat2, lon2 = pixel_to_latlon(row_max, col_max, transform, crs)
    return {
        "lat_min": min(lat1, lat2),
        "lat_max": max(lat1, lat2),
        "lon_min": min(lon1, lon2),
        "lon_max": max(lon1, lon2),
        "center_lat": (lat1 + lat2) / 2,
        "center_lon": (lon1 + lon2) / 2,
    }


def pixel_area_m2(transform) -> float:
    """Return area of one pixel in square metres."""
    return abs(transform.a * transform.e)


# ── Synthetic test data ──────────────────────────────────────────────────────

def create_synthetic_geotiff(
    path: str | Path,
    width: int = 512,
    height: int = 512,
    bands: int = 6,
) -> Path:
    """Create a small synthetic GeoTIFF for pipeline testing."""
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, (bands, height, width)).astype(np.float32)
    # Simulate some vegetation (high NIR band 3, low SWIR)
    data[3] = rng.uniform(0.4, 0.8, (height, width)).astype(np.float32)

    transform = from_bounds(30.0, 25.0, 32.0, 27.0, width, height)

    with rasterio.open(
        str(path), "w",
        driver="GTiff",
        height=height, width=width,
        count=bands, dtype="float32",
        crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        dst.write(data)
    return path
