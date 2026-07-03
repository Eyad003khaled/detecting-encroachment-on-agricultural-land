"""batch_upload.py — Upload RF inference results for all 78 positive sites.

Usage:
    python batch_upload.py              # upload all positive sites
    python batch_upload.py --dry-run    # preview only, no requests sent
    python batch_upload.py --sites site0 site3 site48
"""
from __future__ import annotations
import argparse, pickle, time, json
from pathlib import Path
import numpy as np, rasterio
from rasterio.warp import transform_bounds
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

BASE_URL   = "https://kemet-grad-project-backend-production.up.railway.app"
UPLOAD_EP  = BASE_URL + "/cases/upload-image"
CASES_EP   = BASE_URL + "/cases"

ROOT       = Path(__file__).parent
BA_DIR     = ROOT / "data/KEMET1_BeforeAfter/KEMET1_BeforeAfter_Tiles"
LABELS     = ROOT / "data/ba_labels.json"
MODEL_PATH = ROOT / "models/ba_rf_model.pkl"
OUT_DIR    = ROOT / "outputs"; OUT_DIR.mkdir(exist_ok=True)

ALERT_THRESHOLD  = 0.40
YELLOW_THRESHOLD = 0.23
GEOCODE_DELAY    = 1.1   # Nominatim rate limit: 1 req/sec
SUBMIT_DELAY     = 0.5   # between case submissions

# ── HELPERS ───────────────────────────────────────────────────────────────────
def extract_stats(arr):
    feats = []
    for b in range(arr.shape[0]):
        ch = arr[b].ravel(); ch = ch[np.isfinite(ch)]
        feats += [ch.mean(), ch.std(),
                  np.percentile(ch,10), np.percentile(ch,25), np.percentile(ch,50),
                  np.percentile(ch,75), np.percentile(ch,90)]
    return np.array(feats)

def pair_features(d1, d2):
    fd = extract_stats(d2 - d1)
    return np.concatenate([fd, [float(np.nanmean(d2[0]-d1[0])),
                                float(np.nanmean(d2[1]-d1[1]))]])

def find_clusters(d1, d2, px_ha, min_ha=0.5):
    mask = ((d1[0]>0.25) & (d2[0]<0.25) & (d2[1]>d1[1]+0.08)).astype(np.uint8)
    labeled, n = ndimage.label(mask)
    out = []
    for i in range(n):
        ha = round((labeled==(i+1)).sum() * px_ha, 2)
        if ha >= min_ha: out.append(ha)
    return sorted(out, reverse=True)

def save_rgb_png(data, path, title):
    imgs = []
    for ch in [data[1], data[0], data[2]]:
        lo = np.nanpercentile(ch, 2); hi = np.nanpercentile(ch, 98)
        imgs.append(np.clip((ch-lo)/(hi-lo+1e-9), 0, 1))
    rgb = (np.stack(imgs, axis=-1) * 255).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(5,5), facecolor="#0d1117")
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title, color="white", fontsize=11, pad=4); ax.axis("off")
    plt.tight_layout(pad=0.3)
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0d1117"); plt.close()

def upload_image(img_path: Path) -> str:
    with open(img_path, "rb") as f:
        r = requests.post(UPLOAD_EP, files={"file": (img_path.name, f, "image/png")}, timeout=30)
    r.raise_for_status()
    body = r.json()
    rel = (body.get("url") or body.get("path") or
           body.get("imageUrl") or body.get("image_url") or body.get("filename"))
    if rel is None:
        raise ValueError(f"No URL key in upload response: {body}")
    if rel.startswith("http"): return rel
    return BASE_URL + ("" if rel.startswith("/") else "/") + rel

def reverse_geocode(lat, lon, retries=3) -> tuple[str, str]:
    """Return (governorate, markaz) with retry on failure."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={"format": "json", "lat": lat, "lon": lon,
                        "zoom": 8, "accept-language": "en"},
                headers={"User-Agent": "KEMET1-uploader/1.0"},
                timeout=10,
            )
            r.raise_for_status()
            addr = r.json().get("address", {})
            gov  = (addr.get("state") or addr.get("county") or
                    addr.get("region") or "")
            mrkz = (addr.get("state_district") or addr.get("county") or
                    addr.get("city") or addr.get("town") or
                    addr.get("village") or "")
            gov = gov.replace(" Governorate", "").strip()
            return gov, mrkz
        except Exception as e:
            print(f"      [geocode attempt {attempt}/{retries} failed] {e}")
            if attempt < retries:
                time.sleep(2)
    print(f"      [geocode] all retries failed — submitting with empty gov/markaz")
    return "", ""

# ── PER-SITE PROCESSING ───────────────────────────────────────────────────────
def process_site(site: str, bundle, dry_run: bool) -> dict:
    before_path = BA_DIR / f"{site}_before_2024.tif"
    after_path  = BA_DIR / f"{site}_after_2025.tif"
    if not before_path.exists() or not after_path.exists():
        return {"site": site, "status": "SKIPPED", "reason": "tiles not found"}

    # Load
    with rasterio.open(before_path) as src:
        d1 = src.read().astype(np.float32); res = src.res
        wgs = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        px_ha = (res[0]*res[1])/10_000
        by = before_path.stem.split("_")[-1]
    with rasterio.open(after_path) as src:
        d2 = src.read().astype(np.float32)
        ay = after_path.stem.split("_")[-1]

    clat = (wgs[1]+wgs[3])/2; clon = (wgs[0]+wgs[2])/2

    # RF inference (use Platt-calibrated model if available)
    clf = bundle.get("calibrated_model") or bundle["model"]
    scaler = bundle.get("scaler")
    feats = pair_features(d1, d2).reshape(1,-1)
    if scaler: feats = scaler.transform(feats)
    prob = float(clf.predict_proba(feats)[0,1])
    spec = float(np.nanmean(np.maximum(d1[0]-d2[0], 0)))
    fusion = round(0.65*prob + 0.35*spec, 4)
    alarm = ("High"   if fusion >= ALERT_THRESHOLD  else
             "Medium" if fusion >= YELLOW_THRESHOLD else "Low")
    clusters = find_clusters(d1, d2, px_ha)
    total_m2 = round(sum(clusters)*10_000, 2)

    # Images
    before_png = OUT_DIR / f"{site}_before.png"
    after_png  = OUT_DIR / f"{site}_after.png"
    save_rgb_png(d1, before_png, f"BEFORE ({by})")
    save_rgb_png(d2, after_png,  f"AFTER ({ay}) — {alarm}")

    # Geocode (respects Nominatim rate limit — caller adds delay between sites)
    gov, mrkz = reverse_geocode(clat, clon)

    payload = {
        "lat":              round(clat, 6),
        "long":             round(clon, 6),
        "governorate":      gov,
        "markaz":           mrkz,
        "ai_confidence":    round(fusion, 4),
        "before_image_url": "",
        "after_image_url":  "",
        "before_date":      f"{by}-01-01",
        "after_date":       f"{ay}-01-01",
        "area_lost_m2":     total_m2,
        "risk_level":       alarm,
    }

    if dry_run:
        payload["before_image_url"] = f"<{before_png.name}>"
        payload["after_image_url"]  = f"<{after_png.name}>"
        return {"site": site, "status": "DRY_RUN", "payload": payload}

    # Upload images
    payload["before_image_url"] = upload_image(before_png)
    time.sleep(0.3)
    payload["after_image_url"] = upload_image(after_png)

    # Submit case
    r = requests.post(CASES_EP, json=payload, timeout=30)
    r.raise_for_status()
    return {"site": site, "status": "OK", "http": r.status_code,
            "risk_level": alarm, "governorate": gov, "response": r.json()}

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", help="Specific sites (default: all 78 positive)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.sites:
        sites = args.sites
    else:
        labels = json.load(open(LABELS))
        sites = [r["site"] for r in labels if r["label"] == "pos"]
    sites = sorted(sites)

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing {len(sites)} sites...\n")
    bundle = pickle.load(open(MODEL_PATH, "rb"))

    results, failed = [], []
    for i, site in enumerate(sites, 1):
        print(f"[{i:02d}/{len(sites)}] {site}", end="  ", flush=True)
        try:
            res = process_site(site, bundle, args.dry_run)
            results.append(res)
            gov = res.get("governorate") or res.get("payload", {}).get("governorate", "")
            print(f"✓  risk={res.get('risk_level') or res.get('payload',{}).get('risk_level','')}  gov={gov!r}")
        except Exception as e:
            print(f"✗  ERROR: {e}")
            failed.append({"site": site, "error": str(e)})

        # Nominatim requires >= 1 req/sec — wait between sites
        if i < len(sites):
            time.sleep(GEOCODE_DELAY)

    log = OUT_DIR / "batch_upload_results.json"
    log.write_text(json.dumps({"submitted": len(results), "failed": len(failed),
                                "results": results, "errors": failed}, indent=2))
    print(f"\n{'='*50}")
    print(f"Done: {len(results)} submitted, {len(failed)} failed")
    if failed:
        print(f"Failed sites: {[f['site'] for f in 