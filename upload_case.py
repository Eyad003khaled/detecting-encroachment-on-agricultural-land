"""upload_case.py — RF inference + backend submission for one site.

Usage:
    python upload_case.py site0
    python upload_case.py site0 --dry-run
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
MODEL_PATH = ROOT / "models/ba_rf_model.pkl"
OUT_DIR    = ROOT / "outputs"; OUT_DIR.mkdir(exist_ok=True)

ALERT_THRESHOLD  = 0.40
YELLOW_THRESHOLD = 0.23

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

def reverse_geocode(lat, lon):
    """Return (governorate, markaz). Prints raw address for debugging."""
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format": "json", "lat": lat, "lon": lon,
                    "zoom": 8, "accept-language": "en"},
            headers={"User-Agent": "KEMET1-uploader/1.0"},
            timeout=10,
        )
        data = r.json()
        addr = data.get("address", {})
        print(f"      [geocode raw] {addr}")

        # Egypt: state = "Kafr el-Sheikh Governorate", state_district = markaz
        gov  = (addr.get("state") or
                addr.get("county") or
                addr.get("region") or "")
        mrkz = (addr.get("state_district") or
                addr.get("county") or
                addr.get("city") or
                addr.get("town") or
                addr.get("village") or "")

        # Strip " Governorate" suffix if present
        gov = gov.replace(" Governorate", "").strip()

        return gov, mrkz
    except Exception as e:
        print(f"      [geocode error] {e}")
        return "", ""

def run(site: str, dry_run: bool = False):
    before_path = BA_DIR / f"{site}_before_2024.tif"
    after_path  = BA_DIR / f"{site}_after_2025.tif"
    if not before_path.exists() or not after_path.exists():
        raise FileNotFoundError(f"Tiles not found for {site} in {BA_DIR}")

    print(f"[1/5] Loading tiles for {site}...")
    with rasterio.open(before_path) as src:
        d1 = src.read().astype(np.float32); res = src.res
        wgs = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        px_ha = (res[0]*res[1])/10_000
        by = before_path.stem.split("_")[-1]
    with rasterio.open(after_path) as src:
        d2 = src.read().astype(np.float32)
        ay = after_path.stem.split("_")[-1]

    clat = (wgs[1]+wgs[3])/2; clon = (wgs[0]+wgs[2])/2
    before_date = f"{by}-01-01" if by.isdigit() else "2024-01-01"
    after_date  = f"{ay}-01-01" if ay.isdigit() else "2025-01-01"

    print("[2/5] Running RF inference...")
    bundle = pickle.load(open(MODEL_PATH, "rb"))
    clf = bundle.get("calibrated_model") or bundle["model"]
    scaler = bundle.get("scaler")
    feats = pair_features(d1, d2).reshape(1,-1)
    if scaler: feats = scaler.transform(feats)
    prob = float(clf.predict_proba(feats)[0,1])
    spec = float(np.nanmean(np.maximum(d1[0]-d2[0], 0)))
    fusion = round(0.65*prob + 0.35*spec, 4)
    alarm = "High" if fusion >= ALERT_THRESHOLD else ("Medium" if fusion >= YELLOW_THRESHOLD else "Low")
    clusters = find_clusters(d1, d2, px_ha)
    total_m2 = round(sum(clusters)*10_000, 2)
    print(f"      prob={prob:.4f}  fusion={fusion}  risk={alarm}  area={total_m2:.0f} m2")

    print("[3/5] Generating before/after images...")
    before_png = OUT_DIR / f"{site}_before.png"
    after_png  = OUT_DIR / f"{site}_after.png"
    save_rgb_png(d1, before_png, f"BEFORE ({by})")
    save_rgb_png(d2, after_png,  f"AFTER ({ay}) — {alarm}")

    print("[4/5] Reverse geocoding...")
    gov, mrkz = reverse_geocode(clat, clon)
    print(f"      lat={clat:.5f}  lon={clon:.5f}  governorate={gov!r}  markaz={mrkz!r}")

    payload = {
        "lat":              round(clat, 6),
        "long":             round(clon, 6),
        "governorate":      gov,
        "markaz":           mrkz,
        "ai_confidence":    round(fusion, 4),
        "before_image_url": "",
        "after_image_url":  "",
        "before_date":      before_date,
        "after_date":       after_date,
        "area_lost_m2":     total_m2,
        "risk_level":       alarm,
    }

    if dry_run:
        payload["before_image_url"] = f"<would upload {before_png.name}>"
        payload["after_image_url"]  = f"<would upload {after_png.name}>"
        print("\n[DRY RUN] Payload:")
        print(json.dumps(payload, indent=2))
        return

    print("[5/5] Uploading images and submitting case...")
    payload["before_image_url"] = upload_image(before_png)
    print(f"      before_image_url = {payload['before_image_url']}")
    time.sleep(0.3)
    payload["after_image_url"] = upload_image(after_png)
    print(f"      after_image_url  = {payload['after_image_url']}")

    print("\n      Payload being submitted:")
    print(json.dumps(payload, indent=2))

    r = requests.post(CASES_EP, json=payload, timeout=30)
    r.raise_for_status()
    print(f"\nCase submitted ({r.status_code}):")
    print(json.dumps(r.json(), indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("site", help="Site name e.g. site0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.site, dry_run=args.dry_run)
