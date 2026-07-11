# -*- coding: utf-8 -*-
"""app.py - KEMET1 RF inference web server."""
from __future__ import annotations
import base64, io, os, pickle, tempfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

ROOT             = Path(__file__).parent
MODEL_PATH       = ROOT / "models" / "ba_rf_model.pkl"
STATIC_DIR       = ROOT / "static"
ALERT_THRESHOLD  = 0.40
YELLOW_THRESHOLD = 0.23

print("Loading RF model...", end=" ", flush=True)
_bundle = pickle.load(open(MODEL_PATH, "rb"))
_clf    = _bundle.get("calibrated_model") or _bundle["model"]
_scaler = _bundle.get("scaler")
print("done.")

def _extract_stats(arr):
    feats = []
    for b in range(arr.shape[0]):
        ch = arr[b].ravel(); ch = ch[np.isfinite(ch)]
        feats += [ch.mean(), ch.std(),
                  np.percentile(ch,10), np.percentile(ch,25),
                  np.percentile(ch,50), np.percentile(ch,75),
                  np.percentile(ch,90)]
    return np.array(feats)

def _pair_features(d1, d2):
    fd = _extract_stats(d2 - d1)
    return np.concatenate([fd, [float(np.nanmean(d2[0]-d1[0])),
                                float(np.nanmean(d2[1]-d1[1]))]])

def _find_clusters(d1, d2, px_ha, min_ha=0.5):
    mask = ((d1[0]>0.25)&(d2[0]<0.25)&(d2[1]>d1[1]+0.08)).astype(np.uint8)
    labeled, n = ndimage.label(mask)
    slices = ndimage.find_objects(labeled)
    pairs = []
    for i in range(n):
        ha = round((labeled==(i+1)).sum()*px_ha, 2)
        if ha >= min_ha:
            rs, cs = slices[i]
            pairs.append((ha,(rs.start,cs.start,rs.stop,cs.stop)))
    pairs.sort(key=lambda x: -x[0])
    if pairs:
        areas, boxes = zip(*pairs)
        return list(areas), list(boxes)
    return [], []

def _to_b64(data, title, boxes=None):
    imgs = []
    for ch in [data[1], data[0], data[2]]:
        lo=np.nanpercentile(ch,2); hi=np.nanpercentile(ch,98)
        imgs.append(np.clip((ch-lo)/(hi-lo+1e-9),0,1))
    rgb = (np.stack(imgs,axis=-1)*255).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(5,5), facecolor="#0d1117")
    ax.imshow(rgb, interpolation="nearest")
    if boxes:
        for (r0,c0,r1,c1) in boxes:
            ax.add_patch(patches.Rectangle((c0,r0),c1-c0,r1-r0,
                linewidth=1.5,edgecolor="#FF6B00",facecolor="none"))
    ax.set_title(title, color="white", fontsize=11, pad=4)
    ax.axis("off"); plt.tight_layout(pad=0.3)
    buf=io.BytesIO()
    plt.savefig(buf,format="png",dpi=120,bbox_inches="tight",facecolor="#0d1117")
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

app = FastAPI(title="KEMET1 RF Inference")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    p = STATIC_DIR / "index.html"
    if not p.exists(): raise HTTPException(404, "index.html not found")
    return p.read_text(encoding="utf-8")

@app.post("/predict")
async def predict(before: UploadFile=File(...), after: UploadFile=File(...)):
    b1=await before.read(); b2=await after.read()
    with tempfile.NamedTemporaryFile(suffix=".tif",delete=False) as f: f.write(b1); p1=Path(f.name)
    with tempfile.NamedTemporaryFile(suffix=".tif",delete=False) as f: f.write(b2); p2=Path(f.name)
    try:
        with rasterio.open(p1) as src:
            d1=src.read().astype(np.float32); res=src.res
            wgs=transform_bounds(src.crs,"EPSG:4326",*src.bounds)
            px_ha=(res[0]*res[1])/10_000; nb=src.count
        with rasterio.open(p2) as src:
            d2=src.read().astype(np.float32)
        if d1.shape!=d2.shape:
            raise HTTPException(422,f"Shape mismatch: {d1.shape} vs {d2.shape}")
        if nb<6:
            raise HTTPException(422,f"Expected 6-band TIF, got {nb}")
    except HTTPException: raise
    except Exception as e: raise HTTPException(422,f"Could not read TIF: {e}")
    finally: p1.unlink(missing_ok=True); p2.unlink(missing_ok=True)

    feats=_pair_features(d1,d2).reshape(1,-1)
    if _scaler: feats=_scaler.transform(feats)
    prob=float(_clf.predict_proba(feats)[0,1])
    spec=float(np.nanmean(np.maximum(d1[0]-d2[0],0)))
    fusion=round(0.65*prob+0.35*spec,4)
    alarm=("High" if fusion>=ALERT_THRESHOLD else "Medium" if fusion>=YELLOW_THRESHOLD else "Low")
    clusters,boxes=_find_clusters(d1,d2,px_ha)
    total_m2=round(sum(clusters)*10_000,2)
    clat=round((wgs[1]+wgs[3])/2,6); clon=round((wgs[0]+wgs[2])/2,6)
    return JSONResponse({"risk_level":alarm,"rf_prob":round(prob,4),
        "fusion_score":fusion,"clusters":len(clusters),
        "area_m2":total_m2,"area_feddan":round(total_m2/4200,2),
        "lat":clat,"lon":clon,
        "before_img":_to_b64(d1,"BEFORE"),
        "after_img":_to_b64(d2,f"AFTER - {alarm}",boxes=boxes)})

if __name__=="__main__":
    port=int(os.environ.get("PORT",8000))
    uvicorn.run(app,host="0.0.0.0",port=port,reload=False)
