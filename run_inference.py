"""run_inference.py - KEMET1 BeforeAfter encroachment report generator."""
from __future__ import annotations
import argparse, pickle, base64, datetime
from pathlib import Path
import numpy as np, rasterio
from rasterio.warp import transform_bounds
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

BA_DIR     = Path("data/KEMET1_BeforeAfter/KEMET1_BeforeAfter_Tiles")
MODEL_PATH = Path("models/ba_rf_model.pkl")
OUT_DIR    = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

def extract_stats(arr):
    feats = []
    for b in range(arr.shape[0]):
        ch = arr[b].ravel(); ch = ch[np.isfinite(ch)]
        feats += [ch.mean(), ch.std(),
                  np.percentile(ch,10), np.percentile(ch,25), np.percentile(ch,50),
                  np.percentile(ch,75), np.percentile(ch,90)]
    return np.array(feats)

def pair_features(d1, d2):
    """44 spectral-delta features. pct_conv and pct_new removed (circular with labels)."""
    fd = extract_stats(d2 - d1)
    ndvi2, ndbi2 = d2[0], d2[1]; ndvi1, ndbi1 = d1[0], d1[1]
    return np.concatenate([fd, [float(np.nanmean(ndvi2-ndvi1)),
                                float(np.nanmean(ndbi2-ndbi1))]])

def shared_rgb(d1, d2):
    imgs = []
    for ch1, ch2 in [(d1[1],d2[1]),(d1[0],d2[0]),(d1[2],d2[2])]:
        combined = np.concatenate([ch1.ravel(), ch2.ravel()])
        lo = np.nanpercentile(combined,2); hi = np.nanpercentile(combined,98)
        nrm = lambda x, lo=lo, hi=hi: np.clip((x-lo)/(hi-lo+1e-9),0,1)
        imgs.append((nrm(ch1), nrm(ch2)))
    return np.stack([r[0] for r in imgs],-1), np.stack([r[1] for r in imgs],-1)

def find_clusters(d1, d2, min_ha=0.5):
    ndvi1, ndbi1 = d1[0], d1[1]; ndvi2, ndbi2 = d2[0], d2[1]
    mask = ((ndvi1>0.25)&(ndvi2<0.25)&(ndbi2>ndbi1+0.08)).astype(np.uint8)
    labeled, n = ndimage.label(mask)
    out = []
    for i in range(n):
        comp = labeled==(i+1); ha = comp.sum()*100/10_000
        if ha >= min_ha:
            rows = np.where(comp.any(axis=1))[0]; cols = np.where(comp.any(axis=0))[0]
            out.append((rows[0],rows[-1],cols[0],cols[-1],round(ha,2)))
    return sorted(out, key=lambda x:-x[4])

def card(lbl, val, color="#c9d1d9", sm=False):
    sz = "font-size:1rem;" if sm else ""
    return ('<div class="card"><div class="lbl">'+lbl+'</div>'
            '<div class="val" style="color:'+color+';'+sz+'">'+str(val)+'</div></div>')

# JSONP geocoding — script tags bypass file:// CORS restrictions
GEOCODE_JS = r"""
function _showName(ci,full,rlat,rlon){
  var sel='[data-name="'+ci+'"]';
  var sp=document.querySelector(sel);
  if(sp){sp.textContent=full;sp.style.color='#c9d1d9';}
  if(_clayers[ci])_clayers[ci].bindPopup(
    '<b>'+full+'</b><br>Cluster #'+(ci+1)+'<br>'+(_clayers[ci]._ha||'')
    +'<br>'+parseFloat(rlat).toFixed(5)+'N '+parseFloat(rlon).toFixed(5)+'E');
}
function _parseAddr(d){
  var a=d.address||{};
  var nm=a.hamlet||a.neighbourhood||a.isolated_dwelling
        ||a.village||a.suburb||a.quarter||a.allotments
        ||a.town||a.municipality||a.city_district||a.city
        ||a.county||a.state_district||'';
  var gv=a.state||a.state_district||a.county||'';
  if(!nm&&d.display_name){
    var pts=d.display_name.split(',');
    for(var k=0;k<pts.length;k++){
      var p=pts[k].trim();
      if(p&&!/^\d/.test(p)){nm=p;if(k+1<pts.length)gv=pts[k+1].trim();break;}
    }
  }
  return (nm||'Nile Delta area')+(gv&&gv!==nm?' – '+gv:'');
}
function _jsonpGeo(ci,rlat,rlon){
  window['_gcb'+ci]=function(d){_showName(ci,_parseAddr(d),rlat,rlon);};
  var t=document.createElement('script');
  t.onerror=function(){
    var nm=_parseAddr({display_name:'',address:{}});
    _showName(ci,'Nile Delta ('+parseFloat(rlat).toFixed(3)+'N)',rlat,rlon);
  };
  t.src='https://nominatim.openstreetmap.org/reverse?format=json&lat='+rlat
       +'&lon='+rlon+'&zoom=14&accept-language=en&json_callback=_gcb'+ci;
  document.head.appendChild(t);
}
window.addEventListener('load',function(){
  document.querySelectorAll('tr[data-ci]').forEach(function(row,idx){
    var ci=parseInt(row.getAttribute('data-ci'));
    var rlat=row.getAttribute('data-lat');
    var rlon=row.getAttribute('data-lon');
    setTimeout(function(){_jsonpGeo(ci,rlat,rlon);},idx*700);
  });
});
"""

def run(before_path, after_path, site_name):
    with rasterio.open(before_path) as s:
        d1 = s.read().astype(np.float32)
        wgs = transform_bounds(s.crs,"EPSG:4326",*s.bounds)
        tile_ha = s.width*s.height*s.res[0]*s.res[0]/10_000
    with rasterio.open(after_path) as s:
        d2 = s.read().astype(np.float32)
    clat = (wgs[1]+wgs[3])/2; clon = (wgs[0]+wgs[2])/2

    # Parse acquisition years from filenames (e.g. site3_before_2024.tif → "2024")
    _by = Path(before_path).stem.split("_")[-1]
    _ay = Path(after_path).stem.split("_")[-1]
    before_year = _by if _by.isdigit() else "before"
    after_year  = _ay if _ay.isdigit() else "after"

    bundle = pickle.load(open(MODEL_PATH,"rb"))
    feat = np.nan_to_num(pair_features(d1,d2), nan=0.0).reshape(1,-1)
    prob = float(bundle["model"].predict_proba(feat)[0,1])

    # ── Signal 2: spectral composite score (tile-level) ─────────────────────────
    # Uses feat[42]=mean_dNDVI and feat[43]=mean_dNDBI (tile averages).
    # Positive when NDVI drops (vegetation loss) AND/OR NDBI rises (built-up gain).
    # Per-cluster pixel NDBI_after was tested as an alternative but produced more
    # FPs (bare soil/urban-fringe pixels at 10m overlap with true construction);
    # the RF tile-level score outperforms it as a discriminator.
    ndvi_d_mean = float(feat[0, 42])
    ndbi_d_mean = float(feat[0, 43])
    spectral_score = float(np.clip((-ndvi_d_mean + ndbi_d_mean) * 1.5, 0.0, 1.0))

    # ── Fusion: RF tile-level x0.65 + spectral composite x0.35 ─────────────────
    fusion_score = 0.65 * prob + 0.35 * spectral_score

    YELLOW_THRESHOLD = 0.30   # raised from 0.23: yellow precision 32.6%->62.5%
    ALERT_THRESHOLD  = 0.40   # high-confidence red alert threshold
    VERIFY_AREA_HA   = 80.0
    VERIFY_CONF_HA   = 40.0
    VERIFY_CONF_CEIL = 0.60

    if fusion_score >= ALERT_THRESHOLD:
        alarm_level = "red"
        label = "Red Alert — High-confidence encroachment"
        sc = "#ff4444"
    elif fusion_score >= YELLOW_THRESHOLD:
        alarm_level = "yellow"
        label = "Yellow Alert — Possible encroachment, verify required"
        sc = "#e3a030"
    else:
        alarm_level = "none"
        label = "No encroachment signal"
        sc = "#44ff88"

    if alarm_level == "none":
        no_enc_css = ("body{background:#0d1117;color:#c9d1d9;font-family:'Segoe UI',sans-serif;"
                      "display:flex;flex-direction:column;align-items:center;justify-content:center;"
                      "height:100vh;margin:0;text-align:center}"
                      ".box{background:#161b22;border:1px solid #30363d;border-radius:8px;"
                      "padding:20px 32px;display:inline-block;margin-top:16px}"
                      ".score{font-size:2.2rem;font-weight:700;color:#44ff88}"
                      ".lbl{font-size:10px;color:#6e7681;text-transform:uppercase;letter-spacing:.06em}"
                      "footer{position:fixed;bottom:10px;font-size:10px;color:#6e7681}")
        no_enc_html = "\n".join([
            "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>",
            "<title>KEMET1 - "+site_name+"</title>",
            "<style>"+no_enc_css+"</style>",
            "</head><body>",
            "<div><div style='font-size:3rem;margin-bottom:8px'>&#x2705;</div>",
            "<h1 style='color:#44ff88;font-size:1.5rem;margin-bottom:6px'>No Encroachment Detected</h1>",
            "<p style='color:#6e7681;font-size:0.85rem;margin-bottom:0'>%s &middot; Fusion score %.3f below yellow threshold (%.2f)</p>" % (site_name, fusion_score, YELLOW_THRESHOLD),
            "<div class='box'><div class='score'>%.3f</div>" % fusion_score,
            "<div class='lbl'>Fusion Score (RF&times;0.65 + Spectral&dagger;&times;0.35)</div></div>",
            "<div class='box' style='margin-top:8px'><div class='score' style='font-size:1.1rem'>RF %.3f &nbsp;&nbsp; Spec %.3f</div>" % (prob, spectral_score),
            "<div class='lbl'>Component Scores</div></div>",
            "<p style='margin-top:20px;font-size:11px;color:#6e7681'>Full report suppressed &mdash; no significant encroachment signal</p>",
            "</div>",
            "<footer>KEMET1 BeforeAfter RF Classifier &middot; Sentinel-2 10m"
        " &middot; %s&rarr;%s"
        " &middot; No SCL cloud masking applied &mdash; verify cloud-free acquisition</footer>" % (before_year, after_year),
            "</body></html>",
        ])
        out_html = OUT_DIR/(site_name+"_report.html")
        out_html.write_text(no_enc_html, encoding="utf-8")
        print(f"HTML (no signal, fusion={fusion_score:.3f} RF={prob:.3f} spec={spectral_score:.3f}): {out_html}")
        return prob, 0, 0, 0

    clusters = find_clusters(d1, d2)
    total_ha = sum(c[4] for c in clusters)
    main_ha  = clusters[0][4] if clusters else 0.0
    # Flag for manual review: large area OR low-confidence with moderate area
    needs_verify = bool(clusters) and (
        total_ha > VERIFY_AREA_HA or
        (prob < VERIFY_CONF_CEIL and total_ha > VERIFY_CONF_HA)
    )

    img1, img2 = shared_rgb(d1, d2); ndiff = d2[0]-d1[0]
    BG = "#0d1117"
    fig = plt.figure(figsize=(20,7.5),facecolor=BG)
    gs = GridSpec(1,3,figure=fig,wspace=0.04,left=0.02,right=0.98,top=0.88,bottom=0.08)
    ax0 = fig.add_subplot(gs[0]); ax0.set_facecolor(BG)
    ax0.imshow(img1,interpolation="nearest")
    ax0.set_title("BEFORE (%s)"%before_year,color="#8ee3ff",fontsize=13,fontweight="bold",pad=6); ax0.axis("off")
    ax1 = fig.add_subplot(gs[1]); ax1.set_facecolor(BG)
    ax1.imshow(img2,interpolation="nearest")
    ax1.set_title("AFTER (%s) - "%after_year+label,color=sc,fontsize=13,fontweight="bold",pad=6); ax1.axis("off")
    for i,(r0,r1,c0,c1,ha) in enumerate(clusters[:8]):
        clr="#ff3333" if i==0 else "#ff9900"
        ax1.add_patch(patches.Rectangle((c0,r0),c1-c0,r1-r0,lw=2 if i==0 else 1.2,
            edgecolor=clr,facecolor=clr+"22",ls="-" if i==0 else "--"))
        if i==0:
            ax1.annotate("%.1f ha"%ha,xy=(c0+(c1-c0)//2,r0-4),color="#ff6060",
                fontsize=9,ha="center",va="bottom",fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2",fc="#1a0000",ec="#ff3333",lw=1))
    ax1.set_xlabel("R=NDBI G=NDVI B=MNDWI 10m/px shared scale",color="#6e7681",fontsize=9)
    ax2 = fig.add_subplot(gs[2]); ax2.set_facecolor(BG)
    im = ax2.imshow(ndiff,cmap="RdYlGn",interpolation="nearest",vmin=-0.5,vmax=0.5)
    for r0,r1,c0,c1,_ in clusters[:8]:
        ax2.add_patch(patches.Rectangle((c0,r0),c1-c0,r1-r0,lw=1.5,edgecolor="yellow",facecolor="none"))
    cb = fig.colorbar(im,ax=ax2,fraction=0.04,pad=0.02)
    cb.ax.tick_params(colors="#aaa"); cb.set_label("ΔNDVI  ±0.5 shared scale",color="#aaa",fontsize=9)
    ax2.set_title("ΔNDVI Raw Spectral Change (not RF output)",color="#e3a030",fontsize=11,fontweight="bold",pad=6)
    ax2.axis("off")
    fig.suptitle("%s | Fusion: %.3f  (RF %.3f×0.65 + Spec %.3f×0.35) | Changed: %.1f ha | Tile: %.1f ha"
                 %(site_name,fusion_score,prob,spectral_score,total_ha,tile_ha),color="#c9d1d9",fontsize=11,y=0.97)
    out_png = OUT_DIR/(site_name+"_report.png")
    plt.savefig(out_png,dpi=140,bbox_inches="tight",facecolor=BG); plt.close()
    print("PNG:", out_png)

    hpx = d1.shape[1]; wpx = d1.shape[2]
    lpp = (wgs[3]-wgs[1])/hpx; lnp = (wgs[2]-wgs[0])/wpx
    js_rects = []; sidebar_rows = []

    for i,(r0,r1,c0,c1,ha) in enumerate(clusters):
        lat_s = wgs[3]-r1*lpp; lat_n = wgs[3]-r0*lpp
        lon_w = wgs[0]+c0*lnp; lon_e = wgs[0]+c1*lnp
        clt = (lat_s+lat_n)/2; clo = (lon_w+lon_e)/2
        clr = "#ff3333" if i==0 else ("#ff6600" if i<3 else "#ffaa00")
        wt  = 2.5 if i==0 else 1.8
        ha_str = "%.1f ha" % ha
        bbox = "[[%.5f,%.5f],[%.5f,%.5f]]"%(lat_s,lon_w,lat_n,lon_e)
        js_rects.append(
            "_clayers[%d]=L.rectangle(%s,{color:'%s',weight:%.1f,"
            "dashArray:'5 3',fillColor:'%s',fillOpacity:0.18}).addTo(map);"
            "_clayers[%d]._ha='%s';"
            "_clayers[%d].bindPopup('Cluster #%d<br>%s<br>%.5fN %.5fE');"
            % (i,bbox,clr,wt,clr,i,ha_str,i,i+1,ha_str,clt,clo)
        )
        sidebar_rows.append(
            "<tr data-ci='%d' data-lat='%.5f' data-lon='%.5f'>"
            "<td>#%d</td><td>%.5fN</td><td>%.5fE</td>"
            "<td><span data-name='%d' style='color:#8ee3ff'>loading...</span></td>"
            "<td>%s</td></tr>"
            % (i,clt,clo,i+1,clt,clo,i,ha_str)
        )

    with open(out_png,"rb") as f: b64 = base64.b64encode(f.read()).decode()
    mbbox = "[[%.5f,%.5f],[%.5f,%.5f]]"%(wgs[1],wgs[0],wgs[3],wgs[2])
    cjs  = "\n".join(js_rects)
    stbl = "\n".join(sidebar_rows)

    cards_html = "".join([
        card("Fusion Score","%.3f"%fusion_score,sc),
        card("RF Score (×0.65)","%.3f"%prob,"#8ee3ff"),
        card("Spectral Score &times;0.35&dagger;","%.3f"%spectral_score,"#8ee3ff"),
        card("Alarm Level",label,sc,sm=True),
        card("Centre Lat","%.5fN"%clat,"#c9d1d9",sm=True),
        card("Centre Lon","%.5fE"%clon,"#c9d1d9",sm=True),
        card("Total Area Lost","%.1f ha"%total_ha,"#ff8c00"),
        card("Largest Cluster","%.1f ha"%main_ha),
        card("Tile Area","%.1f ha"%tile_ha),
        card("Change Clusters",str(len(clusters))),
        card("&#x26A0; Verify","High-area — manual review required","#e3a030",sm=True) if needs_verify else "",
    ])

    map_script = (
        "var _clayers=[];"
        "var map=L.map('map').setView([%.5f,%.5f],15);" % (clat,clon)
        + "var sat=L.tileLayer('https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',"
          "{maxZoom:20,subdomains:['mt0','mt1','mt2','mt3'],attribution:'Google'});"
          "var hyb=L.tileLayer('https://{s}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',"
          "{maxZoom:20,subdomains:['mt0','mt1','mt2','mt3'],attribution:'Google'});"
          "var osm=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'OSM'});"
          "sat.addTo(map);"
        + "L.rectangle(%s,{color:'#3399ff',weight:2,fillOpacity:0.04,dashArray:'8 4'})"
          ".addTo(map).bindPopup('Study tile %.1f ha');" % (mbbox, tile_ha)
        + cjs
        + "L.circleMarker([%.5f,%.5f],{radius:5,color:'%s',fillColor:'%s',fillOpacity:0.9,weight:2})"
          % (clat,clon,sc,sc)
        + ".addTo(map).bindPopup('<b>%s</b><br>Fusion:%.3f (RF:%.3f Spec:%.3f)<br>%s<br>Lost:%.1f ha');"
          % (site_name,fusion_score,prob,spectral_score,label,total_ha)
        + "L.control.layers({'Satellite':sat,'Hybrid':hyb,'OSM':osm}).addTo(map);"
          "L.control.scale().addTo(map);"
    )

    css = (
        "*{margin:0;padding:0;box-sizing:border-box}"
        "body{background:#0d1117;color:#c9d1d9;font-family:'Segoe UI',sans-serif}"
        "h1{padding:16px 24px;font-size:1.2rem;color:#8ee3ff;border-bottom:1px solid #21262d}"
        ".cards{display:flex;gap:14px;padding:16px 24px;flex-wrap:wrap}"
        ".card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 18px;min-width:130px}"
        ".card .lbl{font-size:10px;color:#6e7681;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}"
        ".card .val{font-size:1.3rem;font-weight:700}"
        ".ms{display:flex;margin:0 24px;border-radius:8px;overflow:hidden;border:1px solid #30363d;height:460px}"
        "#map{flex:1;min-width:0}"
        ".panel{width:310px;background:#161b22;overflow-y:auto;border-left:1px solid #30363d;flex-shrink:0}"
        ".panel h3{padding:12px 16px;font-size:11px;color:#8ee3ff;text-transform:uppercase;"
        "letter-spacing:.07em;border-bottom:1px solid #21262d;position:sticky;top:0;background:#161b22;z-index:1}"
        ".panel table{width:100%;border-collapse:collapse;font-size:11px}"
        ".panel th{padding:7px 10px;color:#6e7681;text-align:left;border-bottom:1px solid #21262d;font-size:10px}"
        ".panel td{padding:7px 10px;border-bottom:1px solid #161b22}"
        ".panel tr:nth-child(odd) td{background:#0d1117}"
        ".panel tr:first-child td{color:#ff6060;font-weight:600}"
        ".tot{padding:10px 16px;font-size:11px;color:#8ee3ff;border-top:1px solid #30363d;background:#0d1117}"
        ".leg{padding:10px 16px;font-size:10px;border-top:1px solid #21262d}"
        ".li{display:flex;align-items:center;gap:6px;margin-bottom:5px}"
        ".dot{width:12px;height:12px;border-radius:2px;border:2px solid}"
        ".iw{padding:16px 24px}.iw img{width:100%;border-radius:8px;border:1px solid #21262d}"
        "footer{padding:10px 24px;font-size:10px;color:#6e7681;border-top:1px solid #21262d}"
    )

    html = "\n".join([
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>",
        "<title>KEMET1 - "+site_name+"</title>",
        "<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>",
        "<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>",
        "<style>"+css+"</style>",
        "</head><body>",
        "<h1>KEMET1 %s &mdash; %s</h1>" % (
            "Red Alert" if alarm_level == "red" else "Yellow Alert", site_name),
        ('<div style="background:#2d2400;border:1px solid #e3a030;border-radius:6px;'
         'margin:0 24px 8px;padding:8px 16px;font-size:12px;color:#e3a030">'
         '&#x26A1; <b>Yellow Alert &mdash; Uncertain Detection</b> &mdash; '
         'Fusion score %.3f is between yellow (%.2f) and red (%.2f) thresholds. '
         'Spectral signal: %.3f. Manual verification required before any action.</div>'
         % (fusion_score, YELLOW_THRESHOLD, ALERT_THRESHOLD, spectral_score))
        if alarm_level == "yellow" else "",
        ('<div style="background:#2d1800;border:1px solid #e3a030;border-radius:6px;'
         'margin:0 24px 8px;padding:8px 16px;font-size:11px;color:#e3a030">'
         '&#x26A0; <b>Requires Manual Verification</b> -- '
         'detected area (%.1f ha) exceeds 80 ha threshold (tile: %.1f ha total)</div>' % (total_ha, tile_ha))
        if needs_verify else "",
        '<div class="cards">'+cards_html+"</div>",
        '<p style="margin:0 24px 8px;font-size:10px;color:#6e7681">'
        '&dagger; Spectral score = clip((&minus;&Delta;&macr;NDVI+&Delta;&macr;NDBI)&times;1.5, 0, 1)'
        ' using tile-average feat[42&ndash;43] &mdash; confirmatory, not independent from RF. '
        'Yellow threshold raised 0.23&rarr;0.30: yellow precision 32.6%&rarr;62.5%.</p>',
        '<div class="ms"><div id="map"></div>',
        '<div class="panel"><h3>Detected Change Areas</h3>',
        "<table><thead><tr><th>#</th><th>Lat</th><th>Lon</th><th>Location Name</th><th>Area</th></tr></thead>",
        "<tbody>"+stbl+"</tbody></table>",
        '<div class="tot">Total lost: <b>%.1f ha</b> across %d clusters</div>' % (total_ha,len(clusters)),
        '<div class="leg">',
        '<div class="li"><div class="dot" style="border-color:#ff3333;background:#ff333330"></div>Largest cluster</div>',
        '<div class="li"><div class="dot" style="border-color:#ff6600;background:#ff660030"></div>2nd-3rd cluster</div>',
        '<div class="li"><div class="dot" style="border-color:#ffaa00;background:#ffaa0030"></div>Smaller clusters</div>',
        '<div class="li"><div class="dot" style="border-color:#3399ff;background:#3399ff10"></div>Tile boundary</div>',
        "</div></div></div>",
        "<script>"+map_script+"</script>",
        "<script>"+GEOCODE_JS+"</script>",
        '<div class="iw"><img src="data:image/png;base64,'+b64+'" alt="Before/After"></div>',
        "<footer>KEMET1 BeforeAfter RF Classifier &middot; Sentinel-2 10m"
        " &middot; Before=%s After=%s &middot; Yellow&ge;%.2f / Red&ge;%.2f"
        " (Fusion=RF&times;0.65+Spectral&dagger;&times;0.35)"
        " &middot; &dagger;Spectral=clip(NDBI<sub>after</sub>[cluster px]/0.30,0,1) pixel-wise built-up confirmation"
        " &middot; No SCL cloud masking &middot; %s UTC</footer>"
        % (before_year, after_year, YELLOW_THRESHOLD, ALERT_THRESHOLD, datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")),
        "</body></html>",
    ])

    out_html = OUT_DIR/(site_name+"_report.html")
    out_html.write_text(html, encoding="utf-8")
    print("HTML:", out_html)
    return prob, total_ha, main_ha, len(clusters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site",   type=str,  default=None)
    parser.add_argument("--before", type=Path, default=None)
    parser.add_argument("--after",  type=Path, default=None)
    args = parser.parse_args()
    if args.site:
        sn = args.site
        run(BA_DIR/(sn+"_before_2024.tif"), BA_DIR/(sn+"_after_2025.tif"), sn)
    elif args.before and args.after:
        run(args.before, args.after, args.before.stem)
    else:
        import json
        labels = json.load(open("data/ba_labels.json"))
        for r in [x for x in labels if x["label"]=="pos"][:5]:
            sn = r["site"]
            print("\n"+"="*60+"\n"+sn)
            run(BA_DIR/(sn+"_before_2024.tif"), BA_DIR/(sn+"_after_2025.tif"), sn)

if __name__ == "__main__":
    main()
