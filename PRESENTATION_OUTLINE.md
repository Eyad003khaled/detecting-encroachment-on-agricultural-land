# 📊 PRESENTATION OUTLINE
## End-to-End System for Food Security — Detecting Buildings on Agriculture Areas

**Prepared By**: Khaled Ramadan Ali  
**Team**: Data & AI Team  
**Date**: April 2026

---

## Slide 1 — Title

> **End-to-End System for Food Security**  
> **Detecting Buildings on Agricultural Land Using Satellite Imagery & Deep Learning**
>
> Khaled Ramadan Ali · Data & AI Team  
> March – May 2026

---

## Slide 2 — The Problem

- 🌾 Agricultural land is being illegally consumed by construction
- Manual monitoring is slow, expensive, and limited in coverage
- Need: automated, scalable, satellite-based detection system

**Key Question**: *Can we automatically detect where buildings were built on farmland using free satellite data?*

---

## Slide 3 — The Solution (High-Level)

An **8-step AI pipeline** that:
1. Downloads satellite images (before & after)
2. Cleans them (remove clouds)
3. Detects what changed
4. Identifies farmland
5. Finds buildings on that farmland
6. Produces a colored map + report

**Final Output**: Colored map where 🔴 = buildings on farms, 🟡 = changed vegetation, 🟢 = safe farmland

---

## Slide 4 — Pipeline Overview (Visual Flow)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  STEP 01    │────▶│  STEP 02    │────▶│  STEP 03    │────▶│  STEP 04    │
│  Satellite  │     │  Cloud      │     │  Cloud      │     │  Spectral   │
│  Download   │     │  Detection  │     │  Removal    │     │  Indices    │
│ ──────────  │     │ ──────────  │     │ ──────────  │     │ ──────────  │
│ Sentinel-2  │     │ U-Net       │     │ OpenCV      │     │ NumPy       │
│ via GEE     │     │ + ResNet34  │     │ Telea       │     │ NDVI, NDBI  │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                   │
       ┌───────────────────────────────────────────────────────────┘
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  STEP 05    │────▶│  STEP 06    │────▶│  STEP 07    │────▶│  STEP 08    │
│  Change     │     │  Agriculture│     │  Building   │     │  Final      │
│  Detection  │     │  Segment.   │     │  Detection  │     │  Output     │
│ ──────────  │     │ ──────────  │     │ ──────────  │     │ ──────────  │
│ ChangeFormer│     │ SegFormer   │     │ SAM         │     │ Colored Map │
│ (LEVIR-CD)  │     │ (ADE20K)    │     │ (SpaceNet)  │     │ + Report    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## Slide 5 — Datasets Used (One Per Step)

| Step | Dataset | Link | Purpose |
|------|---------|------|---------|
| 01 | Sentinel-2 L2A | [GEE Collection](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) | 10m satellite imagery |
| 02 | 38-Cloud | [GitHub](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset) | Cloud mask labels |
| 05 | LEVIR-CD | [Download Page](https://justchenhao.github.io/LEVIR/) | Building change pairs |
| 06 | ADE20K | [HuggingFace](https://huggingface.co/nvidia/segformer-b4-finetuned-ade-512-512) | 150 semantic classes |
| 07 | SpaceNet v2 | [SpaceNet](https://spacenet.ai/spacenet-buildings-dataset-v2/) | Building footprints |

*Steps 03, 04, 08 use no external dataset — they process outputs from earlier steps.*

---

## Slide 6 — Models Used (One Per Step)

| Step | Model | Architecture | Pretrained Source |
|------|-------|-------------|-------------------|
| 02 | U-Net + ResNet34 | Encoder-Decoder CNN | ImageNet (auto-download) |
| 05 | ChangeFormer | Siamese Transformer | LEVIR-CD (manual download) |
| 06 | SegFormer-B4 | Hierarchical Transformer + MLP | ADE20K (auto-download via HuggingFace) |
| 07 | SAM | Vision Transformer (ViT) | SA-1B — 1.1 billion masks |

---

## Slide 7 — Step 01: Data Acquisition

**Goal**: Download multi-temporal satellite images

- **Source**: Sentinel-2 L2A via Google Earth Engine
- **Resolution**: 10m per pixel, 13 spectral bands
- **Method**: Define AOI + date range → filter clouds < 20% → median composite → export GeoTIFF
- **Output**: Two GeoTIFFs — `T1` (before) and `T2` (after)

---

## Slide 8 — Step 02: Cloud Detection

**Goal**: Find cloud pixels

- **Model**: U-Net with ResNet34 encoder (ImageNet pretrained)
- **Dataset**: 38-Cloud — 8,400 patches with cloud masks
- **Process**: Tile 512×512 → normalize → inference → sigmoid → threshold 0.45
- **Output**: Binary cloud mask (H × W)

---

## Slide 9 — Step 03: Cloud Removal

**Goal**: Reconstruct surface under clouds

- **Method**: OpenCV Telea inpainting (classical algorithm)
- **Process**: For each band → inpaint cloud pixels using nearest clear neighbors
- **No model weights needed** — runs on CPU instantly
- **Output**: Clean, cloud-free GeoTIFF

---

## Slide 10 — Step 04: Spectral Indices

**Goal**: Highlight vegetation and built-up areas

- **NDVI** = (NIR − Red) / (NIR + Red) → vegetation health
- **NDBI** = (SWIR − NIR) / (SWIR + NIR) → built-up density
- **MNDWI** = (Green − SWIR) / (Green + SWIR) → water bodies
- **Pure math** — no neural network, just NumPy arithmetic

---

## Slide 11 — Step 05: Change Detection

**Goal**: Find pixels that changed between T1 and T2

- **Model**: ChangeFormer — Siamese Transformer
- **Dataset**: LEVIR-CD — 637 image pairs with building-change labels
- **Process**: Feed T1 + T2 through dual Transformer branches → difference decoder
- **Output**: Binary change map — 1=changed, 0=unchanged

---

## Slide 12 — Step 06: Agriculture Segmentation

**Goal**: Identify farmland pixels in T1

- **Model**: SegFormer-B4 from NVIDIA (via HuggingFace)
- **Dataset**: ADE20K — 150 classes including field, earth, grass
- **Process**: Inference → 150-class semantic map → filter agriculture class IDs [9, 29, 92, 94, 96]
- **Output**: Binary agriculture mask — 255=farmland, 0=other

---

## Slide 13 — Step 07: Building Detection

**Goal**: Detect buildings in areas that are BOTH changed AND agricultural

- **Model**: SAM (Segment Anything — Meta AI)
- **Dataset**: SpaceNet v2 — building polygons from satellite images
- **Process**: Crop T2 to (changed ∩ agricultural) → SAM auto-segmentation → filter by area
- **Output**: Building footprint mask + polygons + confidence scores

---

## Slide 14 — Step 08: Final Output

**Goal**: Combine all masks into one visual result

**Color Rules:**
- 🔴 `building ∩ farmland ∩ changed` = Buildings on farmland
- 🟡 `changed ∩ farmland ∩ NOT building` = Changed vegetation
- 🟢 `farmland ∩ NOT changed` = Stable agriculture

**Exports**: Colored PNG, GeoTIFF, GeoJSON polygons, JSON report (area lost in hectares, % encroachment, building count)

---

## Slide 15 — Data Flow Summary

```
Sentinel-2  ──▶  Cloud Mask  ──▶  Clean Image  ──▶  Spectral Indices
                                       │                    │
                                       ▼                    ▼
                                  Agri Mask ◀──  Change Map
                                       │              │
                                       ▼              ▼
                                  Building Mask (changed ∩ agri)
                                       │
                                       ▼
                                  🗺️ FINAL COLORED MAP
                                  🔴 Buildings on farm
                                  🟡 Changed vegetation
                                  🟢 Stable agriculture
```

---

## Slide 16 — Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Data** | Google Earth Engine, rasterio, GDAL |
| **Deep Learning** | PyTorch, HuggingFace Transformers |
| **Models** | U-Net, ChangeFormer, SegFormer-B4, SAM |
| **Classical** | OpenCV, NumPy |
| **Geo** | GeoPandas, Folium, QGIS |

---

## Slide 17 — Timeline

| Phase | Weeks | Steps | Deliverable |
|-------|-------|-------|-------------|
| **Data Prep** | W1–W5 | Steps 01–04 | Clean, cloud-free image pairs with spectral indices |
| **AI Core** | W6–W11 | Steps 05–07 | Change map + agriculture mask + building detections |
| **Output** | W11–W12 | Step 08 | Final colored map, GeoJSON, JSON report |

---

## Slide 18 — Demo / Results

> *Show example output: colored satellite image with RED/YELLOW/GREEN overlay*
>
> - Total agricultural area monitored: X hectares
> - Buildings detected on farmland: Y
> - Area lost to construction: Z hectares
> - Encroachment percentage: W%

---

## Slide 19 — Conclusion

✅ **Automated** — no manual image analysis needed  
✅ **Scalable** — works on any region with Sentinel-2 coverage  
✅ **Free data** — uses only open satellite imagery  
✅ **Pretrained models** — no training from scratch required  
✅ **Actionable output** — GeoJSON polygons for authorities  

---

## Slide 20 — Q&A

> Thank you!
>
> GitHub: `Graduation_project_AI_system/`  
> Contact: Khaled Ramadan Ali
