# Food Security ML Pipeline

## End-to-End System for Detecting Buildings on Agricultural Land

An 8-step ML pipeline that compares satellite imagery from two time periods to detect where buildings have been constructed on agricultural land.

### Pipeline Steps

| # | Step | Model / Tool | Output |
|---|------|-------------|--------|
| 01 | Data Acquisition | Google Earth Engine | GeoTIFF (T1 & T2) |
| 02 | Cloud Detection | U-Net + ResNet34 | Cloud probability + binary mask |
| 03 | Cloud Removal | OpenCV Telea | Cloud-free GeoTIFFs |
| 04 | Spectral Indices | NumPy | NDVI, NDBI, MNDWI arrays |
| 05 | Change Detection | ChangeFormer | Binary change map |
| 06 | Agriculture Seg. | SegFormer-B4 | Farmland mask |
| 07 | Building Detection | SAM + YOLOv8-seg | Building mask + polygons |
| 08 | Final Output | OpenCV + GeoPandas | Colored PNG, GeoTIFF, GeoJSON, report |

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run with test data (no model weights needed for fallback mode)
python run.py --test

# 3. Run with real satellite images
python run.py --t1 data/raw/T1/image.tif --t2 data/raw/T2/image.tif

# 4. Download from Google Earth Engine
python run.py --gee
```

### Project Structure

```
├── config/settings.py           # All configurations
├── src/
│   ├── utils/                   # Geo I/O, tiling, logging
│   ├── step_01_data_acquisition/
│   ├── step_02_cloud_detection/
│   ├── step_03_cloud_removal/
│   ├── step_04_spectral_indices/
│   ├── step_05_change_detection/
│   ├── step_06_agriculture_segmentation/
│   ├── step_07_building_detection/
│   └── step_08_final_output/
├── pipeline.py                  # Orchestrator
├── run.py                       # CLI entry point
└── requirements.txt
```

### Output Color Legend

- **RED**: Buildings detected on farmland (encroachment)
- **YELLOW**: Vegetation change (no building)
- **GREEN**: Stable agricultural land

### Model Weights

Download pretrained weights to the `weights/` directory:

- **ChangeFormer**: [github.com/wgcban/ChangeFormer](https://github.com/wgcban/ChangeFormer) → `weights/ChangeFormer_LEVIR.pth`
- **SAM (vit_b)**: [Download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) → `weights/sam_vit_b_01ec64.pth`
- **SegFormer-B4**: Downloads automatically from HuggingFace
- **YOLOv8-seg**: Downloads automatically from Ultralytics

### Recommended Datasets

| Dataset | Use Case | Source |
|---------|----------|--------|
| LEVIR-CD | Change detection training | [justchenhao/LEVIR](https://justchenhao.github.io/LEVIR/) |
| CloudSEN12 | Cloud detection training | [Zenodo](https://zenodo.org/record/7431205) |
| SpaceNet v2 | Building detection | [spacenet.ai](https://spacenet.ai/) |
| ADE20K | Agriculture segmentation | HuggingFace (auto) |

