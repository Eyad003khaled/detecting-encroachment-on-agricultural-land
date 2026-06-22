# 🚀 How To Run: Food Security AI System

This manual explains exactly how to set up the environment, prepare the models, and run the pipeline from start to finish to get your final results.

---

## 🛠️ Phase 1: Environment Setup

Because this project uses complex geospatial libraries (like `rasterio` and `geopandas`), the safest way to install it on Windows is using **Anaconda/Miniconda**.

### 1. Create a Conda Environment
Open your **Anaconda Prompt** and run:
```bash
conda create -n food_project python=3.10 -y
conda activate food_project
```

### 2. Install Geospatial Libraries via Conda
This safely installs the C++ GDAL dependencies underneath avoiding Windows crash errors:
```bash
conda install -c conda-forge geopandas rasterio fiona shapely -y
```

### 3. Install Deep Learning Libraries via Pip
Now install the rest of the AI framework from your requirements file:
```bash
cd d:\ML_DS_DA\anti_project\Graduation_project_AI_system
pip install torch torchvision numpy opencv-python segmentation-models-pytorch transformers ultralytics segment-anything einops tqdm
```

---

## 🧠 Phase 2: Download Model Weights

Before running the actual script, you must manually place two highly specific pretrained model weights into your `weights/` folder. 
*(Note: Empty folders will automatically be created when you first run the test script, or you can create the `weights/` folder yourself).*

1. **ChangeFormer (Step 05)**:
   - Download the LEVIR-CD pretrained weights.
   - Save inside the weights folder as: `weights/ChangeFormer_LEVIR.pth`
2. **SAM (Step 07)**:
   - Download the Meta AI Segment Anything ViT-B weights (approx 375MB).
   - [Click here to download sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
   - Save inside the weights folder as: `weights/sam_vit_b_01ec64.pth`

*(The other models like U-Net and SegFormer will automatically download their own weights via the internet during their first run).*

---

## 🏃 Phase 3: Running The Pipeline

We have provided a central entry point script called `run.py`. Make sure your terminal is inside the `Graduation_project_AI_system` folder.

### Option A: The "Dry Run" Test (Extremely Fast Verification)
To ensure your environment is working without needing huge real satellites images, run the built-in synthetic test mode:
```bash
python run.py --test
```
*What it does: Creates tiny fake T1/T2 GeoTIFF images and pushes them through all 8 steps to verify no errors occur.*

### Option B: Offline Mode (Using your own GeoTIFFs)
If you already have a "Before" (T1) and "After" (T2) satellite image of the same location:
```bash
python run.py --t1 data/raw/T1/my_before_image.tif --t2 data/raw/T2/my_after_image.tif
```
*Note: The script expects these to be Multi-Band GeoTIFFs (specifically Sentinel-2 equivalents with bands B2, B3, B4, B8, B11).*

### Option C: Google Earth Engine Auto-Download Mode
To let the script automatically query Google Earth Engine, find images from 2022 and 2024 (configurable in `config/settings.py`), and process them:
```bash
# First, you must authenticate GEE on your system
earthengine authenticate

# Then run the script
python run.py --gee
```

---

## 📂 Phase 4: Understanding The Outputs

Once the pipeline reaches **Step 08**, it will finalize the execution. You can find all your results inside the `outputs/` folder.

You will see the following files generated:

1. **`final_report.json`**:
   - A data report containing the specific mathematical statistics (e.g. Total Hectares of stable farmland, Total Hectares of changed land, Exact count of illegal buildings detected).
2. **`encroachment_polygons.geojson`**:
   - The geometric vector shapes of the detected illegal buildings. You can drag and drop this file into **QGIS** or **Google Earth Pro** to physically see the building shapes on the map!
3. **`final_colored_map.png`** (and `.tif`):
   - A beautiful visualization image mapping out the land:
     - 🔴 **RED**: Illegal Buildings on farmland (Encroachment)
     - 🟡 **YELLOW**: Vegetation changes (Could be seasonal harvest, no building found)
     - 🟢 **GREEN**: Stable, healthy agricultural land.

*Check the `logs/pipeline.log` file at any time if you wish to see how fast each of the 8 steps performed.*
