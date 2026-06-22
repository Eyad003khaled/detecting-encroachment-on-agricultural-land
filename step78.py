import numpy as np
from src.utils.geo_utils import read_geotiff, get_rgb_from_multiband
from src.step_07_building_detection.detect_buildings import run as run_07
from src.step_08_final_output.generate_output import run as run_08
from src.step_05_change_detection.detect_changes import run as run_05

t2_clean, t2_meta = read_geotiff("data/processed/T2_cloud_free.tif")
t1_clean, t1_meta = read_geotiff('data/processed/T1_cloud_free.tif')
result_05 = run_05(t1_clean, t2_clean, t1_meta)
change_map = result_05['change_map']
agri_mask, _ = read_geotiff("data/processed/agriculture_mask.tif")
agri_mask = agri_mask[0]
print("Agri pixels:", np.count_nonzero(agri_mask))
result_07 = run_07(t2_clean, change_map, agri_mask, t2_meta)
t2_rgb = get_rgb_from_multiband(t2_clean)
result_08 = run_08(
    t2_rgb,
    change_map,
    agri_mask,
    result_07["building_mask"],
    result_07.get("polygons", []),
    t2_meta,
)
print("DONE!")