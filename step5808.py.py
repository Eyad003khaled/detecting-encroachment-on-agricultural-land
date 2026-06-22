import numpy as np
from src.utils.geo_utils import read_geotiff, get_rgb_from_multiband, write_geotiff
from src.step_05_change_detection.detect_changes import run as run_05
from src.step_08_final_output.generate_output import run as run_08

t1_clean, t1_meta = read_geotiff('data/processed/T1_cloud_free.tif')
t2_clean, t2_meta = read_geotiff('data/processed/T2_cloud_free.tif')
agri_mask, _ = read_geotiff('data/processed/agriculture_mask.tif')
agri_mask = agri_mask[0]
building_mask, _ = read_geotiff('data/processed/building_mask.tif')
building_mask = building_mask[0]

print('Running change detection with new threshold...')
result_05 = run_05(t1_clean, t2_clean, t1_meta)
change_map = result_05['change_map']
print('Changed pixels:', np.count_nonzero(change_map))

t2_rgb = get_rgb_from_multiband(t2_clean)
result_08 = run_08(t2_rgb, change_map, agri_mask, building_mask, [], t2_meta)
print('DONE! Check outputs/final_colored.png')