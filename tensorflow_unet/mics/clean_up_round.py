# 512_dynamic_unet/run_3/bs_12/dr_0,9/w_3/h_6/ep_48_
# 512_dynamic_unet/run_3/bs_8/dr_0,95/w_4/h_6/ep_48_
# 512_dynamic_unet/run_3/bs_8/dr_0,95/w_4/h_6/ep_40_
# 512_dynamic_unet/run_3/bs_8/dr_0,9/w_3/h_6/ep_40_
# 512_dynamic_unet/run_3/bs_12/dr_0,95/w_4/h_6/ep_40_
# 512_dynamic_unet/run_3/bs_8/dr_0,95/w_4/h_6/
# 512_dynamic_unet/run_3/bs_12/dr_0,9/w_3/h_6/ep_32_
# 512_dynamic_unet/run_3/bs_8/dr_0,95/w_3/h_6/ep_40_
# 512_dynamic_unet/run_3/bs_8/dr_0,95/w_2/h_6/ep_32_
# 512_dynamic_unet/run_3/bs_12/dr_0,9/w_3/h_6/
# 512_dynamic_unet/run_3/bs_12/dr_0,95/w_3/h_6/ep_32_
# 512_dynamic_unet/run_3/bs_12/dr_0,9/w_3/h_6/ep_40_
# 512_dynamic_unet/run_3/bs_12/dr_0,95/w_3/h_6/
# 512_dynamic_unet/run_3/bs_12/dr_0,95/w_3/h_6/ep_24_
# 512_dynamic_unet/run_3/bs_8/dr_0,9/w_3/h_6/
# 512_dynamic_unet/run_3/bs_8/dr_0,95/w_3/h_5/
# 512_dynamic_unet/run_3/bs_8/dr_0,9/w_3/h_6/ep_24_
# 512_dynamic_unet/run_3/bs_8/dr_0,95/w_2/h_6/ep_48_
# 512_dynamic_unet/run_3/bs_8/dr_0,95/w_3/h_5/ep_48_
# 512_dynamic_unet/run_3/bs_12/dr_0,9/w_2/h_6/ep_48_
# 512_dynamic_unet/run_3/bs_8/dr_0,95/w_3/h_5/ep_40_
# 512_dynamic_unet/run_3/bs_12/dr_0,9/w_3/h_5/
# 512_dynamic_unet/run_3/bs_8/dr_0,9/w_3/h_5/ep_48_
# 512_dynamic_unet/run_3/bs_8/dr_0,95/w_2/h_6/
# 512_dynamic_unet/run_3/bs_8/dr_0,95/w_5/h_6/
# 512_dynamic_unet/run_3/bs_8/dr_0,9/w_4/h_6/ep_40_
# 512_dynamic_unet/run_3/bs_12/dr_0,95/w_3/h_6/ep_48_
# 512_dynamic_unet/run_3/bs_8/dr_0,95/w_2/h_6/ep_40_
# 512_dynamic_unet/run_3/bs_8/dr_0,9/w_4/h_5/ep_32_
# 512_dynamic_unet/run_3/bs_12/dr_0,95/w_3/h_6/ep_40_
# 512_dynamic_unet/run_3/bs_8/dr_0,9/w_4/h_6/ep_24_
# 512_dynamic_unet/run_3/bs_12/dr_0,95/w_2/h_6/ep_24_
# 512_dynamic_unet/run_3/bs_8/dr_0,9/w_4/h_5/ep_40_
# 512_dynamic_unet/run_3/bs_8/dr_0,95/w_2/h_6/ep_24_
# 512_dynamic_unet/run_3/bs_8/dr_0,9/w_3/h_6/ep_48_
# 512_dynamic_unet/run_3/bs_8/dr_0,9/w_2/h_6/
import sys
import shutil
import os
from pathlib import Path


def remove_unmatched_files(dir_a, dir_b):
    """
    Remove files from dir_a that don't have a filename match in dir_b.

    Parameters:
        dir_a (str): Path to the source directory (files to potentially remove).
        dir_b (str): Path to the reference directory (files to compare against).
    """
    # Get set of filenames in dir_b
    b_filenames = set(os.listdir(dir_b))

    # Iterate over files in dir_a
    for filename in os.listdir(dir_a):
        a_file_path = os.path.join(dir_a, filename)
        
        # Only check files (not directories)
        if os.path.isfile(a_file_path) and filename not in b_filenames:
            print(f"Removing: {a_file_path}")
            os.remove(a_file_path)
            
            
            
            
            
remove_unmatched_files('dataset/cross_sections/images/images', 'dataset/cross_sections/local')            
            
# def get_first_dirs(path_str, num):
#     path = Path(path_str)
#     return str(Path(*path.parts[:num]))



# raw_models = [line.strip() for line in sys.stdin if line.strip()]
# mapped_models = [get_first_dirs(model, 6) for model in raw_models]

# # Get base directory (first 2 parts of first path)
# run_dir = Path(get_first_dirs(raw_models[0], 2))

# # Delete directories in run_dir that aren't in mapped_models
# for subdir in run_dir.glob("*/*/*/*"):
#     candidate = str(subdir)
#     if candidate not in mapped_models:
#         print(f"Removing: {candidate}")
#         shutil.rmtree(subdir, ignore_errors=True)

# print(f"Done cleaning in: {run_dir}")

# print(run_dir)