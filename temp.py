import numpy as np
import glob
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import json


num_jobs = 20
input_dir = "updated_arr_0/esim_reds_filtered_patched/train/5_0.55_0.005_50_70000_300000/ssd_feat_normed"
output_dir = "updated_arr_0/esim_reds_filtered_patched/train/5_0.55_0.005_50_70000_300000/ssd_feat_normed2"
os.makedirs(output_dir, exist_ok=True)

files = glob.glob(os.path.join(input_dir, "*.npz"))

def process_file(file):
    data = np.load(file)
    arr = data['backbone_feature']

    filename = os.path.basename(file)
    out_path = os.path.join(output_dir, filename)
    # np.save(out_path.replace('.npz', ''), arr)
    np.savez_compressed(out_path, arr)


# joblib with tqdm
Parallel(n_jobs=num_jobs)(
    delayed(process_file)(f) for f in tqdm(files, desc="Processing files")
)

print("Done!")