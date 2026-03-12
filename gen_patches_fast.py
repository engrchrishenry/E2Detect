import argparse
import os
import sys
import numpy as np
import glob
import shutil
import math
import multiprocessing
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm


def process_file(vox_path, im_path):
    f_name = os.path.basename(vox_path)
    f_name, _ = os.path.splitext(f_name)

    # ========== VOX ==========
    vox = np.load(vox_path)['arr_0']
    c = 0
    for i in range(0, im_h - patch_h + 1, step_h):
        for j in range(0, im_w - patch_w + 1, step_w):
            c += 1
            patch = vox[:, i:i+patch_h, j:j+patch_w]
            np.savez_compressed(f'{out_path}/vox/{f_name}_{c}', patch)

    # ========== IMAGE ==========
    im = Image.open(im_path)
    im = np.array(im)
    c = 0
    for i in range(0, im_h - patch_h + 1, step_h):
        for j in range(0, im_w - patch_w + 1, step_w):
            c += 1
            patch = im[i:i+patch_h, j:j+patch_w, :]
            Image.fromarray(patch).save(f'{out_path}/images/{f_name}_{c}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate patches from event voxels for training/testin via using multiple cores.')
    parser.add_argument('--base_path', type=str, required=True,
                        help='Base path for input data (should contain subfolders: vox, images)')
    parser.add_argument('--out_path', type=str, required=True,
                        help='Output path for patched data (will contain subfolders: vox, images)')
    parser.add_argument('--ip_size', type=str, required=True,
                        help='Size of the input images (width:height). E.g., 533:300')
    parser.add_argument('--patch_size', type=str, required=True,
                        help='Size of the patches (width:height). E.g., 300:300')
    parser.add_argument('--cores', type=int, default=-1,
                        help='Number of cores to use. -1 -> use all cores.')
    
    args = parser.parse_args()

    im_w, im_h = map(int, args.ip_size.split(':'))
    base_path = args.base_path
    out_path = args.out_path
    patch_w, patch_h = map(int, args.patch_size.split(':'))
    cores = multiprocessing.cpu_count() if args.cores == -1 else args.cores

    voxs = sorted(glob.glob(f'{base_path}/vox/*.npz'))
    ims = sorted(glob.glob(f'{base_path}/images/*.png'))

    for sub_dir in ["images", "vox"]:
        os.makedirs(f"{out_path}/{sub_dir}", exist_ok=True)

    # Compute required number of patches
    num_patches_w, num_patches_h = math.ceil(im_w/patch_w), math.ceil(im_h/patch_h)

    # Compute independent steps
    step_w = (im_w - patch_w) // (num_patches_w - 1) if num_patches_w > 1 else patch_w
    step_h = (im_h - patch_h) // (num_patches_h - 1) if num_patches_h > 1 else patch_h

    # Run in parallel
    Parallel(n_jobs=cores)(
        delayed(process_file)(vox, im)
        for vox, im in tqdm(zip(voxs, ims), total=len(voxs), desc="Processing files")
    )

