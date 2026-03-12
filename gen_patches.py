import argparse
import os
import sys
import numpy as np
import glob
import shutil
import math
from PIL import Image
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate patches from event voxels for training/testing.')
    parser.add_argument('--base_path', type=str, required=True,
                        help='Base path for input data (should contain subfolders: vox, images)')
    parser.add_argument('--out_path', type=str, required=True,
                        help='Output path for patched data (will contain subfolders: vox, images)')
    parser.add_argument('--ip_size', type=str, required=True,
                        help='Size of the input images (width:height). E.g., 533:300')
    parser.add_argument('--patch_size', type=str, required=True,
                        help='Size of the patches (width:height). E.g., 300:300')
    args = parser.parse_args()

    im_w, im_h = map(int, args.ip_size.split(':'))
    base_path = args.base_path
    out_path = args.out_path
    patch_w, patch_h = map(int, args.patch_size.split(':'))

    voxs = sorted(glob.glob(f'{base_path}/vox/*.npz'))
    ims = sorted(glob.glob(f'{base_path}/images/*.png'))

    for sub_dir in ["images", "vox"]:
        os.makedirs(f"{out_path}/{sub_dir}", exist_ok=True)

    # Compute required number of patches
    num_patches_w, num_patches_h = math.ceil(im_w/patch_w), math.ceil(im_h/patch_h)
    print (num_patches_w, num_patches_h)

    # Compute independent steps
    step_w = (im_w - patch_w) // (num_patches_w - 1) if num_patches_w > 1 else patch_w
    step_h = (im_h - patch_h) // (num_patches_h - 1) if num_patches_h > 1 else patch_h

    for vox, im, in tqdm(zip(voxs, ims), total=len(voxs)):
        f_name = os.path.basename(vox)
        f_name, _ = os.path.splitext(f_name)
        vox = np.load(vox)['arr_0']
        c = 0
        for i in range(0, im_h - patch_h + 1, step_h):
            for j in range(0, im_w - patch_w + 1, step_w):
                c += 1
                patch = vox[:, i:i+patch_h, j:j+patch_w]
                np.savez_compressed(f'{out_path}/vox/{f_name}_{c}', patch)

        im = Image.open(im)
        im = np.array(im)
        c = 0
        for i in range(0, im_h - patch_h + 1, step_h):
            for j in range(0, im_w - patch_w + 1, step_w):
                c += 1
                patch = im[i:i+patch_h, j:j+patch_w, :]
                Image.fromarray(patch).save(f'{out_path}/images/{f_name}_{c}.png')
        
        
