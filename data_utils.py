import torch
# import torchvision
import torch.utils.data
import numpy as np
import os
import glob


def norm_vox_log(arr, clip_min, clip_max, activation):
    # Clip values between clip_min and clip_max
    clipped_array = np.clip(arr, clip_min, clip_max)

    # Normalize values to 0-1
    min_value = np.min(clipped_array)
    max_value = np.max(clipped_array)
    
    if activation == 'sigmoid':
        epsilon = 1e-10
        normalized_array = (clipped_array - min_value) / (max_value - min_value + epsilon)
    if activation == 'tanh':
        normalized_array = (((clipped_array - min_value) / (max_value - min_value))*2)-1
    return clipped_array, normalized_array


def norm_vox_e2vid(vox):
    nonzero_ev = (vox != 0)
    num_nonzeros = nonzero_ev.sum()
    if num_nonzeros > 0:
        # compute mean and stddev of the **nonzero** elements of the event tensor
        # we do not use PyTorch's default mean() and std() functions since it's faster
        # to compute it by hand than applying those funcs to a masked array
        mean = vox.sum() / num_nonzeros
        stddev = torch.sqrt((vox ** 2).sum() / num_nonzeros - mean ** 2)
        mask = nonzero_ev.float()
        vox = mask * (vox - mean) / stddev
    return vox


def recon_norm_log(arr, clip_min, clip_max):
    return arr * (clip_max - clip_min) + clip_min


class Event_to_SSDFeature_Dataset(torch.utils.data.Dataset):
    def __init__(self, vox_path, ssd_feat_path, clip_vox, clip_ssd_feat, activation):
        # self.vox_paths = sorted(glob.glob(f'{vox_path}/*.npz', recursive=True))
        # self.ssd_feat_paths = sorted(glob.glob(f'{ssd_feat_path}/*.npz', recursive=True))
        self.vox_paths = sorted(
            file
            for path in vox_path
            for file in glob.glob(os.path.join(path, "*.npz"), recursive=True)
        )
        self.ssd_feat_paths = sorted(
            file
            for path in ssd_feat_path
            for file in glob.glob(os.path.join(path, "*.npz"), recursive=True)
        )
        self.clip_min_vox, self.clip_max_vox = clip_vox[0], clip_vox[1]
        if clip_ssd_feat is not None:
            self.clip_min_ssd_feat, self.clip_max_ssd_feat = clip_ssd_feat
        else:
            self.clip_min_ssd_feat, self.clip_max_ssd_feat = None, None
        self.activation = activation

    def __len__(self):
        return len(self.vox_paths)

    def __getitem__(self, idx):
        vox_path = self.vox_paths[idx]
        ssd_feat_path = self.ssd_feat_paths[idx]
        name = os.path.basename(ssd_feat_path)
        name, ext = os.path.splitext(name) 
        
        vox = np.load(vox_path)['arr_0']
        ssd_feat = np.load(ssd_feat_path)['arr_0']

        vox = norm_vox_e2vid(torch.from_numpy(vox))
        vox = vox.cpu().numpy()
        _, vox = norm_vox_log(vox, self.clip_min_vox, self.clip_max_vox, self.activation)
        if self.clip_min_ssd_feat is not None and self.clip_max_ssd_feat is not None:
            ssd_feat = np.clip(ssd_feat, a_min=self.clip_min_ssd_feat, a_max=self.clip_max_ssd_feat)
        vox = torch.from_numpy(vox).float()
        ssd_feat = torch.from_numpy(ssd_feat).float()

        return vox, ssd_feat, name


if __name__ == "__main__":
    vox_path = 'path_to_voxels_dir'
    feat_path = 'path_to_feats_dir'

    train_data_load = Event_to_SSDFeature_Dataset(vox_path, feat_path, clip_vox=[-1, 1], clip_ssd_feat=[-1, 1], activation='sigmoid')
    training_loader = torch.utils.data.DataLoader(train_data_load, batch_size=2, shuffle=False, num_workers=2)
    for vox, feat, name in training_loader:
        print (vox.shape, feat.shape, name)
        exit(0)

