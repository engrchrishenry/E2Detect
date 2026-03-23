import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from datetime import datetime
from model.model_chris import E2VID
from models_biren.EDSR_2 import TSFNet_Chris
from utils.loading_utils import load_model
from data_utils import Event_Camera_Dataset_SSD_patches
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import time
from pytorch_msssim import SSIM, ssim
from torch.optim.lr_scheduler import CosineAnnealingLR
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import matplotlib.pyplot as plt
import data_utils
from scipy.io import savemat
from PIL import Image
from train_tsfnet_ssd import cosine_similarity_metric

path_to_model = 'logs_ssd/10_cos_raw_ncaltech_Sunday_02_November_2025_04h_23m_31s/ckpt/best.pth'
vox_path_valid = '/storage4tb/PycharmProjects/rpg_e2vid/output/n_caltech101_processed/test/resized_300_300/events_voxels_resized'
ssd_feat_path = '/storage4tb/PycharmProjects/SSD/demo/gt_feat_ncaltech101/test/ssd_feat_normed'
out_path = 'output/pred_feat_ncaltech_cos_normed/test' # pred_log_tsfne pred_log_e2vid

batch_size = 16
vox_clip = [-4.42, 4.23]
ssd_clip = [None, None] # None
num_workers = 3
activation = 'sigmoid'

valid_data_load = Event_Camera_Dataset_SSD_patches(vox_path_valid, ssd_feat_path, vox_clip, ssd_clip, activation)
valid_loader = torch.utils.data.DataLoader(valid_data_load, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print("Validation set samples: ", len(valid_loader))

# model = E2VID()
model = TSFNet_Chris()
model.load_state_dict(torch.load(path_to_model))
mse_check = nn.MSELoss()
if torch.cuda.is_available():
    model.cuda()
model.eval()

if not os.path.exists(out_path):
    os.makedirs(out_path)

total_mse = 0.0
total_cos_metric = 0.0
with torch.no_grad():
    for idx, (voxs, logs, names) in enumerate(valid_loader):
        if idx % 10 == 0:
            print (f'{idx}/{len(valid_loader)}')
        if torch.cuda.is_available():
            voxs = voxs.cuda()
            logs = logs.cuda()

        outputs = model(voxs)
        if ssd_clip[0] is not None and ssd_clip[1] is not None:
            outputs = torch.clamp(outputs, min=ssd_clip[0], max=ssd_clip[1])

        total_mse += mse_check(outputs, logs).item()
        total_cos_metric += cosine_similarity_metric(outputs, logs).item()
        
        outputs = outputs.squeeze().detach().cpu().numpy()
        if outputs.ndim == 3:
            np.savez_compressed(f'{out_path}/{names[0]}.npz', patch=outputs)
            # for idx2, channel in enumerate(outputs):
            #     plt.imshow(channel, cmap = 'gray')
            #     plt.axis('off')
            #     plt.tight_layout()
            #     plt.savefig(f'{out_path}/{names[0]}_{idx2}.png', bbox_inches='tight', pad_inches=0)
            #     plt.close()
        else:
            for idx2, o in enumerate(outputs):
                np.savez_compressed(f'{out_path}/{names[idx2]}.npz', patch=o)
                # for idx2, channel in enumerate(o):
                #     plt.imshow(channel, cmap = 'gray')
                #     plt.axis('off')
                #     plt.tight_layout()
                #     plt.savefig(f'{out_path}/{names[idx]}_{idx2}.png', bbox_inches='tight', pad_inches=0)
                #     plt.close()

    average_mse = total_mse / len(valid_loader)
    average_cos = total_cos_metric / len(valid_loader)

    print('Evaluating Network.....')
    print('Test set: MSE: {:.7f}, COS SIM: {:.7f}'.format(
        average_mse,
        average_cos
    ))

