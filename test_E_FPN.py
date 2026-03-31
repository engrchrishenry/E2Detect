import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import os
from models.E_FPN import E_FPN
from data_utils import Event_to_SSDFeature_Dataset
import torch.utils.data
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import data_utils
from train_E_FPN import cosine_similarity_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing script for TSFNet_E2SIFT")
    parser.add_argument("--vox_path", type=str, nargs="+",
                        help="One or more paths to directories containing test voxel .npz files")
    parser.add_argument("--feat_path", type=str, nargs="+",
                        help="One or more paths to directories containing test SSD backbone feature .npz files")
    parser.add_argument("--weights", type=str, 
                        help='Path to trained weights')
    parser.add_argument("--out_path", type=str,
                        help='Path to output predicted LoG pyramid')
    parser.add_argument("--vox_clip", type=float, nargs=2, metavar=('min', 'max'),
                        help='Min and max clipping value for event voxels')
    parser.add_argument("--feat_clip", type=float, nargs=2, metavar=('min', 'max'),
                        help='Min and max clipping value for SSD backbone feature')
    parser.add_argument("--dct_min", type=str,
                        help='Path to dct_min.npy (generated via get_dct_min_max.py)')
    parser.add_argument("--dct_max", type=str, 
                        help='Path to dct_max.npy (generated via get_dct_min_max.py)')
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--n_workers", type=int, default=4,
                        help='Number of workers for data loading')

    args = parser.parse_args()

    test_data_load = Event_to_SSDFeature_Dataset(args.vox_path, args.feat_path, args.vox_clip, args.feat_clip, 'sigmoid')
    test_loader = torch.utils.data.DataLoader(test_data_load, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    print("Test batches: ", len(test_loader))

    model = E_FPN(args.dct_min, args.dct_max)
    model.load_state_dict(torch.load(args.weights)) #, strict=False
    mse_check = nn.MSELoss()
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if not os.path.exists(f'{args.out_path}/pred_feat'):
            os.makedirs(f'{args.out_path}/pred_feat')

    total_mse = 0.0
    total_cos_metric = 0.0
    with torch.no_grad():
        for idx, (voxs, feats, names) in enumerate(test_loader):
            print (f'Processing batch: {idx+1}/{len(test_loader)}')
            if torch.cuda.is_available():
                voxs = voxs.cuda()
                feats = feats.cuda()

            outputs = model(voxs)
            
            # feat_clip_min, feat_clip_max = args.feat_clip
            # if feat_clip_min is not None and feat_clip_max is not None:
            #     outputs = torch.clamp(outputs, min=feat_clip_min, max=feat_clip_max)

            total_mse += mse_check(outputs, feats).item()
            total_cos_metric += cosine_similarity_metric(outputs, feats).item()
            
            outputs = outputs.squeeze().detach().cpu().numpy()
            if outputs.ndim == 3: # for batch size of 1, outputs will have shape (4, H, W) instead of (B, 4, H, W)
                np.savez_compressed(f'{args.out_path}/pred_feat/{names[0]}.npz', outputs)
            else:
                for idx2, o in enumerate(outputs):
                    np.savez_compressed(f'{args.out_path}/pred_feat/{names[idx2]}.npz', o)

        average_mse = total_mse / len(test_loader)
        average_cos = total_cos_metric / len(test_loader)

        print('Evaluating Network.....')
        print('Test set: MSE: {:.7f}, COS SIM: {:.7f}'.format(
            average_mse,
            average_cos
        ))

        with open(f"{args.out_path}/results.txt", "w") as f:
                f.write(f"Test set: MSE: {average_mse:.7f}, COS SIM: {average_cos:.7f}")


