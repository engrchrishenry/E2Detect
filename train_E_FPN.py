import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from datetime import datetime
from models.E_FPN import E_FPN
from data_utils import Event_to_SSDFeature_Dataset
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
matplotlib.use('Agg') # Use non-GUI backend for servers or training


def plot_preds(outputs, gts):
    idx = random.randint(0, len(outputs) - 1)
    output = outputs[idx]  # (N, C, H, W)
    gt = gts[idx]          # (N, C, H, W)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Plot GTs
    gt_proj = gt.mean(dim=0).detach().cpu().numpy()    # mean projection
    im1 = axes[0, 0].imshow(gt_proj, cmap="inferno")
    axes[0, 0].set_title(f'GT mean')
    fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Plot Predictions
    pred_proj = output.mean(dim=0).detach().cpu().numpy()
    im2 = axes[0, 1].imshow(pred_proj, cmap="inferno")
    axes[0, 1].set_title(f'Pred mean')
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    gt_proj = gt.var(dim=0, unbiased=False).detach().cpu().numpy()    # variance projection
    im3 = axes[1, 0].imshow(gt_proj, cmap="inferno")
    axes[1, 0].set_title(f'GT var')
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Plot Predictions
    pred_proj = output.var(dim=0, unbiased=False).detach().cpu().numpy()
    im4 = axes[1, 1].imshow(pred_proj, cmap="inferno")
    axes[1, 1].set_title(f'Pred var')
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


class CosineLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(CosineLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred, target: [B, C, H, W]
        B, C, H, W = pred.shape

        # reshape to [B, H*W, C]
        pred = pred.permute(0, 2, 3, 1).reshape(B, -1, C)
        target = target.permute(0, 2, 3, 1).reshape(B, -1, C)

        # cosine similarity along channel dimension
        cos_sim = F.cosine_similarity(pred, target, dim=-1, eps=self.eps)  # [B, ]

        # 1 - similarity, then mean over batch and spatial positions
        return torch.mean(1 - cos_sim)


def cosine_similarity_metric(pred, target, eps=1e-8):
    # pred, target: [B, C, H, W]
    B, C, H, W = pred.shape

    # reshape to [B, H*W, C]
    pred = pred.permute(0, 2, 3, 1).reshape(B, -1, C)
    target = target.permute(0, 2, 3, 1).reshape(B, -1, C)

    # cosine similarity along channels
    cos_sim = F.cosine_similarity(pred, target, dim=-1, eps=eps)  # [B, H*W]

    # average per image, then average across batch
    return cos_sim.mean()


def train(epoch):
    start = time.time()
    model.train()
    total_loss = 0.0
    running_cos_loss = 0.0
    total_cos_metric = 0.0
    for batch_index, (voxs, feats, _) in enumerate(train_loader):
        if torch.cuda.is_available():
            voxs = voxs.cuda()
            feats = feats.cuda()
        
        optimizer.zero_grad()
        outputs = model(voxs)

        cos_loss_value = cos_loss(outputs, feats)

        total_cos_metric += cosine_similarity_metric(outputs, feats).item()

        loss = cos_loss_value
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        running_cos_loss += cos_loss_value.item()

        scheduler.step()

        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1

        if batch_index % 100 == 0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.batch_size + len(voxs),
                total_samples=len(train_loader.dataset)
            ))

        # update training loss for each iteration
        writer.add_scalar('Train Loss/Cosine (iteration)', loss.item(), n_iter)
        writer.add_scalar('LR/iteration', optimizer.param_groups[0]['lr'], n_iter)

    writer.add_figure('preds',
                        plot_preds(outputs, feats),
                        global_step=epoch)

    for name, param in model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
    
    avg_loss = total_loss / len(train_loader)
    running_cos_loss = running_cos_loss / len(train_loader)
    average_cos_metric = total_cos_metric / len(train_loader)

    finish = time.time()
    writer.add_scalar('Train Loss/Cosine (epoch)', running_cos_loss, epoch)
    writer.add_scalar('Cos Simi/Train', average_cos_metric, epoch)
    writer.add_scalar('LR/epoch', optimizer.param_groups[0]['lr'], epoch)
    print('epoch {} training time consumed: {:.5f}s, train epoch loss = {:.5f}'.format(epoch, finish - start, avg_loss))


def eval_training(epoch=0):
    start = time.time()
    model.eval()

    valid_loss = 0.0
    valid_loss_cos = 0.0
 
    total_mse = 0.0
    total_cos_metric = 0.0
    with torch.no_grad():
        for (voxs, feats, _) in valid_loader:
            if torch.cuda.is_available():
                voxs = voxs.cuda()
                feats = feats.cuda()

            outputs = model(voxs)

            cos_loss_value = cos_loss(outputs, feats)

            total_cos_metric += cosine_similarity_metric(outputs, feats)

            loss = cos_loss_value
            
            valid_loss += loss.item()
            valid_loss_cos += cos_loss_value.item()
            
        finish = time.time()
        
        avg_loss_val = valid_loss / len(valid_loader)
        valid_loss_cos = valid_loss_cos / len(valid_loader)

        average_cos_metric = total_cos_metric / len(valid_loader)

        print('Evaluating Network.....')
        print('Test set: Epoch: {}, Average loss: {:.4f}, Cos Simi: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            avg_loss_val,
            average_cos_metric,
            finish - start
        ))

        writer.add_figure('preds_val',
                        plot_preds(outputs, feats),
                        global_step=epoch)

        writer.add_scalar('Test Loss/COS', valid_loss_cos, epoch)
        writer.add_scalar('Cos Simi/Test', average_cos_metric, epoch)
        return avg_loss_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for E-FPN for SSD backbone feature recovery")
    parser.add_argument("--vox_path", type=str, nargs="+",
                        help="One or more paths to directories containing training voxel .npz files")
    parser.add_argument("--feat_path", type=str, nargs="+",
                        help="One or more paths to directories containing training SSD backbone feature .npz files")
    parser.add_argument("--vox_path_valid", type=str, nargs="+",
                        help="One or more paths to directories containing validation voxel .npz files")
    parser.add_argument("--feat_path_valid", type=str, nargs="+",
                        help="One or more paths to directories containing validation SSD backbone feature .npz files")
    parser.add_argument("--out_path", type=str, default='./logs/',
                        help='Path to output logs')
    parser.add_argument("--vox_clip", type=float, nargs=2, metavar=('min', 'max'),
                        help='Min and max clipping value for event voxels')
    parser.add_argument("--feat_clip", type=float, nargs=2, metavar=('min', 'max'),
                        help='Min and max clipping value for SSD backbone feature')
    parser.add_argument("--dct_min", type=str, 
                        help='Path to dct_min.npy (generated via get_dct_min_max.py)')
    parser.add_argument("--dct_max", type=str, 
                        help='Path to dct_max.npy (generated via get_dct_min_max.py)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--init_lr", type=float, default=0.0001,
                        help="Initial learning rate")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help='GPU ID to use for training/validation')
    parser.add_argument("--n_workers", type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument("--id", type=str, default='1',
                        help='Set a unique ID for output logs directory')

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    print("\nloading dataset ...")
    train_data_load = Event_to_SSDFeature_Dataset(args.vox_path, args.feat_path, args.vox_clip, args.feat_clip, 'sigmoid')
    train_loader = torch.utils.data.DataLoader(train_data_load, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    print(f"Iteration per epoch: {len(train_loader)}")
    valid_data_load = Event_to_SSDFeature_Dataset(args.vox_path_valid, args.feat_path_valid, args.vox_clip, args.feat_clip, 'sigmoid')
    valid_loader = torch.utils.data.DataLoader(valid_data_load, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    print("Validation set samples: ", len(valid_loader))

    model = E_FPN(args.dct_min, args.dct_max)
    
    num_params = sum(param.numel() for param in model.parameters())
    print(f'Number of model parameters = {num_params} ≈ {num_params/1e6:.2f}M')

    # loss function
    cos_loss = CosineLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr, betas=(0.9, 0.999))

    total_steps = args.epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # output path
    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    date_time = datetime.now().strftime(DATE_FORMAT)
    args.out_path = f'{args.out_path}/{args.id}_{date_time}'
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    ckpt_path = args.out_path + '/ckpt/'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    if torch.cuda.is_available():
        model.cuda()

    writer = SummaryWriter(log_dir=ckpt_path)

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        loss = eval_training(epoch)

        print('saving weights.')

        if loss < best_loss:
            best_loss = loss

            torch.save(model.state_dict(), os.path.join(ckpt_path, 'best.pth'))
            with open(os.path.join(ckpt_path, 'details.txt'), 'w') as f:
                f.write(f'val_loss = {loss}, epoch = {epoch}, lr = {args.init_lr}, batch_size = {args.batch_size}')
            f.close()

        torch.save(model.state_dict(), os.path.join(ckpt_path, f'{epoch}_best.pth'))
        with open(os.path.join(ckpt_path, f'{epoch}_details.txt'), 'w') as f:
            f.write(f'val_loss = {loss}, epoch = {epoch}, lr = {args.init_lr}, batch_size = {args.batch_size}')
        f.close()

    writer.close()

