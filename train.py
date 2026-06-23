import os
import torch
import random
import dataset
import numpy as np
from para import Parameter
from torch.utils.data import DataLoader
import Model
import loss
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import r2_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_psnr_ssim(pred, target):
    """pred, target: [B, T, 1, H, W]"""
    B, T, _, H, W = pred.shape
    psnr_list, ssim_list = [], []
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    for b in range(B):
        for t in range(T):
            pred_frame = pred_np[b, t, 0]
            target_frame = target_np[b, t, 0]
            psnr_list.append(peak_signal_noise_ratio(target_frame, pred_frame, data_range=1.0))
            ssim_list.append(structural_similarity(target_frame, pred_frame, data_range=1.0))
    return np.mean(psnr_list), np.mean(ssim_list)

def compute_r2_all(pred_list, target_list):
    """
    pred_list, target_list: list of tensors [B, T, C, H, W] or [T, C, H, W]
    """
    all_pred, all_target = [], []
    for pred, target in zip(pred_list, target_list):
        if pred.ndim == 4:
            pred = pred.unsqueeze(0)  # [1, T, C, H, W]
        if target.ndim == 4:
            target = target.unsqueeze(0)

        B, T, C, H, W = pred.shape
        pred = pred.reshape(B*T, C, H, W)
        target = target.reshape(B*T, C, H, W)

        all_pred.append(pred)
        all_target.append(target)
    all_pred = torch.cat(all_pred, dim=0)    # [total_frames, C, H, W]
    all_target = torch.cat(all_target, dim=0)
    C = all_pred.shape[1]
    r2_scores = []
    for c in range(C):
        pred_flat = all_pred[:, c, :, :].flatten().cpu().detach().numpy()
        target_flat = all_target[:, c, :, :].flatten().cpu().detach().numpy()
        r2_scores.append(r2_score(target_flat, pred_flat))
    return r2_scores

def train(model, criterion, optimizer, train_loader, val_loader, num_epochs=20, save_path='checkpoints/best_model.pth'):

    best_val_psnr = -1
    best_r2 = -100
    log_path = ""

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = {'total': 0, 'restoration': 0, 'estimation': 0, 're_degraded': 0, 'physical': 0}
        train_psnr_list, train_ssim_list = [], []
        cn2_pred_list, target_cn2_list = [], []

        pbar = tqdm(train_loader, desc=f"[Train Epoch {epoch}]")
        for degraded, target_img, target_cn2 in pbar:
            degraded = degraded.to(device)
            target_img = target_img.to(device)
            target_cn2 = target_cn2.to(device)

            optimizer.zero_grad()
            restored, cn2_pred = model(degraded)
            losses = criterion(restored, cn2_pred, target_img, target_cn2, degraded)
            losses['total'].backward()
            optimizer.step()

            for k in train_losses.keys():
                train_losses[k] += losses.get(k, torch.tensor(0)).item() * degraded.size(0)

            psnr, ssim = compute_psnr_ssim(restored, target_img)
            train_psnr_list.append(psnr)
            train_ssim_list.append(ssim)
            cn2_pred_list.append(cn2_pred)
            target_cn2_list.append(target_cn2)

        n_train = len(train_loader.dataset)
        for k in train_losses:
            train_losses[k] /= n_train
        train_psnr = np.mean(train_psnr_list)
        train_ssim = np.mean(train_ssim_list)
        train_r2 = compute_r2_all(cn2_pred_list, target_cn2_list)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}, LR = {current_lr:.6f}")

        model.eval()
        val_losses = {'total': 0, 'restoration': 0, 'estimation': 0, 're_degraded': 0, 'physical': 0}
        val_psnr_list, val_ssim_list = [], []
        cn2_pred_list, target_cn2_list = [], []
        with torch.no_grad():
            for degraded, target_img, target_cn2 in val_loader:
                degraded = degraded.to(device)
                target_img = target_img.to(device)
                target_cn2 = target_cn2.to(device)

                restored, cn2_pred = model(degraded)
                losses = criterion(restored, cn2_pred, target_img, target_cn2, degraded)

                for k in val_losses.keys():
                    val_losses[k] += losses.get(k, torch.tensor(0)).item() * degraded.size(0)

                psnr, ssim = compute_psnr_ssim(restored, target_img)
                val_psnr_list.append(psnr)
                val_ssim_list.append(ssim)
                cn2_pred_list.append(cn2_pred)
                target_cn2_list.append(target_cn2)

        n_val = len(val_loader.dataset)
        for k in val_losses:
            val_losses[k] /= n_val
        val_psnr = np.mean(val_psnr_list)
        val_ssim = np.mean(val_ssim_list)
        val_r2 = compute_r2_all(cn2_pred_list, target_cn2_list)

        print(f"\nEpoch [{epoch}/{num_epochs}]")
        print(f"  Train Loss: total={train_losses['total']:.4f}\n")
        print(f"  Train PSNR={train_psnr:.2f} SSIM={train_ssim:.4f} R2={np.round(train_r2, 3)}\n")
        print(f"  Val   Loss: total={val_losses['total']:.4f}\n")
        print(f"  Val   PSNR={val_psnr:.2f} SSIM={val_ssim:.4f} R2={np.round(val_r2, 3)}\n")

        with open(log_path, 'a') as f:
            f.write(f"\nEpoch [{epoch}/{num_epochs}]\n")
            f.write(f"  Train Loss: total={train_losses['total']:.4f}\n")
            f.write(f"  Train PSNR={train_psnr:.2f} SSIM={train_ssim:.4f} R2={np.round(train_r2, 3)}\n")
            f.write(f"  Val   Loss: total={val_losses['total']:.4f}\n")
            f.write(f"  Val   PSNR={val_psnr:.2f} SSIM={val_ssim:.4f} R2={np.round(val_r2, 3)}\n")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), model_path + '/epoch_'+str(epoch)+'.pth')
        if val_psnr > best_val_psnr and np.mean(val_r2) > best_r2:
            best_r2 = np.mean(val_r2)
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), model_path + '/best'+'.pth')
            print(f"Best model saved at epoch {epoch}")


if __name__ == '__main__':
    para = Parameter().args
    torch.manual_seed(para.seed)
    torch.cuda.manual_seed(para.seed)
    random.seed(para.seed)
    np.random.seed(para.seed)

    train_dataset = dataset.TrainDataset(para, 'train/', crop_size=128)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=para.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )
    print('length_of_train: ', len(train_dataset))

    valid_dataset = dataset.ValidDataset(para, 'test/')
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )
    print('length_of_valid: ', len(valid_dataset))

    model = Model.MutualPLENetwork(
        num_levels=1,
        num_shared_experts=2,
        num_task_experts=2,
        expert_dim=64
    ).cuda()

    criterion = loss.MutualLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
    model_path = ''
    os.makedirs(model_path, exist_ok=True)
    train(model, criterion, optimizer, train_loader, valid_loader, num_epochs=100, save_path = model_path)
