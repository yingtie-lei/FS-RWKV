import sys
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets import PairedImageDataset
from losses import TotalLoss
from model.RWKV_wo_dwtnet import RWKV_WO_DWTNET
from model.RWKV import RWKV
from model.RWKV_wo_shift import RWKV_WO_SHIFT
from model.RWKV_wo_both import RWKV_WO_BOTH
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


def set_seed(seed=3407):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_A_dir', required=True)
    parser.add_argument('--train_B_dir', required=True)
    parser.add_argument('--val_A_dir', required=True)
    parser.add_argument('--val_B_dir', required=True)
    parser.add_argument('--train_pairing_txt', required=True)
    parser.add_argument('--val_pairing_txt', required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=3407, help='Random seed for reproducibility')
    return parser.parse_args()


def main():
    args = get_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ==================== Dataset ====================
    transform_ops = [
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ]

    train_dataset = PairedImageDataset(
        args.train_A_dir, args.train_B_dir, args.train_pairing_txt,
        transforms_=transform_ops, img_size=args.img_size, use_mixup=True, use_albumentations=True
    )
    val_dataset = PairedImageDataset(
        args.val_A_dir, args.val_B_dir, args.val_pairing_txt,
        transforms_=transform_ops, img_size=args.img_size
    )

    # Set worker_init_fn for DataLoader reproducibility
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
        worker_init_fn=worker_init_fn, generator=torch.Generator().manual_seed(args.seed)
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # ==================== Model and Loss ====================
    model = RWKV().cuda()
    criterion = TotalLoss(lambda_ssim=0.4, lambda_edge=0.3).cuda()

    metric_psnr = PeakSignalNoiseRatio().cuda()
    metric_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()

    # ==================== Optimizer and Scheduler ====================
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ==================== Logs ====================
    best_ssim = 0.0
    best_psnr = 0.0
    best_val_loss = float('inf')
    train_log = {'train_loss': [], 'val_loss': [], 'psnr': [], 'ssim': []}

    log_file = os.path.join(args.checkpoint_dir, "train_log.txt")

    # ==================== Training Loop ====================
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        loss_log = {'smooth_L1': 0, 'L_ssim': 0, 'L_edge': 0}

        for batch in tqdm(train_loader, desc=f"[Epoch {epoch+1}]"):
            inp = batch['A'].cuda()
            target = batch['B'].cuda()

            output = model(inp)
            loss, log_dict = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            for k in loss_log:
                loss_log[k] += log_dict[k]

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | " +
              " | ".join([f"{k.upper()}: {loss_log[k]/len(train_loader):.4f}" for k in loss_log]))

        # ==================== Validation ====================
        print(f'========Validating==========')
        model.eval()
        val_loss = 0
        val_log = {'smooth_L1': 0, 'L_ssim': 0, 'L_edge': 0}
        psnr_total = 0
        ssim_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader):
                inp = batch['A'].cuda()
                target = batch['B'].cuda()
                output = model(inp)

                loss, log_dict = criterion(output, target)
                val_loss += loss.item()
                for k in val_log:
                    val_log[k] += log_dict[k]

                psnr_total += metric_psnr(output, target).item()
                ssim_total += metric_ssim(output, target).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_psnr = psnr_total / len(val_loader)
        avg_ssim = ssim_total / len(val_loader)

        print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | " +
              " | ".join([f"{k.upper()}: {val_log[k]/len(val_loader):.4f}" for k in val_log]) +
              f" | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}")

        # 记录日志
        train_log['train_loss'].append(avg_train_loss)
        train_log['val_loss'].append(avg_val_loss)
        train_log['psnr'].append(avg_psnr)
        train_log['ssim'].append(avg_ssim)

        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{avg_psnr:.4f},{avg_ssim:.4f}\n")

        scheduler.step()

        # Save best model based on SSIM
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f'best_ssim_{epoch}.pth'))
            print(f"Saved best SSIM model! Best SSIM: {best_ssim:.4f}")
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(),os.path.join(args.checkpoint_dir, f'best_psnr_{epoch}.pth'))
            print(f"Saved best PSNR model! Best PSNR: {best_psnr:.4f}")
        # Save best model based on Val Loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f'best_loss_{epoch}_model.pth'))
            print(f"Saved best Loss model! Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()