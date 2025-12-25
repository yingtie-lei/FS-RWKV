import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from datasets import PairedImageDataset
from model.RWKV import RWKV
# from model.RWKV_wo_dwtnet import RWKV_WO_DWTNET
# from model.RWKV_wo_shift import RWKV_WO_SHIFT
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics import MeanSquaredError
import lpips  # pip install lpips
from PIL import Image
import numpy as np
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_A_dir', required=True)
    parser.add_argument('--test_B_dir', required=True)
    parser.add_argument('--test_pairing_txt', required=True)
    parser.add_argument('--checkpoint', required=True, help='path to model checkpoint')
    parser.add_argument('--save_dir', default='./results', help='where to save outputs')
    parser.add_argument('--img_size', type=int, default=256)
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ==================== Dataset ====================
    transform_ops = [
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ]
    test_dataset = PairedImageDataset(
        args.test_A_dir, args.test_B_dir, args.test_pairing_txt,
        transforms_=transform_ops, img_size=args.img_size
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ==================== Model ====================
    model = RWKV().cuda()
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    # ==================== Metrics ====================
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).cuda()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    mse_metric = MeanSquaredError().cuda()
    lpips_metric = lpips.LPIPS(net='vgg').cuda()  # LPIPS with VGG backbone

    psnr_list, ssim_list, rmse_list, lpips_list = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inp = batch['A'].cuda().clamp(0,1)
            # print(f"[DEBUG] input value: min={inp.min()}, max={inp.max()}")
            target = batch['B'].cuda().clamp(0,1)
            name = batch['B_name'][0]

            output = model(inp).clamp(0,1)
            psnr = psnr_metric(output, target)
            ssim = ssim_metric(output, target)
            rmse = torch.sqrt(mse_metric(output, target))
            lpips_val = lpips_metric(output, target)


            psnr_list.append(psnr.item())
            ssim_list.append(ssim.item())
            rmse_list.append(rmse.item())
            lpips_list.append(lpips_val.item())

            # Save fake_B
            # Save fake_B
            out_img = output.squeeze().detach().cpu().numpy()  # 变为[H,W]
            # print(f"[DEBUG] output before scaling: min={out_img.min()}, max={out_img.max()}, dtype={out_img.dtype}")

            out_img_uint8 = (out_img * 255.0).round().clip(0,255).astype(np.uint8)
            # print(f"[DEBUG] output after scaling to uint8: min={out_img_uint8.min()}, max={out_img_uint8.max()}, dtype={out_img_uint8.dtype}")

            im_pil = Image.fromarray(out_img_uint8, mode='L')
            im_pil.save(os.path.join(args.save_dir, name))

            # 可选: 保存后再加载看下min/max
            check_img = np.array(Image.open(os.path.join(args.save_dir, name)))
            # print(f"[DEBUG] reloaded image: min={check_img.min()}, max={check_img.max()}, dtype={check_img.dtype}")


    psnr_arr = np.array(psnr_list)
    ssim_arr = np.array(ssim_list)
    rmse_arr = np.array(rmse_list)
    lpips_arr = np.array(lpips_list)

    print(f"Average PSNR: {psnr_arr.mean():.4f} ± {psnr_arr.std(ddof=1):.4f}")
    print(f"Average SSIM: {ssim_arr.mean():.4f} ± {ssim_arr.std(ddof=1):.4f}")
    print(f"Average RMSE: {rmse_arr.mean():.4f} ± {rmse_arr.std(ddof=1):.4f}")
    print(f"Average LPIPS: {lpips_arr.mean():.4f} ± {lpips_arr.std(ddof=1):.4f}")


if __name__ == '__main__':
    main()
