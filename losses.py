import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

# ==========================
# 1. SSIM
# ==========================


# ==========================
# 2. Edge Loss (Sobel-based)
# ==========================
def edge_map(img):
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)

    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)

    grad_x = F.conv2d(img, sobel_x, padding=1)
    grad_y = F.conv2d(img, sobel_y, padding=1)

    edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    return edge

def edge_loss(pred, target):
    return F.l1_loss(edge_map(pred), edge_map(target))


# ==========================
# 3. Frequency SNR Loss
# ==========================
def frequency_snr_loss(pred, target, eps=1e-8):
    """
    Frequency-domain 1/SNR Loss with log scaling.
    Args:
        pred, target: [B, C, H, W] tensors.
        eps: stability term.
    Returns:
        scalar loss.
    """
    pred_fft = torch.fft.fft2(pred, dim=(-2, -1), norm="ortho")
    target_fft = torch.fft.fft2(target, dim=(-2, -1), norm="ortho")

    diff = target_fft - pred_fft

    signal_power = torch.sum(torch.abs(target_fft) ** 2, dim=(-2, -1))
    noise_power = torch.sum(torch.abs(diff) ** 2, dim=(-2, -1))

    # 1/SNR and log scaling
    inv_snr = (noise_power + eps) / (signal_power + eps)
    loss = torch.log(1 + inv_snr)
    return loss.mean()




# ==========================
# 4. Total Loss Wrapper
# ==========================
class TotalLoss(nn.Module):
    def __init__(self, lambda_ssim=1.0, lambda_edge=1.0):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge

        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, pred, target):
        l1 = F.smooth_l1_loss(pred, target)

        ssim_score = self.ssim_metric(pred, target)
        ssim = 1 - ssim_score

        edge = edge_loss(pred, target)

        total = l1 + self.lambda_ssim * ssim + self.lambda_edge * edge
        return total, {
            'smooth_L1': l1.item(),
            'L_ssim': ssim.item(),
            'L_edge': edge.item()
        }
