import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import torch.utils.checkpoint as cp
from einops import rearrange
from pytorch_wavelets import DWTForward, DWTInverse
from timm.models.layers import DropPath, trunc_normal_

# Use this line for experiments
T_MAX = 512 * 512

# Only use this for performance testing
# T_MAX = 1024 * 1024


wkv_cuda = load(
    name="wkv",
    sources=[
        "/home/lzmlab/lyt/bibm2025/RWKV-BIBM/model/cuda/wkv_op.cpp",
        "/home/lzmlab/lyt/bibm2025/RWKV-BIBM/model/cuda/wkv_cuda.cu",
    ],
    verbose=True,
    extra_cuda_cflags=[
        "-res-usage",
        "--maxrregcount 60",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        f"-DTmax={T_MAX}",
    ],
)


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


class ContextGuidedTokenShift(nn.Module):
    """Context-guided Token Shift Module following Algorithm 1"""
    
    def __init__(self, dim, learnable_weight=True, init_weight=0.5):
        super().__init__()
        self.dim = dim
        if learnable_weight:
            self.weight = nn.Parameter(torch.tensor(init_weight))
        else:
            self.register_buffer('weight', torch.tensor(init_weight))
        
        # Define offset dictionary D
        self.offset_dict = [
            (0, 1),   # right +1
            (0, -1),  # left -1
            (1, 0),   # down +1
            (-1, 0),  # up -1
            (0, 2),   # right +2
            (0, -2),  # left -2
            (2, 0),   # down +2
            (-2, 0),  # up -2
            (1, 1),   # diagonal down-right +1
            (-1, -1), # diagonal up-left -1
            (1, -1),  # anti-diagonal down-left +1
            (-1, 1),  # anti-diagonal up-right -1
            (2, 2),   # diagonal down-right +2
            (-2, -2), # diagonal up-left -2
            (2, -2),  # anti-diagonal down-left +2
            (-2, 2),  # anti-diagonal up-right -2
        ]
        
        # Pre-compute weights and channel allocation
        self.p_sum = 0.0
        self.weights = []
        for offset in self.offset_dict:
            d_p = abs(offset[0]) + abs(offset[1])  # Manhattan distance
            w_p = 1.0 / d_p if d_p > 0 else 1.0
            self.weights.append(w_p)
            self.p_sum += w_p
        
        # Calculate channel expansion factor
        self.k = dim / self.p_sum

    def forward(self, input, patch_resolution=None):
        B, N, C = input.shape
        
        if patch_resolution is None:
            H = W = int(N ** 0.5)
        else:
            H, W = patch_resolution
            
        # Reshape input to spatial format
        input = input.transpose(1, 2).reshape(B, C, H, W)
        B, _, H, W = input.shape
        output = torch.zeros_like(input)
        
        # Apply context-guided token shift
        c = 0
        for i, (offset, w_p) in enumerate(zip(self.offset_dict, self.weights)):
            # Calculate channel width for this offset
            channel_width = int(self.k * w_p)
            if c + channel_width > C:
                channel_width = C - c
            if channel_width <= 0:
                continue
                
            dy, dx = offset
            
            # Apply shift based on offset direction
            if dy == 0 and dx == 1:  # right +1
                output[:, c:c+channel_width, :, 1:W] = input[:, c:c+channel_width, :, 0:W-1]
            elif dy == 0 and dx == -1:  # left -1
                output[:, c:c+channel_width, :, 0:W-1] = input[:, c:c+channel_width, :, 1:W]
            elif dy == 1 and dx == 0:  # down +1
                output[:, c:c+channel_width, 1:H, :] = input[:, c:c+channel_width, 0:H-1, :]
            elif dy == -1 and dx == 0:  # up -1
                output[:, c:c+channel_width, 0:H-1, :] = input[:, c:c+channel_width, 1:H, :]
            elif dy == 0 and dx == 2:  # right +2
                output[:, c:c+channel_width, :, 2:W] = input[:, c:c+channel_width, :, 0:W-2]
            elif dy == 0 and dx == -2:  # left -2
                output[:, c:c+channel_width, :, 0:W-2] = input[:, c:c+channel_width, :, 2:W]
            elif dy == 2 and dx == 0:  # down +2
                output[:, c:c+channel_width, 2:H, :] = input[:, c:c+channel_width, 0:H-2, :]
            elif dy == -2 and dx == 0:  # up -2
                output[:, c:c+channel_width, 0:H-2, :] = input[:, c:c+channel_width, 2:H, :]
            elif dy == 1 and dx == 1:  # diagonal down-right +1
                output[:, c:c+channel_width, 1:H, 1:W] = input[:, c:c+channel_width, 0:H-1, 0:W-1]
            elif dy == -1 and dx == -1:  # diagonal up-left -1
                output[:, c:c+channel_width, 0:H-1, 0:W-1] = input[:, c:c+channel_width, 1:H, 1:W]
            elif dy == 1 and dx == -1:  # anti-diagonal down-left +1
                output[:, c:c+channel_width, 1:H, 0:W-1] = input[:, c:c+channel_width, 0:H-1, 1:W]
            elif dy == -1 and dx == 1:  # anti-diagonal up-right -1
                output[:, c:c+channel_width, 0:H-1, 1:W] = input[:, c:c+channel_width, 1:H, 0:W-1]
            elif dy == 2 and dx == 2:  # diagonal down-right +2
                output[:, c:c+channel_width, 2:H, 2:W] = input[:, c:c+channel_width, 0:H-2, 0:W-2]
            elif dy == -2 and dx == -2:  # diagonal up-left -2
                output[:, c:c+channel_width, 0:H-2, 0:W-2] = input[:, c:c+channel_width, 2:H, 2:W]
            elif dy == 2 and dx == -2:  # anti-diagonal down-left +2
                output[:, c:c+channel_width, 2:H, 0:W-2] = input[:, c:c+channel_width, 0:H-2, 2:W]
            elif dy == -2 and dx == 2:  # anti-diagonal up-right -2
                output[:, c:c+channel_width, 0:H-2, 2:W] = input[:, c:c+channel_width, 2:H, 0:W-2]
            
            c += channel_width
            if c >= C:
                break
        
        # Reshape back to sequence format
        output = output.reshape(B, C, N).transpose(1, 2)
        
        # Return weighted combination: CTS(x) = ω · o + (1 - ω) · x
        return self.weight * output + (1 - self.weight) * input.reshape(B, C, N).transpose(1, 2)
    

class DWTContextGuidedTokenShift(nn.Module):
    """
    DWT-based Context-guided Token Shift Module using pytorch_wavelets
    Only apply shift to LL (low-frequency) component and original input, then combine them
    """
    
    def __init__(self, dim, wavelet='haar', learnable_weight=True, init_weight=0.5):
        super().__init__()
        self.dim = dim
        self.wavelet = wavelet
        
        if learnable_weight:
            self.weight = nn.Parameter(torch.tensor(init_weight))
        else:
            self.register_buffer('weight', torch.tensor(init_weight))
        
        self.dwt = DWTForward(J=1, mode='zero', wave=wavelet)
        self.idwt = DWTInverse(mode='zero', wave=wavelet)
        
        # Create shift modules for LL component and input
        self.ll_shift = ContextGuidedTokenShift(dim, learnable_weight=True, init_weight=0.5)
        self.input_shift = ContextGuidedTokenShift(dim, learnable_weight=True, init_weight=0.5)
        
        # Use the complete 16-directional offset patterns for both LL and input
        self.offset_dict = [
            (0, 1),   # right +1
            (0, -1),  # left -1
            (1, 0),   # down +1
            (-1, 0),  # up -1
            (0, 2),   # right +2
            (0, -2),  # left -2
            (2, 0),   # down +2
            (-2, 0),  # up -2
            (1, 1),   # diagonal down-right +1
            (-1, -1), # diagonal up-left -1
            (1, -1),  # anti-diagonal down-left +1
            (-1, 1),  # anti-diagonal up-right -1
            (2, 2),   # diagonal down-right +2
            (-2, -2), # diagonal up-left -2
            (2, -2),  # anti-diagonal down-left +2
            (-2, 2),  # anti-diagonal up-right -2
        ]
        
        # Set the same offset patterns for both LL and input shift modules
        self.ll_shift.offset_dict = self.offset_dict
        self.input_shift.offset_dict = self.offset_dict
        
        # Recompute weights and allocations for both modules
        self._recompute_weights()
    
    def _recompute_weights(self):
        """重新计算LL和input shift模块的权重和通道分配"""
        for shift_module in [self.ll_shift, self.input_shift]:
            shift_module.weights = []
            shift_module.p_sum = 0.0
            for offset in shift_module.offset_dict:
                d_p = abs(offset[0]) + abs(offset[1])
                w_p = 1.0 / d_p if d_p > 0 else 1.0
                shift_module.weights.append(w_p)
                shift_module.p_sum += w_p
            shift_module.k = self.dim / shift_module.p_sum
    
    def apply_shift_to_component(self, component, shift_module, patch_resolution):
        """
        对单个频率分量应用shift操作
        
        Args:
            component: 频率分量张量 (B, C, H, W)
            shift_module: 对应的shift模块
            patch_resolution: 空间分辨率 (H, W)
        
        Returns:
            shifted频率分量
        """
        B, C, H, W = component.shape
        N = H * W
        
        # 重塑为序列格式 (B, N, C)
        component_seq = component.reshape(B, C, N).transpose(1, 2)
        
        # 应用shift
        shifted_seq = shift_module(component_seq, patch_resolution)
        
        # 重塑回空间格式 (B, C, H, W)
        shifted_component = shifted_seq.transpose(1, 2).reshape(B, C, H, W)
        
        return shifted_component
    
    def forward(self, input, patch_resolution=None):
        """
        Forward pass using pytorch_wavelets for DWT/IDWT
        Only shift LL component and original input, then combine them
        
        Args:
            input: (B, N, C) where N = H*W
            patch_resolution: (H, W)
        
        Returns:
            shifted output with preserved gradients
        """
        B, N, C = input.shape
        
        if patch_resolution is None:
            H = W = int(N ** 0.5)
        else:
            H, W = patch_resolution
        
        # 1. Apply shift to original input
        shifted_input = self.input_shift(input, patch_resolution)
        
        # 2. Reshape to spatial format for DWT
        input_spatial = input.transpose(1, 2).reshape(B, C, H, W)
        
        # 3. GPU上的可微分DWT - 保持梯度流
        LL, Yh = self.dwt(input_spatial)  # LL: (B,C,H//2,W//2), Yh: [(B,C,3,H//2,W//2)]
        
        # 4. 提取高频分量（保持不变，不进行shift）
        LH = Yh[0][:, :, 0, :, :]  # 水平细节 (B, C, H//2, W//2)
        HL = Yh[0][:, :, 1, :, :]  # 垂直细节 (B, C, H//2, W//2)
        HH = Yh[0][:, :, 2, :, :]  # 对角细节 (B, C, H//2, W//2)
        
        # 5. 获取LL频率分量的空间分辨率
        _, _, H_sub, W_sub = LL.shape
        freq_resolution = (H_sub, W_sub)
        
        # 6. 只对LL分量应用shift（高频分量保持不变）
        LL_shifted = self.apply_shift_to_component(LL, self.ll_shift, freq_resolution)
        
        # 7. 重组高频分量为pytorch_wavelets格式（高频分量未改变）
        Yh_shifted = [torch.stack([LH, HL, HH], dim=2)]
        
        # 8. GPU上的可微分IDWT - 保持梯度流
        dwt_output_spatial = self.idwt((LL_shifted, Yh_shifted))
        
        # 9. 确保输出尺寸与输入一致（处理小波变换的边界效应）
        if dwt_output_spatial.shape[2:] != (H, W):
            dwt_output_spatial = F.interpolate(dwt_output_spatial, size=(H, W), mode='bilinear', align_corners=False)
        
        # 10. Reshape DWT output back to sequence format
        dwt_output = dwt_output_spatial.reshape(B, C, N).transpose(1, 2)
        
        # 11. 加权融合shifted input和DWT output
        # 最终输出 = weight * (shifted_input + dwt_output) + (1 - weight) * original_input
        combined_shifted = shifted_input + dwt_output
        final_output = self.weight * combined_shifted + (1 - self.weight) * input
        
        return final_output
    
# class DWTContextGuidedTokenShift(nn.Module):
#     """
#     DWT-based Context-guided Token Shift Module using pytorch_wavelets
#     Only apply shift to LL (low-frequency) component and original input, then combine them
#     """
    
#     def __init__(self, dim, wavelet='haar'):
#         super().__init__()
#         self.dim = dim
#         self.wavelet = wavelet
        
        
#         self.dwt = DWTForward(J=1, mode='zero', wave=wavelet)
#         self.idwt = DWTInverse(mode='zero', wave=wavelet)
        
#         # Create shift modules for LL component and input
#         self.ll_shift = ContextGuidedTokenShift(dim)
#         self.input_shift = ContextGuidedTokenShift(dim)
        
#         # Use the complete 16-directional offset patterns for both LL and input
#         self.offset_dict = [
#             (0, 1),   # right +1
#             (0, -1),  # left -1
#             (1, 0),   # down +1
#             (-1, 0),  # up -1
#             (0, 2),   # right +2
#             (0, -2),  # left -2
#             (2, 0),   # down +2
#             (-2, 0),  # up -2
#             (1, 1),   # diagonal down-right +1
#             (-1, -1), # diagonal up-left -1
#             (1, -1),  # anti-diagonal down-left +1
#             (-1, 1),  # anti-diagonal up-right -1
#             (2, 2),   # diagonal down-right +2
#             (-2, -2), # diagonal up-left -2
#             (2, -2),  # anti-diagonal down-left +2
#             (-2, 2),  # anti-diagonal up-right -2
#         ]
        
#         # Set the same offset patterns for both LL and input shift modules
#         self.ll_shift.offset_dict = self.offset_dict
#         self.input_shift.offset_dict = self.offset_dict
        
#         # Recompute weights and allocations for both modules
#         self._recompute_weights()
    
#     def _recompute_weights(self):
#         """重新计算LL和input shift模块的权重和通道分配"""
#         for shift_module in [self.ll_shift, self.input_shift]:
#             shift_module.weights = []
#             shift_module.p_sum = 0.0
#             for offset in shift_module.offset_dict:
#                 d_p = abs(offset[0]) + abs(offset[1])
#                 w_p = 1.0 / d_p if d_p > 0 else 1.0
#                 shift_module.weights.append(w_p)
#                 shift_module.p_sum += w_p
#             shift_module.k = self.dim / shift_module.p_sum
    
#     def apply_shift_to_component(self, component, shift_module, patch_resolution):
#         """
#         对单个频率分量应用shift操作
        
#         Args:
#             component: 频率分量张量 (B, C, H, W)
#             shift_module: 对应的shift模块
#             patch_resolution: 空间分辨率 (H, W)
        
#         Returns:
#             shifted频率分量
#         """
#         B, C, H, W = component.shape
#         N = H * W
        
#         # 重塑为序列格式 (B, N, C)
#         component_seq = component.reshape(B, C, N).transpose(1, 2)
        
#         # 应用shift
#         shifted_seq = shift_module(component_seq, patch_resolution)
        
#         # 重塑回空间格式 (B, C, H, W)
#         shifted_component = shifted_seq.transpose(1, 2).reshape(B, C, H, W)
        
#         return shifted_component
    
#     def forward(self, input, patch_resolution=None):
#         """
#         Forward pass using pytorch_wavelets for DWT/IDWT
#         Only shift LL component and original input, then combine them
        
#         Args:
#             input: (B, N, C) where N = H*W
#             patch_resolution: (H, W)
        
#         Returns:
#             shifted output with preserved gradients
#         """
#         B, N, C = input.shape
        
#         if patch_resolution is None:
#             H = W = int(N ** 0.5)
#         else:
#             H, W = patch_resolution
        
#         # 1. Apply shift to original input
#         shifted_input = self.input_shift(input, patch_resolution)
        
#         # 2. Reshape to spatial format for DWT
#         input_spatial = input.transpose(1, 2).reshape(B, C, H, W)
        
#         # 3. GPU上的可微分DWT - 保持梯度流
#         LL, Yh = self.dwt(input_spatial)  # LL: (B,C,H//2,W//2), Yh: [(B,C,3,H//2,W//2)]
        
#         # 4. 提取高频分量（保持不变，不进行shift）
#         LH = Yh[0][:, :, 0, :, :]  # 水平细节 (B, C, H//2, W//2)
#         HL = Yh[0][:, :, 1, :, :]  # 垂直细节 (B, C, H//2, W//2)
#         HH = Yh[0][:, :, 2, :, :]  # 对角细节 (B, C, H//2, W//2)
        
#         # 5. 获取LL频率分量的空间分辨率
#         _, _, H_sub, W_sub = LL.shape
#         freq_resolution = (H_sub, W_sub)
        
#         # 6. 只对LL分量应用shift（高频分量保持不变）
#         LL_shifted = self.apply_shift_to_component(LL, self.ll_shift, freq_resolution)
        
#         # 7. 重组高频分量为pytorch_wavelets格式（高频分量未改变）
#         Yh_shifted = [torch.stack([LH, HL, HH], dim=2)]
        
#         # 8. GPU上的可微分IDWT - 保持梯度流
#         dwt_output_spatial = self.idwt((LL_shifted, Yh_shifted))
        
#         # 9. 确保输出尺寸与输入一致（处理小波变换的边界效应）
#         if dwt_output_spatial.shape[2:] != (H, W):
#             dwt_output_spatial = F.interpolate(dwt_output_spatial, size=(H, W), mode='bilinear', align_corners=False)
        
#         # 10. Reshape DWT output back to sequence format
#         dwt_output = dwt_output_spatial.reshape(B, C, N).transpose(1, 2)
        
#         # 11. 加权融合shifted input和DWT output
#         # 最终输出 = weight * (shifted_input + dwt_output) + (1 - weight) * original_input
#         combined_shifted = shifted_input + dwt_output
        
#         return combined_shifted


def DWTTokenShift(input, shift_pixel=1, patch_resolution=None):
    """
    DWT Token Shift function to replace the original context_shift
    使用pytorch_wavelets，保持完整梯度流
    """
    # Create a DWT shift module with appropriate dimension
    B, N, C = input.shape
    dwt_shift = DWTContextGuidedTokenShift(dim=C, learnable_weight=True, init_weight=0.5)
    dwt_shift = dwt_shift.to(input.device)
    
    return dwt_shift(input, patch_resolution)

# def DWTTokenShift(input, shift_pixel=1, patch_resolution=None):
#     """
#     DWT Token Shift function to replace the original context_shift
#     使用pytorch_wavelets，保持完整梯度流
#     """
#     # Create a DWT shift module with appropriate dimension
#     B, N, C = input.shape
#     dwt_shift = DWTContextGuidedTokenShift(dim=C)
#     dwt_shift = dwt_shift.to(input.device)
    
#     return dwt_shift(input, patch_resolution)


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='DWTTokenShift',
                 shift_pixel=1, init_mode='fancy', 
                 key_norm=True, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        
        # Initialize DWT shift module
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.with_cp = with_cp

    def _init_weights(self, init_mode):
        if init_mode=='fancy':
            with torch.no_grad(): # fancy init
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1)) # 0 to 1
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                
                # fancy time_decay
                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd-1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                # fancy time_first
                zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)
                
                # fancy time_mix
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        elif init_mode=='local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode=='global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        if self.shift_pixel > 0:
            xx = self.shift_func(x, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            B, T, C = x.size()
            self.device = x.device

            sr, k, v = self.jit_func(x, patch_resolution)
            x = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
            if self.key_norm is not None:
                x = self.key_norm(x)
            x = sr * x
            x = self.output(x)
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='DWTTokenShift',
                 shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 key_norm=True, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.with_cp = with_cp
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        
        # Initialize DWT shift module
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad(): # fancy init of time_mix
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            if self.shift_pixel > 0:
                xx = self.shift_func(x, patch_resolution)
                xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
                xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
            else:
                xk = x
                xr = x

            k = self.key(xk)
            k = torch.square(torch.relu(k))
            if self.key_norm is not None:
                k = self.key_norm(k)
            kv = self.value(k)
            x = torch.sigmoid(self.receptance(xr)) * kv
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class CRBv1(nn.Module):
    def __init__(
        self,
        n_embd,
        n_layer,
        layer_id,
        shift_mode="DWTTokenShift",
        shift_pixel=1,
        hidden_rate=4,
        init_mode="fancy",
        key_norm=False,
    ):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(
            n_embd, n_layer, layer_id, 
            shift_mode=shift_mode, shift_pixel=shift_pixel, 
            init_mode=init_mode, key_norm=key_norm
        )

        self.ffn = VRWKV_ChannelMix(
            n_embd, n_layer, layer_id, 
            shift_mode=shift_mode, shift_pixel=shift_pixel, hidden_rate=hidden_rate,
            init_mode=init_mode, key_norm=key_norm
        )

        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        resolution = (h, w)

        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.gamma1 * self.att(self.ln1(x), resolution)

        x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        return x




##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)

class LayerNorm2d(nn.Module):
    """对(B, C, H, W)格式输入的LayerNorm"""
    def __init__(self, num_channels, eps=1e-5):
        super(LayerNorm2d, self).__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # (B, C, H, W) -> (B, H, W, C) -> LN -> (B, C, H, W)
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ===========================
# LSConv 相关模块 
# ===========================
class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)

        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w

class SKA(nn.Module):
    def forward(self, x, w):
        """
        x: [B, C, H, W]
        w: [B, G, K*K, H, W]
        G = C // groups
        """
        B, C, H, W = x.shape
        G = w.shape[1]
        K = int(w.shape[2] ** 0.5)
        pad = K // 2
        out = torch.zeros_like(x)

        x_unfold = torch.nn.functional.unfold(x, kernel_size=K, padding=pad)  # [B, C*K*K, H*W]
        x_unfold = x_unfold.view(B, G, C // G, K * K, H, W)  # [B, G, C//G, K*K, H, W]

        w = w.view(B, G, 1, K * K, H, W)  # [B, G, 1, K*K, H, W]
        out_group = (x_unfold * w).sum(dim=3)  # [B, G, C//G, H, W]
        out = out_group.view(B, C, H, W)
        return out

class LSConv(nn.Module):
    def __init__(self, dim):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x))) + x


# ===========================
# 简化的普通卷积 (修复GroupNorm)
# ===========================
class SimpleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        # 确保GroupNorm的groups不超过channels
        num_groups = min(8, out_channels)
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


# ===========================
# 深度可分离卷积 + LayerNorm
# ===========================
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = LayerNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.norm(self.pointwise(self.depthwise(x))))


# ===========================
# 低频特征处理模块 (普通卷积 + LSConv)
# ===========================
class LowFrequencyProcessor(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        # 普通多尺度卷积
        self.conv_3x3 = SimpleConv(in_channels, base_channels, 3, padding=1)
        self.conv_5x5 = SimpleConv(in_channels, base_channels, 5, padding=2)
        self.conv_7x7 = SimpleConv(in_channels, base_channels, 7, padding=3)
        
        # LSConv处理
        self.lsconv_3x3 = LSConv(base_channels)
        self.lsconv_5x5 = LSConv(base_channels)
        self.lsconv_7x7 = LSConv(base_channels)
        
        # 输出融合
        self.out_conv = nn.Conv2d(base_channels * 3, base_channels, 1, bias=False)
    
    def forward(self, x):
        f1 = self.lsconv_3x3(self.conv_3x3(x))
        f2 = self.lsconv_5x5(self.conv_5x5(x))
        f3 = self.lsconv_7x7(self.conv_7x7(x))
        return self.out_conv(torch.cat([f1, f2, f3], dim=1))


# ===========================
# 高频特征处理模块 (普通卷积 + DepthwiseSeparableConv)
# ===========================
class HighFrequencyProcessor(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        # 普通多尺度卷积
        self.conv_3x3 = SimpleConv(in_channels, base_channels, 3, padding=1)
        self.conv_5x5 = SimpleConv(in_channels, base_channels, 5, padding=2)
        self.conv_7x7 = SimpleConv(in_channels, base_channels, 7, padding=3)
        
        # DepthwiseSeparableConv处理
        self.dsconv_3x3 = DepthwiseSeparableConv(base_channels, base_channels, 3, padding=1)
        self.dsconv_5x5 = DepthwiseSeparableConv(base_channels, base_channels, 3, padding=1)
        self.dsconv_7x7 = DepthwiseSeparableConv(base_channels, base_channels, 3, padding=1)
        
        # 输出融合
        self.out_conv = nn.Conv2d(base_channels * 3, base_channels, 1, bias=False)
    
    def forward(self, x):
        f1 = self.dsconv_3x3(self.conv_3x3(x))
        f2 = self.dsconv_5x5(self.conv_5x5(x))
        f3 = self.dsconv_7x7(self.conv_7x7(x))
        return self.out_conv(torch.cat([f1, f2, f3], dim=1))


# ===========================
# DWT 特征融合 (修复通道数问题)
# ===========================
class DWTFeatureFusion(nn.Module):
    """DWT特征融合"""
    def __init__(self, fuse_channels, delta_channels):
        super().__init__()
        # 确保hidden_channels至少为1
        hidden_channels = max(fuse_channels // 2, 1)

        self.weight_net = nn.Sequential(
            nn.Conv2d(fuse_channels, hidden_channels, 1, bias=False),
            LayerNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, 1, bias=False),
            nn.Sigmoid()
        )

        # 通道对齐模块
        if fuse_channels != delta_channels:
            self.align_conv = nn.Conv2d(fuse_channels, delta_channels, 1, bias=False)
            self.need_align = True
        else:
            self.need_align = False
    
    def forward(self, dwt_fuse, spatial_delta):
        # 计算权重
        gap_fuse = F.adaptive_avg_pool2d(dwt_fuse, 1)
        weights = self.weight_net(gap_fuse)
        w1, w2 = weights[:, 0:1, :, :], weights[:, 1:2, :, :]

        # 通道对齐
        if self.need_align:
            dwt_fuse = self.align_conv(dwt_fuse)

        return w1 * dwt_fuse + w2 * spatial_delta


# ===========================
# 精简的多尺度DWT网络
# ===========================
class SFEB(nn.Module):
    def __init__(self, in_channels=1, base_channels=48, out_channels=1, wavelet='haar', device=None):
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # DWT 变换
        self.dwt = DWTForward(J=1, wave=wavelet, mode='symmetric')
        self.idwt = DWTInverse(wave=wavelet, mode='symmetric')
        
        # 低频处理器 (普通卷积 + LSConv)
        self.low_freq_processor = LowFrequencyProcessor(in_channels, base_channels)
        
        # 高频处理器 (普通卷积 + DepthwiseSeparableConv)
        self.high_freq_processor = HighFrequencyProcessor(in_channels * 3, base_channels)
        
        # 频域特征后处理 (将处理后的特征转回原通道数以便IDWT)
        self.low_back_conv = nn.Conv2d(base_channels, in_channels, 1, bias=False)
        self.high_back_conv = nn.Conv2d(base_channels, in_channels * 3, 1, bias=False)
        
        # 空间域特征 (LSConv)
        self.spatial_conv = SimpleConv(in_channels, base_channels, 3, padding=1)
        self.spatial_lsconv = LSConv(base_channels)
        
        # 最终特征融合 (IDWT重构后的特征 + 空间域特征)
        self.final_fusion = DWTFeatureFusion(in_channels, base_channels)
        
        # 简化输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.Tanh()
        )
        
        self.to(self.device)
    
    def forward(self, x):
        x = x.to(self.device)
        B, C, H, W = x.shape
        
        # 空间域特征 (LSConv)
        f_spatial = self.spatial_lsconv(self.spatial_conv(x))
        
        # DWT变换
        coeffs_low, coeffs_high = self.dwt(x)
        
        # 低频处理 (普通卷积 + LSConv)
        low_features = self.low_freq_processor(coeffs_low)
        # 转回原通道数
        low_features_back = self.low_back_conv(low_features)
        
        # 高频处理 (普通卷积 + DepthwiseSeparableConv)
        coeffs_high_cat = coeffs_high[0].view(B, C * 3, coeffs_high[0].shape[-2], coeffs_high[0].shape[-1])
        high_features = self.high_freq_processor(coeffs_high_cat)
        # 转回原通道数并重构为高频系数格式
        high_features_back = self.high_back_conv(high_features)
        high_coeffs_reconstructed = [high_features_back.view(B, C, 3, high_features_back.shape[-2], high_features_back.shape[-1])]
        
        # IDWT重构回空间域
        reconstructed = self.idwt((low_features_back, high_coeffs_reconstructed))
        
        # 最终融合 (IDWT重构特征 + 空间域特征)
        fused_features = self.final_fusion(reconstructed, f_spatial)
        
        # 输出
        return self.output_conv(fused_features)


class RWKV(nn.Module):
    def __init__(
        self,
        inp_channels=1,
        out_channels=1,
        dim=48,
        num_blocks=[4,6,6,8],
        num_refinement_blocks=4,
    ):

        super(RWKV, self).__init__()

        self.patch_embed = nn.Conv2d(
            inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.encoder_level1 = nn.Sequential(
            *[
                CRBv1(n_embd=dim, n_layer=num_blocks[0], layer_id=i)
                for i in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(
            *[
                CRBv1(n_embd=int(dim * 2**1), n_layer=num_blocks[1], layer_id=i)
                for i in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2**1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(
            *[
                CRBv1(n_embd=int(dim * 2**2), n_layer=num_blocks[2], layer_id=i)
                for i in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(int(dim * 2**2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(
            *[
                CRBv1(n_embd=int(dim * 2**3), n_layer=num_blocks[3], layer_id=i)
                for i in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(int(dim * 2**3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=True  # 上采样通道 + 原编码器通道
        )
        self.decoder_level3 = nn.Sequential(
            *[
                CRBv1(n_embd=int(dim * 2**2), n_layer=num_blocks[2], layer_id=i)
                for i in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=True  # 上采样通道 + 原编码器通道
        )
        self.decoder_level2 = nn.Sequential(
            *[
                CRBv1(n_embd=int(dim * 2**1), n_layer=num_blocks[1], layer_id=i)
                for i in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(
            int(dim * 2**1)
        )  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(
            *[
                CRBv1(n_embd=int(dim * 2**1), n_layer=num_blocks[0], layer_id=i)  # 上采样通道 + 原编码器通道
                for i in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                CRBv1(n_embd=int(dim * 2**1), n_layer=num_refinement_blocks, layer_id=i)  # 保持通道数一致
                for i in range(num_refinement_blocks)
            ]
        )

        ###########################

        self.output = nn.Conv2d(
            int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=True  # 保持通道数一致
        )

        # 为不同尺度创建SFEB (确保输出通道数正确)
        self.enhance_level1 = SFEB(in_channels=dim, base_channels=dim, out_channels=dim)
        self.enhance_level2 = SFEB(in_channels=int(dim * 2**1), base_channels=int(dim * 2**1), out_channels=int(dim * 2**1))
        self.enhance_level3 = SFEB(in_channels=int(dim * 2**2), base_channels=int(dim * 2**2), out_channels=int(dim * 2**2))
        
    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)  # [1, 48, 256, 256]
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # [1, 48, 256, 256]

        inp_enc_level2 = self.down1_2(out_enc_level1)  # [1, 96, 128, 128]
        out_enc_level2 = self.encoder_level2(inp_enc_level2)  # [1, 96, 128, 128]

        inp_enc_level3 = self.down2_3(out_enc_level2)  # [1, 192, 64, 64]
        out_enc_level3 = self.encoder_level3(inp_enc_level3)  # [1, 192, 64, 64]

        inp_enc_level4 = self.down3_4(out_enc_level3)  # [1, 384, 32, 32]
        latent = self.latent(inp_enc_level4)  # [1, 384, 32, 32]

        # Level 3 解码 + DWT增强
        inp_dec_level3 = self.up4_3(latent)  # 上采样: [1, 192, 64, 64]
        enhanced_enc_level3 = self.enhance_level3(out_enc_level3)  # DWT增强: [1, 192, 64, 64]
        inp_dec_level3 = torch.cat([inp_dec_level3, enhanced_enc_level3], 1)  # concat: [1, 384, 64, 64]
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)  # 降维到: [1, 192, 64, 64]
        out_dec_level3 = self.decoder_level3(inp_dec_level3)  # [1, 192, 64, 64]

        # Level 2 解码 + DWT增强
        inp_dec_level2 = self.up3_2(out_dec_level3)  # 上采样: [1, 96, 128, 128]
        enhanced_enc_level2 = self.enhance_level2(out_enc_level2)  # DWT增强: [1, 96, 128, 128]
        inp_dec_level2 = torch.cat([inp_dec_level2, enhanced_enc_level2], 1)  # concat: [1, 192, 128, 128]
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)  # 降维到: [1, 96, 128, 128]
        out_dec_level2 = self.decoder_level2(inp_dec_level2)  # [1, 96, 128, 128]

        # Level 1 解码 + DWT增强
        inp_dec_level1 = self.up2_1(out_dec_level2)  # 上采样: [1, 48, 256, 256]
        enhanced_enc_level1 = self.enhance_level1(out_enc_level1)  # DWT增强: [1, 48, 256, 256]
        inp_dec_level1 = torch.cat([inp_dec_level1, enhanced_enc_level1], 1)  # concat: [1, 96, 256, 256]
        out_dec_level1 = self.decoder_level1(inp_dec_level1)  # 处理: [1, 96, 256, 256]

        out_dec_level1 = self.refinement(out_dec_level1)  # 保持: [1, 96, 256, 256]

        out_dec_level1 = self.output(out_dec_level1) + inp_img  # 输出: [1, out_channels, 256, 256]

        return out_dec_level1
    
if __name__ == '__main__':
    device = 'cuda'
    x = torch.randn(1,1,256,256).to(device)
    model = RWKV().to(device)
    y = model(x)
    print(y.shape)