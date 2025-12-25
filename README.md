# FS-RWKV: Leveraging Frequency Spatial-Aware RWKV for 3T-to-7T MRI Translation

[![Paper](https://img.shields.io/badge/Paper-BIBM%202025-blue)](https://ieeexplore.ieee.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-yellow.svg)](https://developer.nvidia.com/cuda-toolkit)

Official PyTorch implementation of **"FS-RWKV: Leveraging Frequency Spatial-Aware RWKV for 3T-to-7T MRI Translation"** accepted at **IEEE BIBM 2025**.

## 📄 Paper Information

**Authors:**

Yingtie Lei¹, Zimeng Li²*, Chi-Man Pun¹, Yupeng Liu³'⁴, and Xuhang Chen¹'⁵*

**Affiliations:**

¹Faculty of Science and Technology, University of Macau  
²School of Electronic and Communication Engineering, Shenzhen Polytechnic University  
³Department of Cardiology, Guangdong Provincial People's Hospital (Guangdong Academy of Medical Sciences), Southern Medical University  
⁴Guangdong Cardiovascular Institute, Guangdong Provincial People's Hospital, Guangdong Academy of Medical Sciences  
⁵School of Computer Science and Engineering, Huizhou University

**Corresponding Authors:** *Zimeng Li ([li_zimeng@szpu.edu.cn](mailto:li_zimeng@szpu.edu.cn)), Xuhang Chen ([xuhangc@hzu.edu.cn](mailto:xuhangc@hzu.edu.cn))

**Conference:** IEEE International Conference on Bioinformatics and Biomedicine (BIBM) 2025

## 🌟 Highlights

- **Novel RWKV-based Architecture**: First application of RWKV for medical image synthesis with linear complexity
- **Frequency-Spatial Omnidirectional Shift (FSO-Shift)**: Wavelet decomposition + omnidirectional token shifting for global context modeling
- **Structural Fidelity Enhancement Block (SFEB)**: Adaptive fusion of spatial and frequency domain features
- **State-of-the-Art Performance**: Outperforms CNN, Transformer, GAN, and RWKV baselines on both T1w and T2w modalities

## 📊 Results

| Dataset | Modality | PSNR (dB) ↑ | SSIM ↑ | RMSE ↓ |
|---------|----------|-------------|--------|--------|
| UNC     | T1w      | **21.0008** | **0.7258** | **0.0898** |
| UNC     | T2w      | **25.3058** | **0.7807** | **0.0565** |
| BNU     | T1w      | **23.3571** | **0.8388** | **0.0689** |
| BNU     | T2w      | **27.4937** | **0.8624** | **0.0431** |

## 🔧 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (with compatible GPU)
- GCC/G++ compiler for CUDA compilation

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/FS-RWKV.git
cd FS-RWKV

# Create conda environment
conda create -n fsrwkv python=3.8
conda activate fsrwkv

# Install PyTorch (adjust based on your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### CUDA Kernel Compilation

The model requires custom CUDA kernels for the WKV operation. These are automatically compiled on first run via PyTorch's JIT compiler. Ensure:

1. CUDA toolkit is properly installed
2. `nvcc` is accessible in your PATH
3. Update the CUDA source paths in `model/RWKV.py` (lines 18-20) to match your directory structure

```python
wkv_cuda = load(
    name="wkv",
    sources=[
        "/path/to/your/project/model/cuda/wkv_op.cpp",  # Update this
        "/path/to/your/project/model/cuda/wkv_cuda.cu",  # Update this
    ],
    verbose=True,
    ...
)
```

## 📁 Dataset Preparation

### Directory Structure

```
dataset/
├── train/
│   ├── 3T/          # 3T MRI images (input)
│   └── 7T/          # 7T MRI images (target)
├── val/
│   ├── 3T/
│   └── 7T/
└── test/
    ├── 3T/
    └── 7T/
```

### Pairing Files

Create text files that pair 3T and 7T images:

**train_pairs.txt:**
```
3T_image_001.png,7T_image_001.png
3T_image_002.png,7T_image_002.png
...
```

**val_pairs.txt** and **test_pairs.txt** follow the same format.

### Supported Datasets

- **UNC Dataset**: [Chen et al., Scientific Data 2023](https://www.nature.com/articles/s41597-023-02400-6)
- **BNU Dataset**: [Chu et al., Scientific Data 2025](https://www.nature.com/articles/s41597-025-04091-5)

## 🚀 Training

### Basic Training

```bash
python train.py \
    --train_A_dir ./dataset/train/3T \
    --train_B_dir ./dataset/train/7T \
    --val_A_dir ./dataset/val/3T \
    --val_B_dir ./dataset/val/7T \
    --train_pairing_txt ./dataset/train_pairs.txt \
    --val_pairing_txt ./dataset/val_pairs.txt \
    --checkpoint_dir ./checkpoints \
    --batch_size 4 \
    --epochs 200 \
    --lr 2e-4 \
    --img_size 256 \
    --seed 3407
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--train_A_dir` | Directory of 3T training images | Required |
| `--train_B_dir` | Directory of 7T training images | Required |
| `--val_A_dir` | Directory of 3T validation images | Required |
| `--val_B_dir` | Directory of 7T validation images | Required |
| `--train_pairing_txt` | Training pairing file | Required |
| `--val_pairing_txt` | Validation pairing file | Required |
| `--checkpoint_dir` | Directory to save checkpoints | `./checkpoints` |
| `--batch_size` | Training batch size | 4 |
| `--epochs` | Number of training epochs | 200 |
| `--lr` | Initial learning rate | 2e-4 |
| `--img_size` | Image resolution | 256 |
| `--seed` | Random seed for reproducibility | 3407 |

### Training Features

- **Data Augmentation**: Horizontal/vertical flip, rotation, transpose, mixup, cutmix
- **Optimizer**: AdamW with β₁=0.9, β₂=0.999
- **Scheduler**: Cosine annealing with T_max=200, η_min=1e-6
- **Loss Function**: Smooth L1 + SSIM (λ=0.4) + Edge Loss (λ=0.3)
- **Checkpointing**: Saves best models based on SSIM, PSNR, and validation loss

### Training Logs

Training logs are saved to `{checkpoint_dir}/train_log.txt`:
```
epoch,train_loss,val_loss,psnr,ssim
1,0.1234,0.0987,20.5432,0.7123
2,0.1100,0.0912,21.2345,0.7345
...
```

## 🧪 Testing

### Basic Testing

```bash
python test.py \
    --test_A_dir ./dataset/test/3T \
    --test_B_dir ./dataset/test/7T \
    --test_pairing_txt ./dataset/test_pairs.txt \
    --checkpoint ./checkpoints/best_ssim_199.pth \
    --save_dir ./results \
    --img_size 256
```

### Testing Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--test_A_dir` | Directory of 3T test images | Required |
| `--test_B_dir` | Directory of 7T test images | Required |
| `--test_pairing_txt` | Test pairing file | Required |
| `--checkpoint` | Path to model checkpoint | Required |
| `--save_dir` | Directory to save results | `./results` |
| `--img_size` | Image resolution | 256 |

### Evaluation Metrics

The testing script computes:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **RMSE** (Root Mean Squared Error)
- **LPIPS** (Learned Perceptual Image Patch Similarity)

Results are printed with mean ± standard deviation:
```
Average PSNR: 21.0008 ± 0.1234
Average SSIM: 0.7258 ± 0.0123
Average RMSE: 0.0898 ± 0.0045
Average LPIPS: 0.1234 ± 0.0056
```

## 🏗️ Model Architecture

### Key Components

1. **FS-RWKV Block**
   - Frequency Spatial Omnidirectional-Shift (FSO-Shift)
   - Spatial Mix with Bi-WKV attention
   - Channel Mix with Squared ReLU

2. **Structural Fidelity Enhancement Block (SFEB)**
   - Discrete Wavelet Transform (DWT)
   - Multi-scale processing for low/high frequencies
   - LSConv for low-frequency features
   - Depthwise separable convolutions for high-frequency details

3. **U-Net Architecture**
   - 4 encoder levels: [48, 96, 192, 384] channels
   - 4 decoder levels with skip connections
   - SFEB modules between encoder-decoder

### Model Variants

- `model/RWKV.py`: Full FS-RWKV model
- `model/RWKV_wo_dwtnet.py`: Without SFEB (ablation study)
- `model/RWKV_wo_shift.py`: Without FSO-Shift (ablation study)
- `model/RWKV_wo_both.py`: Without both components (ablation study)

## 📝 Citation

If you find this work useful for your research, please cite:

```bibtex
@inproceedings{lei2025fsrwkv,
  title={FS-RWKV: Leveraging Frequency Spatial-Aware RWKV for 3T-to-7T MRI Translation},
  author={Lei, Yingtie and Li, Zimeng and Pun, Chi-Man and Liu, Yupeng and Chen, Xuhang},
  booktitle={IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2025}
}
```

## 🙏 Acknowledgments

This work was supported by:
- Shenzhen Medical Research Fund (Grant No. A2503006)
- National Natural Science Foundation of China (Grant No. 62501412 and 82300277)
- Shenzhen Polytechnic University Research Fund (Grant No. 6025310023K)
- Medical Scientific Research Foundation of Guangdong Province (Grant No. B2025610 and B2023012)
- Science and Technology Development Fund, Macau SAR (Grant No. 0193/2023/RIA3 and 0079/2025/AFJ)
- University of Macau (Grant No. MYRG-GRG2024-00065-FST-UMDF)
- Guangdong Basic and Applied Basic Research Foundation (Grant No. 2024A1515140010)

## 📧 Contact

For questions and discussions, please contact:
- Zimeng Li: [li_zimeng@szpu.edu.cn](mailto:li_zimeng@szpu.edu.cn)
- Xuhang Chen: [xuhangc@hzu.edu.cn](mailto:xuhangc@hzu.edu.cn)

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

**Note**: This is a research project. The synthesized 7T MRI images are for research purposes only and should not be used for clinical diagnosis without proper validation.
