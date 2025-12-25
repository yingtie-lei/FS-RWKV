import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import albumentations as A

class PairedImageDataset(Dataset):
    """Dataset for paired images with explicit pairing file"""
    def __init__(self, A_dir, B_dir, pairing_txt, transforms_=None, 
                 img_size=256, use_mixup=False, use_albumentations=False,batchsize=None):
        self.A_dir = A_dir
        self.B_dir = B_dir
        self.transform = transforms.Compose(transforms_) if transforms_ else None
        self.img_size = img_size
        self.use_mixup = use_mixup
        self.use_albumentations = use_albumentations
        self.batchsize = batchsize
        
        # Read pairing file
        self.pairs = []
        with open(pairing_txt, 'r') as f:
            for line in f:
                a, b = line.strip().split(',')
                self.pairs.append((a, b))
        
        # Albumentations transform
        if self.use_albumentations:
            self.album_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=45, p=0.3),
                A.Transpose(p=0.3),
                A.Resize(height=self.img_size, width=self.img_size),
            ], additional_targets={'target': 'image'})
        else:
            self.album_transform = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a_name, b_name = self.pairs[idx]
        A_path = os.path.join(self.A_dir, a_name)
        B_path = os.path.join(self.B_dir, b_name)
        
        if self.use_albumentations and self.album_transform is not None:
            A_img = np.array(Image.open(A_path).convert("L"))
            B_img = np.array(Image.open(B_path).convert("L"))
            transformed = self.album_transform(image=A_img, target=B_img)
            A_img = Image.fromarray(transformed['image'])
            B_img = Image.fromarray(transformed['target'])
        else:
            A_img = Image.open(A_path).convert("L")
            B_img = Image.open(B_path).convert("L")
        
        if self.transform:
            item_A = self.transform(A_img)
            item_B = self.transform(B_img)
        else:
            item_A = F.to_tensor(A_img)
            item_B = F.to_tensor(B_img)
        
        # Apply mixup/cutmix
        if self.use_mixup and idx > 0 and idx % 3 == 0:
            if random.random() > 0.5:
                item_A, item_B = self.mixup(item_A, item_B, mode='mixup')
            else:
                item_A, item_B = self.mixup(item_A, item_B, mode='cutmix')
        
        return {'A': item_A, 'B': item_B, 'B_name': b_name}
    
    def mixup(self, inp_img, tar_img, mode='mixup'):
        mixup_index = random.randint(0, len(self) - 1)
        mixup_a_name, mixup_b_name = self.pairs[mixup_index]
        mixup_A_path = os.path.join(self.A_dir, mixup_a_name)
        mixup_B_path = os.path.join(self.B_dir, mixup_b_name)
        
        if self.use_albumentations and self.album_transform is not None:
            mixup_A_img = np.array(Image.open(mixup_A_path).convert("L"))
            mixup_B_img = np.array(Image.open(mixup_B_path).convert("L"))
            transformed = self.album_transform(image=mixup_A_img, target=mixup_B_img)
            mixup_A_img = Image.fromarray(transformed['image'])
            mixup_B_img = Image.fromarray(transformed['target'])
        else:
            mixup_A_img = Image.open(mixup_A_path).convert("L")
            mixup_B_img = Image.open(mixup_B_path).convert("L")
        
        if self.transform:
            mixup_inp_img = self.transform(mixup_A_img)
            mixup_tar_img = self.transform(mixup_B_img)
        else:
            mixup_inp_img = F.to_tensor(mixup_A_img)
            mixup_tar_img = F.to_tensor(mixup_B_img)

        alpha = 0.2
        lam = np.random.beta(alpha, alpha)

        if mode == 'mixup':
            inp_img = lam * inp_img + (1 - lam) * mixup_inp_img
            tar_img = lam * tar_img + (1 - lam) * mixup_tar_img
        elif mode == 'cutmix':
            _, img_h, img_w = inp_img.shape
            cx = np.random.uniform(0, img_w)
            cy = np.random.uniform(0, img_h)
            w = img_w * np.sqrt(1 - lam)
            h = img_h * np.sqrt(1 - lam)
            x0 = int(np.round(max(cx - w / 2, 0)))
            x1 = int(np.round(min(cx + w / 2, img_w)))
            y0 = int(np.round(max(cy - h / 2, 0)))
            y1 = int(np.round(min(cy + h / 2, img_h)))
            inp_img[:, y0:y1, x0:x1] = mixup_inp_img[:, y0:y1, x0:x1]
            tar_img[:, y0:y1, x0:x1] = mixup_tar_img[:, y0:y1, x0:x1]

        return inp_img, tar_img