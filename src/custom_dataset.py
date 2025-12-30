import torch
from typing import List, Tuple
from torch.utils.data import Dataset
from pathlib import WindowsPath
import cv2
import numpy as np
import random

class CustomDataset(Dataset):
    def __init__(self, data: List[WindowsPath]):
        self._data = data
        self._lenght = 4 * len(data)
        self._segment_size = len(data)
        self._imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).float()
        self._imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).float()

    def __len__(self) -> int:
        return self._lenght

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:

        segment = idx // self._segment_size
        idx = idx % self._segment_size

        img = cv2.imread(self._data[idx].absolute().as_posix(), cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError(f"Image at path {self._data[idx]} could not be loaded.")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y = torch.tensor(segment).long()

        img = cv2.resize(img, (224, 224),
                 interpolation=cv2.INTER_AREA if img.shape[0]*img.shape[1] > 224*224 else cv2.INTER_LANCZOS4)
        
        if segment == 0:
            pass
        elif segment == 1:
            img = self.add_gaussian_noise(img)
        elif segment == 2:
            img = self.add_gaussian_blur(img)
        elif segment == 3:
            img = self.remove_lighting(img)
        else:
            raise ValueError("Invalid Segment: ", segment)
            
        img = CustomDataset._normalize(CustomDataset._img2ptensor(img),
                                           self._imagenet_mean, self._imagenet_std)
            
        return img, y
        
    @staticmethod    
    def _img2ptensor(img: np.ndarray) -> torch.Tensor:
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img

    @staticmethod
    def _normalize(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (features - mean) / std            

    def add_gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        mean = 0
        std = np.random.uniform(8.0, 35.0)
        
        gauss = np.random.normal(mean, std, img.shape).astype('float32')
        noisy_img = cv2.add(img.astype(np.float32), gauss)
        noisy_img = np.clip(noisy_img, 0, 255.0).astype(np.uint8)

        return noisy_img

    def add_gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        kernels = ((3,3), (5,5), (7,7), (9,9))
        blurred_img = cv2.GaussianBlur(img, random.choice(kernels), 0)
        return blurred_img
    
    @staticmethod
    def _srgb2linear(x: np.ndarray) -> np.ndarray:
        a = 0.055
        return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

    @staticmethod
    def _linear2srgb(x: np.ndarray) -> np.ndarray:
        a = 0.055
        return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * (x ** (1 / 2.4)) - a)

    @staticmethod
    def _reinhard_tonemap(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + x)

    def remove_lighting(self, img: np.ndarray) -> np.ndarray:
        img_f = img.astype(np.float32) / 255.0
        lin = self._srgb2linear(img_f)

        ev = np.random.uniform(1.5, 2.5)
        lin = lin * (2.0 ** (-ev))

        noise_std = np.random.uniform(0.0002, 0.001)
        noise = np.random.normal(0.0, noise_std, lin.shape).astype(np.float32)
        noise -= noise.mean(axis=(0, 1), keepdims=True)

        lin += noise

        h, w = lin.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        rx = (xx - cx) / cx
        ry = (yy - cy) / cy
        r2 = rx * rx + ry * ry

        vig_strength = np.random.uniform(0.2, 0.9)
        vignette = np.exp(-vig_strength * r2)

        ang = np.deg2rad(np.random.uniform(0.0, 360.0))
        g = np.cos(ang) * rx + np.sin(ang) * ry   
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)  
        grad_strength = np.random.uniform(0.05, 0.35)
        gradient = (1.0 - grad_strength) + grad_strength * g

        M = vignette * gradient
        lin *= M[..., None]

        out = self._linear2srgb(np.clip(lin, 0.0, 1.0))
        return np.rint(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)