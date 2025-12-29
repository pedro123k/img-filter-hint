import torch
from typing import List
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

    def __len__(self):
        return self._lenght

    def __getitem__(self, idx):

        segment = idx // self._segment_size
        idx = idx % self._segment_size

        img = cv2.imread(self._data[idx].absolute().as_posix(), cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError(f"Image at path {self._data[idx]} could not be loaded.")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y = torch.tensor(segment).long()

        img = cv2.resize(img, (224, 224),
                 interpolation=cv2.INTER_AREA if img.shape[0]*img.shape[1] > 224*224 else cv2.INTER_LANCZOS4)
        
        if segment == 1:
            img = self.add_gaussian_noise(img)
        elif segment == 2:
            img = self.add_gaussian_blur(img)
        else:
            img = self.remove_lighting(img)
            
        img = CustomDataset._normalize(CustomDataset._img2ptensor(img),
                                           self._imagenet_mean, self._imagenet_std)
            
        return img, y
        
    @staticmethod    
    def _img2ptensor(img: np.ndarray) -> torch.Tensor:
        img = cv2.resize(img, (224, 224),
                 interpolation=cv2.INTER_AREA if img.shape[0]*img.shape[1] > 224*224 else cv2.INTER_LANCZOS4)

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

    def remove_lighting(self, img: np.ndarray) -> np.ndarray:
        img_f = img.astype(np.float32) / 255.0

        scale = np.random.uniform(0.4, 0.9)
        gamma = np.random.uniform(1.2, 2.2)
        out = (img_f * scale) ** gamma

        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return out