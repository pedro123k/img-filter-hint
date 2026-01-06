import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Tuple

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).float()
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).float()

def img_bytes2norm_tensor(img_raw: bytes) -> torch.tensor:
    ndarray_buff = np.frombuffer(img_raw, dtype=np.uint8)
    img = cv2.imdecode(ndarray_buff, cv2.IMREAD_COLOR)

    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    interp = cv2.INTER_AREA if (img.shape[0] * img.shape[1] > 224 * 224) else cv2.INTER_LANCZOS4
    img = cv2.resize(img, (224, 224), interpolation=interp)

    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    t = (t - IMAGENET_MEAN) / IMAGENET_STD

    return t

def predict(model: nn.Module, img_raw: bytes) -> List[Tuple[int, float]]:
    model.eval()

    x = img_bytes2norm_tensor(img_raw)
    device = next(model.parameters()).device
    x = x.to(device)
    x = torch.unsqueeze(x, 0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0] 

    return probs.tolist()
