import torch
from torch import nn as nn 
import torchvision.models as models

class CNNCustomModel(nn.Module):
    def __init__(self):
        super(CNNCustomModel, self).__init__()
        
        self._backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        for param in self._backbone.parameters():
            param.requires_grad = False

        for param in self._backbone.features[-2:].parameters():
            param.requires_grad = True

        num_features = self._backbone.classifier[1].in_features
        self._backbone.classifier[1] = nn.Sequential(
            nn.Linear(num_features, 256),   
            nn.ReLU(),
            nn.Dropout(0.),
            nn.Linear(256, 4)
        )

    def forward(self, x) -> torch.Tensor:
        return self._backbone(x)