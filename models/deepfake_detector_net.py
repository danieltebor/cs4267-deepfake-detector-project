import torch
import torch.nn as nn

class DeepfakeDetectorNet(nn.Module):
    def __init__(self):
        super(DeepfakeDetectorNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=4, stride=4),
            LayerNorm2d(48),
            InvertedBottleneckBlock(48),

            LayerNorm2d(48),
            nn.Conv2d(48, 96, kernel_size=2, stride=2),
            InvertedBottleneckBlock(96),

            LayerNorm2d(96),
            nn.Conv2d(96, 192, kernel_size=2, stride=2),
            InvertedBottleneckBlock(192),
            InvertedBottleneckBlock(192),

            LayerNorm2d(192),
            nn.Conv2d(192, 384, kernel_size=2, stride=2),
            InvertedBottleneckBlock(384),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            LayerNorm2d(384),
            nn.Flatten(),
            nn.Linear(384 * 7 * 7, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)

class InvertedBottleneckBlock(nn.Module):
    def __init__(self, in_channels):
        super(InvertedBottleneckBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels),
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1),
        )

    def forward(self, x):
        residual = x
        x = self.block(x)
        return x + residual

class LayerNorm2d(nn.Module):
    def __init__(self, num_features):
        super(LayerNorm2d, self).__init__()
        self.norm = nn.LayerNorm(num_features, eps=1e-6, elementwise_affine=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)