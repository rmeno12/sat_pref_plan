import torch
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
)


class ConvClassifier(Module):
    def __init__(self, n_classes, in_channels=3) -> None:
        super(ConvClassifier, self).__init__()
        self.conv = Sequential(
            Conv2d(in_channels, 16, kernel_size=3, padding=2, bias=False),
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(2),
            Conv2d(16, 32, kernel_size=3, padding=2, bias=False),
            BatchNorm2d(32),
            ReLU(inplace=True),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=3, padding=2, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
            Conv2d(64, 128, kernel_size=3, padding=2, bias=False),
            MaxPool2d(2),
            BatchNorm2d(128),
            ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.fc = Sequential(
            Linear(128 * 2 * 2, 256),
            ReLU(inplace=True),
            Linear(256, n_classes),
            Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
