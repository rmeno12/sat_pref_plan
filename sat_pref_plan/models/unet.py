from typing import Union

import torch
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
    Upsample,
)

"""taken mostly from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
"""


class DoubleConv(Module):
    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Union[int, None] = None
    ) -> None:
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = Sequential(
            Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, bias=False),
            BatchNorm2d(mid_channels),
            ReLU(inplace=True),
            Conv2d(mid_channels, out_channels, kernel_size=3),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Down, self).__init__()
        self.maxpool_conv = Sequential(
            MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(Module):
    def __init__(
        self, in_channels: int, out_channels: int, bilinear: bool = True
    ) -> None:
        super(Up, self).__init__()

        self.up: Union[Upsample, ConvTranspose2d]
        if bilinear:
            self.up = Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = torch.nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(Module):
    def __init__(
        self, in_channels: int, n_classes: int, bilinear: bool = False
    ) -> None:
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(in_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128 // factor)

        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.out = Conv2d(16, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)

        return x
