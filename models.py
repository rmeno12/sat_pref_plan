import torch
from torch.nn import (
    Module,
    Upsample,
    Conv2d,
    MaxPool2d,
    ReLU,
    Linear,
    Sequential,
    ConvTranspose2d,
    Sigmoid,
    BatchNorm2d,
)

# taken mostly from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py


class DoubleConv(Module):
    def __init__(self, in_channels, out_channels, mid_channels=None) -> None:
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

    def forward(self, x):
        return self.conv(x)


class Down(Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Down, self).__init__()
        self.maxpool_conv = Sequential(
            MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(Module):
    def __init__(self, in_channels, out_channels, bilinear=True) -> None:
        super(Up, self).__init__()

        if bilinear:
            self.up = Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = torch.nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(Module):
    def __init__(self, in_channels, n_classes, bilinear=False) -> None:
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

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)

        return x


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
        print(x.shape)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x