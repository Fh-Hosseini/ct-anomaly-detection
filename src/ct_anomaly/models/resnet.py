import torch
import torch.nn as nn


class Basic3DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet3D18(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1_block1 = BasicBlock3D(64, 64, stride=1)
        self.layer1_block2 = BasicBlock3D(64, 64, stride=1)

        self.layer2_block1 = BasicBlock3D(64, 128, stride=2)
        self.layer2_block2 = BasicBlock3D(128, 128, stride=1)

        self.layer3_block1 = BasicBlock3D(128, 256, stride=2)
        self.layer3_block2 = BasicBlock3D(256, 256, stride=1)

        self.layer4_block1 = BasicBlock3D(256, 512, stride=2)
        self.layer4_block2 = BasicBlock3D(512, 512, stride=1)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1_block1(x)
        x = self.layer1_block2(x)

        x = self.layer2_block1(x)
        x = self.layer2_block2(x)

        x = self.layer3_block1(x)
        x = self.layer3_block2(x)

        x = self.layer4_block1(x)
        x = self.layer4_block2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x