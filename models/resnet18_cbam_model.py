# resnet18_cbam_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------
# CBAM Components (same as before)
# ---------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced = max(in_channels // reduction_ratio, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# ---------------------------------------------------
# BasicBlock with CBAM (inserted before residual addition,
# following the placement used in the original CBAM paper)
# ---------------------------------------------------
class BasicBlockCBAM(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockCBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.cbam = CBAM(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)          # apply CBAM before adding shortcut
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ---------------------------------------------------
# ResNet18 + CBAM
# Uses a CIFAR-style stem (3x3 conv, stride 1, no initial maxpool)
# so it works well on both 32x32 (CIFAR-10) and 28x28 (MNIST) images.
# ---------------------------------------------------
class ResNet18_CBAM(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(ResNet18_CBAM, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * BasicBlockCBAM.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlockCBAM(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlockCBAM.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# Quick sanity test
if __name__ == "__main__":
    # CIFAR-10 style input
    model_cifar = ResNet18_CBAM(in_channels=3, num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    print("CIFAR-10 output shape:", model_cifar(x).shape)

    # MNIST style input
    model_mnist = ResNet18_CBAM(in_channels=1, num_classes=10)
    x = torch.randn(4, 1, 28, 28)
    print("MNIST output shape:", model_mnist(x).shape)

    num_params = sum(p.numel() for p in model_cifar.parameters())
    print(f"Total parameters: {num_params:,}")