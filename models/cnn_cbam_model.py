# cnn_cbam_model.py

import torch
import torch.nn as nn


# ---------------------------------------------------
# CBAM Components
# ---------------------------------------------------

class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction_ratio=16):

        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)


        reduced = max(
            in_channels // reduction_ratio,
            1
        )


        self.mlp = nn.Sequential(

            nn.Conv2d(
                in_channels,
                reduced,
                kernel_size=1,
                bias=False
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                reduced,
                in_channels,
                kernel_size=1,
                bias=False
            )

        )


        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

        avg_out = self.mlp(
            self.avg_pool(x)
        )


        max_out = self.mlp(
            self.max_pool(x)
        )


        return self.sigmoid(
            avg_out + max_out
        )



class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):

        super(SpatialAttention, self).__init__()


        self.conv = nn.Conv2d(

            2,
            1,

            kernel_size=kernel_size,

            padding=kernel_size // 2,

            bias=False
        )


        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

        avg_out = torch.mean(

            x,

            dim=1,

            keepdim=True

        )


        max_out, _ = torch.max(

            x,

            dim=1,

            keepdim=True

        )


        out = torch.cat(

            [avg_out, max_out],

            dim=1

        )


        return self.sigmoid(
            self.conv(out)
        )



class CBAM(nn.Module):

    def __init__(
            self,
            in_channels,
            reduction_ratio=16,
            kernel_size=7
    ):

        super(CBAM, self).__init__()


        self.channel_attention = ChannelAttention(

            in_channels,

            reduction_ratio

        )


        self.spatial_attention = SpatialAttention(

            kernel_size

        )



    def forward(self, x):

        x = x * self.channel_attention(x)

        x = x * self.spatial_attention(x)

        return x



# ---------------------------------------------------
# CNN + CBAM Model
# ---------------------------------------------------

class CNN_CBAM(nn.Module):

    def __init__(
            self,
            in_channels=3,
            num_classes=10,
            img_size=32
    ):

        super(CNN_CBAM, self).__init__()



        # -------------------------------
        # Block 1
        # -------------------------------

        self.block1 = nn.Sequential(

            nn.Conv2d(

                in_channels,

                32,

                kernel_size=3,

                padding=1,

                bias=False

            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(

                kernel_size=2,

                stride=2

            )

        )


        self.cbam1 = CBAM(32)



        # -------------------------------
        # Block 2
        # -------------------------------

        self.block2 = nn.Sequential(

            nn.Conv2d(

                32,

                64,

                kernel_size=3,

                padding=1,

                bias=False

            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(

                kernel_size=2,

                stride=2

            )

        )


        self.cbam2 = CBAM(64)



        # -------------------------------
        # Classifier
        # -------------------------------

        flattened_size = (

            64 *

            (img_size // 4) *

            (img_size // 4)

        )


        self.flatten = nn.Flatten()


        self.fc1 = nn.Linear(

            flattened_size,

            128

        )


        self.relu_fc = nn.ReLU(inplace=True)


        self.dropout = nn.Dropout(

            0.5

        )


        self.fc2 = nn.Linear(

            128,

            num_classes

        )



    def forward(self, x):

        x = self.block1(x)

        x = self.cbam1(x)


        x = self.block2(x)

        x = self.cbam2(x)


        x = self.flatten(x)


        x = self.fc1(x)

        x = self.relu_fc(x)

        x = self.dropout(x)

        x = self.fc2(x)


        return x



# ---------------------------------------------------
# Sanity Test
# ---------------------------------------------------

if __name__ == "__main__":


    # CIFAR-10

    model_cifar = CNN_CBAM(

        in_channels=3,

        num_classes=10,

        img_size=32

    )


    x = torch.randn(

        4,

        3,

        32,

        32

    )


    print(
        "CIFAR-10 output shape:",
        model_cifar(x).shape
    )



    # MNIST

    model_mnist = CNN_CBAM(

        in_channels=1,

        num_classes=10,

        img_size=28

    )


    x = torch.randn(

        4,

        1,

        28,

        28

    )


    print(
        "MNIST output shape:",
        model_mnist(x).shape
    )



    num_params = sum(

        p.numel()

        for p in model_cifar.parameters()

    )


    print(
        f"Total parameters: {num_params:,}"
    )