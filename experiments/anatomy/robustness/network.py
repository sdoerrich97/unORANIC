"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
unORANIC: Unsupervised orthogonalization of anatomy and image-characteristic features

@description:
Architecture of the robustness experiment for the anatomy classification.

@author: Sebastian Doerrich
@copyright: Copyright (c) 2022, Chair of Explainable Machine Learning (xAI), Otto-Friedrich University of Bamberg
@credits: [Sebastian Doerrich]
@license: CC BY-SA
@version: 1.0
@python: Python 3
@maintainer: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
@status: Development
"""

# Import packages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet18, ResNet18_Weights


class Classifier(pl.LightningModule):
    def __init__(self, task, input_dim: tuple, feature_dim: int, nr_classes: int, dropout: float):
        """
        Initialize the Autoencoder following: https://github.com/axkoenig/autoencoder/blob/master/autoencoder.py

        :param task: What classification task.
        :param input_dim: Width, height and channel dimension of the input image.
        :param feature_dim: Dimension of the latent space.
        """
        # Run parent constructor
        super().__init__()

        # Store parameters
        self.task = task
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.nr_classes = nr_classes
        self.dropout = dropout

        # Initialize the encoder
        self.encoder = Encoder(self.input_dim, self.feature_dim, self.dropout)

        # Initialize the classification decoder
        self.classifier = Classification(self.task, self.feature_dim, self.nr_classes, self.dropout)

    def forward(self, x):
        """
        Forward Pass of the Autoencoder.

        :param x: Batch of input images.
        :return: Batch of reconstructed images.
        """

        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)

        return x


class Encoder(pl.LightningModule):
    def __init__(self, input_dim: tuple, feature_dim: int, dropout: float, feature_maps_encoder=32):
        """
        Initialize the Encoder following: https://github.com/axkoenig/autoencoder/blob/master/autoencoder.py

        :param input_dim: Width, height and channel dimension of the input image.
        :param feature_dim: Dimension of the output features.
        :param dropout: What percentage of nodes should be dropped in the dropout layers.
        :param feature_maps_encoder: Size of feature maps in encoder.
        """

        # Call parent constructor
        super().__init__()

        # Store the parameters
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.feature_maps_encoder = feature_maps_encoder

        self.model = nn.Sequential(
            # input 28 x 28 x C_in
            ResidualBlock(self.input_dim[2]),
            DownBlock(self.input_dim[2], self.feature_maps_encoder, kernel_size=4),
            # input 14 x 14 x (32 | 64)
            ResidualBlock(self.feature_maps_encoder),
            DownBlock(self.feature_maps_encoder, self.feature_maps_encoder * 2, kernel_size=4),
            nn.Dropout(self.dropout),
            # input 7 x 7 x (64 | 128)
            ResidualBlock(self.feature_maps_encoder * 2),
            DownBlock(self.feature_maps_encoder * 2, self.feature_maps_encoder * 4, kernel_size=5),
            nn.Dropout(self.dropout),
            # input 3 x 3 x (128 | 256)
            ResidualBlock(self.feature_maps_encoder * 4),
            DownBlock(self.feature_maps_encoder * 4, self.feature_dim, kernel_size=5)
            # input 1 x 1 x (256 | 512)
        )

    def forward(self, X):
        """
        Encode the input X.
        """

        return self.model(X)


class Classification(pl.LightningModule):
    def __init__(self, task, feature_dim: int, nr_classes: int, dropout: float):
        """
        Initialize the Encoder following: https://github.com/axkoenig/autoencoder/blob/master/autoencoder.py

        :param task: What classification task.
        :param feature_dim: Dimension of the output features.
        :param nr_classes: Number of classes.
        :param dropout: What percentage of nodes should be dropped in the dropout layers.
        """

        # Call parent constructor
        super().__init__()

        # Store the parameters
        self.feature_dim = feature_dim
        self.nr_classes = nr_classes
        self.dropout = dropout

        if task == 'multi-label, binary-class':
            activation = nn.Sigmoid()

        else:
            activation = nn.Softmax(dim=1)

        self.model = nn.Sequential(
            nn.Linear(self.feature_dim, self.nr_classes),
            activation
        )

    def forward(self, X):
        """
        Encode the input X.
        """

        return self.model(X)


class DownBlock(nn.Module):
    """Convolution block to decrease image dimension by simultaneously increasing channel dimension"""

    def __init__(self, in_channels, out_channels, kernel_size=4):
        super().__init__()

        # input W x H x C_in
        self.downblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(True)
        )
        # output W//2 x H//2 x C_out

    def forward(self, x):
        return self.downblock(x)


class UpBlock(nn.Module):
    """Convolution block to decrease image dimension by simultaneously increasing channel dimension"""

    def __init__(self, in_channels, out_channels, kernel_size=4):
        super().__init__()

        # input W x H x C_in
        self.upblock = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(True)
        )
        # output 2*W x 2*H x C_out

    def forward(self, x):
        return self.upblock(x)


class ResidualBlock(nn.Module):
    """Residual block containign two times (Convolution => [BN] => ReLU)"""

    def __init__(self, in_channels):
        super().__init__()

        # input W x H x C_in
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
        )
        # output W x H x C_in

    def forward(self, x):
        return self.resblock(x) + x


def ResNet18(in_channels, num_classes):
    """Adapted from kuangliu/pytorch-cifar ."""
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)


def ResNet50(in_channels, num_classes):
    """Adapted from kuangliu/pytorch-cifar ."""
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out