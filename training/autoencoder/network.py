"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
unORANIC: Unsupervised orthogonalization of anatomy and image-characteristic features

@description:
Architecture of the autoencoder.

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
import torch.nn as nn
import pytorch_lightning as pl
import math


class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim: tuple, feature_dim: int, dropout: float, feature_maps_encoder=32, feature_maps_decoder=32):
        """
        Initialize the Autoencoder following: https://github.com/axkoenig/autoencoder/blob/master/autoencoder.py

        :param input_dim: Width, height and channel dimension of the input image.
        :param feature_dim: Dimension of the latent space.
        :param dropout: Dropout regularization rate.
        :param feature_maps_encoder: Size of feature maps in encoder.
        :param feature_maps_decoder: Size of feature maps in decoder.
        """
        # Run parent constructor
        super().__init__()

        # Store parameters
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.feature_maps_encoder = feature_maps_encoder
        self.feature_maps_decoder = feature_maps_decoder

        # Create the encoder
        self.encoder = nn.Sequential(
            # input 28 x 28 x C_in
            ResidualBlock(self.input_dim[2]),
            DownBlock(self.input_dim[2], self.feature_maps_encoder, kernel_size=4),
            # input 14 x 14 x (16 | 32 | 64)
            ResidualBlock(self.feature_maps_encoder),
            DownBlock(self.feature_maps_encoder, self.feature_maps_encoder * 2, kernel_size=4),
            nn.Dropout(self.dropout),
            # input 7 x 7 x (32 | 64 | 128)
            ResidualBlock(self.feature_maps_encoder * 2),
            DownBlock(self.feature_maps_encoder * 2, self.feature_maps_encoder * 4, kernel_size=5),
            nn.Dropout(self.dropout),
            # input 3 x 3 x (64 | 128 | 256)
            ResidualBlock(self.feature_maps_encoder * 4),
            DownBlock(self.feature_maps_encoder * 4, self.feature_dim, kernel_size=5)
            # input 1 x 1 x (128 | 256 | 512)
        )

        # Create the decoder
        self.decoder = nn.Sequential(
            # input 1 x 1 x (128 | 256 | 512)
            UpBlock(self.feature_dim, self.feature_maps_decoder * 4, kernel_size=5),
            ResidualBlock(self.feature_maps_decoder * 4),
            nn.Dropout(self.dropout),
            # input 3 x 3 x (64 |128 | 256)
            UpBlock(self.feature_maps_decoder * 4, self.feature_maps_decoder * 2, kernel_size=5),
            ResidualBlock(self.feature_maps_decoder * 2),
            nn.Dropout(self.dropout),
            # input 7 x 7 x (32 | 64 | 128)
            UpBlock(self.feature_maps_decoder * 2, self.feature_maps_decoder, kernel_size=4),
            ResidualBlock(self.feature_maps_decoder),
            nn.Dropout(self.dropout),
            # input 14 x 14 x (16 | 32 | 64)
            UpBlock(self.feature_maps_decoder, self.input_dim[2], kernel_size=4),
            ResidualBlock(self.input_dim[2]),
            # output 28 x 28 x C_in
        )

    def forward(self, x):
        """
        Forward Pass of the Autoencoder.

        :param x: Batch of input images.
        :return: Batch of reconstructed images.
        """

        x = self.encoder(x)
        x = self.decoder(x)

        return x


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
