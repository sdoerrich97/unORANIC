"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
unORANIC: Unsupervised orthogonalization of anatomy and image-characteristic features

@description:
Architecture of the reconstruction experiment.

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
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
import numpy as np
from itertools import combinations
from sklearn.metrics.pairwise import cosine_distances
from torchmetrics.functional import structural_similarity_index_measure


class DAAF(pl.LightningModule):
    def __init__(self, architecture: str, input_dim: tuple, feature_dim: int, dropout: float):
        """
        Initialize the DAAF model.

        :param architecture: Which network architecture to use (AE or DCAF)
        :param input_dim: Width, height and channel dimension of the input image.
        :param feature_dim: Dimension of the latent space.
        :param dropout: What percentage of nodes should be dropped in the dropout layers.
        """

        # Run parent constructor
        super().__init__()

        # Store the parameters
        self.architecture = architecture
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.dropout = dropout

        # Create the encoder
        if self.architecture == 'AE':
            self.encoder = Encoder(self.input_dim, self.feature_dim, self.dropout)
            self.decoder = Decoder(self.input_dim, self.feature_dim, self.dropout)

        else:
            self.content_encoder = Encoder(self.input_dim, self.feature_dim, self.dropout)
            self.attribute_encoder = Encoder(self.input_dim, self.feature_dim, self.dropout)

            # Create the decoder
            self.content_decoder = Decoder(self.input_dim, self.feature_dim, self.dropout)
            self.decoder = Decoder(self.input_dim, 2 * self.feature_dim, self.dropout)

    def forward(self, X_orig, X):
        """
        Forward pass of the DAAF network.

        :param X_orig: Original input batch.
        :param X: Input batch.

        :return: Reconstructed input image.
        """

        batch_size = X.shape[0]
        X_corruption = torch.sub(X, X_orig)

        if self.architecture == 'AE':
            Z_enc = self.encoder(X)
            X_hat = self.decoder(Z_enc)

            # Calculate the mse loss and the psnr values for the different input images and their reconstructions
            mse_to_orig, ssim_to_orig, psnr_to_orig = self.calculate_mse_ssim_and_psnr(X_orig, X_hat)  # For the corrupted input image
            mse_to_input, ssim_to_input, psnr_to_input = self.calculate_mse_ssim_and_psnr(X, X_hat)  # For the content reconstruction
            mse_to_corruption, ssim_to_corruption, psnr_to_corruption = self.calculate_mse_ssim_and_psnr(X_corruption, X_hat)  # For the reconstruction of the applied corruptions

            return (mse_to_orig, mse_to_input, mse_to_corruption), (ssim_to_orig, ssim_to_input, ssim_to_corruption),\
                   (psnr_to_orig, psnr_to_input, psnr_to_corruption), (X_hat, X_corruption)

        else:
            # Pass the input image through the content encoder to get the content latent representation
            Z_content_enc = self.content_encoder(X)
            Z_content_enc = Z_content_enc.reshape(batch_size, -1)

            # Pass the input image through the attribute encoder to get the attribute latent representation
            Z_attribute_enc = self.attribute_encoder(X)
            Z_attribute_enc = Z_attribute_enc.reshape(batch_size, -1)

            # Concatenate the feature embeddings
            Z = torch.cat((Z_content_enc, Z_attribute_enc), dim=1)

            # Pass the content latent representation through the content decoder to get an content reconstruction
            X_hat_content = self.content_decoder(Z_content_enc.reshape(batch_size, self.feature_dim, 1, 1))

            # Pass the combined latent represnetation through the image decoder to reconstruct the input image
            X_hat = self.decoder(Z.reshape(batch_size, 2 * self.feature_dim, 1, 1))

            # Calculate the applied corruption and an approximation for the attribute image
            X_hat_attribute = torch.sub(X_hat, X_hat_content)

            # Calculate the mse loss and the psnr values for the different input images and their reconstructions
            mse_corruption_to_orig, ssim_corruption_to_orig, psnr_corruption_to_orig = self.calculate_mse_ssim_and_psnr(X_orig, X)
            mse_recons_to_orig, ssim_recons_to_orig, psnr_recons_to_orig = self.calculate_mse_ssim_and_psnr(X_orig, X_hat)
            mse_content_to_orig, ssim_content_to_orig, psnr_content_to_orig = self.calculate_mse_ssim_and_psnr(X_orig, X_hat_content)
            mse_recons_to_input, ssim_recons_to_input, psnr_recons_to_input = self.calculate_mse_ssim_and_psnr(X, X_hat)
            mse_content_to_input, ssim_content_to_input, psnr_content_to_input = self.calculate_mse_ssim_and_psnr(X, X_hat_content)
            mse_attribute, ssim_attribute, psnr_attribute = self.calculate_mse_ssim_and_psnr(X_corruption, X_hat_attribute)

            return (mse_corruption_to_orig, ssim_corruption_to_orig, psnr_corruption_to_orig),\
                   (mse_recons_to_orig, mse_content_to_orig, mse_recons_to_input, mse_content_to_input, mse_attribute),\
                   (ssim_recons_to_orig, ssim_content_to_orig, ssim_recons_to_input, ssim_content_to_input, ssim_attribute),\
                   (psnr_recons_to_orig, psnr_content_to_orig, psnr_recons_to_input, psnr_content_to_input, psnr_attribute),\
                   (X_hat, X_hat_content, X_hat_attribute, X_corruption)

    def calculate_mse_ssim_and_psnr(self, X, X_hat):
        """
        Calculate the MSE loss as well as the PSNR for the given input image(s) and its reconstruction(s).

        :param X: Input image(s).
        :param X_hat: Reconstruction(s).
        :return: MSE and PSNR.
        """

        # Calcualte the ssim value
        ssim = structural_similarity_index_measure(X_hat, X, reduction='elementwise_mean')

        # Calculate the mse value
        mse = F.mse_loss(X_hat, X)
        mse_psnr = F.mse_loss(X_hat, X)

        # Calculate the psnr value
        if mse < 1e-15:
            psnr = 20 * torch.log10(torch.tensor(1).to(X.device))

        else:
            psnr = 20 * torch.log10(torch.tensor(1).to(X.device)) - 10 * torch.log10(mse_psnr)

        # return the values
        return mse, ssim, psnr


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


class Decoder(pl.LightningModule):
    def __init__(self, input_dim: tuple, feature_dim: int, dropout: float, feature_maps_decoder=32):
        """
        Initialize the Encoder following: https://github.com/axkoenig/autoencoder/blob/master/autoencoder.py

        :param input_dim: Width, height and channel dimension of the input image.
        :param feature_dim: Dimension of the output features.
        :param dropout: What percentage of nodes should be dropped in the dropout layers.
        :param feature_maps_decoder: Size of feature maps in decoder.
        """

        # Call parent constructor
        super().__init__()

        # Store the parameters
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.feature_maps_decoder = feature_maps_decoder

        self.model = nn.Sequential(
            # input 1 x 1 x (256 | 512)
            UpBlock(self.feature_dim, self.feature_maps_decoder * 4, kernel_size=5),
            ResidualBlock(self.feature_maps_decoder * 4),
            nn.Dropout(self.dropout),
            # input 3 x 3 x (128 | 256)
            UpBlock(self.feature_maps_decoder * 4, self.feature_maps_decoder * 2, kernel_size=5),
            ResidualBlock(self.feature_maps_decoder * 2),
            nn.Dropout(self.dropout),
            # input 7 x 7 x (64 | 128)
            UpBlock(self.feature_maps_decoder * 2, self.feature_maps_decoder, kernel_size=4),
            ResidualBlock(self.feature_maps_decoder),
            nn.Dropout(self.dropout),
            # input 14 x 14 x (32 | 64)
            UpBlock(self.feature_maps_decoder, self.input_dim[2], kernel_size=4),
            ResidualBlock(self.input_dim[2]),
            # output 28 x 28 x C_in
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