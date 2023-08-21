"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
unORANIC: Unsupervised orthogonalization of anatomy and image-characteristic features

@description:
Architecture of unORANIC.

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
    def __init__(self, input_dim: tuple, feature_dim: int, dropout: float):
        """
        Initialize the DAAF model.

        :param input_dim: Width, height and channel dimension of the input image.
        :param feature_dim: Dimension of the latent space.
        :param dropout: What percentage of nodes should be dropped in the dropout layers.
        """

        # Run parent constructor
        super().__init__()

        # Store the parameters
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.dropout = dropout

        # Create additional parameters
        self.barlow_lambda = 5e-3

        # Create the encoder
        self.content_encoder = Encoder(self.input_dim, self.feature_dim, self.dropout)
        self.attribute_encoder = Encoder(self.input_dim, self.feature_dim, self.dropout)

        # Create the decoder
        self.content_decoder = Decoder(self.input_dim, self.feature_dim, self.dropout)
        self.decoder = Decoder(self.input_dim, 2 * self.feature_dim, self.dropout)

        # Initialize the content batchnorm (normalization layer for the representations z1 and z2)
        # self.bn = nn.BatchNorm1d(self.feature_dim, affine=False)
        self.content_bn = nn.BatchNorm1d(self.feature_dim, affine=False)
        self.attribute_bn = nn.BatchNorm1d(self.feature_dim, affine=False)

    # def forward(self, X_orig, X, X_tuple_content, X_tuple_attribute):
    def forward(self, X_orig, X, X_tuple_content):
        """
        Forward pass of the DAAF network.

        :param X_orig: Original input batch.
        :param X: Input batch.
        :param X_tuple_content: Image tuple for the content branch.
        :param X_tuple_attribute: Image tuple for the attribute branch.

        :return: Reconstructed input image, mean and log variance of the latent variables.
        """

        batch_size = X.shape[0]

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
        X_corruption = torch.sub(X, X_orig)
        X_hat_attribute = torch.sub(X_hat, X_hat_content)

        # Calculate the mse loss and the psnr values for the different input images and their reconstructions
        mse, ssim, psnr = self.calculate_mse_ssim_and_psnr(X, X_hat)  # For the corrupted input image
        mse_content, ssim_content, psnr_content = self.calculate_mse_ssim_and_psnr(X_orig, X_hat_content)  # For the content reconstruction
        mse_attribute, ssim_attribute, psnr_attribute = self.calculate_mse_ssim_and_psnr(X_corruption, X_hat_attribute)  # For the reconstruction of the applied corruptions

        # Run the input through the embeddings consistency loss
        if X_tuple_content:
            # Run the input through the embeddings consistency loss
            consistency_loss_content, mean_cos_dist_content, embeddings_content = self.compute_embedding_consistency(X_tuple_content, content=True)
            #consistency_loss_attribute, mean_cos_dist_attribute, embeddings_attribute = self.compute_embedding_consistency(X_tuple_attribute, content=False)

            # Run the input through the Barlow model
            barlow_loss_content = self.compute_barlow_loss(embeddings_content, content=True)
            #barlow_loss_attribute = self.compute_barlow_loss(embeddings_attribute, content=False)

        else:
            consistency_loss_content, mean_cos_dist_content, embeddings_content, barlow_loss_content = None, None, None, None
            #consistency_loss_attribute, mean_cos_dist_attribute, embeddings_attribute, barlow_loss_attribute = None, None, None, None

        # Return everything
        # return (mse, mse_content, mse_attribute), (ssim, ssim_content, ssim_attribute), (psnr, psnr_content, psnr_attribute), \
        #        (consistency_loss_content, mean_cos_dist_content, barlow_loss_content),\
        #        (consistency_loss_attribute, mean_cos_dist_attribute, barlow_loss_attribute), \
        #        (X_hat.detach().cpu(), X_hat_content.detach().cpu(), X_hat_attribute.detach().cpu(), X_corruption.detach().cpu())

        return (mse, mse_content, mse_attribute), (ssim, ssim_content, ssim_attribute), (psnr, psnr_content, psnr_attribute), \
               (consistency_loss_content, mean_cos_dist_content, barlow_loss_content), \
               (X_hat.detach().cpu(), X_hat_content.detach().cpu(), X_hat_attribute.detach().cpu(), X_corruption.detach().cpu())

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

    def compute_embedding_consistency(self, barlow_batch, content: bool):
        """
        Compute the mean cosine distances and the consistency loss between the embeddings.

        :param barlow_batch:
        :param content:
        :return:
        """

        # Assign the encoder
        if content:
            encoder = self.content_encoder

        else:
            encoder = self.attribute_encoder

        # Store the embeddings and predictions for each input image
        embeddings = []

        # Pass each input through the network
        for X in barlow_batch:
            embed = encoder(X)
            embed = embed.reshape(embed.shape[0], -1)
            embeddings.append(embed)

        # Calculate the consistency loss for each embedding combination
        mean_cos_dist, consistency_loss = 0, 0

        for E1, E2 in combinations(embeddings, 2):
            # Calculate the cosine distance between the current embedding pair
            mean_cos_dist += np.mean(cosine_distances(E1.detach().cpu(), E2.detach().cpu()))

            # Calculate the mean squared error between the current embedding pair
            consistency_loss += F.mse_loss(E1, E2)

        # Average the consistency loss over the number of embedding combinations
        nr_comb_embed = math.comb(len(embeddings), 2)
        mean_cos_dist = (1 / nr_comb_embed) * mean_cos_dist
        consistency_loss = (1 / nr_comb_embed) * consistency_loss

        # Return the consistency loss
        return consistency_loss, mean_cos_dist, embeddings

    def compute_barlow_loss(self, content_embeddings, content: bool):
        """
        Pass the inputs X1, X2, ... Xn through the network and return the loss as well as the encoded features of the
        first input sample.

        :param content_embeddings: content Embeddings [(batch_size, latent_dim), (batch_size, latent_dim), ...]

        :return barlow loss, mean_cosine_distance of the embeddings
        """

        # Assign the batchnorm
        if content:
            bn = self.content_bn

        else:
            bn = self.attribute_bn

        batch_size = content_embeddings[0].shape[0]

        # Calculate the barlow loss
        loss = 0

        for E1, E2 in combinations(content_embeddings, 2):
            # Empirical cross-correlation matrix
            c = bn(E1).T @ bn(E2) / batch_size

            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = self.off_diagonal(c).pow_(2).sum()
            current_combination_loss = on_diag + self.barlow_lambda * off_diag

            # Add the current loss to the over all loss
            loss += current_combination_loss

        # Average the cosine distance
        nr_comb_embed = math.comb(len(content_embeddings), 2)
        loss = (1 / nr_comb_embed) * loss

        # Return Barlow loss
        return loss

    def off_diagonal(self, X):
        """
        Return a flattened view of the off-diagonal elements of a square matrix X for the barlow loss.
        """

        n, m = X.shape
        assert n == m

        return X.flatten()[:-1].reshape(n - 1, n + 1)[:, 1:].flatten()


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