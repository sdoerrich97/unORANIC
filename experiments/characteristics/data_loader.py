"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
unORANIC: Unsupervised orthogonalization of anatomy and image-characteristic features

@description:
Data Loader of the image characteristics classification experiment.

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
import numpy as np
import pickle
from PIL import Image
import torch
import nibabel as nib
import random
from imagecorruptions import get_corruption_names, corrupt
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision


class DataLoaderCustom:
    def __init__(self, dataset: str, data_path: Path, fold: str, image_dim: tuple, batch_size: int, seed: int):
        """
        Initialize the specified dataset as a torchvision dataset.

        :param dataset: Which dataset to load.
        :param data_path: Location of the data samples.
        :param image_dim: Input image dimensions.
        :param corruption: What type of corruption should be applied to all data samples
        :param severity: How strong should the corruption be.
        """

        # Store the parameters
        self.dataset = dataset
        self.data_path = data_path
        self.image_dim = image_dim
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = os.cpu_count()

        self.transform = A.Compose([
            A.Resize(height=self.image_dim[0], width=self.image_dim[1], interpolation=cv2.INTER_AREA, p=1.0),
            A.OneOf([
                A.PixelDropout(p=1),
                A.GaussianBlur(p=1),
                A.ColorJitter(p=1),
                A.Downscale(scale_min=0.5, scale_max=0.9, p=1, interpolation=cv2.INTER_AREA),
                A.GaussNoise(p=1),
                A.InvertImg(p=1),
                A.MotionBlur(p=1),
                A.MultiplicativeNoise(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
                A.Solarize(threshold=128, p=1),
                A.Sharpen(p=1),
            ], p=1.0),
            A.ToFloat(),
            ToTensorV2()
        ])

        self.transform_prime = A.Compose([
            A.Resize(height=self.image_dim[0], width=self.image_dim[1], interpolation=cv2.INTER_AREA, p=1.0),
            A.ToFloat(),
            ToTensorV2()
        ])

        if self.batch_size != 1:
            shuffle = True

        else:
            shuffle = False

        self.train_set = MEDMNIST(self.data_path, self.image_dim, self.transform_prime, self.transform, "train", self.seed)
        self.val_set = MEDMNIST(self.data_path, self.image_dim, self.transform_prime, self.transform, "val", self.seed)
        self.test_set = MEDMNIST(self.data_path, self.image_dim, self.transform_prime, self.transform, "test", self.seed)

        # Create the dataloader instances
        self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.batch_size,
                                       num_workers=self.num_workers, shuffle=shuffle)

        self.val_loader = DataLoader(dataset=self.val_set, batch_size=self.batch_size,
                                     num_workers=self.num_workers, shuffle=False)

        self.test_loader = DataLoader(dataset=self.test_set, batch_size=self.batch_size,
                                      num_workers=self.num_workers, shuffle=False)

    def get_train_loader(self):
        """Get the train loader."""

        return self.train_loader

    def get_val_loader(self):
        """Get the validation loader."""

        return self.val_loader

    def get_test_loader(self):
        """Get the test loader."""

        return self.test_loader


class MEDMNIST(Dataset):
    def __init__(self, data_path: Path, image_dim: tuple, transform_prime: A.Compose, transform: A.Compose, samples_set: str, seed: int):
        """
        Initialize the MedMNIST version
            - Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark
              for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

            - Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni.
              Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image
              classification." Scientific Data, 2023.

        of the given dataset as a torchvision dataset.

        :param data_path: Location of the data samples.
        :param image_dim: Input image dimensions.
        :param transform_pre: Preprocessing pipeline for the input image to be able to apply the corruptions.
        :param transform_post: Postprocessing pipeline for the input image to be able to apply the corruptions.
        :param data_path: Location of the data samples.
        :param samples_set: Name of the set to load.
        """

        # Store the parameters
        self.data_path = data_path
        self.image_dim = image_dim
        self.transform_prime = transform_prime
        self.transform = transform
        self.samples_set = samples_set
        self.seed = seed

        random.seed(self.seed)

        # Get all samples for the current set
        self.samples = np.load(str(self.data_path))[f"{samples_set}_images"]
        self.labels = np.load(str(self.data_path))[f"{samples_set}_labels"]

    def __len__(self):
        """Return the number of samples in the dataset"""

        return len(self.samples)

    def __getitem__(self, index: int):
        """Return one data sample."""

        # Extract the samples path
        sample_name = f"{index}"
        sample_orig = self.samples[index]

        # Apply the corruption to the original sample
        if random.random() < 0.5:
            sample = self.transform(image=sample_orig)["image"]
            label = 1

        else:
            sample = self.transform_prime(image=sample_orig)["image"]
            label = 0

        sample_orig = self.transform_prime(image=sample_orig)["image"]

        # Return the samples
        return sample_orig, sample, np.array([label]), sample_name


# class DataLoaderCustom:
#     def __init__(self, dataset: str, data_path: Path, fold: str, image_dim: tuple, batch_size: int, seed: int):
#         """
#         Initialize the specified dataset as a torchvision dataset.
#
#         :param dataset: Which dataset to load.
#         :param data_path: Location of the data samples.
#         :param fold: Which fold of the 5-fold cross-validaiton is used for testing.
#         :param image_dim: The dimension the images should be padded to if necessary.
#         :param seed: random seed.
#         :param batch_size: Batch size.
#         """
#         # Store the parameters
#         self.dataset = dataset
#         self.data_path = data_path
#         self.fold = fold
#         self.image_dim = image_dim
#         self.batch_size = batch_size
#         self.seed = seed
#         self.num_workers = os.cpu_count()
#
#         # Create the augmentations
#         self.transform_pre = A.Compose([
#             A.Resize(height=32, width=32, interpolation=cv2.INTER_AREA, p=1.0)
#         ])
#
#         self.transform_post = A.Compose([
#             A.Resize(height=28, width=28, interpolation=cv2.INTER_AREA, p=1.0),
#             A.ToFloat(),
#             ToTensorV2()
#         ])
#
#         if self.batch_size != 1:
#             shuffle = True
#
#         else:
#             shuffle = False
#
#         self.train_set = MEDMNIST(self.data_path, self.image_dim, self.transform_pre, self.transform_post, "train", self.seed)
#         self.val_set = MEDMNIST(self.data_path, self.image_dim, self.transform_pre, self.transform_post, "val", self.seed)
#         self.test_set = MEDMNIST(self.data_path, self.image_dim, self.transform_pre, self.transform_post, "test", self.seed)
#
#         # Create the dataloader instances
#         self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.batch_size,
#                                        num_workers=self.num_workers, shuffle=shuffle)
#
#         self.val_loader = DataLoader(dataset=self.val_set, batch_size=self.batch_size,
#                                      num_workers=self.num_workers, shuffle=False)
#
#         self.test_loader = DataLoader(dataset=self.test_set, batch_size=self.batch_size,
#                                       num_workers=self.num_workers, shuffle=False)
#
#     def get_train_loader(self):
#         """Get the train loader."""
#
#         return self.train_loader
#
#     def get_val_loader(self):
#         """Get the validation loader."""
#
#         return self.val_loader
#
#     def get_test_loader(self):
#         """Get the test loader."""
#
#         return self.test_loader
#
#
# class MEDMNIST(Dataset):
#     def __init__(self, data_path: Path, image_dim: tuple, transform_pre: A.Compose, transform_post: A.Compose, samples_set: str, seed: int):
#         """
#         Initialize the MedMNIST version
#             - Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark
#               for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.
#
#             - Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni.
#               Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image
#               classification." Scientific Data, 2023.
#
#         of the given dataset as a torchvision dataset.
#
#         :param data_path: Location of the data samples.
#         :param image_dim: Input image dimensions.
#         :param transform_pre: Preprocessing pipeline for the input image to be able to apply the corruptions.
#         :param transform_post: Postprocessing pipeline for the input image to be able to apply the corruptions.
#         :param samples_set: Name of the set to load.
#         :param seed: random seed.
#         """
#
#         # Store the parameters
#         self.data_path = data_path
#         self.image_dim = image_dim
#         self.transform_pre = transform_pre
#         self.transform_post = transform_post
#         self.samples_set = samples_set
#         self.seed = seed
#
#         # Get all corruptions
#         random.seed(self.seed)
#         self.corruptions = get_corruption_names()
#         self.corruptions = [corruption for corruption in self.corruptions if corruption != 'glass_blur']  # Remove glass blur from the list of corruptions
#
#         # Get all samples for the current set
#         self.samples = list(np.load(str(self.data_path))[f"{samples_set}_images"])
#
#     def __len__(self):
#         """Return the number of samples in the dataset"""
#
#         return len(self.samples)
#
#     def __getitem__(self, index: int):
#         """Return one data sample."""
#
#         # Extract the samples path
#         sample_name = f"{index}"
#         sample_orig = self.samples[index]
#
#         # Apply the preprocessing to the original sample
#         sample = self.transform_pre(image=sample_orig)["image"]
#
#         # With a 50% probability randomly apply a corruption
#         if random.random() < 0.5:
#             apply_corruption = True
#             label_attribute = 1
#
#         else:
#             apply_corruption = False
#             label_attribute = 0
#
#         if apply_corruption:
#             corruption_idx = random.randint(0, 13)
#             severity = random.randint(1, 5)
#             sample = corrupt(sample, corruption_name=self.corruptions[corruption_idx], severity=severity)
#
#         sample = self.transform_post(image=sample)["image"]
#         sample_orig = self.transform_post(image=sample_orig)["image"]
#
#         # Check wether the input image had only one channel
#         if apply_corruption and self.image_dim[2] == 1:
#             sample = torch.mean(sample, dim=0, keepdim=True)
#
#         # Return the samples
#         return sample_orig, sample, np.array([label_attribute]), sample_name
