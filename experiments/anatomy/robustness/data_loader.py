"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
unORANIC: Unsupervised orthogonalization of anatomy and image-characteristic features

@description:
Data Loader of the robustness experiment for the anatomy classification.

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
import torch
import nibabel as nib
import os
from imagecorruptions import corrupt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision


class DataLoaderCustom:
    def __init__(self, dataset: str, data_path: Path, image_dim: tuple, corruption: str, severity: int):
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
        self.corruption = corruption
        self.severity = severity
        self.num_workers = os.cpu_count()

        # Create the augmentations
        self.transform_pre = A.Compose([
            A.Resize(height=32, width=32, interpolation=cv2.INTER_AREA, p=1.0)
        ])

        self.transform_post = A.Compose([
            A.Resize(height=28, width=28, interpolation=cv2.INTER_AREA, p=1.0),
            A.ToFloat(),
            ToTensorV2()
        ])

        nr_samples = np.load(str(self.data_path))["test_images"].shape[0]

        # Load the dataset
        self.test_set = MEDMNIST(self.data_path, self.image_dim, self.transform_pre, self.transform_post, self.corruption, self.severity, "test")

        self.test_loader = DataLoader(dataset=self.test_set, batch_size=nr_samples, num_workers=self.num_workers,
                                      shuffle=False)

    def get_test_loader(self):
        """Get the test loader."""

        return self.test_loader


class MEDMNIST(Dataset):
    def __init__(self, data_path: Path, image_dim: tuple, transform_pre: A.Compose, transform_post: A.Compose, corruption: str, severity: int, samples_set: str):
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
        :param corruption: What type of corruption should be applied to all data samples
        :param samples_set: Name of the set to load.
        """

        # Store the parameters
        self.data_path = data_path
        self.image_dim = image_dim
        self.transform_pre = transform_pre
        self.transform_post = transform_post
        self.corruption = corruption
        self.severity = severity
        self.samples_set = samples_set

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
        sample_orig, label_orig = self.samples[index], self.labels[index]

        # Apply the preprocessing to the original sample
        sample_pre = self.transform_pre(image=sample_orig)["image"]

        # Apply the corruption
        corrupted = corrupt(sample_pre, corruption_name=self.corruption, severity=self.severity)

        # Apply the postprocessing to the original sample
        sample = self.transform_post(image=corrupted)["image"]
        sample_orig = self.transform_post(image=sample_orig)["image"]

        # Check wether the input image had only one channel
        if self.image_dim[2] == 1:
            sample = torch.mean(sample, dim=0, keepdim=True)

        # Return the samples
        return sample_orig, label_orig, sample, sample_name
