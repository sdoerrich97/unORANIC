"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
unORANIC: Unsupervised orthogonalization of anatomy and image-characteristic features

@description:
Data Loader of unORANIC.

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
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision


class DataLoaderCustom:
    def __init__(self, dataset: str, data_path: Path, fold: str, image_dim: tuple, batch_size: int, tuple_size: int):
        """
        Initialize the specified dataset as a torchvision dataset.

        :param dataset: Which dataset to load.
        :param data_path: Location of the data samples.
        :param fold: Which fold of the 5-fold cross-validaiton is used for testing.
        :param image_dim: The dimension the images should be padded to if necessary.
        :param batch_size: Batch size.
        :param tuple_size: Number of samples used for the Barlow Tuple.
        """
        # Store the parameters
        self.dataset = dataset
        self.data_path = data_path
        self.fold = fold
        self.image_dim = image_dim
        self.batch_size = batch_size
        self.tuple_size = tuple_size
        self.num_workers = os.cpu_count()

        # Create the augmentations
        additional_targets = get_additional_targets(self.tuple_size)

        self.transform = A.Compose([
            A.Resize(height=self.image_dim[0], width=self.image_dim[1], interpolation=cv2.INTER_AREA, p=1.0),
            # A.HorizontalFlip(p=0.25),
            # A.VerticalFlip(p=0.25),
            A.PixelDropout(p=0.25),
            # A.Transpose(p=0.25),
            A.GaussianBlur(p=0.25),
            A.ColorJitter(p=0.25),
            A.Downscale(scale_min=0.5, scale_max=0.9, p=0.25, interpolation=cv2.INTER_AREA),
            A.GaussNoise(p=0.25),
            A.InvertImg(p=0.25),
            A.MotionBlur(p=0.25),
            A.MultiplicativeNoise(p=0.25),
            A.RandomBrightnessContrast(p=0.25),
            A.RandomGamma(p=0.25),
            A.Solarize(threshold=128, p=0.25),
            A.Sharpen(p=0.25),
            A.ToFloat(),
            ToTensorV2()
        ],
            additional_targets=additional_targets
        )

        self.transform_prime = A.Compose([
            A.Resize(height=self.image_dim[0], width=self.image_dim[1], interpolation=cv2.INTER_AREA, p=1.0),
            A.ToFloat(),
            ToTensorV2()
        ])

        if self.batch_size != 1:
            shuffle = True

        else:
            self.transform = self.transform_prime
            shuffle = False

        # Load the dataset
        self.train_set = MEDMNIST(self.data_path, self.tuple_size, self.transform_prime, self.transform, "train")
        self.val_set = MEDMNIST(self.data_path, self.tuple_size, self.transform_prime, self.transform, "val")
        self.test_set = MEDMNIST(self.data_path, self.tuple_size, self.transform_prime, self.transform_prime, "test")

        self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=shuffle)

        self.val_loader = DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False)

        self.test_loader = DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False)

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
    def __init__(self, data_path: Path, tuple_size: int, transform_prime, transform, samples_set: str):
        """
        Initialize the MedMNIST version
            - Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark
              for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

            - Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni.
              Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image
              classification." Scientific Data, 2023.

        of the given dataset as a torchvision dataset.

        :param data_path: Location of the data samples.
        :param tuple_size: Number of samples used for the Barlow Tuple.
        :param transform_prime: Loading pipeline for the original image.
        :param transform: Loading pipeline for the input image.
        :param samples_set: Name of the set to load.
        """

        # Store the parameters
        self.data_path = data_path
        self.tuple_size = tuple_size
        self.transform_prime = transform_prime
        self.transform = transform
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

        # Get additional samples for the scanner barlow
        additional_indices = np.random.choice(len(self.samples), self.tuple_size, False, None)  # Get the indices
        # samples_barlow_attribute = [self.samples[idx] for idx in additional_indices]  # Get the corresponding sample paths

        sample = self.transform(image=sample_orig)["image"]
        # Create the sample variants for the content Barlow

        samples_barlow_content = []

        for i in range(self.tuple_size - 1):
            samples_barlow_content.append(self.transform(image=sample_orig)["image"])

        sample_orig = self.transform_prime(image=sample_orig)["image"]

        # # Load the original sample as well as the variants for each encoder path
        # sample_orig, sample, samples_barlow_content, samples_barlow_attribute = \
        #     apply_transformations(sample_orig, samples_barlow_attribute, self.transform_prime, self.transform,
        #                           self.tuple_size)

        # Return the samples
        #return sample_orig, label_orig, sample, samples_barlow_content, samples_barlow_attribute, sample_name
        return sample_orig, label_orig, sample, samples_barlow_content, sample_name


def get_additional_targets(tuple_size):
    """
    Create the additional targets dictionary for the attribute barlow samples, depending on the given Barlow tuple size.

    :param tuple_size: Barlow Tuple size.

    :return: additional targets dictionary for the albumentations transforms.
    """

    # Get the additional_targets, depending on the number of tuples
    if tuple_size == 1:
        additional_targets = {}

    elif tuple_size == 2:
        additional_targets = {'image0': 'image'}

    elif tuple_size == 3:
        additional_targets = {'image0': 'image', 'image1': 'image'}

    elif tuple_size == 4:
        additional_targets = {'image0': 'image', 'image1': 'image', 'image2': 'image'}

    elif tuple_size == 5:
        additional_targets = {'image0': 'image', 'image1': 'image', 'image2': 'image', 'image3': 'image'}

    elif tuple_size == 6:
        additional_targets = {'image0': 'image', 'image1': 'image', 'image2': 'image', 'image3': 'image', 'image4': 'image'}

    else:
        raise SystemExit("Specified tuple size not supported!")

    # Return the additional targets
    return additional_targets


def apply_transformations(sample, samples_barlow_attribute: list, transform_prime, transform, tuple_size: int):
    """
    Apply the current transformations as well as different transformations on the original sample to get a
    transformed version of the original sample as well as its content barlow variants.
    Furthermore, create the scanner barlow variants by applying the same transformation to different samples.

    :return: The transformation of the original sample, the content barlow variants, the scanner barlow variants.
    """

    # Transform the original sample to match the dimension and value range of the variants
    sample_orig = transform_prime(image=sample)["image"]

    # Transform the given original sample as well as the additional samples the same way
    if tuple_size == 1:
        transformed = transform(image=sample)
        sample_augm, samples_barlow_attribute = transformed["image"], []

    elif tuple_size == 2:
        transformed = transform(image=sample, image0=samples_barlow_attribute[0])
        sample_augm, samples_barlow_attribute = transformed["image"], [transformed["image0"]]

    elif tuple_size == 3:
        transformed = transform(image=sample, image0=samples_barlow_attribute[0], image1=samples_barlow_attribute[1])
        sample_augm, samples_barlow_attribute = transformed["image"], [transformed["image0"], transformed["image1"]]

    elif tuple_size == 4:
        transformed = transform(image=sample, image0=samples_barlow_attribute[0], image1=samples_barlow_attribute[1],
                                image2=samples_barlow_attribute[2])
        sample_augm, samples_barlow_attribute = transformed["image"], [transformed["image0"], transformed["image1"],
                                                                       transformed["image2"]]

    elif tuple_size == 5:
        transformed = transform(image=sample, image0=samples_barlow_attribute[0], image1=samples_barlow_attribute[1],
                                image2=samples_barlow_attribute[2], image3=samples_barlow_attribute[3])
        sample_augm, samples_barlow_attribute = transformed["image"], [transformed["image0"], transformed["image1"],
                                                                       transformed["image2"], transformed["image3"]]

    elif tuple_size == 6:
        transformed = transform(image=sample, image0=samples_barlow_attribute[0], image1=samples_barlow_attribute[1],
                                image2=samples_barlow_attribute[2], image3=samples_barlow_attribute[3],
                                image4=samples_barlow_attribute[4])
        sample_augm, samples_barlow_attribute = transformed["image"], [transformed["image0"], transformed["image1"],
                                                                       transformed["image2"], transformed["image3"],
                                                                       transformed["image4"]]

    else:
        raise SystemExit("Specified tuple size not supported!")

    # Create the sample variants for the content Barlow
    samples_barlow_content = []

    for i in range(tuple_size - 1):
        samples_barlow_content.append(transform(image=sample)["image"])

    # Return the original sample as well as the variants for both the scanner and content barlows
    return sample_orig, sample_augm, samples_barlow_content, samples_barlow_attribute
