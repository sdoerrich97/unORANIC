"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
unORANIC: Unsupervised orthogonalization of anatomy and image-characteristic features

@description:
Data Loader of the anatomy classification experiment.

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
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DataLoaderCustom:
    def __init__(self, dataset: str, data_path: Path, fold: str, shots_per_class: str, image_dim: tuple,
                 batch_size: int):
        """
        Initialize the specified dataset as a torchvision dataset.

        :param dataset: Which dataset to load.
        :param data_path: Location of the data samples.
        :param fold: Which fold of the 5-fold cross-validaiton is used for testing.
        :param shots_per_class: How many shots per class.
        :param image_dim: The dimension the images should be padded to if necessary.
        :param batch_size: Batch size.
        """

        # Store the parameters
        self.dataset = dataset
        self.data_path = data_path
        self.fold = fold
        self.shots_per_class = shots_per_class
        self.image_dim = image_dim
        self.batch_size = batch_size
        self.num_workers = os.cpu_count()

        # Create the augmentations
        self.transform = A.Compose([
            A.Resize(height=self.image_dim[0], width=self.image_dim[1], interpolation=cv2.INTER_AREA, p=1.0),
            # A.PixelDropout(p=0.25),
            # A.GaussianBlur(p=0.25),
            # A.ColorJitter(p=0.25),
            # A.Downscale(scale_min=0.5, scale_max=0.9, p=0.25, interpolation=cv2.INTER_AREA),
            # A.GaussNoise(p=0.25),
            # A.InvertImg(p=0.25),
            # A.MotionBlur(p=0.25),
            # A.MultiplicativeNoise(p=0.25),
            # A.RandomBrightnessContrast(p=0.25),
            # A.RandomGamma(p=0.25),
            # A.Solarize(threshold=128, p=0.25),
            # A.Sharpen(p=0.25),
            A.ToFloat(),
            ToTensorV2()
        ])

        if self.batch_size != 1:
            shuffle = True

        else:
            shuffle = False

        self.train_set = MEDMNIST(self.data_path, self.transform, self.shots_per_class, "train")
        self.val_set = MEDMNIST(self.data_path, self.transform, self.shots_per_class, "val")
        self.test_set = MEDMNIST(self.data_path, self.transform, self.shots_per_class, "test")

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
    def __init__(self, data_path: Path, transform: A.Compose, shots_per_class: str, samples_set: str):
        """
        Initialize the MedMNIST version
            - Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark
              for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

            - Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni.
              Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image
              classification." Scientific Data, 2023.

        of the given dataset as a torchvision dataset.

        :param data_path: Location of the data samples.
        :param transform: Loading pipeline for the input image.
        :param shots_per_class: How many shots per class.
        :param samples_set: Name of the set to load.
        """

        # Store the parameters
        self.data_path = data_path
        self.transform = transform
        self.shots_per_class = shots_per_class
        self.samples_set = samples_set

        # Get all samples for the current set
        self.samples = list(np.load(str(self.data_path))[f"{samples_set}_images"])
        self.labels = list(np.load(str(self.data_path))[f"{samples_set}_labels"])

        # Extract as much samples as desired for training the classifier
        if self.samples_set == 'train':
            if self.shots_per_class != 'all':
                # Get the number of samples per label
                num_samples_per_label = int(self.shots_per_class[6:])
                unqiue_labels = sorted(set([tuple(label) for label in self.labels]))

                # Randomly extract the required amount of samples per label from the set
                # self.samples = [sample for label in unqiue_labels for sample in random.sample([s for s, l in zip(self.samples, self.labels) if l == label], min(num_samples_per_label, self.labels.count(label)))]
                self.samples = [sample for label in unqiue_labels for sample in [s for s, l in zip(self.samples, self.labels) if l == label][:min(num_samples_per_label, self.labels.count(label))]]
                self.labels = [np.array(label) for label in unqiue_labels for _ in range(min(num_samples_per_label, self.labels.count(label)))]

    def __len__(self):
        """Return the number of samples in the dataset"""

        return len(self.samples)

    def __getitem__(self, index: int):
        """Return one data sample."""

        # Extract the samples path
        sample_name = f"{index}"
        sample, label = self.samples[index], self.labels[index]

        # Apply the transform the get the input sample
        sample = self.transform(image=sample)["image"]

        # Return the samples
        return sample, label, sample_name
