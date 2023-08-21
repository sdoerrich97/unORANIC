"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
unORANIC: Unsupervised orthogonalization of anatomy and image-characteristic features

@description:
Data Loader of the autoencoder.

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
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision


class DataLoaderCustom:
    def __init__(self, dataset: str, data_path: Path, fold: str, image_dim: tuple, batch_size: int, training: bool):
        """
        Initialize the specified dataset as a torchvision dataset.

        :param dataset: Which dataset to load.
        :param data_path: Location of the data samples.
        :param fold: Which fold of the 5-fold cross-validaiton is used for testing.
        :param image_dim: The dimension the images should be padded to if necessary.
        :param batch_size: Batch size.
        """
        # Store the parameters
        self.dataset = dataset
        self.data_path = data_path
        self.fold = fold
        self.image_dim = image_dim
        self.batch_size = batch_size
        self.num_workers = os.cpu_count()

        # Create the augmentations
        self.transform = A.Compose([
            A.Resize(height=self.image_dim[0], width=self.image_dim[1], interpolation=cv2.INTER_AREA, p=1.0),
            #A.HorizontalFlip(p=0.25),
            #A.VerticalFlip(p=0.25),
            A.PixelDropout(p=0.25),
            #A.Transpose(p=0.25),
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
        ])

        self.transform_prime = A.Compose([
            A.Resize(height=self.image_dim[0], width=self.image_dim[1], interpolation=cv2.INTER_AREA, p=1.0),
            A.ToFloat(),
            ToTensorV2()
        ])

        if training:
            shuffle = True

        else:
            self.transform = self.transform_prime
            shuffle = False

        # Load the dataset
        if self.dataset == 'pathmnist':
            self.train_set = PATHMNIST(self.data_path, self.transform_prime, self.transform, "train")
            self.val_set = PATHMNIST(self.data_path, self.transform_prime, self.transform_prime, "val")
            self.test_set = PATHMNIST(self.data_path, self.transform_prime, self.transform_prime, "test")

        elif self.dataset == 'chestmnist':
            self.train_set = CHESTMNIST(self.data_path, self.transform_prime, self.transform, "train")
            self.val_set = CHESTMNIST(self.data_path, self.transform_prime, self.transform_prime, "val")
            self.test_set = CHESTMNIST(self.data_path, self.transform_prime, self.transform_prime, "test")

        elif self.dataset == 'dermamnist':
            self.train_set = DERMAMNIST(self.data_path, self.transform_prime, self.transform, "train")
            self.val_set = DERMAMNIST(self.data_path, self.transform_prime, self.transform_prime, "val")
            self.test_set = DERMAMNIST(self.data_path, self.transform_prime, self.transform_prime, "test")

        elif self.dataset == 'octmnist':
            self.train_set = OCTMNIST(self.data_path, self.transform_prime, self.transform, "train")
            self.val_set = OCTMNIST(self.data_path, self.transform_prime, self.transform_prime, "val")
            self.test_set = OCTMNIST(self.data_path, self.transform_prime, self.transform_prime, "test")

        elif self.dataset == 'pneumoniamnist':
            self.train_set = PNEUMONIAMNIST(self.data_path, self.transform_prime, self.transform, "train")
            self.val_set = PNEUMONIAMNIST(self.data_path, self.transform_prime, self.transform_prime, "val")
            self.test_set = PNEUMONIAMNIST(self.data_path, self.transform_prime, self.transform_prime, "test")

        elif self.dataset == 'retinamnist':
            self.train_set = RETINAMNIST(self.data_path, self.transform_prime, self.transform, "train")
            self.val_set = RETINAMNIST(self.data_path, self.transform_prime, self.transform_prime, "val")
            self.test_set = RETINAMNIST(self.data_path, self.transform_prime, self.transform_prime, "test")

        elif self.dataset == 'breastmnist':
            self.train_set = BREASTMNIST(self.data_path, self.transform_prime, self.transform, "train")
            self.val_set = BREASTMNIST(self.data_path, self.transform_prime, self.transform_prime, "val")
            self.test_set = BREASTMNIST(self.data_path, self.transform_prime, self.transform_prime, "test")

        elif self.dataset == 'bloodmnist':
            self.train_set = BLOODMNIST(self.data_path, self.transform_prime, self.transform, "train")
            self.val_set = BLOODMNIST(self.data_path, self.transform_prime, self.transform_prime, "val")
            self.test_set = BLOODMNIST(self.data_path, self.transform_prime, self.transform_prime, "test")

        elif self.dataset == 'tissuemnist':
            self.train_set = TISSUEMNIST(self.data_path, self.transform_prime, self.transform, "train")
            self.val_set = TISSUEMNIST(self.data_path, self.transform_prime, self.transform_prime, "val")
            self.test_set = TISSUEMNIST(self.data_path, self.transform_prime, self.transform_prime, "test")

        elif self.dataset == 'organamnist':
            self.train_set = ORGANAMNIST(self.data_path, self.transform_prime, self.transform, "train")
            self.val_set = ORGANAMNIST(self.data_path, self.transform_prime, self.transform_prime, "val")
            self.test_set = ORGANAMNIST(self.data_path, self.transform_prime, self.transform_prime, "test")

        elif self.dataset == 'organcmnist':
            self.train_set = ORGANCMNIST(self.data_path, self.transform_prime, self.transform, "train")
            self.val_set = ORGANCMNIST(self.data_path, self.transform_prime, self.transform_prime, "val")
            self.test_set = ORGANCMNIST(self.data_path, self.transform_prime, self.transform_prime, "test")

        elif self.dataset == 'organsmnist':
            self.train_set = ORGANSMNIST(self.data_path, self.transform_prime, self.transform, "train")
            self.val_set = ORGANSMNIST(self.data_path, self.transform_prime, self.transform_prime, "val")
            self.test_set = ORGANSMNIST(self.data_path, self.transform_prime, self.transform_prime, "test")

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


class PATHMNIST(Dataset):
    def __init__(self, data_path: Path, transform_prime, transform, samples_set: str):
        """
        Initialize the MedMNIST version
            - Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark
              for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

            - Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni.
              Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image
              classification." Scientific Data, 2023.

        of the NCT-CRC-HE-100K (train/val) and CRC-VAL-HE-7K (test) dataset:
            Jakob Nikolas Kather, Johannes Krisam, et al., "Predicting survival from colorectal cancer histology
            slides using deep learning: A retrospective multicenter study," PLOS Medicine, vol. 16, no. 1, pp. 1–22,
            01 2019.

        as a torchvision dataset.

        :param data_path: Location of the data samples.
        :param transform_prime: Loading pipeline for the original image.
        :param transform: Loading pipeline for the input image.
        :param samples_set: Name of the set to load.
        """

        # Store the parameters
        self.data_path = data_path
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

        # Apply the transform the original sample
        sample = self.transform(image=sample_orig)["image"]
        sample_orig = self.transform_prime(image=sample_orig)["image"]

        # Return the samples
        return sample_orig, label_orig, sample, sample_name


class CHESTMNIST(Dataset):
    def __init__(self, data_path: Path, transform_prime, transform, samples_set: str):
        """
        Initialize the MedMNIST version
            - Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark
              for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

            - Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni.
              Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image
              classification." Scientific Data, 2023.

        of the Chest X-ray dataset:
            Xiaosong Wang, Yifan Peng, et al., "Chest x-ray8: Hospital-scale chest x-ray database and benchmarks on
            weakly-supervised classification and localization of common thorax diseases," in CVPR, 2017, pp. 3462–3471.

        as a torchvision dataset.

        :param data_path: Location of the data samples.
        :param transform_prime: Loading pipeline for the original image.
        :param transform: Loading pipeline for the input image.
        :param samples_set: Name of the set to load.
        """

        # Store the parameters
        self.data_path = data_path
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

        # Apply the transform the original sample
        sample = self.transform(image=sample_orig)["image"]
        sample_orig = self.transform_prime(image=sample_orig)["image"]

        # Return the samples
        return sample_orig, label_orig, sample, sample_name


class DERMAMNIST(Dataset):
    def __init__(self, data_path: Path, transform_prime, transform, samples_set: str):
        """
        Initialize the MedMNIST version
            - Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark
              for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

            - Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni.
              Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image
              classification." Scientific Data, 2023.

        of the HAM10000 dataset:
            - Philipp Tschandl, Cliff Rosendahl, et al., "The ham10000 dataset, a large collection of multisource
              dermatoscopic images of common pigmented skin lesions," Scientific data, vol. 5, pp. 180161, 2018.

            - Noel Codella, Veronica Rotemberg, et al., “Skin Lesion Analysis Toward Melanoma Detection 2018: A
              Challenge Hosted by the International Skin Imaging Collaboration (ISIC)”, 2018, arXiv:1902.03368.

        as a torchvision dataset.

        :param data_path: Location of the data samples.
        :param transform_prime: Loading pipeline for the original image.
        :param transform: Loading pipeline for the input image.
        :param samples_set: Name of the set to load.
        """

        # Store the parameters
        self.data_path = data_path
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

        # Apply the transform the original sample
        sample = self.transform(image=sample_orig)["image"]
        sample_orig = self.transform_prime(image=sample_orig)["image"]

        # Return the samples
        return sample_orig, label_orig, sample, sample_name


class OCTMNIST(Dataset):
    def __init__(self, data_path: Path, transform_prime, transform, samples_set: str):
        """
        Initialize the MedMNIST version
            - Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark
              for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

            - Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni.
              Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image
              classification." Scientific Data, 2023.

        of the OCT dataset:
            Daniel S. Kermany, Michael Goldbaum, et al., "Identifying medical diagnoses and treatable diseases by
            image-based deep learning," Cell, vol. 172, no. 5, pp. 1122 – 1131.e9, 2018.

        as a torchvision dataset.

        :param data_path: Location of the data samples.
        :param transform_prime: Loading pipeline for the original image.
        :param transform: Loading pipeline for the input image.
        :param samples_set: Name of the set to load.
        """

        # Store the parameters
        self.data_path = data_path
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

        # Apply the transform the original sample
        sample = self.transform(image=sample_orig)["image"]
        sample_orig = self.transform_prime(image=sample_orig)["image"]

        # Return the samples
        return sample_orig, label_orig, sample, sample_name


class PNEUMONIAMNIST(Dataset):
    def __init__(self, data_path: Path, transform_prime, transform, samples_set: str):
        """
        Initialize the MedMNIST version
            - Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark
              for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

            - Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni.
              Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image
              classification." Scientific Data, 2023.

        of the Pneumonia dataset:
            Daniel S. Kermany, Michael Goldbaum, et al., "Identifying medical diagnoses and treatable diseases by
            image-based deep learning," Cell, vol. 172, no. 5, pp. 1122 – 1131.e9, 2018.
            [https://data.mendeley.com/datasets/rscbjbr9sj/3]

        as a torchvision dataset.

        :param data_path: Location of the data samples.
        :param transform_prime: Loading pipeline for the original image.
        :param transform: Loading pipeline for the input image.
        :param samples_set: Name of the set to load.
        """

        # Store the parameters
        self.data_path = data_path
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

        # Apply the transform the original sample
        sample = self.transform(image=sample_orig)["image"]
        sample_orig = self.transform_prime(image=sample_orig)["image"]

        # Return the samples
        return sample_orig, label_orig, sample, sample_name


class RETINAMNIST(Dataset):
    def __init__(self, data_path: Path, transform_prime, transform, samples_set: str):
        """
        Initialize the MedMNIST version
            - Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark
              for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

            - Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni.
              Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image
              classification." Scientific Data, 2023.

        of the DeepDRiD24 challenge dataset:
            DeepDR Diabetic Retinopathy Image Dataset (DeepDRiD), "The 2nd diabetic retinopathy grading and image
            quality estimation challenge," https://isbi.deepdr.org/data.html, 2020.

        as a torchvision dataset.

        :param data_path: Location of the data samples.
        :param transform_prime: Loading pipeline for the original image.
        :param transform: Loading pipeline for the input image.
        :param samples_set: Name of the set to load.
        """

        # Store the parameters
        self.data_path = data_path
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

        # Apply the transform the original sample
        sample = self.transform(image=sample_orig)["image"]
        sample_orig = self.transform_prime(image=sample_orig)["image"]

        # Return the samples
        return sample_orig, label_orig, sample, sample_name


class BREASTMNIST(Dataset):
    def __init__(self, data_path: Path, transform_prime, transform, samples_set: str):
        """
        Initialize the MedMNIST version
            - Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark
              for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

            - Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni.
              Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image
              classification." Scientific Data, 2023.

        of the BUSI dataset:
            Walid Al-Dhabyani, Mohammed Gomaa, et al., "Dataset of breast ultrasound images," Data in Brief, vol. 28,
            pp. 104863, 2020.

        as a torchvision dataset.

        :param data_path: Location of the data samples.
        :param transform_prime: Loading pipeline for the original image.
        :param transform: Loading pipeline for the input image.
        :param samples_set: Name of the set to load.
        """

        # Store the parameters
        self.data_path = data_path
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

        # Apply the transform the original sample
        sample = self.transform(image=sample_orig)["image"]
        sample_orig = self.transform_prime(image=sample_orig)["image"]

        # Return the samples
        return sample_orig, label_orig, sample, sample_name


class BLOODMNIST(Dataset):
    def __init__(self, data_path: Path, transform_prime, transform, samples_set: str):
        """
        Initialize the MedMNIST version
            - Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark
              for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

            - Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni.
              Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image
              classification." Scientific Data, 2023.

        of the blood dataset:
            Andrea Acevedo, Anna Merino, et al., "A dataset of microscopic peripheral blood cell images for
            development of automatic recognition systems," Data in Brief, vol. 30, pp. 105474, 2020.

        as a torchvision dataset.

        :param data_path: Location of the data samples.
        :param transform_prime: Loading pipeline for the original image.
        :param transform: Loading pipeline for the input image.
        :param samples_set: Name of the set to load.
        """

        # Store the parameters
        self.data_path = data_path
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

        # Apply the transform the original sample
        sample = self.transform(image=sample_orig)["image"]
        sample_orig = self.transform_prime(image=sample_orig)["image"]

        # Return the samples
        return sample_orig, label_orig, sample, sample_name


class TISSUEMNIST(Dataset):
    def __init__(self, data_path: Path, transform_prime, transform, samples_set: str):
        """
        Initialize the MedMNIST version
            - Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark
              for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

            - Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni.
              Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image
              classification." Scientific Data, 2023.

        of the BBBC051 dataset:
            Vebjorn Ljosa, Katherine L Sokolnicki, et al., “Annotated high-throughput microscopy imagesets for
            validation.,” Nature methods, vol. 9, no. 7, pp.637–637, 2012.

        as a torchvision dataset.

        :param data_path: Location of the data samples.
        :param transform_prime: Loading pipeline for the original image.
        :param transform: Loading pipeline for the input image.
        :param samples_set: Name of the set to load.
        """

        # Store the parameters
        self.data_path = data_path
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

        # Apply the transform the original sample
        sample = self.transform(image=sample_orig)["image"]
        sample_orig = self.transform_prime(image=sample_orig)["image"]

        # Return the samples
        return sample_orig, label_orig, sample, sample_name


class ORGANAMNIST(Dataset):
    def __init__(self, data_path: Path, transform_prime, transform, samples_set: str):
        """
        Initialize the MedMNIST version
            - Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark
              for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

            - Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni.
              Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image
              classification." Scientific Data, 2023.

        of the Liver Tumor Segmentation Benchmark (LiTS) - Axial Perspective dataset:
            - Patrick Bilic, Patrick Ferdinand Christ, et al., "The liver tumor segmentation benchmark (lits),"
              arXiv preprint arXiv:1901.04056, 2019.

            - Xuanang Xu, Fugen Zhou, et al., "Efficient multiple organ localization in ct image using 3d region
              proposal network," IEEE Transactions on Medical Imaging, vol. 38, no. 8, pp. 1885–1898, 2019.

        as a torchvision dataset.

        :param data_path: Location of the data samples.
        :param transform_prime: Loading pipeline for the original image.
        :param transform: Loading pipeline for the input image.
        :param samples_set: Name of the set to load.
        """

        # Store the parameters
        self.data_path = data_path
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

        # Apply the transform the original sample
        sample = self.transform(image=sample_orig)["image"]
        sample_orig = self.transform_prime(image=sample_orig)["image"]

        # Return the samples
        return sample_orig, label_orig, sample, sample_name


class ORGANCMNIST(Dataset):
    def __init__(self, data_path: Path, transform_prime, transform, samples_set: str):
        """
        Initialize the MedMNIST version
            - Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark
              for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

            - Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni.
              Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image
              classification." Scientific Data, 2023.

        of the Liver Tumor Segmentation Benchmark (LiTS) - Coronal Perspective dataset:
            - Patrick Bilic, Patrick Ferdinand Christ, et al., "The liver tumor segmentation benchmark (lits),"
              arXiv preprint arXiv:1901.04056, 2019.

            - Xuanang Xu, Fugen Zhou, et al., "Efficient multiple organ localization in ct image using 3d region
              proposal network," IEEE Transactions on Medical Imaging, vol. 38, no. 8, pp. 1885–1898, 2019.

        as a torchvision dataset.

        :param data_path: Location of the data samples.
        :param transform_prime: Loading pipeline for the original image.
        :param transform: Loading pipeline for the input image.
        :param samples_set: Name of the set to load.
        """

        # Store the parameters
        self.data_path = data_path
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

        # Apply the transform the original sample
        sample = self.transform(image=sample_orig)["image"]
        sample_orig = self.transform_prime(image=sample_orig)["image"]

        # Return the samples
        return sample_orig, label_orig, sample, sample_name


class ORGANSMNIST(Dataset):
    def __init__(self, data_path: Path, transform_prime, transform, samples_set: str):
        """
        Initialize the MedMNIST version
            - Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark
              for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

            - Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni.
              Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image
              classification." Scientific Data, 2023.

        of the Liver Tumor Segmentation Benchmark (LiTS) - Sagittal Perspective dataset:
            - Patrick Bilic, Patrick Ferdinand Christ, et al., "The liver tumor segmentation benchmark (lits),"
              arXiv preprint arXiv:1901.04056, 2019.

            - Xuanang Xu, Fugen Zhou, et al., "Efficient multiple organ localization in ct image using 3d region
              proposal network," IEEE Transactions on Medical Imaging, vol. 38, no. 8, pp. 1885–1898, 2019.

        as a torchvision dataset.

        :param data_path: Location of the data samples.
        :param transform_prime: Loading pipeline for the original image.
        :param transform: Loading pipeline for the input image.
        :param samples_set: Name of the set to load.
        """

        # Store the parameters
        self.data_path = data_path
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

        # Apply the transform the original sample
        sample = self.transform(image=sample_orig)["image"]
        sample_orig = self.transform_prime(image=sample_orig)["image"]

        # Return the samples
        return sample_orig, label_orig, sample, sample_name