"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
unORANIC: Unsupervised orthogonalization of anatomy and image-characteristic features

@description:
Main Script of the robustness experiment for the anatomy classification.

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
import argparse
from pathlib import Path
import time
import wandb
import torch
from torch import nn
import yaml
import numpy as np
import cv2
import json
from imagecorruptions import get_corruption_names
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

# Import own scripts
from data_loader import DataLoaderCustom
from network import Classifier, ResNet18


class CorruptionRobustness:
    def __init__(self, configuration):
        """
        Initialize the trainer.
        """

        # Read out the parameters of the training run
        self.dataset = configuration['dataset']
        self.data_path = Path(configuration['data_path'])
        self.input_path = Path(configuration['input_path'])
        self.output_path = Path(configuration['output_path'])
        self.architecture = configuration['architecture']
        self.fold = configuration['fold']
        self.seed = configuration['seed']
        self.device = torch.device(configuration['device'])

        # Read out the hyperparameters of the training run
        self.image_dim = (configuration['image_dim']['height'], configuration['image_dim']['width'],
                          configuration['image_dim']['channel'])
        self.feature_dim = configuration['feature_dim']
        self.task = configuration['task']
        self.shots_per_class = configuration['shots_per_class']
        self.nr_classes = configuration['nr_classes']
        self.dropout = configuration['dropout']

        # Create the path to where the output shall be stored
        self.input_path = self.input_path / self.architecture
        self.output_path = self.output_path / f"{self.feature_dim}" / self.dataset / f"fold_{self.fold}" / self.architecture / self.shots_per_class
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Enable the use of cudnn
        torch.backends.cudnn.benchmark = True

        # Initialize the model
        print("\tInitializing the model ...")
        if self.architecture == 'AE' or self.architecture == 'DCAF':
            # Load the trained encoder checkpoint
            self.checkpoint_file = self.input_path / f"{self.feature_dim}" / self.dataset / f"fold_{self.fold}" / self.architecture / self.shots_per_class / "checkpoint_final.pt"
            self.checkpoint = torch.load(self.checkpoint_file, map_location='cpu')

            # Initialize the model an load the trained weights
            self.model = Classifier(self.task, self.image_dim, self.feature_dim, self.nr_classes, self.dropout)
            self.model.load_state_dict(self.checkpoint['model'])

        elif self.architecture == 'benchmarkRes18_complete' or self.architecture == 'benchmarkRes18_lastlayer':
            # Load the trained encoder checkpoint
            self.checkpoint_file = self.input_path / f"{self.feature_dim}" / self.dataset / f"fold_{self.fold}" / self.architecture / self.shots_per_class / "best_model.pth"
            self.checkpoint = torch.load(self.checkpoint_file, map_location='cpu')

            # Initialize the model an load the trained weights
            self.model = ResNet18(in_channels=self.image_dim[2], num_classes=self.nr_classes)
            self.model.load_state_dict(self.checkpoint['net'])

        else:
            # Load the trained encoder checkpoint
            self.checkpoint_file = self.input_path / f"{self.feature_dim}" / self.dataset / f"fold_{self.fold}" / self.architecture / self.shots_per_class / "best_model.pth"
            self.checkpoint = torch.load(self.checkpoint_file, map_location='cpu')

            self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Load pre-trained ResNet18 model
            self.model.fc = nn.Linear(self.model.fc.in_features, self.nr_classes)
            self.model.load_state_dict(self.checkpoint['net'])

        self.model.to(self.device)
        self.model.requires_grad_(False)

        # Initialize the mse loss
        print("\tInitialize the loss criterion...")
        self.loss_criterion = nn.CrossEntropyLoss().to(self.device)

    def run_robustness_experiment(self):
        """
        Run inference for the DAAF network.
        """

        # Start code
        start_time = time.time()

        # Empty the unused memory cache
        print("\tRun robustness experiment...")
        print("\tEmptying the unused memory cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Set the model to eval
        self.model.eval()

        store_orig_image = True
        all_images, corruptions, severities, all_auc, all_acc = [], [], [], [], []

        # Iterate through all corruptions and test the robustness of the model
        for corruption in get_corruption_names():
            # Skip the broken corruptions
            if corruption == 'glass_blur':
                continue

            # Iterate through all severity degrees and test the robustness of the model
            for severity in range(1, 6):
                data_loader = DataLoaderCustom(self.dataset, self.data_path, self.image_dim, corruption, severity)
                test_loader = data_loader.get_test_loader()

                # Get all samples for the current combination of corruption and severity
                samples_orig, labels_orig, samples_corrupted, sample_names = next(iter(test_loader))

                # Push the corrupted samples to the GPU
                X, Y = samples_corrupted.to(self.device), labels_orig.to(self.device).reshape(-1)

                # Run the input through the classifier to get the predictions
                if self.architecture == 'imagenetRes18_complete' or self.architecture == 'imagenetRes18_lastlayer':
                    X = torch.cat((X, X, X), dim=1)

                Z = self.model(X)

                # Calculate the loss as well as AUC and ACC
                loss = self.loss_criterion(Z, Y)

                AUC = self.getAUC(Y.cpu().numpy(), Z.cpu().numpy(), self.task)
                ACC = self.getACC(Y.cpu().numpy(), Z.cpu().numpy(), self.task)

                print(f"\t\t\tCorruption: {corruption} ({severity})")
                print(f"\t\t\t\tCrossEntropyLoss: {loss}")
                print(f"\t\t\t\tArea-under-the-curve: {AUC}")
                print(f"\t\t\t\tAccuracy: {ACC}")

                # Log the metrics
                wandb.log(
                    {
                        "CrossEntropyLoss": loss,
                        "Accuracy": ACC,
                        "Area-under-the-curve": AUC,
                    })

                # Store one image with the current corruption in the list of all images
                if store_orig_image:
                    all_images.append(samples_orig[0].cpu())
                    corruptions.append('None')
                    severities.append('0')
                    all_auc.append('1')
                    all_acc.append('1')

                all_images.append(X[0].cpu())
                corruptions.append(corruption)
                severities.append(severity)
                all_auc.append(round(AUC, 3))
                all_acc.append(round(ACC, 3))

        # Save one example of the sample set together with all its corrupted variants
        self.save_img_and_corruptions(all_images, corruptions, severities, all_auc, all_acc, self.output_path)

        # Stop the time
        end_time = time.time()
        hours, minutes, seconds = self.calculate_passed_time(start_time, end_time)

        print("\t\t\tElapsed time: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))


    def getAUC(self, y_true, y_score, task):
        """
        AUC metric.

        :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
        :param y_score: the predicted score of each class,
        shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
        :param task: the task of current dataset
        """
        y_true = y_true.squeeze()
        y_score = y_score.squeeze()

        if task == 'multi-label, binary-class':
            auc = 0
            for i in range(y_score.shape[1]):
                label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
                auc += label_auc
            ret = auc / y_score.shape[1]

        elif task == 'binary-class':
            if y_score.ndim == 2:
                y_score = y_score[:, -1]
            else:
                assert y_score.ndim == 1
            ret = roc_auc_score(y_true, y_score)

        else:
            auc = 0
            for i in range(y_score.shape[1]):
                y_true_binary = (y_true == i).astype(float)
                y_score_binary = y_score[:, i]
                auc += roc_auc_score(y_true_binary, y_score_binary)
            ret = auc / y_score.shape[1]

        return ret

    def getACC(self, y_true, y_score, task, threshold=0.5):
        """
        Accuracy metric.

        :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
        :param y_score: the predicted score of each class,
        shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
        :param task: the task of current dataset
        :param threshold: the threshold for multilabel and binary-class tasks
        """
        y_true = y_true.squeeze()
        y_score = y_score.squeeze()

        if task == 'multi-label, binary-class':
            y_pre = y_score > threshold
            acc = 0
            for label in range(y_true.shape[1]):
                label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
                acc += label_acc
            ret = acc / y_true.shape[1]

        elif task == 'binary-class':
            if y_score.ndim == 2:
                y_score = y_score[:, -1]
            else:
                assert y_score.ndim == 1
            ret = accuracy_score(y_true, y_score > threshold)

        else:
            ret = accuracy_score(y_true, np.argmax(y_score, axis=-1))

        return ret


    def save_img_and_corruptions(self, images: list, corruptions: list, severities: list, AUCs: list, ACCs: list, output_path):
        """
        Save the processed input image together with the reconstructed image.

        :param images: List of one example of the test set as well as all corrupted variants of that example.
        :param corruptions: List of the applied corruptions to each image.
        :param severities: List of the severity of each corruption applied to each image.
        :param output_path: Where the images shall be stored.
        """

        evaluation = {}

        # Iterate through all images, corruptions and severities and store the corresponding image
        for image, corruption, severity, AUC, ACC in zip(images, corruptions, severities, AUCs, ACCs):
            # Reorder the and height, width and channel dimensions
            image = image.permute(1, 2, 0).numpy()

            # Normalize the pixel values in the range 0 to 255
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            # Save the images PNG files
            cv2.imwrite(str(output_path / f"{corruption}_{severity}.png"), image)

            # Write the results to the results dictionary
            if corruption in evaluation:
                evaluation[corruption][f'Severity {severity}'] = {'AUC': AUC, 'ACC': ACC}

            else:
                evaluation[corruption] = {f'Severity {severity}': {'AUC': AUC, 'ACC': ACC}}

        # Write the dictionary to the JSON file
        with open((output_path / "evaluation.json"), 'w') as json_file:
            json.dump(evaluation, json_file, indent=4)


    def calculate_passed_time(self, start_time, end_time):
        """
        Calculate the time needed for running the code

        :param: start_time: Start time.
        :param: end_time: End time.
        :return: Duration in hh:mm:ss.ss
        """

        # Calculate the duration
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        # Return the duration in hours, minutes and seconds
        return int(hours), int(minutes), seconds


def main(configuration):
    """
    Run the inference.

    :param configuration: Configuration of the inference run.
    """

    # Initialize the weights & biases project with the given inference configuration
    wandb.init(project="exp_classification_content_robustness", config=configuration)

    # Initialize the inference
    print("Initializing the experiment ...")
    corruptionrobustness = CorruptionRobustness(configuration)

    # Run the inference
    print("Run the experiment...")
    corruptionrobustness.run_robustness_experiment()


if __name__ == '__main__':
    # Start code
    start_time = time.time()

    # Read out the command line parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str, help="Path to the configuration file to use.")
    parser.add_argument("--architecture", required=False, default='DCAF', type=str, help="Which trained encoder to use.")  # Only for bash execution
    parser.add_argument("--shots_per_class", required=False, default='all', type=str, help="How many shots per class.")  # Only for bash

    args = parser.parse_args()
    config_file = args.config_file
    architecture = args.architecture
    shots_per_class = args.shots_per_class

    # Load the default and the hyperparameters and run the training
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        config['architecture'] = architecture
        config['shots_per_class'] = shots_per_class
        main(config)

    # Finished code
    end_time = time.time()

    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))