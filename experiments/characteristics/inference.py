"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
unORANIC: Unsupervised orthogonalization of anatomy and image-characteristic features

@description:
Inference of the image characteristics classification experiment.

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
import torch.nn as nn
import yaml
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# Import own scripts
from data_loader import DataLoaderCustom
from network import Classifier


class Inference:
    def __init__(self, configuration):
        """
        Initialize the trainer.
        """

        # Read out the parameters of the training run
        self.dataset = configuration['dataset']
        self.data_path = Path(configuration['data_path'])
        self.input_path = Path(configuration['input_path'])
        self.architecture = configuration['architecture']
        self.fold = configuration['fold']
        self.seed = configuration['seed']
        self.device = torch.device(configuration['device'])

        # Read out the hyperparameters of the training run
        self.image_dim = (configuration['image_dim']['height'], configuration['image_dim']['width'],
                          configuration['image_dim']['channel'])
        self.feature_dim = configuration['feature_dim']
        self.batch_size = configuration['batch_size']
        self.task = configuration['task']
        self.nr_classes = configuration['nr_classes']
        self.dropout = configuration['dropout']

        # Enable the use of cudnn
        torch.backends.cudnn.benchmark = True

        # Initialize the dataloader
        print("\tInitializing the dataloader...")
        self.data_loader = DataLoaderCustom(self.dataset, self.data_path, self.fold, self.image_dim, self.batch_size, self.seed)
        self.train_loader = self.data_loader.get_train_loader()
        self.val_loader = self.data_loader.get_val_loader()
        self.test_loader = self.data_loader.get_test_loader()

        # Initialize the model
        print("\tInitializing the model ...")
        # Load the trained encoder checkpoint
        self.checkpoint_file = self.input_path / f"{self.feature_dim}" / self.dataset / f"fold_{self.fold}" / self.architecture / "checkpoint_final.pt"
        self.checkpoint = torch.load(self.checkpoint_file, map_location='cpu')

        # Initialize the model and assign the encoder
        self.model = Classifier(self.task, self.image_dim, self.feature_dim, self.nr_classes, self.dropout)
        self.model.load_state_dict(self.checkpoint['model'])
        self.model.to(self.device)
        self.model.requires_grad_(False)

        # Initialize the mse loss
        print("\tInitialize the loss criterion...")
        self.loss_criterion = nn.CrossEntropyLoss().to(self.device)

    def run_inference(self):
        """
        Run inference for the DAAF network.
        """

        # Start code
        start_time = time.time()

        # Empty the unused memory cache
        print("\tRun inference...")
        print("\tEmptying the unused memory cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for set, set_loader in zip(['train', 'val', 'test'], [self.train_loader, self.val_loader, self.test_loader]):
            # Run the inference for the current set
            start_time_set = time.time()
            print(f"\tRun inference for {set}...")

            nr_samples = len(set_loader)
            print(f"\t\tRun inference for {nr_samples} samples...")

            # Create a progress bar to track the training
            pbar_set = tqdm(total=nr_samples, bar_format='{l_bar}{bar}', ncols=80, initial=0, position=0, leave=False)
            pbar_set.set_description(f'\t\t\tProgress')

            loss = 0
            Y_target, Y_predicted = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)

            with torch.no_grad():
                for i, (X_orig, X, Y, _) in enumerate(set_loader):
                    # Set the model to eval
                    self.model.eval()

                    # Push the next input batch to the GPU
                    X, Y = X.to(self.device), Y.to(self.device).reshape(-1)

                    # Run the input through the content model
                    Z = self.model(X)

                    # Calculate the loss
                    loss_sample = self.loss_criterion(Z, Y)

                    print(f"\t\t\tSample: [{i}/{nr_samples}]")
                    print(f"\t\t\t\tCrossEntropyLoss: {loss_sample}")

                    # Store the target and predicted labels
                    loss += loss_sample.detach().cpu()
                    Y_target = torch.cat((Y_target, Y.detach()), dim=0)
                    Y_predicted = torch.cat((Y_predicted, Z.detach()), dim=0)

                    # Update the progress bar
                    pbar_set.update(1)

                # Average the loss over this epoch and compute the metrics
                loss /= (len(set_loader) * self.batch_size)

                ACC = self.getACC(Y_target.cpu().numpy(), Y_predicted.cpu().numpy(), self.task)
                AUC = self.getAUC(Y_target.cpu().numpy(), Y_predicted.cpu().numpy(), self.task)

                print(f"\t\t\tCrossEntropyLoss: {loss}")
                print(f"\t\t\tAccuracy: {ACC}")
                print(f"\t\t\tArea-under-the-curve: {AUC}")

                # Log the metrics' averages
                wandb.log(
                    {
                        "CrossEntropyLoss": loss,
                        "Accuracy": ACC,
                        "Area-under-the-curve": AUC,
                    })

            # Stop the time
            end_time_set = time.time()
            hours_set, minutes_set, seconds_set = self.calculate_passed_time(start_time_set, end_time_set)

            print("\t\t\tElapsed time for set: {:0>2}:{:0>2}:{:05.2f}".format(hours_set, minutes_set, seconds_set))

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

                if len(np.unique(y_true_binary)) > 1:
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
    wandb.init(project="exp_classification_attribute_inference", config=configuration)

    # Initialize the inference
    print("Initializing the inference ...")
    inference = Inference(configuration)

    # Run the inference
    print("Run the inference...")
    inference.run_inference()


if __name__ == '__main__':
    # Start code
    start_time = time.time()

    # Read out the command line parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str, help="Path to the configuration file to use.")
    parser.add_argument("--architecture", required=False, default='DCAF', type=str,
                        help="Which trained encoder to use.")  # Only for bash execution
    parser.add_argument("--shots_per_class", required=False, default='all', type=str, help="How many shots per class.")  # Only for bash

    args = parser.parse_args()
    config_file = args.config_file
    architecture = args.architecture
    shots_per_class = args.shots_per_class

    # Load the default and the hyperparameters and run the training
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        config['inference']['architecture'] = architecture
        config['inference']['shots_per_class'] = shots_per_class

        main(config['inference'])

    # Finished code
    end_time = time.time()

    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
