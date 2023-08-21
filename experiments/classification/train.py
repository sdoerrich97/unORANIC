"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
unORANIC: Unsupervised orthogonalization of anatomy and image-characteristic features

@description:
Training of the simultaneous anatomy & image characteristics classification experiment.

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
import torch.optim as optim
from torch import nn
import yaml
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# Import own scripts
from data_loader import DataLoaderCustom
from network import Classifier


class Trainer:
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
        self.resume_training = configuration['resume_training']['resume']
        self.seed = configuration['seed']
        self.device = torch.device(configuration['device'])
        self.hp_optimization = configuration['hp_optimization']

        # Read out the hyperparameters of the training run
        self.starting_epoch = configuration['starting_epoch']
        self.epochs = configuration['epochs']
        self.image_dim = (configuration['image_dim']['height'], configuration['image_dim']['width'],
                          configuration['image_dim']['channel'])
        self.feature_dim = configuration['feature_dim']
        self.batch_size = configuration['batch_size']
        self.task_content = configuration['task']['content']
        self.task_attribute = configuration['task']['attribute']
        self.nr_classes_content = configuration['nr_classes']['content']
        self.nr_classes_attribute = configuration['nr_classes']['attribute']
        self.optimizer = configuration['optimizer']['optimizer']
        self.learning_rate = configuration['optimizer']['learning_rate']
        self.first_momentum = configuration['optimizer']['first_momentum']
        self.second_momentum = configuration['optimizer']['second_momentum']
        self.scheduler_lr_base = configuration['optimizer']['scheduler']['base_lr']
        self.scheduler_lr_max = configuration['optimizer']['scheduler']['max_lr']
        self.scheduler_step_up = configuration['optimizer']['scheduler']['step_up']

        self.dropout = configuration['dropout']

        # Create the path to where the output shall be stored and initialize the logger
        self.input_path = self.input_path / self.architecture
        self.output_path = self.output_path / f"{self.feature_dim}" / self.dataset / f"fold_{self.fold}" / self.architecture
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Enable the use of cudnn
        torch.backends.cudnn.benchmark = True

        # Initialize the dataloader
        print("\tInitializing the dataloader...")
        self.data_loader = DataLoaderCustom(self.dataset, self.data_path, self.fold, self.image_dim, self.batch_size, self.seed)
        self.train_loader = self.data_loader.get_train_loader()
        self.val_loader = self.data_loader.get_val_loader()

        # Initialize the model
        print("\tInitializing the model ...")
        # Load the trained encoder checkpoint
        self.encoder_checkpoint_file = self.input_path / f"{self.feature_dim}" / self.dataset / f"fold_{self.fold}" / "checkpoint_final.pt"
        self.checkpoint = torch.load(self.encoder_checkpoint_file, map_location='cpu')

        # Initialize the model and assign the encoder
        self.model = Classifier(self.architecture, self.image_dim, self.feature_dim, self.task_content,
                                self.nr_classes_content, self.task_attribute, self.nr_classes_attribute, self.dropout)

        if self.architecture == 'AE':
            self.model.encoder.model.load_state_dict(self.checkpoint['encoder'])

        else:
            self.model.encoder_content.load_state_dict(self.checkpoint['content_encoder'])
            self.model.encoder_attribute.load_state_dict(self.checkpoint['attribute_encoder'])

        # Freeze the encoder
        self.model = self.model.to(self.device)
        if self.architecture == 'AE':
            self.model.encoder.requires_grad_(False)

        else:
            self.model.encoder_content.requires_grad_(False)
            self.model.encoder_attribute.requires_grad_(False)

        self.model.classifier_content.requires_grad_(True)
        self.model.classifier_attribute.requires_grad_(True)

        # Initialize the optimizer for the model
        print("\tInitializing the optimizer for the classifier...")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(self.first_momentum, self.second_momentum))

        # Initialize the learning rate scheduler for the model
        print("\tInitialize the learning rate scheduler for the classifier...")
        self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.scheduler_lr_base,
                                                        max_lr=self.scheduler_lr_max,
                                                        step_size_up=len(self.train_loader) * 5,
                                                        mode='triangular2', cycle_momentum=False)

        # Initialize the loss function
        print("\tInitializing the loss function...")
        self.loss_criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        """
        Train the Classifier network.
        """

        # Start code
        start_time = time.time()

        # Continue from a potential checkpoint
        if self.resume_training:
            checkpoint_file = self.output_path / f"checkpoint_ep{self.starting_epoch}.pt"
            checkpoint = torch.load(checkpoint_file, map_location='cpu')

            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

            self.model.to(self.device)

            if self.architecture == 'AE':
                self.model.encoder.requires_grad_(False)

            else:
                self.model.encoder_content.requires_grad_(False)
                self.model.encoder_attribute.requires_grad_(False)

            self.model.classifier_content.requires_grad_(True)
            self.model.classifier_attribute.requires_grad_(True)

        # Empty the unused memory cache
        print("\tEmptying the unused memory cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Train the classifier for the specified number of epochs
        print(f"\tRun the training for {self.epochs} epochs...")
        for epoch in range(self.starting_epoch, self.epochs + 1):
            # Stop the time
            start_time_epoch = time.time()
            print(f"\t\tEpoch {epoch} of {self.epochs}:")
            pbar_train = tqdm(total=len(self.train_loader), bar_format='{l_bar}{bar}', ncols=80, initial=0, position=0,
                              leave=False)
            pbar_train.set_description(f'\t\t\tTraining')

            # Set the model to train
            self.model.train()

            total_loss, loss_content, loss_attribute = 0, 0, 0
            Y_target_content, Y_predicted_content = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
            Y_target_attribute, Y_predicted_attribute = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)

            for i, (_, X, Y_content, Y_attribute, _) in enumerate(self.train_loader):
                # Zero out the gradients
                self.optimizer.zero_grad()

                # Push the next input batch to the GPU
                X, Y_content, Y_attribute = X.to(self.device), Y_content.to(self.device).reshape(-1), Y_attribute.to(self.device).reshape(-1)

                # Run the input through the model
                Z_content, Z_attribute = self.model(X)

                # Calculate the loss
                loss_current_content = self.loss_criterion(Z_content, Y_content)
                loss_current_attribute = self.loss_criterion(Z_attribute, Y_attribute)
                total_loss_current = loss_current_content + loss_current_attribute

                # Do backprop
                total_loss_current.backward()
                self.optimizer.step()

                # Add the current loss to the overall values of this epoch
                total_loss += total_loss_current.detach().cpu()
                loss_content += loss_current_content.detach().cpu()
                loss_attribute += loss_current_attribute.detach().cpu()

                Y_target_content = torch.cat((Y_target_content, Y_content.detach()), dim=0)
                Y_predicted_content = torch.cat((Y_predicted_content, Z_content.detach()), dim=0)
                Y_target_attribute = torch.cat((Y_target_attribute, Y_attribute.detach()), dim=0)
                Y_predicted_attribute = torch.cat((Y_predicted_attribute, Z_attribute.detach()), dim=0)

                # Update the progress bar
                pbar_train.update(1)

            # Update the learning rate scheduler
            self.scheduler.step()

            # Average the loss over this epoch and compute the metrics
            total_loss /= len(self.train_loader)
            loss_content /= len(self.train_loader)
            loss_attribute /= len(self.train_loader)

            ACC_content = self.getACC(Y_target_content.cpu().numpy(), Y_predicted_content.cpu().numpy(), self.task_content)
            AUC_content = self.getAUC(Y_target_content.cpu().numpy(), Y_predicted_content.cpu().numpy(), self.task_content)

            ACC_attribute = self.getACC(Y_target_attribute.cpu().numpy(), Y_predicted_attribute.cpu().numpy(), self.task_attribute)
            AUC_attribute = self.getAUC(Y_target_attribute.cpu().numpy(), Y_predicted_attribute.cpu().numpy(), self.task_attribute)

            print(f"\t\t\tTrain Loss Total: {total_loss}")
            print(f"\t\t\tTrain Loss Content: {loss_content}")
            print(f"\t\t\tTrain Loss Attribute: {loss_attribute}")
            print(f"\t\t\tTrain Accuracy Content: {ACC_content}")
            print(f"\t\t\tTrain Area-under-the-curve Content: {AUC_content}")
            print(f"\t\t\tTrain Accuracy Attribute: {ACC_attribute}")
            print(f"\t\t\tTrain Area-under-the-curve Attribute: {AUC_attribute}")

            # Run the validation
            with torch.no_grad():
                # Set the model to validation
                self.model.eval()

                # Create a progress bar to track the validation
                pbar_val = tqdm(total=len(self.val_loader), bar_format='{l_bar}{bar}', ncols=80, initial=0, position=0,
                                leave=False)
                pbar_val.set_description(f'\t\t\tValidation')

                total_loss_val, loss_content_val, loss_attribute_val = 0, 0, 0
                Y_target_content_val, Y_predicted_content_val = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
                Y_target_attribute_val, Y_predicted_attribute_val = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)

                for i, (_, X_val, Y_content_val, Y_attribute_val, _) in enumerate(self.val_loader):
                    # Push the next input batch to the GPU
                    X_val, Y_content_val, Y_attribute_val = X_val.to(self.device), Y_content_val.to(self.device).reshape(-1), Y_attribute_val.to(self.device).reshape(-1)

                    # Run the input through the content model
                    Z_content_val, Z_attribute_val = self.model(X_val)

                    # Calculate the loss
                    loss_current_content_val = self.loss_criterion(Z_content_val, Y_content_val)
                    loss_current_attribute_val = self.loss_criterion(Z_attribute_val, Y_attribute_val)
                    total_loss_current_val = loss_current_content_val + loss_current_attribute_val

                    # Add the current loss to the overall values of this epoch
                    total_loss_val += total_loss_current_val.cpu()
                    loss_content_val += loss_current_content_val.cpu()
                    loss_attribute_val += loss_current_attribute_val.cpu()

                    Y_target_content_val = torch.cat((Y_target_content_val, Y_content_val), dim=0)
                    Y_predicted_content_val = torch.cat((Y_predicted_content_val, Z_content_val), dim=0)
                    Y_target_attribute_val = torch.cat((Y_target_attribute_val, Y_attribute_val), dim=0)
                    Y_predicted_attribute_val = torch.cat((Y_predicted_attribute_val, Z_attribute_val), dim=0)

                    # Update the progress bar
                    pbar_val.update(1)

                # Average the loss over this epoch and compute the metrics
                total_loss_val /= len(self.val_loader)
                loss_content_val /= len(self.val_loader)
                loss_attribute_val /= len(self.val_loader)

                ACC_content_val = self.getACC(Y_target_content_val.cpu().numpy(), Y_predicted_content_val.cpu().numpy(), self.task_content)
                AUC_content_val = self.getAUC(Y_target_content_val.cpu().numpy(), Y_predicted_content_val.cpu().numpy(), self.task_content)

                ACC_attribute_val = self.getACC(Y_target_attribute_val.cpu().numpy(), Y_predicted_attribute_val.cpu().numpy(), self.task_attribute)
                AUC_attribute_val = self.getAUC(Y_target_attribute_val.cpu().numpy(), Y_predicted_attribute_val.cpu().numpy(), self.task_attribute)

                print(f"\t\t\tValidation Loss Total: {total_loss_val}")
                print(f"\t\t\tValidation Loss Content: {loss_content_val}")
                print(f"\t\t\tValidation Loss Attribute: {loss_attribute_val}")
                print(f"\t\t\tValidation Accuracy Content: {ACC_content_val}")
                print(f"\t\t\tValidation Area-under-the-curve Content: {AUC_content_val}")
                print(f"\t\t\tValidation Accuracy Attribute: {ACC_attribute_val}")
                print(f"\t\t\tValidation Area-under-the-curve Attribute: {AUC_attribute_val}")

            # Log training and validation loss to wandb
            wandb.log(
                {
                    "Train Loss Total": total_loss,
                    "Train Loss Content": loss_content,
                    "Train Loss Attribute": loss_attribute,
                    "Train Accuracy Content": ACC_content,
                    "Train Area-under-the-curve Content": AUC_content,
                    "Train Accuracy Attribute": ACC_attribute,
                    "Train Area-under-the-curve Attribute": AUC_attribute,

                    "Validation Loss Total": total_loss_val,
                    "Validation Loss Content": loss_content_val,
                    "Validation Loss Attribute": loss_attribute_val,
                    "Validation Accuracy Content": ACC_content_val,
                    "Validation Area-under-the-curve Content": AUC_content_val,
                    "Validation Accuracy Attribute": ACC_attribute_val,
                    "Validation Area-under-the-curve Attribute": AUC_attribute_val,
                }, step=epoch)

            # Save the checkpoint
            print(f"\t\t\tSaving the checkpoint...")
            if not self.hp_optimization:
                # Save the checkpoint
                self.save_model(epoch)

            # Stop the time for the epoch
            end_time_epoch = time.time()
            hours_epoch, minutes_epoch, seconds_epoch = self.calculate_passed_time(start_time_epoch, end_time_epoch)

            print("\t\t\tElapsed time for epoch: {:0>2}:{:0>2}:{:05.2f}".format(hours_epoch, minutes_epoch,
                                                                                seconds_epoch))

        # Stop the time
        end_time = time.time()
        hours, minutes, seconds = self.calculate_passed_time(start_time, end_time)

        print("\t\t\tElapsed time for epoch: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))

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

    def save_model(self, epoch: int, save_idx=50):
        """
        Save the model every save_idx epochs.

        :param epoch: Current epoch.
        :param save_idx: Which epochs to save.
        """

        if epoch % save_idx == 0 or epoch == self.epochs:
            # Create the checkpoint
            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }

            # Save the checkpoint
            if epoch % save_idx == 0:
                torch.save(checkpoint, self.output_path / f"checkpoint_ep{epoch}.pt")

            # Save the final model
            if epoch == self.epochs:
                torch.save(checkpoint, self.output_path / "checkpoint_final.pt")

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


def main(training_configuration=None):
    """
    Initialize the classifier and execute either a training or a hyperparameter optimization run with the provided
    configuration.
    - If a training run is executed the parameter 'training_configuration' contains the respective training parameters.
    - If a hyperparameter optimization run is executed the parameter 'training_configuration' is not used.

    :param training_configuration: Dictionary containing the parameters and hyperparameters of the training run.
    """

    # Initialize either the given training or the hyperparameter optimization weights & biases project
    if training_configuration is not None:
        if training_configuration["resume_training"]["resume"]:
            # Resume the weights & biases project for the specified training run
            wandb.init(project="exp_classification_content_and_attribute", id=training_configuration["resume_training"]["wandb_id"], resume="allow", config=training_configuration)

        else:
            # Initialize a weights & biases project for a training run with the given training configuration
            wandb.init(project="exp_classification_content_and_attribute", config=training_configuration)

    # Run the hyperparameter optimization run
    else:
        # Initialize a weights & biases project for a hyperparameter optimization run
        wandb.init(project="sweep_exp_classification_content_and_attribute")

        # Load the beforehand configured sweep configuration
        training_configuration = wandb.config

    # Initialize the Trainer
    print("Initializing the trainer...")
    trainer = Trainer(training_configuration)

    # Run the training
    print("Train the model...")
    trainer.train()


if __name__ == '__main__':
    # Start code
    start_time = time.time()

    # Read out the command line parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str, help="Path to the configuration file to use.")
    parser.add_argument("--input_path", required=False, default='/data/sids-ml/checkpoints/train', type=str, help="Parent directory to where the trained encoder is stored.")  # Only for bash execution
    parser.add_argument("--architecture", required=False, default='DCAF', type=str,  help="Which trained encoder to use.")  # Only for bash execution
    parser.add_argument("--sweep", required=False, default=False, type=bool, help="Whether to run hyperparameter tuning or just training.")

    args = parser.parse_args()
    config_file = args.config_file
    input_path = args.input_path
    architecture = args.architecture
    sweep = args.sweep

    # Load the default and the hyperparameters and run the training
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        config['training']['input_path'] = input_path
        config['training']['architecture'] = architecture

        if sweep:
            # Configure the sweep
            sweep_id = wandb.sweep(sweep=config['hyperparameter_tuning'], project="sweep_exp_classification_content_and_attribute")

            # Start the sweep
            wandb.agent(sweep_id, function=main, count=25)

        else:
            # If no hyperparameter optimization shall be performed run the training
            main(config['training'])

    # Finished code
    end_time = time.time()

    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))