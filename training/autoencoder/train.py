"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
unORANIC: Unsupervised orthogonalization of anatomy and image-characteristic features

@description:
Training of the autoencoder.

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
import torch.nn as nn
import yaml
from tqdm import tqdm
from torchmetrics import StructuralSimilarityIndexMeasure

# Import own scripts
from data_loader import DataLoaderCustom
from network import Autoencoder


class Trainer:
    def __init__(self, configuration):
        """
        Initialize the trainer.
        """

        # Read out the parameters of the training run
        self.dataset = configuration['dataset']
        self.data_path = Path(configuration['data_path'])
        self.output_path = Path(configuration['output_path'])
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
        self.optimizer = configuration['optimizer']['optimizer']
        self.learning_rate = configuration['optimizer']['learning_rate']
        self.first_momentum = configuration['optimizer']['first_momentum']
        self.second_momentum = configuration['optimizer']['second_momentum']
        self.scheduler_lr_base = configuration['optimizer']['scheduler']['base_lr']
        self.scheduler_lr_max = configuration['optimizer']['scheduler']['max_lr']
        self.scheduler_step_up = configuration['optimizer']['scheduler']['step_up']
        self.dropout = configuration['dropout']

        # Create the path to where the output shall be stored and initialize the logger
        self.output_path = self.output_path / f"{self.feature_dim}" / self.dataset / f"fold_{self.fold}"
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Enable the use of cudnn
        torch.backends.cudnn.benchmark = True

        # Initialize the dataloader
        print("\tInitializing the dataloader...")
        self.data_loader = DataLoaderCustom(self.dataset, self.data_path, self.fold, self.image_dim, self.batch_size, training=True)
        self.train_loader = self.data_loader.get_train_loader()
        self.val_loader = self.data_loader.get_val_loader()

        # Initialize the model
        print("\tInitializing the model ...")
        self.model = Autoencoder(self.image_dim, self.feature_dim, self.dropout)
        self.model.to(self.device)
        self.model.requires_grad_(True)

        # Initialize the optimizer and learning scheduler for the attribute VAE
        print("\tInitializing the optimizer and lr scheduler for the attribute VAE...")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(self.first_momentum,
                                                                                                 self.second_momentum))
        self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.scheduler_lr_base,
                                                     max_lr=self.scheduler_lr_max,
                                                     step_size_up=len(self.train_loader) * 5,
                                                     mode='triangular2', cycle_momentum=False)

        # Initialize the loss criterion
        print("\tInitialize the loss criterion...")
        self.loss_criterion_mse = nn.MSELoss()
        self.loss_criterion_ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='elementwise_mean').to(self.device)

    def train(self):
        """
        Train the content encoder-decoder pair in an autencoder-like fashion.
        """

        # Start code
        start_time = time.time()

        # Continue from a potential checkpoint
        if self.resume_training:
            print("Load checkpoint...")
            checkpoint_file = self.output_path / f"checkpoint_ep{self.starting_epoch}.pt"
            checkpoint = torch.load(checkpoint_file, map_location='cpu')

            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

            self.model.to(self.device)
            self.model.requires_grad_(True)

        print("Run training...")

        # Empty the unused memory cache
        print("\tEmptying the unused memory cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Train the model for the specified number of epochs
        print(f"\tRun the training for {self.epochs} epochs...")
        for epoch in range(self.starting_epoch, self.epochs + 1):
            # Stop the time
            start_time_epoch = time.time()

            print(f"\t\tEpoch {epoch} of {self.epochs}:")
            pbar_train = tqdm(total=len(self.train_loader), bar_format='{l_bar}{bar}', ncols=80, initial=0, position=0, leave=False)
            pbar_train.set_description(f'\t\t\tTraining')

            # Set the model to train
            self.model.train()

            mse, ssim, psnr = 0, 0, 0
            for i, (_, _, X, _) in enumerate(self.train_loader):
                # Zero out the gradients
                self.optimizer.zero_grad()

                # Push the next input batch to the GPU
                X = X.to(self.device)

                # Run the input through the Autoencoder
                X_hat = self.model(X)

                # Calculate the loss of the current batch
                mse_batch = self.loss_criterion_mse(X_hat, X)
                ssim_batch = (1 - self.loss_criterion_ssim(X_hat, X))

                # Do backprop
                mse_batch.backward()
                self.optimizer.step()

                # Peak Signal-to-Noise Ratio (PSNR)
                if not mse_batch < 1e-15:
                    psnr_batch = 20 * torch.log10(torch.tensor(1).to(mse_batch.device)) - 10 * torch.log10(mse_batch)

                else:
                    psnr_batch = torch.tensor(50).to(mse_batch.device)

                # Add the loss, psnr and ssim to the overall values for this epoch
                ssim += -(ssim_batch.detach().cpu() - 1)
                mse += mse_batch.detach().cpu()
                psnr += psnr_batch.detach().cpu()

                # Update the progress bar
                pbar_train.update(1)

            # Update the learning rate scheduler
            self.scheduler.step()

            # Average the loss and psnr over the batches of this epoch
            ssim /= len(self.train_loader)
            mse /= len(self.train_loader)
            psnr /= len(self.train_loader)

            print(f"\t\t\tTrain SSIM: {ssim}")
            print(f"\t\t\tTrain MSE: {mse}")
            print(f"\t\t\tTrain PSNR: {psnr}")

            # Run the validation
            with torch.no_grad():
                # Set the model to validation
                self.model.eval()

                # Create a progress bar to track the validation
                pbar_val = tqdm(total=len(self.val_loader), bar_format='{l_bar}{bar}', ncols=80, initial=0, position=0, leave=False)
                pbar_val.set_description(f'\t\t\tValidation')

                mse_val, ssim_val, psnr_val = 0, 0, 0
                for i, (_, _, X_val, _) in enumerate(self.val_loader):
                    # Push the next input batch to the GPU
                    X_val = X_val.to(self.device)

                    # Run the input through model
                    X_hat_val = self.model(X_val)

                    # Calculate the loss of the current batch
                    mse_batch_val = self.loss_criterion_mse(X_hat_val, X_val)
                    ssim_batch_val = self.loss_criterion_ssim(X_hat_val, X_val)

                    # Peak Signal-to-Noise Ratio (PSNR)
                    if not mse_batch_val < 1e-15:
                        psnr_batch_val = 20 * torch.log10(torch.tensor(1).to(mse_batch_val.device)) - 10 * torch.log10(mse_batch_val)

                    else:
                        psnr_batch_val = torch.tensor(50).to(mse_batch_val.device)

                    # Add the loss, psnr and ssim to the overall values for this epoch
                    ssim_val += ssim_batch_val
                    mse_val += mse_batch_val
                    psnr_val += psnr_batch_val

                    # Update the progress bar
                    pbar_val.update(1)

                # Average the loss, mse, kl_div and psnr over the batches of this epoch
                ssim_val /= len(self.val_loader)
                mse_val /= len(self.val_loader)
                psnr_val /= len(self.val_loader)

                print(f"\t\t\tValidation SSIM: {ssim_val}")
                print(f"\t\t\tValidation MSE: {mse_val}")
                print(f"\t\t\tValidation PSNR: {psnr_val}")

            # Log training and validation loss to wandb
            print(f"\t\t\tLogging to weights&biases...")
            wandb.log(
                {
                    "Train SSIM": ssim,
                    "Train MSE": mse,
                    "Train PSNR": psnr,
                    "Validation SSIM": ssim_val,
                    "Validation MSE": mse_val,
                    "Validation PSNR": psnr_val
                })

            # Save the checkpoint
            print(f"\t\t\tSaving the checkpoint...")
            if not self.hp_optimization:
                # Save the checkpoint
                self.save_model(epoch)

            # Stop the time for the epoch
            end_time_epoch = time.time()
            hours_epoch, minutes_epoch, seconds_epoch = self.calculate_passed_time(start_time_epoch, end_time_epoch)

            print("\t\t\tElapsed time for epoch: {:0>2}:{:0>2}:{:05.2f}".format(hours_epoch, minutes_epoch, seconds_epoch))

        # Stop the time
        end_time = time.time()
        hours, minutes, seconds = self.calculate_passed_time(start_time, end_time)

        print("\t\t\tElapsed time for Training: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))

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
                'encoder': self.model.encoder.state_dict(),
                'decoder': self.model.decoder.state_dict(),
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
    Initialize the trainer and execute either a training or a hyperparameter optimization run with the provided
    configuration.
    - If a training run is executed the parameter 'training_configuration' contains the respective training parameters.
    - If a hyperparameter optimization run is executed the parameter 'training_configuration' is not used.

    :param training_configuration: Dictionary containing the parameters and hyperparameters of the training run.
    """

    # Initialize either the given training or the hyperparameter optimization weights & biases project
    if training_configuration is not None:
        if training_configuration["resume_training"]["resume"]:
            # Resume the weights & biases project for the specified training run
            wandb.init(project="pretrain_content_encoder_decoder", id=training_configuration["resume_training"]["wandb_id"], resume="must", config=training_configuration)
        else:
            # Initialize a weights & biases project for a training run with the given training configuration
            wandb.init(project="pretrain_content_encoder_decoder", config=training_configuration)

    # Run the hyperparameter optimization run
    else:
        # Initialize a weights & biases project for a hyperparameter optimization run
        wandb.init(project="sweep_pretrain_content_encoder_decoder")

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
    parser.add_argument("--fold", required=False, default='0', type=str, help="Which fold to use as test fold.")
    parser.add_argument("--sweep", required=False, default=False, type=bool, help="Whether to run hyperparameter tuning or just training.")

    args = parser.parse_args()
    config_file = args.config_file
    fold = args.fold
    sweep = args.sweep

    # Load the default and the hyperparameters and run the training
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        config['training']['fold'] = fold

        if sweep:
            # Configure the sweep
            sweep_id = wandb.sweep(sweep=config['hyperparameter_tuning'], project="sweep_pretrain_content")

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