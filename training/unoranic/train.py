"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
unORANIC: Unsupervised orthogonalization of anatomy and image-characteristic features

@description:
Training of unORANIC.

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
import yaml
from tqdm import tqdm

# Import own scripts
from data_loader import DataLoaderCustom
from network import DAAF


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
        self.tuple_size = configuration['tuple_size']
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
        self.data_loader = DataLoaderCustom(self.dataset, self.data_path, self.fold, self.image_dim, self.batch_size,
                                            self.tuple_size)
        self.train_loader = self.data_loader.get_train_loader()
        self.val_loader = self.data_loader.get_val_loader()

        # Load the pretrained decoder and initialize the model
        print("\tInitializing the model ...")
        self.model = DAAF(self.image_dim, self.feature_dim, self.dropout)
        self.model.to(self.device)
        self.model.requires_grad_(True)

        # Initialize the optimizer and learning scheduler for the attribute VAE
        print("\tInitializing the optimizer and lr scheduler for the attribute VAE...")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(self.first_momentum,
                                                                                                 self.second_momentum))
        self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.scheduler_lr_base,
                                                     max_lr=self.scheduler_lr_max, step_size_up=len(self.train_loader) * 5,
                                                     mode='triangular2', cycle_momentum=False)

        # self.optimizer_content_decoder = torch.optim.Adam(self.model.content_decoder.parameters(),
        #                                                   lr=self.learning_rate, betas=(self.first_momentum, self.second_momentum))
        # self.scheduler_content_decoder = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.scheduler_lr_base,
        #                                                              max_lr=self.scheduler_lr_max,
        #                                                              step_size_up=len(self.train_loader) * 5,
        #                                                              mode='triangular2', cycle_momentum=False)

        self.lambda_reconstruction = 1
        self.lambda_robustness = 1

    def train(self):
        """
        Train the DAAF network.
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
            # self.optimizer_content_decoder.load_state_dict(checkpoint['optimizer_content_decoder'])
            # self.scheduler_content_decoder.load_state_dict(checkpoint['scheduler_content_decoder'])

            self.model.to(self.device)
            self.model.requires_grad_(True)

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

            total_loss, reconstruction_loss, robustness_loss = 0, 0, 0
            mse_input, mse_content, mse_attribute = 0, 0, 0
            ssim_input, ssim_content, ssim_attribute = 0, 0, 0
            psnr_input, psnr_content, psnr_attribute = 0, 0, 0
            consistency_loss_content, mean_cos_dist_content, barlow_loss_content = 0, 0, 0
            # consistency_loss_attribute, mean_cos_dist_attribute, barlow_loss_attribute = 0, 0, 0

            # for i, (X_orig, _, X, X_content_tuples, X_attribute_tuples, _) in enumerate(self.train_loader):
            for i, (X_orig, _, X, X_content_tuples, _) in enumerate(self.train_loader):
                # Zero out the gradients
                self.optimizer.zero_grad()
                # self.optimizer_content_decoder.zero_grad()

                # # Create the barlow batch for the content and attribute encoder
                # X_barlow_content, X_barlow_attribute = [X.clone().to(self.device)], [X.clone().to(self.device)]
                #
                # for X_content_batch, X_attribute_batch in zip(X_content_tuples, X_attribute_tuples):
                #     X_barlow_content.append(X_content_batch.to(self.device))
                #     X_barlow_attribute.append(X_attribute_batch.to(self.device))

                # Create the barlow batch for the content and attribute encoder
                X_barlow_content = [X.clone().to(self.device)]

                for X_content_batch in X_content_tuples:
                    X_barlow_content.append(X_content_batch.to(self.device))

                # Map the next input tensors to the available device
                X_orig = X_orig.to(self.device)
                X = X.to(self.device)

                # Run the input through the model
                # (mse_input_batch, mse_content_batch, mse_attribute_batch),\
                # (ssim_input_batch, ssim_content_batch, ssim_attribute_batch),\
                # (psnr_input_batch, psnr_content_batch, psnr_attribute_batch), \
                # (consistency_loss_content_batch, mean_cos_dist_content_batch, barlow_loss_content_batch),\
                # (consistency_loss_attribute_batch, mean_cos_dist_attribute_batch, barlow_loss_attribute_batch) , _ = \
                #     self.model(X_orig, X, X_barlow_content, X_barlow_attribute)
                (mse_input_batch, mse_content_batch, mse_attribute_batch), \
                (ssim_input_batch, ssim_content_batch, ssim_attribute_batch),\
                (psnr_input_batch, psnr_content_batch, psnr_attribute_batch), \
                (consistency_loss_content_batch, mean_cos_dist_content_batch, barlow_loss_content_batch), _ = \
                    self.model(X_orig, X, X_barlow_content)

                # Create the final losses as a combination of the individual losses
                reconstruction_loss_batch = mse_input_batch + mse_content_batch
                robustness_loss_batch = consistency_loss_content_batch # + barlow_loss_content_batch # + consistency_loss_attribute_batch # + barlow_loss_content_batch + barlow_loss_attribute_batch
                total_loss_batch = self.lambda_reconstruction * reconstruction_loss_batch + self.lambda_robustness * robustness_loss_batch

                # Do backprop of the model loss and of the loss for the content decoder
                # total_loss_batch.backward(retain_graph=True)
                total_loss_batch.backward()
                self.optimizer.step()

                # mse_content_batch.backward()
                # self.optimizer_content_decoder.step()

                # Add everything up for this epoch
                total_loss += total_loss_batch.detach().cpu()
                reconstruction_loss += reconstruction_loss_batch.detach().cpu()
                robustness_loss += robustness_loss_batch.detach().cpu()

                mse_input += mse_input_batch.detach().cpu()
                mse_content += mse_content_batch.detach().cpu()
                mse_attribute += mse_attribute_batch.detach().cpu()
                ssim_input += ssim_input_batch.detach().cpu()
                ssim_content += ssim_content_batch.detach().cpu()
                ssim_attribute += ssim_attribute_batch.detach().cpu()
                psnr_input += psnr_input_batch.detach().cpu()
                psnr_content += psnr_content_batch.detach().cpu()
                psnr_attribute += psnr_attribute_batch.detach().cpu()

                consistency_loss_content += consistency_loss_content_batch.detach().cpu()
                #consistency_loss_attribute += consistency_loss_attribute_batch.detach().cpu()
                mean_cos_dist_content += mean_cos_dist_content_batch
                #mean_cos_dist_attribute += mean_cos_dist_attribute_batch
                barlow_loss_content += barlow_loss_content_batch.detach().cpu()
                #barlow_loss_attribute += barlow_loss_attribute_batch.detach().cpu()

                # Update the progress bar
                pbar_train.update(1)

            # Update the learning rate schedulers
            self.scheduler.step()
            # self.scheduler_content_decoder.step()

            # Average it
            total_loss /= len(self.train_loader)
            reconstruction_loss /= len(self.train_loader)
            robustness_loss /= len(self.train_loader)

            mse_input /= len(self.train_loader)
            mse_content /= len(self.train_loader)
            mse_attribute /= len(self.train_loader)
            ssim_input /= len(self.train_loader)
            ssim_content /= len(self.train_loader)
            ssim_attribute /= len(self.train_loader)
            psnr_input /= len(self.train_loader)
            psnr_content /= len(self.train_loader)
            psnr_attribute /= len(self.train_loader)

            consistency_loss_content /= len(self.train_loader)
            barlow_loss_content /= len(self.train_loader)
            mean_cos_dist_content /= len(self.train_loader)
            #consistency_loss_attribute /= len(self.train_loader)
            #barlow_loss_attribute /= len(self.train_loader)
            #mean_cos_dist_attribute /= len(self.train_loader)

            print(f"\t\t\tTrain Total Loss: {total_loss}")
            print(f"\t\t\tTrain Reconstruction Loss: {reconstruction_loss}")
            print(f"\t\t\tTrain Robustness Loss: {robustness_loss}")
            print(f"\t\t\tTrain MSE Input: {mse_input}")
            print(f"\t\t\tTrain MSE Content: {mse_content}")
            print(f"\t\t\tTrain MSE Attribute: {mse_attribute}")
            print(f"\t\t\tTrain SSIM Input: {ssim_input}")
            print(f"\t\t\tTrain SSIM Content: {ssim_content}")
            print(f"\t\t\tTrain SSIM Attribute: {ssim_attribute}")
            print(f"\t\t\tTrain PSNR Input: {psnr_input}")
            print(f"\t\t\tTrain PSNR Content: {psnr_content}")
            print(f"\t\t\tTrain PSNR Attribute: {psnr_attribute}")
            print(f"\t\t\tTrain Consistency Loss Content: {consistency_loss_content}")
            print(f"\t\t\tTrain Barlow Loss Content: {barlow_loss_content}")
            print(f"\t\t\tTrain Mean Cosine Distance Content: {mean_cos_dist_content}")
            #print(f"\t\t\tTrain Consistency Loss Attribute: {consistency_loss_attribute}")
            #print(f"\t\t\tTrain Barlow Loss Attribute: {barlow_loss_attribute}")
            #print(f"\t\t\tTrain Mean Cosine Distance Attribute: {mean_cos_dist_attribute}")

            # Run the validation
            with torch.no_grad():
                # Set the model to validation
                self.model.eval()

                # Create a progress bar to track the validation
                pbar_val = tqdm(total=len(self.val_loader), bar_format='{l_bar}{bar}', ncols=80, initial=0, position=0,
                                leave=False)
                pbar_val.set_description(f'\t\t\tValidation')

                total_loss_val, reconstruction_loss_val, robustness_loss_val = 0, 0, 0
                mse_input_val, mse_content_val, mse_attribute_val = 0, 0, 0
                ssim_input_val, ssim_content_val, ssim_attribute_val = 0, 0, 0
                psnr_input_val, psnr_content_val, psnr_attribute_val = 0, 0, 0
                consistency_loss_content_val, mean_cos_dist_content_val, barlow_loss_content_val = 0, 0, 0
                #consistency_loss_attribute_val, mean_cos_dist_attribute_val, barlow_loss_attribute_val = 0, 0, 0

                #for i, (X_orig_val, _, X_val, X_content_tuples_val, X_attribute_tuples_val, _) in enumerate(self.val_loader):
                for i, (X_orig_val, _, X_val, X_content_tuples_val, _) in enumerate(self.val_loader):
                    # # Create the barlow batch for the content and attribute encoder
                    # X_barlow_content_val, X_barlow_attribute_val = [X_val.clone().to(self.device)], [X_val.clone().to(self.device)]
                    #
                    # for X_content_batch_val, X_attribute_batch_val in zip(X_content_tuples_val, X_attribute_tuples_val):
                    #     X_barlow_content_val.append(X_content_batch_val.to(self.device))
                    #     X_barlow_attribute_val.append(X_attribute_batch_val.to(self.device))

                    # Create the barlow batch for the content and attribute encoder
                    X_barlow_content_val = [X_val.clone().to(self.device)]
                    for X_content_batch_val in X_content_tuples_val:
                        X_barlow_content_val.append(X_content_batch_val.to(self.device))

                    # Map the next input tensors to the available device
                    X_orig_val = X_orig_val.to(self.device)
                    X_val = X_val.to(self.device)

                    # Run the input through the model
                    # (mse_input_batch_val, mse_content_batch_val, mse_attribute_batch_val), \
                    # (ssim_input_batch_val, ssim_content_batch_val, ssim_attribute_batch_val), \
                    # (psnr_input_batch_val, psnr_content_batch_val, psnr_attribute_batch_val), \
                    # (consistency_loss_content_batch_val, mean_cos_dist_content_batch_val, barlow_loss_content_batch_val), \
                    # (consistency_loss_attribute_batch_val, mean_cos_dist_attribute_batch_val, barlow_loss_attribute_batch_val), _ = \
                    #     self.model(X_orig_val, X_val, X_barlow_content_val, X_barlow_attribute_val)
                    (mse_input_batch_val, mse_content_batch_val, mse_attribute_batch_val), \
                    (ssim_input_batch_val, ssim_content_batch_val, ssim_attribute_batch_val), \
                    (psnr_input_batch_val, psnr_content_batch_val, psnr_attribute_batch_val), \
                    (consistency_loss_content_batch_val, mean_cos_dist_content_batch_val, barlow_loss_content_batch_val), _ = self.model(X_orig_val, X_val, X_barlow_content_val)

                    # Create the final losses as a combination of the individual losses
                    reconstruction_loss_batch_val = mse_input_batch_val + mse_content_batch_val # + mse_attribute_batch_val
                    robustness_loss_batch_val = consistency_loss_content_batch_val # + barlow_loss_content_batch_val # + consistency_loss_attribute_batch_val # + barlow_loss_content_batch_val + barlow_loss_attribute_batch_val
                    total_loss_batch_val = self.lambda_reconstruction * reconstruction_loss_batch_val + self.lambda_robustness * robustness_loss_batch_val

                    # Add everything up for this epoch
                    total_loss_val += total_loss_batch_val.detach().cpu()
                    reconstruction_loss_val += reconstruction_loss_batch_val.detach().cpu()
                    robustness_loss_val += robustness_loss_batch_val.detach().cpu()

                    mse_input_val += mse_input_batch_val.detach().cpu()
                    mse_content_val += mse_content_batch_val.detach().cpu()
                    mse_attribute_val += mse_attribute_batch_val.detach().cpu()
                    ssim_input_val += ssim_input_batch_val.detach().cpu()
                    ssim_content_val += ssim_content_batch_val.detach().cpu()
                    ssim_attribute_val += ssim_attribute_batch_val.detach().cpu()
                    psnr_input_val += psnr_input_batch_val.detach().cpu()
                    psnr_content_val += psnr_content_batch_val.detach().cpu()
                    psnr_attribute_val += psnr_attribute_batch_val.detach().cpu()

                    consistency_loss_content_val += consistency_loss_content_batch_val.detach().cpu()
                    #consistency_loss_attribute_val += consistency_loss_attribute_batch_val.detach().cpu()
                    mean_cos_dist_content_val += mean_cos_dist_content_batch_val
                    #mean_cos_dist_attribute_val += mean_cos_dist_attribute_batch_val
                    barlow_loss_content_val += barlow_loss_content_batch_val.detach().cpu()
                    #barlow_loss_attribute_val += barlow_loss_attribute_batch_val.detach().cpu()

                    # Update the progress bar
                    pbar_val.update(1)

                # Average it
                total_loss_val /= len(self.val_loader)
                reconstruction_loss_val /= len(self.val_loader)
                robustness_loss_val /= len(self.val_loader)

                mse_input_val /= len(self.val_loader)
                mse_content_val /= len(self.val_loader)
                mse_attribute_val /= len(self.val_loader)
                ssim_input_val /= len(self.val_loader)
                ssim_content_val /= len(self.val_loader)
                ssim_attribute_val /= len(self.val_loader)
                psnr_input_val /= len(self.val_loader)
                psnr_content_val /= len(self.val_loader)
                psnr_attribute_val /= len(self.val_loader)

                consistency_loss_content_val /= len(self.val_loader)
                barlow_loss_content_val /= len(self.val_loader)
                mean_cos_dist_content_val /= len(self.val_loader)
                #consistency_loss_attribute_val /= len(self.val_loader)
                #barlow_loss_attribute_val /= len(self.val_loader)
                #mean_cos_dist_attribute_val /= len(self.val_loader)

                print(f"\t\t\tValidation Total Loss: {total_loss_val}")
                print(f"\t\t\tValidation Reconstruction Loss: {reconstruction_loss_val}")
                print(f"\t\t\tValidation Robustness Loss: {robustness_loss_val}")
                print(f"\t\t\tValidation MSE Input: {mse_input_val}")
                print(f"\t\t\tValidation MSE Content: {mse_content_val}")
                print(f"\t\t\tValidation MSE Attribute: {mse_attribute_val}")
                print(f"\t\t\tValidation SSIM Input: {ssim_input_val}")
                print(f"\t\t\tValidation SSIM Content: {ssim_content_val}")
                print(f"\t\t\tValidation SSIM Attribute: {ssim_attribute_val}")
                print(f"\t\t\tValidation PSNR Input: {psnr_input_val}")
                print(f"\t\t\tValidation PSNR Content: {psnr_content_val}")
                print(f"\t\t\tValidation PSNR Attribute: {psnr_attribute_val}")
                print(f"\t\t\tValidation Consistency Loss Content: {consistency_loss_content_val}")
                print(f"\t\t\tValidation Barlow Loss Content: {barlow_loss_content_val}")
                print(f"\t\t\tValidation Mean Cosine Distance Content: {mean_cos_dist_content_val}")
                #print(f"\t\t\tValidation Consistency Loss Attribute: {consistency_loss_attribute_val}")
                #print(f"\t\t\tValidation Barlow Loss Attribute: {barlow_loss_attribute_val}")
                #print(f"\t\t\tValidation Mean Cosine Distance Attribute: {mean_cos_dist_attribute_val}")

            # Log training and validation loss to wandb
            print(f"\t\t\tLogging to weights&biases...")
            wandb.log(
                {
                    "Train Total Loss": total_loss,
                    "Train Reconstruction Loss": reconstruction_loss,
                    "Train Robustness Loss": robustness_loss,
                    "Train MSE Input": mse_input,
                    "Train MSE Content": mse_content,
                    "Train MSE Attribute": mse_attribute,
                    "Train SSIM Input": ssim_input,
                    "Train SSIM Content": ssim_content,
                    "Train SSIM Attribute": ssim_attribute,
                    "Train PSNR Input": psnr_input,
                    "Train PSNR Content": psnr_content,
                    "Train PSNR Attribute": psnr_attribute,
                    "Train Consistency Loss Content": consistency_loss_content,
                    "Train Barlow Loss Content": barlow_loss_content,
                    "Train Mean Cosine Distance Content": mean_cos_dist_content,
                    #"Train Consistency Loss Attribute": consistency_loss_attribute,
                    #"Train Barlow Loss Attribute": barlow_loss_attribute,
                    #"Train Mean Cosine Distance Attribute": mean_cos_dist_attribute,

                    "Validation Total Loss": total_loss_val,
                    "Validation Reconstruction Loss": reconstruction_loss_val,
                    "Validation Robustness Loss": robustness_loss_val,
                    "Validation MSE Input": mse_input_val,
                    "Validation MSE Content": mse_content_val,
                    "Validation MSE Attribute": mse_attribute_val,
                    "Validation SSIM Input": ssim_input_val,
                    "Validation SSIM Content": ssim_content_val,
                    "Validation SSIM Attribute": ssim_attribute_val,
                    "Validation PSNR Input": psnr_input_val,
                    "Validation PSNR Content": psnr_content_val,
                    "Validation PSNR Attribute": psnr_attribute_val,
                    "Validation Consistency Loss Content": consistency_loss_content_val,
                    "Validation Barlow Loss Content": barlow_loss_content_val,
                    "Validation Mean Cosine Distance Content": mean_cos_dist_content_val,
                    #"Validation Consistency Loss Attribute": consistency_loss_attribute_val,
                    #"Validation Barlow Loss Attribute": barlow_loss_attribute_val,
                    #"Validation Mean Cosine Distance Attribute": mean_cos_dist_attribute_val,
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
                'content_encoder': self.model.content_encoder.state_dict(),
                'attribute_encoder': self.model.attribute_encoder.state_dict(),
                'decoder': self.model.decoder.state_dict(),
                'content_decoder': self.model.content_decoder.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                # 'optimizer_content_decoder': self.optimizer_content_decoder.state_dict(),
                # 'scheduler_content_decoder': self.scheduler_content_decoder.state_dict(),
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
            wandb.init(project="train_daaf", id=training_configuration["resume_training"]["wandb_id"], resume="allow", config=training_configuration)

        else:
            # Initialize a weights & biases project for a training run with the given training configuration
            wandb.init(project="train_daaf", config=training_configuration)

    # Run the hyperparameter optimization run
    else:
        # Initialize a weights & biases project for a hyperparameter optimization run
        wandb.init(project="sweep_train_daaf")

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
            sweep_id = wandb.sweep(sweep=config['hyperparameter_tuning'], project="sweep_train_daaf")

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