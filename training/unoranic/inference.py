"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
unORANIC: Unsupervised orthogonalization of anatomy and image-characteristic features

@description:
Inference of unORANIC.

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
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import own scripts
from data_loader import DataLoaderCustom
from network import DAAF


class Inference:
    def __init__(self, configuration):
        """
        Initialize the trainer.
        """

        # Read out the parameters of the training run
        self.dataset = configuration['dataset']
        self.data_path = Path(configuration['data_path'])
        self.input_path = Path(configuration['input_path'])
        self.output_path = Path(configuration['output_path'])
        self.fold = configuration['fold']
        self.seed = configuration['seed']
        self.device = torch.device(configuration['device'])

        # Read out the hyperparameters of the training run
        self.image_dim = (configuration['image_dim']['height'], configuration['image_dim']['width'], configuration['image_dim']['channel'])
        self.feature_dim = configuration['feature_dim']
        self.dropout = configuration['dropout']

        # Create the path to where the output shall be stored and initialize the logger
        self.output_path = self.output_path / f"{self.feature_dim}" / self.dataset / f"fold_{self.fold}"
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Enable the use of cudnn
        torch.backends.cudnn.benchmark = True

        # Initialize the dataloader
        print("\tInitializing the dataloader...")
        self.data_loader = DataLoaderCustom(self.dataset, self.data_path, self.fold, self.image_dim, 1, 1)
        self.train_loader = self.data_loader.get_train_loader()
        self.val_loader = self.data_loader.get_val_loader()
        self.test_loader = self.data_loader.get_test_loader()

        # Initialize the model
        print("\tInitializing the model ...")
        # Load the trained daaf model
        self.checkpoint_file = self.input_path / f"{self.feature_dim}" / self.dataset / f"fold_{self.fold}" / "checkpoint_final.pt"
        self.checkpoint = torch.load(self.checkpoint_file, map_location='cpu')

        self.model = DAAF(self.image_dim, self.feature_dim, self.dropout)
        self.model.load_state_dict(self.checkpoint['model'])
        self.model.to(self.device)
        self.model.requires_grad_(False)

        # Initialize the mse loss
        print("\tInitialize the loss criterion...")
        self.mse = nn.MSELoss()

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

            # Create the output path for the current set
            output_path = self.output_path / set
            output_path.mkdir(parents=True, exist_ok=True)

            nr_samples = len(set_loader)
            print(f"\t\tRun inference for {nr_samples} samples...")

            # Create a progress bar to track the training
            pbar_set = tqdm(total=nr_samples, bar_format='{l_bar}{bar}', ncols=80, initial=0, position=0, leave=False)
            pbar_set.set_description(f'\t\t\tProgress')

            mse_input, mse_content, mse_attribute = 0, 0, 0
            ssim_input, ssim_content, ssim_attribute = 0, 0, 0
            psnr_input, psnr_content, psnr_attribute = 0, 0, 0
            with torch.no_grad():
                # for i, (X_orig, _, X, _, _, X_name) in enumerate(set_loader):
                for i, (X_orig, _, X, _, X_name) in enumerate(set_loader):
                    # Set the model to eval
                    self.model.eval()

                    # Send the test sample to the GPU
                    X_orig = X_orig.to(self.device)
                    X = X.to(self.device)

                    # Run the first input sample through the model
                    # (mse_input_sample, mse_content_sample, mse_attribute_sample), \
                    # (ssim_input_sample, ssim_content_sample, ssim_attribute_sample), \
                    # (psnr_input_sample, psnr_content_sample, psnr_attribute_sample), \
                    # _, _, (X_hat, X_hat_content, X_hat_attribute, X_corruption) = \
                    #     self.model(X_orig, X, [], [])

                    (mse_input_sample, mse_content_sample, mse_attribute_sample), \
                    (ssim_input_sample, ssim_content_sample, ssim_attribute_sample), \
                    (psnr_input_sample, psnr_content_sample, psnr_attribute_sample), \
                    _, (X_hat, X_hat_content, X_hat_attribute, X_corruption) = \
                        self.model(X_orig, X, [])

                    # Add everything up
                    mse_input += mse_input_sample.detach().cpu()
                    mse_content += mse_content_sample.detach().cpu()
                    mse_attribute += mse_attribute_sample.detach().cpu()

                    ssim_input += ssim_input_sample.detach().cpu()
                    ssim_content += ssim_content_sample.detach().cpu()
                    ssim_attribute += ssim_attribute_sample.detach().cpu()

                    psnr_input += psnr_input_sample.detach().cpu()
                    psnr_content += psnr_content_sample.detach().cpu()
                    psnr_attribute += psnr_attribute_sample.detach().cpu()

                    # Save the current x-x_hat-pair for visual comparison
                    # self.save_img_and_reconstructed_image(X_orig.cpu(), X.cpu(), X_hat.cpu(), X_hat_content.cpu(), X_hat_attribute.cpu(), X_corruption, X_name, output_path)

                    print(f"\t\t\tSample: [{i}/{nr_samples}]")
                    print(f"\t\t\t\tMSE Input: {mse_input_sample}")
                    print(f"\t\t\t\tMSE Content: {mse_content_sample}")
                    print(f"\t\t\t\tMSE Attribute: {mse_attribute_sample}")
                    print(f"\t\t\t\tSSIM Input: {ssim_input_sample}")
                    print(f"\t\t\t\tSSIM Content: {ssim_content_sample}")
                    print(f"\t\t\t\tSSIM Attribute: {ssim_attribute_sample}")
                    print(f"\t\t\t\tPSNR Input: {psnr_input_sample}")
                    print(f"\t\t\t\tPSNR Content: {psnr_content_sample}")
                    print(f"\t\t\t\tPSNR Attribute: {psnr_attribute_sample}")

                    # Log training and validation loss to wandb
                    wandb.log(
                        {
                            "MSE Input": mse_input_sample,
                            "MSE Content": mse_content_sample,
                            "MSE Attribute": mse_attribute_sample,
                            "SSIM Input": ssim_input_sample,
                            "SSIM Content": ssim_content_sample,
                            "SSIM Attribute": ssim_attribute_sample,
                            "PSNR Input": psnr_input_sample,
                            "PSNR Content": psnr_content_sample,
                            "PSNR Attribute": psnr_attribute_sample,
                        })

                # Average it
                mse_input /= nr_samples
                mse_content /= nr_samples
                mse_attribute /= nr_samples

                ssim_input /= nr_samples
                ssim_content /= nr_samples
                ssim_attribute /= nr_samples

                psnr_input /= nr_samples
                psnr_content /= nr_samples
                psnr_attribute /= nr_samples

                print(f"\t\t\tAverage MSE Input: {mse_input}")
                print(f"\t\t\tAverage MSE Content: {mse_content}")
                print(f"\t\t\tAverage MSE Attribute: {mse_attribute}")
                print(f"\t\t\tAverage SSIM Input: {ssim_input}")
                print(f"\t\t\tAverage SSIM Content: {ssim_content}")
                print(f"\t\t\tAverage SSIM Attribute: {ssim_attribute}")
                print(f"\t\t\tAverage PSNR Input: {psnr_input}")
                print(f"\t\t\tAverage PSNR Content: {psnr_content}")
                print(f"\t\t\tAverage PSNR Attribute: {psnr_attribute}")

                # Log the metrics' averages
                wandb.log(
                    {
                        "Average MSE Input": mse_input,
                        "Average MSE Content": mse_content,
                        "Average MSE Attribute": mse_attribute,
                        "Average SSIM Input": ssim_input,
                        "Average SSIM Content": ssim_content,
                        "Average SSIM Attribute": ssim_attribute,
                        "Average PSNR Input": psnr_input,
                        "Average PSNR Content": psnr_content,
                        "Average PSNR Attribute": psnr_attribute,
                    })

            # Stop the time
            end_time_set = time.time()
            hours_set, minutes_set, seconds_set = self.calculate_passed_time(start_time_set, end_time_set)

            print("\t\t\tElapsed time for set: {:0>2}:{:0>2}:{:05.2f}".format(hours_set, minutes_set, seconds_set))

        # Stop the time
        end_time = time.time()
        hours, minutes, seconds = self.calculate_passed_time(start_time, end_time)

        print("\t\t\tElapsed time: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))

    def save_img_and_reconstructed_image(self, X_orig, X, X_hat, X_hat_content, X_hat_attribute, X_corruption, X_name, output_path):
        """
        Save the processed input image together with the reconstructed image.

        :param X_orig: Original image
        :param X: Input image
        :param X_hat: Reconstructed image.
        :param X_hat_content: content reconstructed image.
        :param X_hat_attribute: attribute reconstructed image.
        :param X_corruption: Corruption applied to the original image
        :param X_name: Name of the sample.
        :param output_path: Where the images shall be stored.
        """

        # Get the sample name
        sample_name = X_name[0]
        channel_dim = X_orig.shape[1]

        # Remove the batch dimension and reorder the and height, width and channel dimensions
        X_orig = X_orig.squeeze(0).permute(1, 2, 0).numpy()
        X = X.squeeze(0).permute(1, 2, 0).numpy()
        X_hat = X_hat.squeeze(0).permute(1, 2, 0).numpy()
        X_hat_content = X_hat_content.squeeze(0).permute(1, 2, 0).numpy()
        X_hat_attribute = X_hat_attribute.squeeze(0).permute(1, 2, 0).numpy()
        X_corruption = X_corruption.squeeze(0).permute(1, 2, 0).numpy()

        # Normalize the pixel values in the range 0 to 255
        X_orig = cv2.normalize(X_orig, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        X = cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        X_hat = cv2.normalize(X_hat, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        X_hat_content = cv2.normalize(X_hat_content, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        X_hat_attribute = cv2.normalize(X_hat_attribute, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        X_corruption = cv2.normalize(X_corruption, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Save the images PNG as files
        cv2.imwrite(str(output_path / f"{sample_name}_orig.png"), X_orig)
        cv2.imwrite(str(output_path / f"{sample_name}_input.png"), X)
        cv2.imwrite(str(output_path / f"{sample_name}_recons.png"), X_hat)
        cv2.imwrite(str(output_path / f"{sample_name}_recons_content.png"), X_hat_content)
        cv2.imwrite(str(output_path / f"{sample_name}_recons_attribute.png"), X_hat_attribute)
        cv2.imwrite(str(output_path / f"{sample_name}_corruption.png"), X_corruption)

        # Save all images combined in one figure
        # Create the figure
        fig, axes = plt.subplots(nrows=2, ncols=3)

        # Plot the images
        if channel_dim == 1:
            axes[0, 0].imshow(X_orig, cmap='gray')
            axes[0, 0].set_title("Original", loc='center')
            axes[0, 0].axis("off")

            axes[0, 1].imshow(X, cmap='gray')
            axes[0, 1].set_title("Variant", loc='center')
            axes[0, 1].axis("off")

            axes[0, 2].imshow(X_corruption, cmap='gray')
            axes[0, 2].set_title("Attribute Difference", loc='center')
            axes[0, 2].axis("off")

            axes[1, 0].imshow(X_hat_content, cmap='gray')
            axes[1, 0].set_title("Content\nReconstruction", loc='center')
            axes[1, 0].axis("off")

            axes[1, 1].imshow(X_hat, cmap='gray')
            axes[1, 1].set_title("Image\nReconstruction", loc='center')
            axes[1, 1].axis("off")

            axes[1, 2].imshow(X_hat_attribute, cmap='gray')
            axes[1, 2].set_title("Attribute\nReconstruction", loc='center')
            axes[1, 2].axis("off")
        else:
            axes[0, 0].imshow(X_orig)
            axes[0, 0].set_title("Original", loc='center')
            axes[0, 0].axis("off")

            axes[0, 1].imshow(X)
            axes[0, 1].set_title("Variant", loc='center')
            axes[0, 1].axis("off")

            axes[0, 2].imshow(X_corruption)
            axes[0, 2].set_title("Attribute Difference", loc='center')
            axes[0, 2].axis("off")

            axes[1, 0].imshow(X_hat_content)
            axes[1, 0].set_title("Content\nReconstruction", loc='center')
            axes[1, 0].axis("off")

            axes[1, 1].imshow(X_hat)
            axes[1, 1].set_title("Image\nReconstruction", loc='center')
            axes[1, 1].axis("off")

            axes[1, 2].imshow(X_hat_attribute)
            axes[1, 2].set_title("Attribute\nReconstruction", loc='center')
            axes[1, 2].axis("off")

        # Save and close the figure
        fig.savefig(output_path / f"{sample_name}_all.png", dpi=300)
        plt.close(fig)


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
    wandb.init(project="train_daaf_inference", config=configuration)

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
    parser.add_argument("--fold", required=False, default='0', type=str, help="Which fold to use as test fold.")

    args = parser.parse_args()
    config_file = args.config_file
    fold = args.fold

    # Load the default and the hyperparameters and run the training
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        config['inference']['fold'] = fold

        main(config['inference'])

    # Finished code
    end_time = time.time()

    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
