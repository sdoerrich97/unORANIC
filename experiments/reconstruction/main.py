"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
unORANIC: Unsupervised orthogonalization of anatomy and image-characteristic features

@description:
Main Script of the reconstruction experiment.

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
from imagecorruptions import get_corruption_names
import cv2
import matplotlib.pyplot as plt

# Import own scripts
from data_loader import DataLoaderCustom
from network import DAAF


class Reconstruction:
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
        self.image_dim = (configuration['image_dim']['height'], configuration['image_dim']['width'], configuration['image_dim']['channel'])
        self.feature_dim = configuration['feature_dim']
        self.dropout = configuration['dropout']

        # Create the path to where the output shall be stored and initialize the logger
        self.input_path = self.input_path / self.architecture
        self.output_path = self.output_path / f"{self.feature_dim}" / self.dataset / f"fold_{self.fold}" / self.architecture
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Enable the use of cudnn
        torch.backends.cudnn.benchmark = True

        # Initialize the model
        print("\tInitializing the model ...")
        # Load the trained daaf model
        self.checkpoint_file = self.input_path / f"{self.feature_dim}" / self.dataset / f"fold_{self.fold}" / "checkpoint_final.pt"
        self.checkpoint = torch.load(self.checkpoint_file, map_location='cpu')

        self.model = DAAF(self.architecture, self.image_dim, self.feature_dim, self.dropout)

        if self.architecture == 'AE':
            self.model.encoder.model.load_state_dict(self.checkpoint['encoder'])
            self.model.decoder.model.load_state_dict(self.checkpoint['decoder'])

        else:
            self.model.load_state_dict(self.checkpoint['model'], strict=False)

        self.model.to(self.device)
        self.model.requires_grad_(False)

    def run_reconstruction_experiment(self):
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

        # Iterate through all corruptions and test the robustness of the model
        # for corruption in get_corruption_names():
        #     # Skip the broken corruptions
        #     if corruption == 'glass_blur':
        #         continue
        #
        #     # Iterate through all severity degrees and test the robustness of the model
        #     for severity in range(1, 6):
        #         data_loader = DataLoaderCustom(self.dataset, self.data_path, self.image_dim, corruption, severity)
        #         test_loader = data_loader.get_test_loader()
        #
        #         # Create the output path for the corruption-severity combination
        #         output_path = self.output_path / corruption / f"severity-{severity}"
        #         output_path.mkdir(parents=True, exist_ok=True)
        #
        #         # Get all samples for the current combination of corruption and severity
        #         X_orig, X, X_name = next(iter(test_loader))
        #
        #         # Push the corrupted samples to the GPU
        #         X_orig, X = X_orig.to(self.device), X.to(self.device)
        #
        #         if self.architecture == 'AE':
        #             # Pass the input through the model
        #             (mse_recons_to_orig, mse_recons_to_input, mse_attribute), (ssim_recons_to_orig, ssim_recons_to_input, ssim_attribute), \
        #             (psnr_recons_to_orig, psnr_recons_to_input, psnr_attribute), (X_hat, X_corruption) = self.model(X_orig, X)
        #
        #             # Average the metrics
        #             mse_recons_to_orig /= len(test_loader)
        #             mse_recons_to_input /= len(test_loader)
        #             mse_attribute /= len(test_loader)
        #             ssim_recons_to_orig /= len(test_loader)
        #             ssim_recons_to_input /= len(test_loader)
        #             ssim_attribute /= len(test_loader)
        #             psnr_recons_to_orig /= len(test_loader)
        #             psnr_recons_to_input /= len(test_loader)
        #             psnr_attribute /= len(test_loader)
        #
        #             print(f"\t\tCorruption: {corruption} ({severity})")
        #             print(f"\t\t\tMSE Reconstruction to Original: {mse_recons_to_orig}")
        #             print(f"\t\t\tMSE Reconstruction to Corrupted Input: {mse_recons_to_input}")
        #             print(f"\t\t\tMSE Reconstruction to Applied Corruption: {mse_attribute}")
        #             print(f"\t\t\tSSIM Reconstruction to Original: {ssim_recons_to_orig}")
        #             print(f"\t\t\tSSIM Reconstruction to Corrupted Input: {ssim_recons_to_input}")
        #             print(f"\t\t\tSSIM Reconstruction to Applied Corruption: {ssim_attribute}")
        #             print(f"\t\t\tPSNR Reconstruction to Original: {psnr_recons_to_orig}")
        #             print(f"\t\t\tPSNR Reconstruction to Corrupted Input: {psnr_recons_to_input}")
        #             print(f"\t\t\tPSNR Reconstruction to Applied Corruption: {psnr_attribute}")
        #
        #             # Log the metrics
        #             wandb.log(
        #                 {
        #                     "MSE Reconstruction to Original": mse_recons_to_orig,
        #                     "MSE Reconstruction to Corrupted Input": mse_recons_to_input,
        #                     "MSE Attribute to Applied Corruption": mse_attribute,
        #                     "SSIM Reconstruction to Original": ssim_recons_to_orig,
        #                     "SSIM Reconstruction to Corrupted Input": ssim_recons_to_input,
        #                     "SSIM Attribute to Applied Corruption": ssim_attribute,
        #                     "PSNR Reconstruction to Original": psnr_recons_to_orig,
        #                     "PSNR Reconstruction to Corrupted Input": psnr_recons_to_input,
        #                     "PSNR Attribute to Applied Corruption": psnr_attribute
        #                 })
        #
        #             # Save one example of the sample set together with all its corrupted variants
        #             self.save_img_and_reconstructions_for_AE(X_orig.cpu(), X.cpu(), X_hat.cpu(), X_corruption.cpu(), X_name, output_path)
        #
        #         else:
        #             (mse_recons_to_orig, mse_content_to_orig, mse_recons_to_input, mse_content_to_input, mse_attribute), \
        #             (ssim_recons_to_orig, ssim_content_to_orig, ssim_recons_to_input, ssim_content_to_input, ssim_attribute), \
        #             (psnr_recons_to_orig, psnr_content_to_orig, psnr_recons_to_input, psnr_content_to_input, psnr_attribute), \
        #             (X_hat, X_hat_content, X_hat_attribute, X_corruption) = \
        #                 self.model(X_orig, X)
        #
        #             # Average the metrics
        #             mse_recons_to_orig /= len(test_loader)
        #             mse_content_to_orig /= len(test_loader)
        #             mse_recons_to_input /= len(test_loader)
        #             mse_content_to_input /= len(test_loader)
        #             mse_attribute /= len(test_loader)
        #
        #             ssim_recons_to_orig /= len(test_loader)
        #             ssim_content_to_orig /= len(test_loader)
        #             ssim_recons_to_input /= len(test_loader)
        #             ssim_content_to_input /= len(test_loader)
        #             ssim_attribute /= len(test_loader)
        #
        #             psnr_recons_to_orig /= len(test_loader)
        #             psnr_content_to_orig /= len(test_loader)
        #             psnr_recons_to_input /= len(test_loader)
        #             psnr_content_to_input /= len(test_loader)
        #             psnr_attribute /= len(test_loader)
        #
        #             print(f"\t\tCorruption: {corruption} ({severity})")
        #             print(f"\t\t\tMSE Reconstruction to Original: {mse_recons_to_orig}")
        #             print(f"\t\t\tMSE Content to Original: {mse_content_to_orig}")
        #             print(f"\t\t\tMSE Reconstruction to Corrupted Input: {mse_recons_to_input}")
        #             print(f"\t\t\tMSE Content to Corrupted Input: {mse_content_to_input}")
        #             print(f"\t\t\tMSE Attribute to Applied Corruption: {mse_attribute}")
        #
        #             print(f"\t\t\tSSIM Reconstruction to Original: {ssim_recons_to_orig}")
        #             print(f"\t\t\tSSIM Content to Original: {ssim_content_to_orig}")
        #             print(f"\t\t\tSSIM Reconstruction to Corrupted Input: {ssim_recons_to_input}")
        #             print(f"\t\t\tSSIM Content to Corrupted Input: {ssim_content_to_input}")
        #             print(f"\t\t\tSSIM Attribute to Applied Corruption: {ssim_attribute}")
        #
        #             print(f"\t\t\tPSNR Reconstruction to Original: {psnr_recons_to_orig}")
        #             print(f"\t\t\tPSNR Content to Original: {psnr_content_to_orig}")
        #             print(f"\t\t\tPSNR Reconstruction to Corrupted Input: {psnr_recons_to_input}")
        #             print(f"\t\t\tPSNR Content to Corrupted Input: {psnr_content_to_input}")
        #             print(f"\t\t\tPSNR Attribute to Applied Corruption: {psnr_attribute}")
        #
        #             # Log the metrics
        #             wandb.log(
        #                 {
        #                     "MSE Reconstruction to Original": mse_recons_to_orig,
        #                     "MSE Content to Original": mse_content_to_orig,
        #                     "MSE Reconstruction to Corrupted Input": mse_recons_to_input,
        #                     "MSE Content to Corrupted Input": mse_content_to_input,
        #                     "MSE Attribute to Applied Corruption": mse_attribute,
        #
        #                     "SSIM Reconstruction to Original": ssim_recons_to_orig,
        #                     "SSIM Content to Original": ssim_content_to_orig,
        #                     "SSIM Reconstruction to Corrupted Input": ssim_recons_to_input,
        #                     "SSIM Content to Corrupted Input": ssim_content_to_input,
        #                     "SSIM Attribute to Applied Corruption": ssim_attribute,
        #
        #                     "PSNR Reconstruction to Original": psnr_recons_to_orig,
        #                     "PSNR Content to Original": psnr_content_to_orig,
        #                     "PSNR Reconstruction to Corrupted Input": psnr_recons_to_input,
        #                     "PSNR Content to Corrupted Input": psnr_content_to_input,
        #                     "PSNR Attribute to Applied Corruption": psnr_attribute
        #                 })
        #
        #             # Save one example of the sample set together with all its corrupted variants
        #             self.save_img_and_reconstructions_for_DCAF(X_orig.cpu(), X.cpu(), X_hat.cpu(), X_hat_content.cpu(),
        #                                                        X_hat_attribute.cpu(), X_corruption.cpu(), X_name,
        #                                                        output_path)

        corruptions = ['PixelDropout', 'GaussianBlur', 'ColorJitter', 'Downscale', 'GaussNoise', 'InvertImg',
                       'MotionBlur', 'MultiplicativeNoise', 'RandomBrightnessContrast', 'RandomGamma', 'Solarize',
                       'Sharpen']

        for corruption in corruptions:
            data_loader = DataLoaderCustom(self.dataset, self.data_path, self.image_dim, corruption, self.seed)
            test_loader = data_loader.get_test_loader()

            # Create the output path for the corruption-severity combination
            output_path = self.output_path / corruption
            output_path.mkdir(parents=True, exist_ok=True)

            # Get all samples for the current combination of corruption and severity
            X_orig, X, X_name = next(iter(test_loader))

            # Push the corrupted samples to the GPU
            X_orig, X = X_orig.to(self.device), X.to(self.device)

            if self.architecture == 'AE':
                # Pass the input through the model
                (mse_recons_to_orig, mse_recons_to_input, mse_attribute), (ssim_recons_to_orig, ssim_recons_to_input, ssim_attribute), \
                (psnr_recons_to_orig, psnr_recons_to_input, psnr_attribute), (X_hat, X_corruption) = self.model(X_orig, X)

                # Average the metrics
                mse_recons_to_orig /= len(test_loader)
                mse_recons_to_input /= len(test_loader)
                mse_attribute /= len(test_loader)
                ssim_recons_to_orig /= len(test_loader)
                ssim_recons_to_input /= len(test_loader)
                ssim_attribute /= len(test_loader)
                psnr_recons_to_orig /= len(test_loader)
                psnr_recons_to_input /= len(test_loader)
                psnr_attribute /= len(test_loader)

                print(f"\t\tCorruption: {corruption}")
                print(f"\t\t\tMSE Reconstruction to Original: {mse_recons_to_orig}")
                print(f"\t\t\tMSE Reconstruction to Corrupted Input: {mse_recons_to_input}")
                print(f"\t\t\tMSE Reconstruction to Applied Corruption: {mse_attribute}")
                print(f"\t\t\tSSIM Reconstruction to Original: {ssim_recons_to_orig}")
                print(f"\t\t\tSSIM Reconstruction to Corrupted Input: {ssim_recons_to_input}")
                print(f"\t\t\tSSIM Reconstruction to Applied Corruption: {ssim_attribute}")
                print(f"\t\t\tPSNR Reconstruction to Original: {psnr_recons_to_orig}")
                print(f"\t\t\tPSNR Reconstruction to Corrupted Input: {psnr_recons_to_input}")
                print(f"\t\t\tPSNR Reconstruction to Applied Corruption: {psnr_attribute}")

                # Log the metrics
                wandb.log(
                    {
                        "MSE Reconstruction to Original": mse_recons_to_orig,
                        "MSE Reconstruction to Corrupted Input": mse_recons_to_input,
                        "MSE Attribute to Applied Corruption": mse_attribute,
                        "SSIM Reconstruction to Original": ssim_recons_to_orig,
                        "SSIM Reconstruction to Corrupted Input": ssim_recons_to_input,
                        "SSIM Attribute to Applied Corruption": ssim_attribute,
                        "PSNR Reconstruction to Original": psnr_recons_to_orig,
                        "PSNR Reconstruction to Corrupted Input": psnr_recons_to_input,
                        "PSNR Attribute to Applied Corruption": psnr_attribute
                    })

                # Save one example of the sample set together with all its corrupted variants
                self.save_img_and_reconstructions_for_AE(X_orig.cpu(), X.cpu(), X_hat.cpu(), X_corruption.cpu(), X_name, output_path)

            else:
                (mse_corruption_to_orig, ssim_corruption_to_orig, psnr_corruption_to_orig), \
                (mse_recons_to_orig, mse_content_to_orig, mse_recons_to_input, mse_content_to_input, mse_attribute), \
                (ssim_recons_to_orig, ssim_content_to_orig, ssim_recons_to_input, ssim_content_to_input, ssim_attribute), \
                (psnr_recons_to_orig, psnr_content_to_orig, psnr_recons_to_input, psnr_content_to_input, psnr_attribute), \
                (X_hat, X_hat_content, X_hat_attribute, X_corruption) = \
                    self.model(X_orig, X)

                # Average the metrics
                mse_corruption_to_orig /= len(test_loader)
                ssim_corruption_to_orig /= len(test_loader)
                psnr_corruption_to_orig /= len(test_loader)


                mse_recons_to_orig /= len(test_loader)
                mse_content_to_orig /= len(test_loader)
                mse_recons_to_input /= len(test_loader)
                mse_content_to_input /= len(test_loader)
                mse_attribute /= len(test_loader)

                ssim_recons_to_orig /= len(test_loader)
                ssim_content_to_orig /= len(test_loader)
                ssim_recons_to_input /= len(test_loader)
                ssim_content_to_input /= len(test_loader)
                ssim_attribute /= len(test_loader)

                psnr_recons_to_orig /= len(test_loader)
                psnr_content_to_orig /= len(test_loader)
                psnr_recons_to_input /= len(test_loader)
                psnr_content_to_input /= len(test_loader)
                psnr_attribute /= len(test_loader)

                print(f"\t\tCorruption: {corruption}")
                print(f"\t\t\tMSE Corrupted Input to Original: {mse_corruption_to_orig}")
                print(f"\t\t\tSSIM Corrupted Input to Original: {ssim_corruption_to_orig}")
                print(f"\t\t\tPSNR Corrupted Input to Original: {psnr_corruption_to_orig}")

                print(f"\t\t\tMSE Reconstruction to Original: {mse_recons_to_orig}")
                print(f"\t\t\tMSE Content to Original: {mse_content_to_orig}")
                print(f"\t\t\tMSE Reconstruction to Corrupted Input: {mse_recons_to_input}")
                print(f"\t\t\tMSE Content to Corrupted Input: {mse_content_to_input}")
                print(f"\t\t\tMSE Attribute to Applied Corruption: {mse_attribute}")

                print(f"\t\t\tSSIM Reconstruction to Original: {ssim_recons_to_orig}")
                print(f"\t\t\tSSIM Content to Original: {ssim_content_to_orig}")
                print(f"\t\t\tSSIM Reconstruction to Corrupted Input: {ssim_recons_to_input}")
                print(f"\t\t\tSSIM Content to Corrupted Input: {ssim_content_to_input}")
                print(f"\t\t\tSSIM Attribute to Applied Corruption: {ssim_attribute}")

                print(f"\t\t\tPSNR Reconstruction to Original: {psnr_recons_to_orig}")
                print(f"\t\t\tPSNR Content to Original: {psnr_content_to_orig}")
                print(f"\t\t\tPSNR Reconstruction to Corrupted Input: {psnr_recons_to_input}")
                print(f"\t\t\tPSNR Content to Corrupted Input: {psnr_content_to_input}")
                print(f"\t\t\tPSNR Attribute to Applied Corruption: {psnr_attribute}")

                # Log the metrics
                wandb.log(
                    {
                        "MSE Corrupted Input to Original": mse_corruption_to_orig,
                        "SSIM Corrupted Input to Original": ssim_corruption_to_orig,
                        "PSNR Corrupted Input to Original": psnr_corruption_to_orig,

                        "MSE Reconstruction to Original": mse_recons_to_orig,
                        "MSE Content to Original": mse_content_to_orig,
                        "MSE Reconstruction to Corrupted Input": mse_recons_to_input,
                        "MSE Content to Corrupted Input": mse_content_to_input,
                        "MSE Attribute to Applied Corruption": mse_attribute,

                        "SSIM Reconstruction to Original": ssim_recons_to_orig,
                        "SSIM Content to Original": ssim_content_to_orig,
                        "SSIM Reconstruction to Corrupted Input": ssim_recons_to_input,
                        "SSIM Content to Corrupted Input": ssim_content_to_input,
                        "SSIM Attribute to Applied Corruption": ssim_attribute,

                        "PSNR Reconstruction to Original": psnr_recons_to_orig,
                        "PSNR Content to Original": psnr_content_to_orig,
                        "PSNR Reconstruction to Corrupted Input": psnr_recons_to_input,
                        "PSNR Content to Corrupted Input": psnr_content_to_input,
                        "PSNR Attribute to Applied Corruption": psnr_attribute
                    })

                # Save one example of the sample set together with all its corrupted variants
                self.save_img_and_reconstructions_for_DCAF(X_orig.cpu(), X.cpu(), X_hat.cpu(), X_hat_content.cpu(),
                                                           X_hat_attribute.cpu(), X_corruption.cpu(), X_name,
                                                           output_path)

        # Stop the time
        end_time = time.time()
        hours, minutes, seconds = self.calculate_passed_time(start_time, end_time)

        print("\t\t\tElapsed time: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))

    def save_img_and_reconstructions_for_AE(self, X_orig, X, X_hat, X_corruption, X_name, output_path):
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
        X_orig = X_orig[0].permute(1, 2, 0).numpy()
        X = X[0].permute(1, 2, 0).numpy()
        X_hat = X_hat[0].permute(1, 2, 0).numpy()
        X_corruption = X_corruption[0].permute(1, 2, 0).numpy()

        # Normalize the pixel values in the range 0 to 255
        X_orig = cv2.normalize(X_orig, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        X = cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        X_hat = cv2.normalize(X_hat, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        X_corruption = cv2.normalize(X_corruption, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Save the images PNG as files
        #cv2.imwrite(str(output_path / f"{sample_name}_orig.png"), X_orig)
        #cv2.imwrite(str(output_path / f"{sample_name}_input.png"), X)
        #cv2.imwrite(str(output_path / f"{sample_name}_recons.png"), X_hat)
        #cv2.imwrite(str(output_path / f"{sample_name}_corruption.png"), X_corruption)

        if channel_dim == 1:
            plt.imshow(X_orig, cmap='gray')
            plt.axis("off")
            plt.savefig(output_path / f"{sample_name}_orig.png", dpi=300)

            plt.imshow(X, cmap='gray')
            plt.axis("off")
            plt.savefig(output_path / f"{sample_name}_input.png", dpi=300)

            plt.imshow(X_hat, cmap='gray')
            plt.axis("off")
            plt.savefig(output_path / f"{sample_name}_recons.png", dpi=300)

            plt.imshow(X_corruption, cmap='gray')
            plt.axis("off")
            plt.savefig(output_path / f"{sample_name}_corruption.png", dpi=300)

        else:
            plt.imshow(X_orig)
            plt.axis("off")
            plt.savefig(output_path / f"{sample_name}_orig.png", dpi=300)

            plt.imshow(X)
            plt.axis("off")
            plt.savefig(output_path / f"{sample_name}_input.png", dpi=300)

            plt.imshow(X_hat)
            plt.axis("off")
            plt.savefig(output_path / f"{sample_name}_recons.png", dpi=300)

            plt.imshow(X_corruption)
            plt.axis("off")
            plt.savefig(output_path / f"{sample_name}_corruption.png", dpi=300)

        # Save all images combined in one figure
        # Create the figure
        fig, axes = plt.subplots(nrows=2, ncols=3)

        # Plot the images
        if channel_dim == 1:
            axes[0, 0].imshow(X_orig, cmap='gray')
            axes[0, 0].set_title("Original", loc='center')
            axes[0, 0].axis("off")

            axes[0, 1].imshow(X, cmap='gray')
            axes[0, 1].set_title("Corrupted Variant", loc='center')
            axes[0, 1].axis("off")

            axes[0, 2].imshow(X_corruption, cmap='gray')
            axes[0, 2].set_title("Difference Input", loc='center')
            axes[0, 2].axis("off")

            axes[1, 1].imshow(X_hat, cmap='gray')
            axes[1, 1].set_title("Image\nReconstruction", loc='center')
            axes[1, 1].axis("off")

            axes[1, 0].axis("off")
            axes[1, 2].axis("off")

        else:
            axes[0, 0].imshow(X_orig)
            axes[0, 0].set_title("Original", loc='center')
            axes[0, 0].axis("off")

            axes[0, 1].imshow(X)
            axes[0, 1].set_title("Corrupted Variant", loc='center')
            axes[0, 1].axis("off")

            axes[0, 2].imshow(X_corruption)
            axes[0, 2].set_title("Difference Input", loc='center')
            axes[0, 2].axis("off")

            axes[1, 1].imshow(X_hat)
            axes[1, 1].set_title("Image\nReconstruction", loc='center')
            axes[1, 1].axis("off")

            axes[1, 0].axis("off")
            axes[1, 2].axis("off")

        # Save and close the figure
        fig.tight_layout(h_pad=2)
        fig.savefig(output_path / f"{sample_name}_all.png", dpi=300)
        plt.close(fig)

    def save_img_and_reconstructions_for_DCAF(self, X_orig, X, X_hat, X_hat_content, X_hat_attribute, X_corruption,
                                              X_name, output_path):
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
        X_orig = X_orig[0].permute(1, 2, 0).numpy()
        X = X[0].permute(1, 2, 0).numpy()
        X_hat = X_hat[0].permute(1, 2, 0).numpy()
        X_hat_content = X_hat_content[0].permute(1, 2, 0).numpy()
        X_hat_attribute = X_hat_attribute[0].permute(1, 2, 0).numpy()
        X_corruption = X_corruption[0].permute(1, 2, 0).numpy()

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

        if channel_dim == 1:
            plt.imshow(X_orig, cmap='gray')
            plt.axis("off")
            plt.savefig(output_path / f"{sample_name}_orig.png", dpi=300)

            plt.imshow(X, cmap='gray')
            plt.axis("off")
            plt.savefig(output_path / f"{sample_name}_input.png", dpi=300)

            plt.imshow(X_hat, cmap='gray')
            plt.axis("off")
            plt.savefig(output_path / f"{sample_name}_recons.png", dpi=300)

            plt.imshow(X_hat_content, cmap='gray')
            plt.axis("off")
            plt.savefig(output_path / f"{sample_name}_recons_content.png", dpi=300)

            plt.imshow(X_hat_attribute, cmap='gray')
            plt.axis("off")
            plt.savefig(output_path / f"{sample_name}_recons_attribute.png", dpi=300)

            plt.imshow(X_corruption, cmap='gray')
            plt.axis("off")
            plt.savefig(output_path / f"{sample_name}_corruption.png", dpi=300)

        else:
            plt.imshow(X_orig)
            plt.axis("off")
            plt.savefig(output_path / f"{sample_name}_orig.png", dpi=300)

            plt.imshow(X)
            plt.axis("off")
            plt.savefig(output_path / f"{sample_name}_corrupted.png", dpi=300)

            plt.imshow(X_hat)
            plt.axis("off")
            plt.savefig(output_path / f"{sample_name}_reconstructed.png", dpi=300)

            plt.imshow(X_hat_content)
            plt.axis("off")
            plt.savefig(output_path / f"{sample_name}_recons_content.png", dpi=300)

            plt.imshow(X_hat_attribute)
            plt.axis("off")
            plt.savefig(output_path / f"{sample_name}_recons_attribute.png", dpi=300)

            plt.imshow(X_corruption)
            plt.axis("off")
            plt.savefig(output_path / f"{sample_name}_corruption.png", dpi=300)

        # Save all images combined in one figure
        # Create the figure
        fig, axes = plt.subplots(nrows=2, ncols=3)

        # Plot the images
        if channel_dim == 1:
            axes[0, 0].imshow(X_orig, cmap='gray')
            axes[0, 0].set_title("Original", loc='center')
            axes[0, 0].axis("off")

            axes[0, 1].imshow(X, cmap='gray')
            axes[0, 1].set_title("Corrupted Variant", loc='center')
            axes[0, 1].axis("off")

            axes[0, 2].imshow(X_corruption, cmap='gray')
            axes[0, 2].set_title("Difference Input", loc='center')
            axes[0, 2].axis("off")

            axes[1, 0].imshow(X_hat_content, cmap='gray')
            axes[1, 0].set_title("Anatomical\nReconstruction", loc='center')
            axes[1, 0].axis("off")

            axes[1, 1].imshow(X_hat, cmap='gray')
            axes[1, 1].set_title("Image\nReconstruction", loc='center')
            axes[1, 1].axis("off")

            axes[1, 2].imshow(X_hat_attribute, cmap='gray')
            axes[1, 2].set_title("Difference\nReconstructions", loc='center')
            axes[1, 2].axis("off")
        else:
            axes[0, 0].imshow(X_orig)
            axes[0, 0].set_title("Original", loc='center')
            axes[0, 0].axis("off")

            axes[0, 1].imshow(X)
            axes[0, 1].set_title("Corrupted Variant", loc='center')
            axes[0, 1].axis("off")

            axes[0, 2].imshow(X_corruption)
            axes[0, 2].set_title("Difference Input", loc='center')
            axes[0, 2].axis("off")

            axes[1, 0].imshow(X_hat_content)
            axes[1, 0].set_title("Anatomical\nReconstruction", loc='center')
            axes[1, 0].axis("off")

            axes[1, 1].imshow(X_hat)
            axes[1, 1].set_title("Image\nReconstruction", loc='center')
            axes[1, 1].axis("off")

            axes[1, 2].imshow(X_hat_attribute)
            axes[1, 2].set_title("Difference\nReconstruction", loc='center')
            axes[1, 2].axis("off")

        # Save and close the figure
        fig.tight_layout(h_pad=2)
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
    Run the reconstruction.

    :param configuration: Configuration of the reconstruction experiment.
    """

    # Initialize a weights & biases project for a training run with the given training configuration
    wandb.init(project="exp_reconstruction", config=configuration)

    # Initialize the reconstruction
    print("Initializing the reconstruction ...")
    reconstruction = Reconstruction(configuration)

    # Run the inference
    print("Run the reconstruction...")
    reconstruction.run_reconstruction_experiment()


if __name__ == '__main__':
    # Start code
    start_time = time.time()

    # Read out the command line parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str, help="Path to the configuration file to use.")
    parser.add_argument("--input_path", required=False, default='/data/sids-ml/checkpoints/train', type=str,
                        help="Parent directory to where the trained encoder is stored.")  # Only for bash execution
    parser.add_argument("--fold", required=False, default='0', type=str, help="Which fold to use as test fold.")
    parser.add_argument("--architecture", required=False, default='DCAF', type=str,  help="Which trained encoder to use.")  # Only for bash execution

    args = parser.parse_args()
    config_file = args.config_file
    input_path = args.input_path
    fold = args.fold
    architecture = args.architecture

    # Load the default and the hyperparameters and run the training
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        config['input_path'] = input_path
        config['fold'] = fold
        config['architecture'] = architecture

        main(config)

    # Finished code
    end_time = time.time()

    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
