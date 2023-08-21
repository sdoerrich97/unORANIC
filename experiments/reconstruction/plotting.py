"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
unORANIC: Unsupervised orthogonalization of anatomy and image-characteristic features

@description:
Visualization of the reconstruction experiment.

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

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Set the plot style
sns.set_style('white')
sns.set_context("paper")
sns.set_style("whitegrid")

# Data for the current dataset
data_dictionary = {
    'bloodmnist': {
        'name': 'BloodMNIST',
        'corruptions': ['PixelDropout', 'GaussianBlur', 'ColorJitter', 'Downscale', 'GaussNoise', 'InvertImg', 'MotionBlur', 'MultiplicativeNoise', 'BrightnessContrast', 'Gamma', 'Solarize', 'Sharpen'],
        'AE / Ours': [22.235, 23.693, 18.116, 27.305, 30.5, 4.543, 22.232, 29.82, 14.679, 28.604, 4.83, 18.983],
        'Ours Anatomy Branch': [26.528, 26.052, 25.986, 26.578, 26.598, 26.71, 25.31, 26.704, 25.675, 26.747, 26.533, 26.593]
    }
}

for dataset in data_dictionary.keys():
    # Get all the data for the current dataset
    data = data_dictionary[dataset]

    # Create a DataFrame for the data
    data_frame = {
        'Corruptions': data['corruptions'] * 2,
        'Average PSNR': data['AE / Ours'] + data['Ours Anatomy Branch'],
        'Model': ['AE / Ours'] * len(data['corruptions']) + ['Ours Anatomy Branch'] * len(data['corruptions'])
    }
    df = pd.DataFrame(data_frame)

    # Create the plot using relplot
    fig = sns.relplot(data=df, x='Corruptions', y='Average PSNR', hue='Model', style='Model', kind='line', markers=True, markersize=10, linewidth=2)

    # Set title and axes labels
    #fig.fig.suptitle("Corruption Robustness of the Reconstructions")
    fig.set_xlabels("Corruptions")
    fig.set_ylabels("Average PSNR")
    plt.xticks(rotation=45, ha='right')

    fig_gcf = plt.gcf()
    fig_gcf.set_size_inches(12, 6)

    fig.savefig(Path(f"reconstruction_corruptions_plot" + ".png"), dpi=300)
    fig.savefig(Path(f"reconstruction_corruptions_plot" + ".pdf"), dpi=300)

    # Close the figure
    plt.close(fig.fig)