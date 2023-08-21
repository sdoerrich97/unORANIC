"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
unORANIC: Unsupervised orthogonalization of anatomy and image-characteristic features

@description:
Visualization of the autoencoder performance experiment.

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

# Data for different datasets (PSNR value, Number of training samples)
dataset_names = ['BreastMNIST', 'RetinaMNIST', 'PneumoniaMNIST', 'DermaMNIST', 'BloodMNIST', 'OrganCMNIST', 'OrganSMNIST', 'OrganAMNIST', 'PathMNIST', 'OCTMNIST', 'ChestMNIST', 'TissueMNIST']
nr_trainingsamples = [546, 1080, 4708, 7007, 11959, 13000, 13940, 34581, 78468, 89996, 97477, 165466]
psnr_test = [22.039, 30.122, 30.861, 31.985, 28.215, 22.176, 22.442, 21.346, 28.049, 35.501, 34.942, 38.149]
color_or_gray = ['gray', 'color', 'gray', 'color', 'color', 'gray', 'gray', 'gray', 'color', 'gray', 'gray', 'gray']

# Figure 1:
# Create a DataFrame for the data
data = {'# Training Samples': nr_trainingsamples, 'PSNR': psnr_test, 'Dataset': dataset_names}
df = pd.DataFrame(data)

# Define marker colors and line styles
marker_colors = sns.color_palette("hls", len(dataset_names))
line_style = 'dashed'

# Create the plot using relplot
fig = sns.relplot(data=df, x='# Training Samples', y='PSNR', hue='Dataset', style='Dataset', kind='scatter', s=100)

# Set title and axes labels
fig.set_xlabels("# Training Samples")
fig.set_ylabels("PSNR Value")

fig.savefig(Path('AE_recons_all_datasets' + ".png"), dpi=300)
fig.savefig(Path('AE_recons_all_datasets' + ".pdf"), dpi=300)

# Close the figure
plt.close(fig.fig)

# Figure 2:
# Create a DataFrame for the data
data = {'# Training Samples': nr_trainingsamples, 'PSNR': psnr_test, 'Dataset': dataset_names, 'Color or Gray': color_or_gray}
df = pd.DataFrame(data)

# Create the plot using relplot
fig = sns.relplot(data=df, x='# Training Samples', y='PSNR', col='Color or Gray', hue='Dataset', style='Dataset', kind='line',
                  markers=True, markersize=10, linewidth=2)

# Set title and axes labels
fig.set_xlabels("# Training Samples")
fig.set_ylabels("PSNR Value")

fig.savefig(Path('AE_recons_color_gray' + ".png"), dpi=300)
fig.savefig(Path('AE_recons_color_gray' + ".pdf"), dpi=300)

# Close the figure
plt.close(fig.fig)