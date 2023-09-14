# unORANIC: Unsupervised Orthogonalization of Anatomy and Image-Characteristic Features
This repository contains the code and resources for the paper titled ["unORANIC: Unsupervised Orthogonalization of Anatomy and Image-Characteristic Features"](https://arxiv.org/abs/2308.15507). The paper introduces an innovative unsupervised approach that leverages an adapted loss function to drive the orthogonalization of anatomy and image-characteristic features. The method is designed to enhance the generalizability and robustness of medical image analysis, specifically addressing challenges related to corruption and variations in imaging domains.

## Overview
In recent years, deep learning algorithms have shown promise in medical image analysis, including segmentation, classification, and anomaly detection. However, their adoption in clinical practice is hindered by challenges arising from domain shifts and variations in imaging parameters, corruption artifacts, and more. To tackle these challenges, the paper introduces unORANIC, which focuses on orthogonalizing anatomy and image-characteristic features in an unsupervised manner.

## Key Features
- **Unsupervised Orthogonalization**: unORANIC employs a novel loss function to facilitate the orthogonalization of anatomy and image-characteristic features, resulting in improved generalization and robustness.

- **Versatile for Diverse Modalities and Tasks**: The method is adaptable to a wide range of medical imaging modalities and tasks, making it highly versatile.

- **No Domain Knowledge or Labels Required**: Unlike many existing methods, unORANIC does not rely on domain knowledge, paired data samples, or labels, making it more flexible and applicable.

- **Corruption Robustness**: The approach can effectively handle potentially corrupted input images, producing reconstruction outputs with domain-invariant anatomy.

## Usage
This repository contains the implementation of unORANIC, along with resources to reproduce the results presented in the paper. The codebase is organized for clarity and ease of use, with comprehensive documentation to guide you through the process.

## Getting Started
To get started with unORANIC, follow these steps:

1. Clone this repository to your local machine.
2. The training scripts can be found under "train", whereas the scripts responsible for the experiments presented in the paper can be found under "experiments".
3. Each script provides comprehensive documentation to explain its usage and application.

## Results
The efficacy of unORANIC is demonstrated through extensive experimentation on various datasets. Results include evaluation of classification accuracy, corruption detection, and revision capabilities. The approach shows promise in improving the generalizability and robustness of practical applications in medical image analysis.

## Citation
If you find this work useful in your research, please consider citing our paper:

@misc{doerrich2023unoranic,
      title={unORANIC: Unsupervised Orthogonalization of Anatomy and Image-Characteristic Features}, 
      author={Sebastian Doerrich and Francesco Di Salvo and Christian Ledig},
      year={2023},
      eprint={2308.15507},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
