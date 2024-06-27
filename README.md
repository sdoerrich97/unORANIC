# unORANIC: Unsupervised Orthogonalization of Anatomy and Image-Characteristic Features
This repository contains the code and resources for the paper titled ["unORANIC: Unsupervised Orthogonalization of Anatomy and Image-Characteristic Features"](https://link.springer.com/chapter/10.1007/978-3-031-45673-2_7). The paper introduces an innovative unsupervised approach that leverages an adapted loss function to drive the orthogonalization of anatomy and image-characteristic features. The method is designed to enhance the generalizability and robustness of medical image analysis, specifically addressing challenges related to corruption and variations in imaging domains.

## Overview
In recent years, deep learning algorithms have shown promise in medical image analysis, including segmentation, classification, and anomaly detection. However, their adoption in clinical practice is hindered by challenges arising from domain shifts and variations in imaging parameters, corruption artifacts, and more. To tackle these challenges, the paper introduces unORANIC, which focuses on orthogonalizing anatomy and image-characteristic features in an unsupervised manner.

![approach_anim](https://github.com/sdoerrich97/unORANIC/assets/98497332/af2f5f2a-92c2-42a5-8013-c156ae5ce69b)

## Key Features
- **Unsupervised Orthogonalization**: unORANIC employs a novel loss function to facilitate the orthogonalization of anatomy and image-characteristic features, resulting in improved generalization and robustness.

- **Versatile for Diverse Modalities and Tasks**: The method is adaptable to a wide range of medical imaging modalities and tasks, making it highly versatile.

- **No Domain Knowledge or Labels Required**: Unlike many existing methods, unORANIC does not rely on domain knowledge, paired data samples, or labels, making it more flexible and applicable.

- **Corruption Robustness**: The approach can effectively handle potentially corrupted input images, producing reconstruction outputs with domain-invariant anatomy.

## Results
The efficacy of unORANIC is demonstrated through extensive experimentation on various datasets. Results include evaluation of classification accuracy, corruption detection, and revision capabilities. The approach shows promise in improving the generalizability and robustness of practical applications in medical image analysis.

![Results](https://github.com/sdoerrich97/unORANIC/assets/98497332/a34fdf73-672a-4499-904b-7bf82d8cc794)

## Getting Started
To get started with unORANIC, follow these steps:

1. Clone this repository to your local machine.
2. The training scripts can be found under "train", whereas the scripts responsible for the experiments presented in the paper can be found under "experiments".
3. Each script provides comprehensive documentation to explain its usage and application.

## Citation
If you find this work useful in your research, please consider citing our paper:
- [Publication](https://link.springer.com/chapter/10.1007/978-3-031-45673-2_7)
- [Preprint](https://arxiv.org/abs/2308.15507)

```
@InProceedings{doerrich2023unoranic,
   author="Doerrich, Sebastian and Di Salvo, Francesco and Ledig, Christian",
   title="unORANIC: Unsupervised Orthogonalization of Anatomy and Image-Characteristic Features",
   booktitle="Machine Learning in Medical Imaging",
   year="2024",
   pages="62--71",
   isbn="978-3-031-45673-2"
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
