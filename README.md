# unORANIC: Unsupervised Orthogonalization of Anatomy and Image-Characteristic Features
Official code repository for the paper ["unORANIC: Unsupervised Orthogonalization of Anatomy and Image-Characteristic Features"](https://link.springer.com/chapter/10.1007/978-3-031-45673-2_7) [MICCAI - MLMI - 2023].

## Overview
We introduce unORANIC, an unsupervised approach that uses an adapted loss function to drive the orthogonalization of anatomy and image-characteristic features. The method is versatile for diverse modalities and tasks, as it does not require domain knowledge, paired data samples, or labels. During test time unORANIC is applied to potentially corrupted images, orthogonalizing their anatomy and characteristic components, to subsequently reconstruct corruption-free images, showing their domain-invariant anatomy only. This feature orthogonalization further improves generalization and robustness against corruptions. We confirm this qualitatively and quantitatively on 5 distinct datasets by assessing unORANIC’s classification accuracy, corruption detection and revision capabilities. Our approach shows promise for enhancing the generalizability and robustness of practical applications in medical image analysis.

![approach_anim](https://github.com/sdoerrich97/unORANIC/assets/98497332/d11ff430-076e-4eb3-a5dd-fbf296a8ae51)

Figure 1: We consider the input images as bias-free and uncorrupted ($I$). We further define $A_S$ as a random augmentation that distorts an input image $I$ for the purpose to generate a synthetic, corrupted version $S$ of that image. Such a synthetic image $S$ is obtained via the augmentation $A_S$ applied to $I$ and subsequently fed to both the anatomy encoder $E_A$ and the characteristic encoder $E_C$ simultaneously. The resulting embeddings are concatenated (⊕) and forwarded to a convolutional decoder $D$ to create the reconstruction $\hat{S}$ with its specific characteristics such as contrast level or brightness. By removing these characteristic features in the encoded embeddings of $E_A$, we can reconstruct a distortion-free version ($\hat{I}_A$) of the original input image $I$. To allow this behavior, the anatomy encoder, $E_A$, is actively enforced to learn acquisition- and corruption-robust representations while the characteristic encoder $E_C$ retains image-specific details.

## Key Features
- **Unsupervised Orthogonalization**: unORANIC employs a novel loss function to facilitate the orthogonalization of anatomy and image-characteristic features, resulting in improved generalization and robustness.

- **Versatile for Diverse Modalities and Tasks**: The method is adaptable to a wide range of medical imaging modalities and tasks, making it highly versatile.

- **No Domain Knowledge or Labels Required**: Unlike many existing methods, unORANIC does not rely on domain knowledge, paired data samples, or labels, making it more flexible and applicable.

- **Corruption Robustness**: The approach can effectively handle potentially corrupted input images, producing reconstruction outputs with domain-invariant anatomy.

## Results
The efficacy of unORANIC is demonstrated through extensive experimentation on various datasets. Results include evaluation of classification accuracy, corruption detection, and revision capabilities. The approach shows promise in improving the generalizability and robustness of practical applications in medical image analysis.

![Results](https://github.com/sdoerrich97/unORANIC/assets/98497332/a34fdf73-672a-4499-904b-7bf82d8cc794)

Figure 2: Left: Corruption revision results of unORANIC for a set of corruptions. Right: Visualization of unORANIC’s corruption revision for a solarization corruption. The method reconstructs $I$ via $\hat{I}_A$ while seeing only the corrupted input $S$.

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
