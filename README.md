# GsRCL: Gaussian noise augmented scRNA-seq contrastive learning framework
We proposed a novel cell-type identification method, namely GsRCL, which exploits the well-known Gaussian noise to augment the original gene expression profiles in order to learn contrastive learning feature representations. We investigated the performance of this Gaussian noise data augmentation method on contrastive learning and downstream supervised learning tasks. Due to its simplicity and the concern on the well-known overfitting issue, we investigated the effect of the contrastive learning-based feature extractor against the downstream classifier. The experimental results suggest that our GsRCL method successfully improved the accuracy of cell-type identification and outperformed the random genes masking data augmentation method in [contrastive-sc](https://doi.org/10.1186/s12859-021-04210-8). A further analysis reveals that the self-supervised learning-based contrastive learning framework has higher robustness than the supervised learning-based contrastive learning framework against the overfitting issue.

# Usage
This repository contains the implementation of GsRCL. The impelementation is built in Python3 using Scikit-learn, Scanpy, and the deep learning library Pytorch. 

## Requirements
- torch
- scikit-learn
- scanpy
- numpy==1.22

## Tutorial
The impelemtation is design to parallelise training workloads on GPUs, hence, the implementation should run on a machine with at least two GPUs. We recommend running the impelemtation on a machine with 16 cores, 60 GB memory and 4 GPUs. The `config.py` file holds all the required settings to run the implementation. To run this impelementation yon need to execute `python3 runner.py`.

### The pipeline
The implementation runs a pipeline with the following main steps:
```
self._benchmarking()
self._contrastive_learning()
self._representations_performance()
self._save_results()
```

`self._benchmarking()` Runs a classic supervised learning task by training a SVM classifier on the row counts. Grid search with cross-validation is used for hyperparameters tuning. The results are reported using `MCC`, `AP`, and `F1` which considered as the benchmark for a given particular scRNA-seq dataset.

`self._contrastive_learning()` This is the main step in the pipeline, it conducts two contrastive learning paradigms SimCLR[(Chen et al., 2020)](http://proceedings.mlr.press/v119/chen20j.html) and SupCon[(Khosla et al., 2020)](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html). Ecah paradigm is run with different data augmentation methods: A Gaussian noise augmentation and five varients of random genes masking. Because this step is run in a 5-fold cross-validation manner, the implamantation parallelises the 5 folds training on the available number of GPUs. When the training completes, model selection is conducted using the validations set, only the best performaing model is kept while the others are discarded.

`self._representations_performance()` From the previous step the best model is used to obtain the learnt representations. For fair comparison, the performance of those representations is measured in the exact same way as the bechmarking step.

`self._save_results()` This step only dumps the results collector object.

### The code
Here we will briefly talk about each `.py` file in the code folder.



# Dataset
The scRNA-seq datasets can be downloaded from [Zenodo](https://zenodo.org/record/3357167#.YzxZg9jMJdg). The binary class labels are available in the **Binary classification tasks** folder.

# Acknowledgements
The authors acknowledge the academic research grants provided by Google Cloud. The authors also acknowledge the support by the Department of Computer Science and Information System and the Birkbeck GTA programme.
