# GsRCL: Gaussian noise augmented scRNA-seq contrastive learning framework
We proposed a novel cell-type identification method, namely GsRCL, which exploits the well-known Gaussian noise to augment the original gene expression profiles in order to learn contrastive learning feature representations. We investigated the performance of this Gaussian noise data augmentation method on contrastive learning and downstream supervised learning tasks. Due to its simplicity and the concern on the well-known overfitting issue, we investigated the effect of the contrastive learning-based feature extractor against the downstream classifier. The experimental results suggest that our GsRCL method successfully improved the accuracy of cell-type identification and outperformed the random genes masking data augmentation method in [contrastive-sc](https://doi.org/10.1186/s12859-021-04210-8). A further analysis reveals that the self-supervised learning-based contrastive learning framework has higher robustness than the supervised learning-based contrastive learning framework against the overfitting issue.

If you find this repository helpful, please cite our work:
```
@article {gsrcl,
	author = {Alsaggaf, Ibrahim and Buchan, Daniel and Wan, Cen},
	title = {Improving cell-type identification with Gaussian noise-augmented single-cell RNA-seq contrastive learning},
	elocation-id = {2022.10.06.511191},
	year = {2022},
	doi = {10.1101/2022.10.06.511191},
	URL = {https://www.biorxiv.org/content/early/2022/10/08/2022.10.06.511191},
	journal = {bioRxiv}
}
```

# Usage
This repository contains the implementation of GsRCL. The impelementation is built in Python3 using Scikit-learn, Scanpy, and the deep learning library Pytorch. 

## Requirements
- torch
- scikit-learn
- scanpy
- numpy==1.22

## Tutorial
The impelemtation is design to parallelise training workloads on GPUs, hence, the implementation should run on a machine with at least two GPUs. We recommend running the impelemtation on a machine with 16 cores, 60 GB memory and 4 GPUs. The `config.py` file holds all the required settings to run the implementation. 

To run this impelementation yon need to do the following:
1. Make sure all the requirements stated above are installed.
2. Navigate to your working directory where the `.py` files are stored (e.g. src).
3. Move the required dataset into the working directory (e.g. src/inter-dataset/PbmcBench_Seq-Well).
4. Modify the dataset information in the `config.py` file accordingly.
5. Execute `python3 runner.py`.

### The config file
Here we provide a discription of each part in `config.py`

Dataset information
```
PATH_ = The main folder (e.g. 'inter-dataset')
DATASET = The name of the folder that includes the required dataset (e.g. 'PbmcBench_Seq-Well')
FILE_X =  The name of File X (e.g. 'SW_pbmc1.csv')
FILE_Y = The name of file y (e.g. 'CD4+ T cell_VR_Labels.csv')

The full path of file X should be 'inter-dataset/PbmcBench_Seq-Well/SW_pbmc1.csv' 
and file y should be 'inter-dataset/PbmcBench_Seq-Well/CD4+ T cell_VR_Labels.csv'
```

Contrastive learning settings
```
CL_LOSSES = The contrastive losses (e.g. ['simclr', 'supcon'])
CL_AUG_METHODS = The augmentation methods (e.g. ['noise', 'random_masking'])
CL_EPOCHS = The number of corresponding epochs for each contrastive learning paradigm (e.g. [3000, 1000])
CL_STEP = The number of corresponding steps to measure a model validation performance (e.g. [40, 20])
CL_BATCH_SIZE = The batch size (e.g. 8192)
TOP_N_GENES = The selected topmost variable genes (e.g. [500, 1000, 3000, 5000])
LAYERS = The corresponding encoder and projection head layers' sizes w.r.t. the number of variable genes
          (e.g. [(384, 256, 128, 64), (384, 256, 128, 64), (2048, 1024, 512, 128), (2048, 1024, 512, 128)])
          
500 topmost genes (384, 256, 128, 64)
1000 topmost genes (384, 256, 128, 64)
3000 topmost genes (2048, 1024, 512, 128)
5000 topmost genes  (2048, 1024, 512, 128)
```

Task settings
```
ID_ = A unique timestamp (e.g. str(round(time.time())))
RESULTS_PATH = The folder to save the results in (e.g. 'results')
CLASSIFIERS = The used classifier and its package name (e.g. [('sklearn.svm', 'SVC')])
PARAMS_GRID = The hyperparameters used for grid search 
              (e.g. {'SVC': {'gamma': ['scale', 0.0001, 0.001, 0.01], 'C': [0.1, 1.0, 10.0, 100.0]}})
CV = The number of CV folds (e.g. 5)
TRAIN_SIZE = The training set size (e.g. 0.8)
METRIC = The main metric (e.g. 'mcc' or 'avg_precision' or 'f1')
WORLD_SIZE = The number of GPUs (e.g. torch.cuda.device_count() or 4)
NUM_CORES = The number of cpus for the grid search (os.cpu_count() // CV or 10)
```

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
Here we briefly describe each `.py` file in the **code** folder.

`cl.py` Runs a contrastive learning paradigm (SimCLR or Supcon).

`collector.py` Creates a results collector object.

`config.py` The pipeline configuration file.

`data.py` Reads and preprocess the given scRNA-seq dataset.

`experiment_builder.py` Builds the pipeline and execut it.

`losses.py` Includes the contrastive learning loss in both SimCLR and SupCon.

`metrics.py` Includes the metrics adopted in this implementation: MCC, AP, and F1.

`networks.py` Includes the encoder architecture.

`ovr.py` This is used to conduct the one-vs-rest approach, hence, creating several binary classification tasks.

`performance.py` Measures the performance in `self._benchmarking()` and `self._representations_performance()`.

`utils.py` Includes some helper functions.

`validators.py` Conducts model selection against the validation set.

# Dataset
The scRNA-seq datasets can be downloaded from [Zenodo](https://zenodo.org/record/3357167#.YzxZg9jMJdg). The binary class labels that are used in this work are available in the **Binary classification tasks** folder.

# Acknowledgements
The authors acknowledge the academic research grants provided by Google Cloud. The authors also acknowledge the support by the Department of Computer Science and Information System and the Birkbeck GTA programme.
