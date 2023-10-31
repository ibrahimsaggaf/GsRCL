# Improving cell type identification with Gaussian noise-augmented single-cell RNA-seq contrastive learning

This is a Python implementation of the GsRCL method reported in

Alsaggaf, I. Buchan, D. and Wan, C. (2023) Improving cell type identification with Gaussian noise-augmented single-cell RNA-seq contrastive learning.

# Usage
This repository contains the implementation of GsRCL. The implementation is built in Python3 (version 3.8.10) using Scikit-learn, Scanpy, and the deep learning library Pytorch. 

## Requirements
- torch==1.11.0
- scikit-learn==1.3.2
- scanpy==1.9.1
- numpy==1.22

## Tutorial
The implementation is designed to parallelise training workloads on GPUs, hence, the implementation should run on a machine with at least two GPUs. We recommend running the implementation on a machine with 16 cores, 60 GB memory and 4 GPUs. The `config.py` file holds all the required settings to run the implementation. 

To run this implementation you need to do the following:
1. Make sure all the requirements stated above are installed.
2. Navigate to your working directory where the `.py` files are stored (e.g. src).
3. Move the required dataset(s) into the working directory (e.g. src/inter-dataset/PbmcBench_Seq-Well).
4. Modify the dataset information in the `config.py` file accordingly.
5. Execute `python3 runner.py`.

### The config file
Here we provide a description of each part in `config.py`

Dataset information
```
PATH_ = The main folder (e.g. 'Binary scRNA-seq datasets')
DATASET = The name of the folder that includes the required dataset (e.g. 'PbmcBench_Seq-Well')
FILE_X =  The name of File X (e.g. 'data.csv'), Dimensions should be cells x genes
FILE_Y = The name of file y (e.g. 'CD4+ T cell_VR_Labels.csv')

The full path of file X should be 'Binary scRNA-seq datasets/PbmcBench_Seq-Well/data.csv' 
and file y should be 'Binary scRNA-seq datasets/PbmcBench_Seq-Well/CD4+ T cell_VR_Labels.csv'
This setting is helpful when having different datasets from different sources. Otherwise, PATH_ and DATASET would be fixed.
```

Contrastive learning settings
```
CL_LOSSES = The contrastive losses (e.g. ['simclr', 'supcon'])
CL_AUG_METHODS = The augmentation methods (e.g. ['noise', 'random_masking'])
CL_EPOCHS = The number of corresponding epochs for each contrastive learning paradigm (e.g. [3000, 1000])
CL_STEP = The number of corresponding steps to measure a model validation performance (e.g. 50)
CL_BATCH_SIZE = The batch size (e.g. 8192)
TOP_N_GENES = The selected topmost variable genes (e.g. [500, 1000, 3000, 5000])
LAYERS = The corresponding encoder and projection head layers' sizes w.r.t. the number of topmost variable genes
         (e.g. [(384, 256, 128, 64), (384, 256, 128, 64), (2048, 1024, 512, 128), (2048, 1024, 512, 128)])
          
500 topmost genes (384, 256, 128, 64)
1000 topmost genes (384, 256, 128, 64)
3000 topmost genes (2048, 1024, 512, 128)
5000 topmost genes  (2048, 1024, 512, 128)
```

Task settings
```
ID_ = A unique timestamp (e.g. str(round(time.time())))
RESULTS_PATH = The folder to save the results in (e.g. '/results')
CLASSIFIERS = The used classifier and its package name (e.g. [('sklearn.svm', 'SVC')])
PARAMS_GRID = The hyperparameters used for grid search 
              (e.g. {'SVC': {'gamma': ['scale', 0.0001, 0.001, 0.01], 'C': [0.1, 1.0, 10.0, 100.0]}})
CV = The number of CV folds (e.g. 5)
TRAIN_SIZE = The training set size (e.g. 0.8)
METRIC = The main metric (e.g. 'mcc' or 'avg_precision' or 'f1'). In this work, we set METRIC = 'mcc'.
WORLD_SIZE = The number of GPUs (e.g. torch.cuda.device_count() or 4)
NUM_CORES = The number of cpus for the grid search (os.cpu_count() // CV or 8)
```

### The pipeline
The implementation runs a pipeline with the following main steps:
```
self._benchmarking()
self._contrastive_learning()
self._representations_performance()
self._save_results()
```

`self._benchmarking()` Runs a classic supervised learning task by training an SVM classifier on the row counts. A nested grid search with cross-validation is run for hyperparameters tuning. The results are reported using `MCC`, `AP`, and `F1` which are considered as the benchmark for a given particular scRNA-seq dataset.

`self._contrastive_learning()` This is the main step in the pipeline, it conducts two contrastive learning paradigms SimCLR [(Chen et al., 2020)](http://proceedings.mlr.press/v119/chen20j.html) and SupCon [(Khosla et al., 2020)](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html). Ecah paradigm is run with different data augmentation methods: A Gaussian noise-based augmentation and five variants of random genes masking [(Ciortan and Defrance, 2021)](https://doi.org/10.1186/s12859-021-04210-8). Because this step is run in a 5-fold cross-validation manner, the implementation parallelises the 5 folds training on the available number of GPUs. Analogous to `self._benchmarking()`, the validation performance is measured after every `CL_STEP` training epochs. When the training completes, model selection is conducted using the validations set, only the best-performing model is kept while the others are discarded.

`self._representations_performance()` From the previous step the best model is used to obtain the learnt representations. For a fair comparison, the performance of those representations is measured in the exact same setting as the benchmarking step (i.e. `self._benchmarking()`).

`self._save_results()` This step only dumps the results collector object.

### The code
Here we briefly describe each `.py` file in the **code** folder.

`cl.py` Runs a contrastive learning paradigm (SimCLR or Supcon).

`collector.py` Creates a results collector object.

`config.py` The pipeline configuration file.

`data.py` Reads and preprocesses the given scRNA-seq dataset.

`experiment_builder.py` Builds the pipeline and executes it.

`losses.py` Includes the contrastive learning loss in both SimCLR and SupCon.

`metrics.py` Includes the metrics adopted in this implementation: MCC, AP, and F1.

`networks.py` Includes the encoder architecture.

`ovr.py` This is used to conduct the one-vs-rest approach, hence, creating several binary classification tasks.

`performance.py` Measures the performance in `self._benchmarking()` and `self._representations_performance()`.

`utils.py` Includes some helper functions.

`validators.py` Conducts model selection against the validation set.

# Data availability
The scRNA-seq datasets used in this work can be downloaded from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8087611.svg)](https://doi.org/10.5281/zenodo.8087611).

# Pretrained encoders for GsRCL
The pretrained encoders and the SVM classifiers can be downloaded from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10050548.svg)](https://doi.org/10.5281/zenodo.10050548)

Here we provide an example on how to identify new cells (query) using a reference from the following table:

| Reference name | Organ/tissue | Paltform | # Reference Genes | # Cell types |
| ------------- | ------------ | -------- | ------------------ | ------------ |
| PbmcBench_10Xv2 | PBMC | 10X Chromium v2 | 22280 | 9 |
| PbmcBench_10Xv3 | PBMC | 10X Chromium v3 | 21905 | 8 |
| PbmcBench_Drop-Seq | PBMC | Drop-Seq | 19922 | 9 |
| PbmcBench_inDrop | PBMC | inDrop | 17159 | 7 |
| PbmcBench_Seq-Well | PBMC | Seq-Well | 21059 | 7 |
| Baron Human | Human pancreas | inDrop | 17499 | 14 |
| Muraro | Human pancreas | CEL-Seq2 | 18915 | 9 |
| Segerstolpe | Human pancreas | SMART-Seq2 | 22757 | 13 |
| Xin | Human pancreas | SMARTer | 33889 | 4 |
| Adam | Mouse kidney | Drop-Seq | 23797 | 8 |
| Baron Mouse | Mouse pancreas | inDrop | 14861 | 13 |
| Klein | Mouse Embryonic Stem Cell | inDrop | 24047 | 4 |
| Romanov | Mouse hypothalamus | Unknown | 21143 | 7 |
| Quake_10x_Bladder | Mouse bladder | 10x Genomics | 16867 | 4 |
| Quake_10x_Limb_Muscle | Mouse limb muscle | 10x Genomics | 16512 | 6 |
| Quake_Smart-seq2_Diaphragm | Mouse diaphragm | SMART-Seq2 | 17973 | 5 |
| Quake_Smart-seq2_Limb_Muscle | Mouse limb Muscle | SMART-Seq2 | 18320 | 6 |
| Quake_Smart-seq2_Lung | Mouse lung | SMART-Seq2 | 19390 | 11 |
| Quake_Smart-seq2_Trachea | Mouse trachea | SMART-Seq2 | 19992 | 4 |

To identify new cells you need to do the following:
1. Make sure all the requirements stated above are installed.
2. Navigate to your working directory where the `.py` files are stored (e.g. src).
3. Provide a set of query genes and their expression profiles (a sample is given in the **Example** folder).
4. Execute the following script:
```
from identify import Identify

path = The working directory (e.g. 'src')
reference = A reference from the above table (e.g. 'Quake_10x_Bladder')
query = The set of query genes CSV file (see the given sample 'query_genes.csv' in the Example folder)
values = The raw expression profiles CSV file (see the given sample 'query_values.csv' in the Example folder)

obj = Identify(reference, path)
obj.predict_proba(query, values)
obj.save_results()
```
Given the provided samples in the **Example** folder, the script should print out the following:
```
>>> Downloading pre-trained encoders ...
>>> Unzipping ...
>>> Cross referencing 21143 query genes against 23341 reference genes ...
>>> 1380 out of 21143 genes are not found in reference, hence 1380 genes removed from query.
>>> 3578 out of 23341 genes in reference are not found in query, hence 3578 genes added to query with zero values.
>>> Obtaining probabilities for each cell-type ...
```
The script will first download the pretrained encoders for the selected reference. Then, it will cross reference the given query genes against the reference genes. After that, for each query cell a set of probabilities is obtained, where each probability is associated with a cell type in the selected reference. finally, those probabilities are saved in a CSV file (see `probabilities.csv` in the **Example** folder).

# Acknowledgements
The authors acknowledge the academic research grants provided by Google Cloud. The authors also acknowledge the support by the Department of Computer Science and Information Systems and the Birkbeck GTA programme.
