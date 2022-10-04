'''
The selected topmost variable genes and their corresponding 
encoder and projection head layers' sizes:
500 topmost genes (384, 256, 128, 64)
1000 topmost genes (384, 256, 128, 64)
3000 topmost genes (2048, 1024, 512, 128)
5000 topmost genes  (2048, 1024, 512, 128)
'''

import os
import time
import torch

# Dataset information
PATH_ = 'Binary classification tasks'
DATASET = 'PbmcBench_Seq-Well'
FILE_X = 'SW_pbmc1.csv'
FILE_Y = 'CD4+ T cell_VR_Labels.csv'

#Contrastive learning settings
CL_LOSSES = ['simclr', 'supcon']
CL_AUG_METHODS = ['noise', 'random_masking']
CL_EPOCHS = [1000, 3000]
CL_STEP = [20, 40]
CL_BATCH_SIZE = 8192
TOP_N_GENES = [500, 1000, 3000, 5000]
LAYERS = [(384, 256, 128, 64), (384, 256, 128, 64), 
          (2048, 1024, 512, 128), (2048, 1024, 512, 128)
]

# Task settings
ID_ = str(round(time.time()))
RESULTS_PATH = 'results'
CLASSIFIERS = [
    ('sklearn.svm', 'SVC'),
]
PARAMS_GRID = {
    'SVC': {
        'gamma': ['scale', 0.0001, 0.001, 0.01],
        'C': [0.1, 1.0, 10.0, 100.0]
    },
}
CV = 5
TRAIN_SIZE = 0.8
METRIC = 'mcc'
WORLD_SIZE = torch.cuda.device_count()
NUM_CORES = os.cpu_count() // 2
