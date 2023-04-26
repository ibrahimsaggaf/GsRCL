'''
This code is modefied based on the impelemtation by 
https://github.com/xuebaliang/scziDesk
'''

import csv
import numpy as np
import scanpy as sc
from pathlib import Path
from sklearn.model_selection import train_test_split

from h5 import h5_reader


class BaseData:
    def __init__(self, file_X, file_y, dataset, path, train_size, seed, top_n_genes=None):
        self.top_n_genes = top_n_genes
        if self.top_n_genes is None:
            self.scaler = self._log_transformation
        else:
            self.scaler = self._normalise
        
        if file_X.endswith('.h5'):
            self.X = self.scaler(h5_reader(Path(path, dataset, file_X)))
        
        else:
            self.X = self._csv_reader(Path(path, dataset, file_X))
            
        self.y = self._csv_reader(Path(path, dataset, file_y), labels=True)
        self.dim = self.X.shape[1]
        self.train_size = train_size
        self.seed = seed


    def _log_transformation(self, X, epsilon=1e-6):
        return np.log(X + 1.0 + epsilon)

    
    def _normalise(self, X):
        adata = sc.AnnData(X)
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=self.top_n_genes, subset=True)
        sc.pp.scale(adata)
        X = adata.X.astype(np.float32)
    
        return X 


    def _csv_reader(self, dataset, labels=False):
        rows = []
        with open(dataset, 'r') as file:
            reader = csv.reader(file)
            if labels:
                row = reader.__next__()
                return np.array(row, dtype=np.int32)

            _ = reader.__next__()
            for row in reader:
                rows.append(row[1:])

            return self.scaler(np.array(rows, dtype=np.float32))
        
    
    def split(self):
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(self.X, self.y, 
                                                                              train_size=self.train_size, 
                                                                              random_state=self.seed, 
                                                                              shuffle=True, 
                                                                              stratify=self.y)
        del self.X, self.y

