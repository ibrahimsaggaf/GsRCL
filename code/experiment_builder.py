import os
from datetime import datetime
from pathlib import Path
from joblib import dump

import torch
import numpy as np

import config
from collector import res_collector
from data import BaseData
from performance import CVPerformance
from cl import ContrastiveLearning, CLAugmentation
from utils import get_folds, clean_model_checkpoints


class Pipeline:
    def __init__(self, clfs, params_grid, metric, metrics_list, c_losses_list, cl_aug_methods_list, 
                 file_X, file_y, dataset, path, cv, train_size, seed=None):
        self.clfs = clfs
        self.params_grid = params_grid
        self.metric = metric
        self.metrics_list = metrics_list
        self.c_losses_list = c_losses_list
        self.cl_aug_methods_list = cl_aug_methods_list
        self.file_X = file_X
        self.file_y = file_y
        self.dataset = dataset
        self.path = path
        self.train_size = train_size
        self.seed = seed
        if self.seed is None:
            self.seed = np.random.randint(low=1, high=1e6)
        self.folds = get_folds(cv=cv, seed=self.seed)
        self.best_models = []
        
        res_collector.seed=self.seed
        res_collector.start_datetime=datetime.now()
        
        if not os.path.isdir(Path(config.RESULTS_PATH)):
            os.mkdir(Path(config.RESULTS_PATH))
          
    def _benchmarking(self):
        self.data = BaseData(file_X=self.file_X, file_y=self.file_y, dataset=self.dataset, path=self.path, train_size=self.train_size, 
                             seed=self.seed)
        self.data.split()
        spl_benchmark = CVPerformance(data=self.data, folds=self.folds, clfs=self.clfs, params_grids=self.params_grid, 
                                      metric=self.metric, metrics_list=self.metrics_list, id_=res_collector.id_)
        spl_benchmark.measure()
        spl_benchmark.results_['runtime'] = datetime.now()
        res_collector.benchmark = spl_benchmark.results_ 
    

    def _contrastive_learning(self):
        results= {}

        for c_loss in self.c_losses_list:
            results[c_loss] = {}

            for aug_method in self.cl_aug_methods_list:
                if aug_method == 'random_masking':
                    for top_n_genes, layers in zip(config.TOP_N_GENES, config.LAYERS):
                        data = BaseData(file_X=self.file_X, file_y=self.file_y, dataset=self.dataset, path=self.path, train_size=self.train_size, 
                                        seed=self.seed, top_n_genes=top_n_genes)
                        data.split()
                        cl_ = ContrastiveLearning(data=data, metric=self.metric, folds=self.folds, id_=res_collector.id_, dim_enc=layers[0], 
                                                  in_dim_proj=layers[1], dim_proj=layers[2], out_dim=layers[3], epochs=config.CL_EPOCHS, 
                                                  step=config.CL_STEP, c_loss=c_loss, batch_size=config.CL_BATCH_SIZE, 
                                                  aug_method=f'{aug_method}_{top_n_genes}', aug_func=None, dropout=0.9)
                        cl_.mp_fit()
                        results[c_loss].update({
                            f'{aug_method}_{top_n_genes}': cl_.results_
                        })
                else:
                    aug_func = CLAugmentation(aug_method)
                    cl_ = ContrastiveLearning(data=self.data, metric=self.metric, folds=self.folds, id_=res_collector.id_, dim_enc=4096, 
                                              in_dim_proj=2048, dim_proj=512, out_dim=128, epochs=config.CL_EPOCHS, step=config.CL_STEP, 
                                              c_loss=c_loss, batch_size=config.CL_BATCH_SIZE, aug_method=aug_method, aug_func=aug_func, 
                                              dropout=None)    
                    cl_.mp_fit()
                    results[c_loss].update({
                        aug_method: cl_.results_
                    })
            
            for idx in range(self.folds.n_splits):
                self.best_models.extend([results[c_loss][m][idx]['best'] for m in results[c_loss].keys()])

            clean_model_checkpoints(self.best_models, Path(config.RESULTS_PATH, res_collector.id_))

        results['runtime'] = datetime.now()
        res_collector.contrastive_learning = results

                
    def _representations_performance(self):
        results= {}
        for c_loss in self.c_losses_list:
            results[c_loss] = {}

            for aug_method in res_collector.contrastive_learning[c_loss].keys():
                if 'random_masking' in aug_method:
                    top_n_genes = int(aug_method.split('_')[-1])
                    data = BaseData(file_X=self.file_X, file_y=self.file_y, dataset=self.dataset, path=self.path, train_size=self.train_size, 
                                    seed=self.seed, top_n_genes=top_n_genes)
                    data.split()
                    cl_ = CVPerformance(data=data, folds=self.folds, clfs=self.clfs, params_grids=self.params_grid,
                                        metric=self.metric, metrics_list=self.metrics_list, id_=res_collector.id_, 
                                        task='cl', results_collector=res_collector.contrastive_learning[c_loss][aug_method])
                else:
                    cl_ = CVPerformance(data=self.data, folds=self.folds, clfs=self.clfs, params_grids=self.params_grid,
                                        metric=self.metric, metrics_list=self.metrics_list, id_=res_collector.id_, 
                                        task='cl', results_collector=res_collector.contrastive_learning[c_loss][aug_method])

                cl_.measure()
                results[c_loss].update({
                    aug_method: cl_.results_
                })

        results['runtime'] = datetime.now()
        res_collector.representations = results


    def _save_results(self):
        dump(res_collector, 
             Path(config.RESULTS_PATH, 
                  f'results_collector_{config.DATASET}_{config.METRIC}_{config.ID_}.joblib'), 
             compress='zlib'
        )


    def run(self):
        torch.manual_seed(self.seed)
    
        self._benchmarking()
        self._contrastive_learning()
        self._representations_performance()
        self._save_results()




        
            
       
