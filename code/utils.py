import os
import shutil
import importlib
from pathlib import Path
import math
import torch
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold

from metrics import metrics_list
from networks import Encoder
 

def clean_model_checkpoints(best, path):
    for file in os.listdir(path):
        if file not in best and not os.path.isdir(Path(path, file)):
            os.remove(Path(path, file))


def merge_dict(list_of_dicts):
    merged = {}
    for dict_ in list_of_dicts:
        merged.update(dict_)
    return merged
    

def to_dict_of_lists(list_of_dicts, labels=True):
    merged = {}
    for key in list_of_dicts[0].keys():
        merged[key] = [d[key] for d in list_of_dicts]

    if labels:
        labels = np.hstack(merged['labels'])
        probs = np.hstack(merged['probs'])
        del merged['labels'], merged['probs']
        return merged, labels, probs
    
    return merged       
        

def get_folds(cv, seed):
    return StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    
 
def shuffle(X, y=None):
    shuffled = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
    if y is not None:
        return X[shuffled], y[shuffled]
        
    return X[shuffled]


def get_metric_scorer(metric_name, metric_pkg='metrics'):
    if metric_name not in metrics_list:
        raise NotImplementedError(f'The {metric_name} metric is not implemeneted')
        
    metric = getattr(importlib.import_module(metric_pkg), metric_name)
    return metric, make_scorer(metric)


def get_clf(clf_pkg, clf_name, params={}):
    clf = getattr(importlib.import_module(clf_pkg), clf_name)
    return clf(**params)


def save_encoder_params(idx, id_, cl_loss, aug_method, frozen_params, frozen_kwargs, epoch, path):
    checkpoint = f'{idx}_{cl_loss}_{aug_method}_checkpoint_{epoch}.pt'
        
    if not os.path.isdir(Path(path, id_)):
        os.mkdir(Path(path, id_))
        
    torch.save({
        'forzen_params': frozen_params,
        'frozen_kwargs': frozen_kwargs,
        'epoch': epoch
    }, Path(path, id_, checkpoint))
    
    return checkpoint


def load_encoder(id_, checkpoint, path):
    device = torch.device('cpu')
    params = torch.load(Path(path, id_, checkpoint), map_location=device)
    encoder = Encoder(**params['frozen_kwargs'])
    encoder.load_state_dict(params['forzen_params'])
    return encoder


def get_hiddens(checkpoint, id_, data, path, folds=None):
        frozen_encoder = load_encoder(id_, checkpoint, path)
        frozen_encoder.eval()

        with torch.no_grad():
            if folds is not None:
                train_fold, test_fold = folds
                train_h = frozen_encoder(torch.tensor(data.train_X[train_fold]))
                test_h = frozen_encoder(torch.tensor(data.train_X[test_fold]))

                return train_h, test_h

            else:
                val_h = frozen_encoder(torch.tensor(data.val_X))

                return val_h
