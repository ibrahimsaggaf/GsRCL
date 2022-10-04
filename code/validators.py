import numpy as np
from sklearn.model_selection import GridSearchCV

import config
from utils import get_clf, get_metric_scorer, shuffle
from utils import save_encoder_params 


class EncoderValidator:
    def __init__(self, data, train_fold, metric, folds, idx, id_, c_loss, aug_method):
        self.data = data
        self.train_fold = train_fold
        self.metric = metric
        self.folds = folds
        self.idx = idx
        self.id_ = id_
        self.c_loss = c_loss
        self.aug_method = aug_method


    def _get_score(self, train_h, val_h):
            metric, scorer = get_metric_scorer(self.metric)
            train_h, train_y = shuffle(train_h.detach().numpy(), self.data.train_y[self.train_fold])
            clf = get_clf('sklearn.svm', 'SVC')
            grid_search = GridSearchCV(clf, config.PARAMS_GRID['SVC'], cv=self.folds, scoring=scorer, n_jobs=config.NUM_CORES)         
            grid_search.fit(train_h, train_y)
            preds = grid_search.best_estimator_.predict(val_h)
            return metric(self.data.val_y, preds)


    def validate(self, train_h, val_h, frozen_params, frozen_kwargs, epoch):
        results = {
            'val_score': self._get_score(train_h, val_h),
            'encoder_params': save_encoder_params(self.idx, self.id_, self.c_loss, self.aug_method, frozen_params, 
                                                  frozen_kwargs, epoch, config.RESULTS_PATH)
        }

        return results


    def get_best_encoder(self, results):
        scores = {}
        for epoch, res in results.items():
            scores[epoch] = res['val_score']
        
        best_epoch = max(scores.items(), key=lambda v: v[1])[0]
        
        return results[best_epoch]['encoder_params']