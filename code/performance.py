import numpy as np
from sklearn.model_selection import GridSearchCV

import config
from utils import get_metric_scorer, get_clf
from utils import to_dict_of_lists, get_hiddens
from metrics import pre_rec_curve, standard_error, mean_per_metric


class CVPerformance:
    def __init__(self, data, folds, clfs, params_grids, metric, metrics_list, id_, 
                 task='spl', results_collector=None, n_jobs=None):
        self.data = data
        self.folds = folds
        self.clfs = clfs
        self.params_grids = params_grids
        self.metric = metric
        self.metrics_list = metrics_list
        self.id_ = id_
        self.task = task
        self.results_collector = results_collector
        
        self.n_jobs = n_jobs
        if self.n_jobs is None:
            self.n_jobs = len(self.clfs)
        
        self.results_ = {}


    def _run_grid_search(self, train_X, train_y, test_X, test_fold, clf, params_grid, folds, benchmarking):
        clf_ = get_clf(*clf, {'probability': True}) if clf[1] == 'SVC' else get_clf(*clf)
        metric, scorer = get_metric_scorer(self.metric)
        grid_search = GridSearchCV(clf_, params_grid, cv=folds, scoring=scorer, n_jobs=config.NUM_CORES)         
        grid_search.fit(train_X, train_y)

        test_X = self.data.train_X[test_fold] if test_X is None else test_X

        preds = grid_search.best_estimator_.predict(test_X)
        probs = grid_search.best_estimator_.predict_proba(test_X)[:, 1]
        test_score = {self.metric: metric(self.data.train_y[test_fold], probs if self.metric == 'avg_precision' else preds)}
        
        other_metrics = self.metrics_list.copy()
        other_metrics.remove(self.metric)
        for other_metric in other_metrics:
            other_metric_, _ = get_metric_scorer(other_metric)
            test_score.update({
                other_metric: other_metric_(self.data.train_y[test_fold], probs if other_metric == 'avg_precision' else preds)
            })
        test_score.update({
            'labels': self.data.train_y[test_fold],
            'probs': probs
            })

        if benchmarking and clf[1] == 'SVC':
            preds = grid_search.best_estimator_.predict(self.data.val_X)
            val_score = metric(self.data.val_y, preds)
            
            return grid_search.best_params_, grid_search.best_score_, test_score, val_score
            
        return grid_search.best_params_, grid_search.best_score_, test_score, None


    def _spl_cv_performance(self, clf, params_grid):
        results = {clf[1]: {}}
        scores = []

        for idx, (train_fold, test_fold) in enumerate(self.folds.split(self.data.train_X, self.data.train_y)):
            X, y, benchmarking = self.data.train_X[train_fold], self.data.train_y[train_fold], True

            best_params, best_score, test_score, val_score = self._run_grid_search(X, y, None, test_fold, clf, 
                                                                                   params_grid, self.folds, 
                                                                                   benchmarking)
            scores.append(test_score)
            results[clf[1]].update({
                idx: {
                    'best_params': best_params,
                    'best_score': best_score,
                    'test_score': test_score
                }
            })
            if val_score:
                results[clf[1]][idx].update({'val_score': val_score})

        scores, labels, probs = to_dict_of_lists(scores)
        results[clf[1]].update({
            'standard_error': standard_error(scores),
            'mean': mean_per_metric(scores),
            'cv_scores': scores,
            'pre_rec_curve': pre_rec_curve(labels, probs)
        })

        return results


    def _cl_cv_performance(self, clf, params_grid):
        results = {clf[1]: {}}
        clustering_res = []
        scores = []

        for idx, (train_fold, test_fold) in enumerate(self.folds.split(self.data.train_X, self.data.train_y)):
            checkpoint = self.results_collector[idx]['best']
            train_h, test_h = get_hiddens(checkpoint, self.id_, self.data, config.RESULTS_PATH, (train_fold, test_fold))

            best_params, best_score, test_score, _ = self._run_grid_search(train_h, self.data.train_y[train_fold], test_h, 
                                                                           test_fold, clf, params_grid, self.folds, False)
            scores.append(test_score)
            results[clf[1]].update({
                idx: {
                    'best_params': best_params,
                    'best_score': best_score,
                    'test_score': test_score
                }
            })
                                                                    
        scores, labels, probs = to_dict_of_lists(scores)
        results[clf[1]].update({
                'standard_error': standard_error(scores),
                'mean': mean_per_metric(scores),
                'cv_scores': scores,
                'pre_rec_curve': pre_rec_curve(labels, probs)
        })

        return results


    def measure(self):
        for clf in self.clfs:
            if self.task == 'spl':
                results = self._spl_cv_performance(clf, self.params_grids[clf[1]])
                self.results_.update(results)
            elif self.task == 'cl':
                results = self._cl_cv_performance(clf, self.params_grids[clf[1]])
                self.results_.update(results)
