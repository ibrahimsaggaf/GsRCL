import numpy as np
from sklearn.metrics import precision_recall_curve, matthews_corrcoef
from sklearn.metrics import f1_score, average_precision_score


metrics_list = ['avg_precision', 'mcc', 'f1']

def pre_rec_curve(y, probs):
    precision, recall, _ = precision_recall_curve(y, probs)
    avg_p = avg_precision(y, probs)
    return precision, recall, avg_p


def avg_precision(y, probs):
    return average_precision_score(y, probs)


def mcc(y, preds):
    return matthews_corrcoef(y, preds)


def f1(y, preds):
    return f1_score(y, preds)


def standard_error(cv_results):
    sd = {}
    for metric, results in cv_results.items():
        sd[metric] = np.std(results) / np.sqrt(len(results))
    return sd
    
    
def mean_per_metric(cv_results):
    mean = {}
    for metric, results in cv_results.items():
        mean[metric] = np.mean(results)
    return mean