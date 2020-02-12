import pandas as pd
import torch as pt
import numpy as np
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


def reshape(x):
    sample, labels, landmarks = x
    sample = sample.view(-1)
    return sample.float(), labels.float()


def evaluate(preds, labels):

    labels = labels.cpu().float().byte()
    preds = preds.cpu().float()
    results = []
    for i in range(labels.shape[-1]):
        cls_labels = labels[:, i]
        cls_preds = preds[:, i]

        if cls_labels.sum() == 0:
            results.append({'class': i, 'auc': np.nan})
        else:
            results.append({'class': i, 'auc': roc_auc_score(cls_labels, cls_preds, None)})
    results = pd.DataFrame(results)
    return results


criterion = pt.nn.BCEWithLogitsLoss()
n_classes = 40
outdir = 'results/classification/celeba'
