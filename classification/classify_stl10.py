import pandas as pd
import torch as pt
from sklearn.metrics import f1_score


def reshape(x):
    sample, label = x
    one_hot = pt.zeros(10)
    one_hot[label.item()] = 1
    sample = sample.view(-1)
    return sample.float(), label.long()


def evaluate(preds, labels):

    labels = labels.cpu().float()
    preds = preds.cpu().float().max(1)[1]
    results = []
    for i, f1 in enumerate(f1_score(labels, preds, average=None)):
        results.append({'class': i, 'auc': f1, 'f1': f1})
    results = pd.DataFrame(results)
    return results


n_classes = 10
criterion = pt.nn.CrossEntropyLoss()
outdir = 'results/classification/stl-10'
