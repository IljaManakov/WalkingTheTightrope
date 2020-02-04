import pandas as pd
import torch as pt
import numpy as np
from classifier import classify_and_evaluate


def reshape(x):
    sample, label = x
    sample = sample.view(-1)
    return sample, label


def evaluate(trainer, valloader):

    samples, labels = next(iter(valloader))
    with pt.no_grad():
        preds = pt.sigmoid(trainer.model(samples.cuda().float()).cpu())

    labels = labels.byte()
    thresh = preds.mean(dim=-1, keepdim=True)
    preds[preds < thresh] = 0
    preds[preds >= thresh] = 1
    preds = preds.byte()

    results = []
    for i, (sub_pred, sub_label) in enumerate(zip(pt.t(preds), pt.t(labels))):
        tp = (sub_pred & sub_label).sum().float().item()
        fp = (sub_pred & ~sub_label).sum().float().item()
        fn = (~sub_pred & sub_label).sum().float().item()
        if not sub_label.sum():
            p = r = f1 = np.nan
        elif tp > 0:
            p = tp/(tp+fp)
            r = tp/(tp+fn)
            f1 = 2*p*r/(p+r)
        else:
            p = r = f1 = 0
        results.append({'class': i, 'f1':f1, 'precision':p, 'recall':r,
                        'true positive':int(tp),  'false positive':int(fp), 'false negative':int(fn)})
    results = pd.DataFrame(results)
    return results


sizes = (4, 8, 16, 128)
modes = ('train', 'test')
dataset = 'pokemon'
machine = 'bengio'
n_classes = 18
im_size = 128
n_epochs = 200
n_workers = 0
criterion = pt.nn.BCEWithLogitsLoss()
user = 'ilja'
outdir = f'/mnt/network/results/bengio/{user}/ae_bottleneck'
folder = f'/home/{user}/Datasets/ae_bottleneck'

results = classify_and_evaluate(dataset=dataset, sizes=sizes, modes=modes, n_classes=n_classes, im_size=im_size,
                                outdir=outdir, evaluate=evaluate, reshape=reshape, folder=folder, criterion=criterion,
                                n_epochs=n_epochs, n_workers=n_workers)

