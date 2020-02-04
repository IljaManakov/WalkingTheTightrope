import pandas as pd
import torch as pt
import numpy as np
from classifier import classify_and_evaluate


def reshape(x):
    global regress
    sample, labels, landmarks = x
    sample = sample.view(-1)
    return (sample, labels.float()) if not regress else (sample, landmarks)


def evaluate_attributes(trainer, valloader):

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
        tn = (~sub_pred & ~sub_label).sum().float().item()
        if not sub_label.sum():
            p = r = f1 = np.nan
        elif tp > 0:
            p = tp/(tp+fp)
            r = tp/(tp+fn)
            f1 = 2*p*r/(p+r)
        else:
            p = r = f1 = 0
        results.append({'class': i, 'f1': f1, 'precision': p, 'recall': r,
                        'true positive': int(tp),  'false positive': int(fp),
                        'false negative': int(fn), 'true negative': int(tn)})
    results = pd.DataFrame(results)
    return results


def evaluate_landmarks(trainer, valloader):
    criterion = pt.nn.MSELoss(reduction='none')
    samples, labels = next(iter(valloader))
    with pt.no_grad():
        preds = trainer.model(samples.cuda().float()).cpu()

    loss = criterion(preds, labels)
    results = {'left eye': loss[:, :2].mean().item(), 'right eye': loss[:, 2:4].mean().item(),
               'nose': loss[:, 4:6].mean().item(),
               'left mouth': loss[:, 6:8].mean().item(), 'right mouth': loss[:, 8:].mean().item()}
    results = pd.DataFrame([results])
    return results



#sizes = (3, 6, 12)
sizes = (3, 6, 12, 96)
modes = ('train', 'test')
dataset = 'celeba'
machine = 'hinton'
im_size = 96
n_epochs = 50
n_workers = 0
regress = False
evaluate = evaluate_attributes if not regress else evaluate_landmarks
criterion = pt.nn.BCEWithLogitsLoss() if not regress else pt.nn.MSELoss()
n_classes = 40 if not regress else 10
user = 'ilja'
outdir = f'/mnt/network/results/hinton/{user}/ae_bottleneck/'
folder = f'/home/{user}/Datasets/ae_bottleneck'

results = classify_and_evaluate(dataset=dataset, sizes=sizes, modes=modes, n_classes=n_classes, im_size=im_size,
                                outdir=outdir, evaluate=evaluate, reshape=reshape, folder=folder, criterion=criterion,
                                n_epochs=n_epochs, n_workers=n_workers)

