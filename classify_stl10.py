
import pandas as pd
import torch as pt
import sys
from classifier import classify_and_evaluate


def reshape(x):
    sample, label = x
    one_hot = pt.zeros(10)
    one_hot[label.item()] = 1
    sample = sample.view(-1)
    return sample, one_hot


def evaluate(trainer, valloader):
    emb = pt.nn.Embedding(10, 10)
    emb.weight.data = pt.eye(10)
    emb.weight.requires_grad = False

    samples, labels = next(iter(valloader))
    with pt.no_grad():
        preds = pt.softmax(trainer.model(samples.cuda().float()).cpu(), -1)

    labels = labels.byte()
    # preds[preds < 0.5] = 0
    # preds[preds >= 0.5] = 1
    preds = emb(preds.max(-1).indices).byte()

    results = []
    for i, (sub_pred, sub_label) in enumerate(zip(pt.t(preds), pt.t(labels))):
        tp = (sub_pred & sub_label).sum().float().item()
        fp = (sub_pred & ~sub_label).sum().float().item()
        fn = (~sub_pred & sub_label).sum().float().item()
        p = tp/(tp+fp)
        r = tp/(tp+fn)
        f1 = 2*p*r/(p+r)
        results.append({'class': i, 'f1':f1, 'precision':p, 'recall':r,
                        'true positive':int(tp),  'false positive':int(fp), 'false negative':int(fn)})
    results = pd.DataFrame(results)
    return results


sizes = (3, 6, 12, 96)
modes = ['train']
dataset = 'stl10'
machine = 'hinton'
im_size = 96
n_classes = 10
n_epochs = 200
n_workers = 0
criterion = pt.nn.BCEWithLogitsLoss()
user = 'ilja'
outdir = f'/mnt/network/results/hinton/ilja/ae_bottleneck'
folder = f'/home/{user}/Datasets/ae_bottleneck'

results = classify_and_evaluate(dataset=dataset, sizes=sizes, modes=modes, n_classes=n_classes, im_size=im_size,
                                outdir=outdir, evaluate=evaluate, reshape=reshape, folder=folder, criterion=criterion,
                                n_epochs=n_epochs, n_workers=n_workers)


