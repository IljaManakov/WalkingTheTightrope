import os
import torch as pt
from torch.utils.data import DataLoader
from torch.nn import Module, Linear
import pandas as pd
from trainer.utils import set_seed
from datasets import Representations
from sklearn.exceptions import ConvergenceWarning
from fastai.basics import Learner, DataBunch, accuracy, AUROC, DatasetType
from fastai.callbacks import EarlyStoppingCallback
from functools import partial
import argparse
from trainer import Config

import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)


class Classifier(Module):

    def __init__(self, sizes, activation=pt.relu, activation_on_final_layer=False):
        """
        simple feed-forward NN
        @param sizes: list of sizes of the layers of the network, first entry is the input size
        @param activation: activation function to use after intermediat FC layers
        @param activation_on_final_layer: bool whether to apply activation at the output
        """
        super(Classifier, self).__init__()
        self.activation = activation
        self.activation_on_final_layer = activation_on_final_layer
        self.layers = []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.layers.append(Linear(n_in, n_out))
            self.add_module(f'layer{i}', self.layers[-1])

    def forward(self, x):
        """
        forward pass through the network
        @param x: input tensor of shape (batch, sizes[0])
        @return: prediction of shape (batch, sizes[-1])
        """
        for layer in self.layers:
            x = layer(x)
            if layer is not self.layers[-1] or self.activation_on_final_layer:
                x = self.activation(x)

        return x


def classify_and_evaluate(data_filename, n_classes, outdir, reshape, evaluate, criterion, n_workers=0, cpu=False):
    """
     trains a linear classifier on the provided data for three different seeds and performs evaluation
    @param data_filename: filename of the file containing the exported representations
    @param n_classes: number of classes in the labels
    @param outdir: directory in which to write results
    @param reshape: function for reshaping the latent code
    @param evaluate: function for evaluating the classifier
    @param criterion: loss function for the classifier
    @param n_workers: number of workers for the dataloaders
    @param cpu: bool indicating whether to train on cpu
    @return: results as pandas dataframe
    """

    results = []
    for s in (0, 1, 2):

        # fix seed
        set_seed(s)

        # prepare data 60-20-20
        fraction = 0.6
        trainset = Representations(data_filename, fraction=fraction, transformation=reshape, shuffle=s+1)
        valset = Representations(data_filename, fraction=fraction-1, transformation=reshape, shuffle=s+1)
        valset.reduce_size(end=0.5)
        testset = Representations(data_filename, fraction=(fraction-1)/2, transformation=reshape, shuffle=s+1)

        # wrap with dataloaders
        trainloader = DataLoader(trainset, batch_size=128, num_workers=n_workers)
        valloader = DataLoader(valset, batch_size=len(valset), num_workers=n_workers)
        testloader = DataLoader(testset, batch_size=len(testset), num_workers=n_workers)
        data = DataBunch(trainloader, valid_dl=valloader, test_dl=testloader)

        # train classifier
        in_size = next(iter(trainloader))[0].shape[1:]
        model = Classifier([in_size, n_classes]) if cpu else Classifier([in_size, n_classes]).cuda()
        learner = Learner(data=data, model=model, loss_func=criterion,
                          callback_fns=[partial(EarlyStoppingCallback, monitor='valid_loss',
                                                min_delta=0.001, patience=1)])
        learner.fit_one_cycle(100)

        # run evaluation
        with pt.no_grad():
            result = evaluate(*learner.get_preds(ds_type=DatasetType.Test))
        result['run'] = s
        results.append(result)

        pt.cuda.empty_cache()

    results = pd.concat(results, ignore_index=True)
    results.to_csv(os.path.join(outdir, f'results.csv'))
    return results


if __name__ == '__main__':

    # parse cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', choices=['pokemon', 'celeba', 'stl-10'], type=str,
                        required=True, metavar='config name', help='configuration file for the classifier training')
    parser.add_argument('--dataset_file', '-d', type=str, metavar='filename', required=True,
                        help='filename of the file containing the exported representations')
    parser.add_argument('--outdir', '-o', type=str, metavar='directory',
                        help='directory in which to write results (overrides config)')
    parser.add_argument('--cpu', default=False, action='store_true',
                        help='run training on cpu')
    args = parser.parse_args()

    # load config file
    config_file = f'classification/classify_{args.config}.py'
    config = Config.from_file(config_file)

    # modify config based on cmd arguments
    if args.outdir:
        config.outdir = args.outdir
    if args.cpu:
        setattr(config, 'cpu', True)

    # commence training
    classify_and_evaluate(**config)
