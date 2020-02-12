from datasets import Representations
from autoencoder import ConvAE2d
import torch
from trainer.utils import rgetattr

################################################
#           TRAINER RELATED VARIABLES          #
################################################

MODEL = ConvAE2d
DATASET = Representations
OPTIMIZER = torch.optim.Adam
LOSS = torch.nn.MSELoss
LOGDIR = 'results/pokemon/'

try:
    from apex.fp16_utils import FP16_Optimizer
    APEX = FP16_Optimizer
    apex = {
        'dynamic_loss_scale': True,
        'dynamic_loss_args': {'init_scale': 2 ** 10},
        'verbose': False
    }
    dtype = torch.float16
except ImportError:
    dtype = torch.float32


model = {
    'channels': [3*4**x for x in range(6)],
    'n_residual': (2, 2),
    'kernel_size': (3, 3),
    'activation': torch.relu,
    'affine': True,
    'padding': torch.nn.ZeroPad2d,
    'input_channels': 3
}

dataset = {
    'file': 'code_train.pt',  # adjust to correct filename
    'fraction': 1,
}

dataloader = {
    'batch_size': 32,
    'num_workers': 4
}

loss = {
    # loss keyword arguments go here
}

optimizer = {
    'lr': 0.0005
}

trainer = {
    'storage': 'storage.hdf5',
    'split_sample': lambda x: (x, x.float()),
    'transformation': lambda x: x[0],
    'loss_decay': 0.6
}

cuda =True
seed = 0

################################################
#       EXPERIMENT RELATED VARIABLES           #
################################################

validationset = {
    'file': 'codes_test.pt',  # adjust to correct filename
    'fraction': 1,
}

superfluous_strides = 0
n_samples = min(8, dataloader['batch_size'])
n_epochs = 1000
save_interval = n_epochs//10


def remove_stride(trainer):
    global superfluous_strides
    global model

    n_convs = len(model['channels']) - 1
    for i in range(superfluous_strides):
        rgetattr(trainer.model, f'conv{n_convs - i}.convolution').stride = (1, 1)
        rgetattr(trainer.model, f'dconv{i+1}.upsampling').scale_factor = 1.0

    return trainer


mod_trainer = ['remove_stride']
