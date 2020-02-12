from trainer import Trainer, events, Config
from trainer.handlers import EventSave
from torch.utils.data import DataLoader
import argparse


def run_training(config):
    # init trainer
    trainer = Trainer.from_config(config, altered=True)
    for modification in config.mod_trainer:
        modification = getattr(config, modification)
        trainer = modification(trainer)

    # validation
    validation_loader = DataLoader(config.DATASET(**config.validationset), **config.dataloader)

    # setup events
    # trainer.register_event_handler(events.EACH_EPOCH, trainer, name='sample', sample=sample)
    trainer.register_event_handler(events.EACH_EPOCH, trainer.validate, dataloader=validation_loader)
    # trainer.register_event_handler(events.EACH_STEP, EventSave(), interval=config.save_interval)

    # train classifier
    if hasattr(config, 'n_steps'):
        trainer.train(n_steps=config.n_steps, resume=True)
    else:
        trainer.train(n_epochs=config.n_epochs, resume=True)


if __name__ == '__main__':

    # parse cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', choices=['pokemon', 'celeba', 'stl-10', 'scaling_celeba'], nargs=1, type=str,
                        required=True, metavar='config name', help='configuration file for the AE training')
    parser.add_argument('--bottleneck_size', '-s', choices=['small', 'mid', 'large'], nargs=1, type=str, metavar='size string',
                        help='choose bottleneck size (height x width) setting of the AE (overrides config)')
    parser.add_argument('--channels', '-ch', type=int, nargs=1, metavar='n_channels',
                        help='choose number of channels in the bottleneck (overrides config)')
    parser.add_argument('--seed', type=int, nargs=1, metavar='int',
                        help='seed for the model initialization and training run')
    parser.add_argument('--cpu', type=int, nargs=0,
                        help='run training on cpu')
    parser.add_argument('--epochs', type=int, nargs=1, metavar='int',
                        help='choose number of epochs for training (overrides config)')
    parser.add_argument('--steps', type=int, nargs=1, metavar='int',
                        help='choose number of steps for training (overrides config and overrules epochs)')
    args = parser.parse_args()

    # load config file
    config_file = f'configs/{args.config}_config.py'
    config = Config.from_file(config_file)

    # modify config based on cmd arguments
    if hasattr(args, 'bottleneck_size'):
        strides_to_remove = {'small': 0, 'mid': 1, 'large': 2}
        config.superfluous_strides = strides_to_remove[args.bottleneck_size]
    if hasattr(args, 'channels'):
        config.model['channels'][-1] = args.channels
    if hasattr(args, 'seed'):
        config.seed = args.seed
    if hasattr(args, 'cpu'):
        config.cuda = False
    if hasattr(args, 'epochs'):
        config.n_epochs = args.epochs
    if hasattr(args, 'steps'):
        setattr(config, 'steps', args.steps)

    # commence training
    run_training(config)


