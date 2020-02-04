from trainer import Trainer, events, Config
from trainer.handlers import EventSave
from torch.utils.data import DataLoader


def run_training(config):
    # init trainer
    trainer = Trainer.from_config(config, altered=True)
    for modification in config.mod_trainer:
        modification = getattr(config, modification)
        trainer = modification(trainer)

    # validation
    validation_loader = DataLoader(config.DATASET(**config.validationset), **config.dataloader)
    sample = next(iter(validation_loader))
    sample = [s[0:config.n_samples] for s in sample]

    # setup monitoring
    # trainer.register_event_handler(events.EACH_EPOCH, trainer, name='sample', sample=sample)
    trainer.register_event_handler(events.EACH_STEP, trainer.validate, dataloader=validation_loader, interval=config.save_interval//10)
    trainer.register_event_handler(events.EACH_STEP, EventSave(), interval=config.save_interval)

    # train classifier
    if hasattr(config, 'n_steps'):
        trainer.train(n_steps=config.n_steps, resume=True)
    else:
        trainer.train(n_epochs=config.n_epochs, resume=True)


if __name__ == '__main__':

    config_file = '/home/config.py'
    config = Config.from_file(config_file)
    run_training(config)


