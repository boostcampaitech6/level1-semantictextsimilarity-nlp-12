import argparse
import random

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_loaders.data_loaders import Dataloader
from model.model import Model
from model.gru_model import GRUModel

# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_logger', default=False, type=bool)
    parser.add_argument('--config', default='./configs/KR-ELECTRA-discriminator.yml', type=str)
    parser.add_argument('--add_gru', default=True, type=bool)
    args = parser.parse_args(args=[])

    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(cfg['model_name'], cfg['batch_size'], cfg['shuffle'], cfg['train_path'], 
                            cfg['dev_path'],cfg['test_path'], cfg['predict_path'], cfg['max_sentence_length'])
    
    if (args.add_gru==True):
        model = GRUModel(cfg['model_name'], cfg['learning_rate'])
    else:
        model = Model(cfg['model_name'], cfg['learning_rate'])
    print(model)

    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=1,
        save_top_k=1,
        monitor="val_pearson",      #'val_loss' val_pearson
        mode='max',                 #'min' max
        filename="sts-{epoch:02d}-{val_pearson:.3f}",
    )

    if args.wandb_logger:
        trainer = pl.Trainer(
            accelerator=cfg['accelerator'], 
            devices=1, 
            max_epochs=cfg['max_epoch'], 
            callbacks=[checkpoint_callback],
            log_every_n_steps=1,
            logger=WandbLogger(project=f"sts-{cfg['name']}")
        )
    else:
        trainer = pl.Trainer(
            accelerator=cfg['accelerator'], 
            devices=1, 
            max_epochs=cfg['max_epoch'], 
            callbacks=[checkpoint_callback],
            log_every_n_steps=1,
        )

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)