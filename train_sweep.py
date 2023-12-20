import argparse
import random

import yaml
import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_loaders.data_loaders import Dataloader
from model.model import Model

# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/KR-ELECTRA-discriminator.yml', type=str)
    args = parser.parse_args(args=[])

    sweep_config = {
        'method': 'random',
        'parameters': {
            'lr':{
                'distribution': 'uniform',  
                'min':1e-6,                 
                'max':1e-4                 
            },
            # 'batch_size': {
                # 'values': [64, 128, 256]
            # },
            # 'epochs': {
                # 'values': [10, 30, 60]
            # }
        },
        'metric': {
            'name':'val_pearson', 
            'goal':'maximize'
        }
    }

    def sweep_train(sweep_config=None):
        wandb.init(config=sweep_config)

        with open(args.config, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        dataloader = Dataloader(cfg['model_name'], cfg['batch_size'], cfg['shuffle'], cfg['train_path'], 
                                cfg['dev_path'],cfg['test_path'], cfg['predict_path'], cfg['max_sentence_length'])
    
        model = Model(cfg['model_name'], wandb.config.lr)
        print(model)

        checkpoint_callback = ModelCheckpoint(
            every_n_train_steps=1,
            save_top_k=1,
            monitor="val_pearson",      #'val_loss' val_pearson
            mode='max',                 #'min' max
            filename="sts-{epoch:02d}-{val_pearson:.3f}",
        )
        
        trainer = pl.Trainer(
            accelerator=cfg['accelerator'], 
            devices=1, 
            max_epochs=cfg['max_epoch'], 
            callbacks=[checkpoint_callback],
            log_every_n_steps=1,
            logger=WandbLogger(project=f"sts-{cfg['name']}")
        )
        
        # Train part
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)


    sweep_id = wandb.sweep(
        sweep=sweep_config,     
        project='tuning-1'  
    )
    wandb.agent(
        sweep_id=sweep_id,      
        function=sweep_train,   
        count=5                
    )
    

    
    