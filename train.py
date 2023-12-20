import argparse
import random
import os
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data_loaders.data_loaders import Dataloader
from model.model import Model

# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

# config.yaml 불러오기
CONFIG_PATH = './config/'

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

        return config
    
config = load_config("test_config.yaml")

# dataloader 초기화
dataloader = Dataloader(config["model_name"], config["batch_size"], config["shuffle"], config["train_path"], config["dev_path"],
                            config["test_path"], config["predict_path"])

# model 초기화
if config["fine_tuning"]:
    checkpoint_name = config["checkpoint_name"]
    PATH = f'/data/ephemeral/home/level1-semantictextsimilarity-nlp-12/lightning_logs/version_40/checkpoints/{checkpoint_name}.ckpt'
    model = Model.load_from_checkpoint(PATH)
else:
    model = Model(config["model_name"], float(config["learning_rate"]))
print(model)

checkpoint_callback = ModelCheckpoint(
    every_n_train_steps=1,
    save_top_k=1,
    monitor="val_pearson",      #'val_loss' val_pearson
    mode='max',                 #'min' max
    filename="sts-{epoch:02d}-{val_pearson:.2f}",
)

trainer = pl.Trainer(
    accelerator="gpu", 
    devices=1, 
    max_epochs=config["max_epoch"], 
    callbacks=[checkpoint_callback],
    log_every_n_steps=1
)

# Train part
trainer.fit(model=model, datamodule=dataloader)
trainer.test(model=model, datamodule=dataloader)
