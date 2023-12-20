import argparse
import random

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data_loaders.data_loaders import Dataloader
from model.model import Model

from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(project='sts-origin_datasets',
                           name = 'xlm-roberta-large',
                           log_model='all',
                           )
# 시간 기록
import time
from datetime import timedelta
start_time = time.time()

# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='xlm-roberta-large', type=str)
    # parser.add_argument('--model_name', default='monologg/koelectra-base-v3-discriminator', type=str)
    # parser.add_argument('--model_name', default='snunlp/KR-ELECTRA-discriminator', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_epoch', default=50, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--train_path', default='../data/train.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    parser.add_argument('--fine_tuning', default=False, type=bool)
    args = parser.parse_args(args=[])

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)

    if args.fine_tuning:
        checkpoint_name = 'sts-epoch=33-val_pearson=0.917'
        PATH = f'/data/ephemeral/home/code/lightning_logs/version_2/checkpoints/{checkpoint_name}.ckpt'
        model = Model.load_from_checkpoint(PATH)
    else:
        model = Model(args.model_name, args.learning_rate)
    print(model)

    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=1,
        save_top_k=1,
        monitor="val_pearson",      #'val_loss' val_pearson
        mode='max',                 #'min' max
        filename="sts-origin_dataset-{epoch:02d}-{val_pearson:.3f}",
    )

    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1, 
        max_epochs=args.max_epoch, 
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        logger=wandb_logger # WandB 추가
    )

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

end_time = time.time()
execution_time = end_time - start_time
execution_timedelta = timedelta(seconds=execution_time)
print(f"코드 실행 시간: {execution_timedelta}")