import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

from data_loaders.data_loaders import Dataloader
from model.model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--train_path', default='../data/train.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    args = parser.parse_args(args=[])

    models_config = [
        {
            # 'path':'/data/ephemeral/home/code/lightning_logs/version_2/checkpoints/sts-epoch=33-val_pearson=0.917.ckpt', # cleaning
            'path':'/data/ephemeral/home/code/lightning_logs/version_6/checkpoints/sts-epoch=18-val_pearson=0.925.ckpt',
            'model_name':'xlm-roberta-large'
        },
        {
            # 'path':'/data/ephemeral/home/code/lightning_logs/version_3/checkpoints/sts-epoch=40-val_pearson=0.927.ckpt',
            'path':'/data/ephemeral/home/code/lightning_logs/version_7/checkpoints/sts-epoch=28-val_pearson=0.914.ckpt',
            'model_name':'monologg/koelectra-base-v3-discriminator'
        },
        {
            # 'path':'/data/ephemeral/home/code/lightning_logs/version_4/checkpoints/sts-epoch=43-val_pearson=0.936.ckpt',
            'path':'/data/ephemeral/home/code/lightning_logs/version_8/checkpoints/sts-epoch=26-val_pearson=0.932.ckpt',
            'model_name':'snunlp/KR-ELECTRA-discriminator'
        },
    ]
    OUT_PATH = '../output/output_ensemble_data_switching_1.csv'

    predictions_list = []
    for config in models_config:
        dataloader = Dataloader(config['model_name'], args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
        trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1, log_every_n_steps=1)

        model = Model.load_from_checkpoint(config['path'])
        trainer.test(model=model, datamodule=dataloader)

        predictions = trainer.predict(model=model, datamodule=dataloader)
        predictions = list(round(float(i), 1) for i in torch.cat(predictions))
        predictions_list.append(predictions)

    predictions_final = [round(sum(x)/len(models_config),1) for x in zip(*predictions_list)]
    
    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('../data/sample_submission.csv')
    output['target'] = predictions_final
    output.to_csv(OUT_PATH, index=False)