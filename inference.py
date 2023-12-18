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
    parser.add_argument('--model_name', default='monologg/koelectra-base-v3-discriminator', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--train_path', default='../data/train.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    args = parser.parse_args(args=[])

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=args.max_epoch, log_every_n_steps=1)

    # Inference part
    models_path = [
        '/data/ephemeral/home/level1-semantictextsimilarity-nlp-12/lightning_logs/version_36/checkpoints/sts-epoch=14-val_pearson=0.92.ckpt',
        '/data/ephemeral/home/level1-semantictextsimilarity-nlp-12/lightning_logs/version_50/checkpoints/sts-epoch=43-val_pearson=0.92.ckpt'
    ]
    OUT_PATH = './output_ensemble.csv'

    predictions_list = []
    for path in models_path:
        model = Model.load_from_checkpoint(path)
        model.eval()
        trainer.test(model=model, datamodule=dataloader)

        predictions = trainer.predict(model=model, datamodule=dataloader)
        predictions = list(round(float(i), 1) for i in torch.cat(predictions))
        predictions_list.append(predictions)

    predictions_final = [round(sum(x)/len(models_path),1) for x in zip(*predictions_list)]
    
    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('../data/sample_submission.csv')
    output['target'] = predictions_final
    output.to_csv(OUT_PATH, index=False)
