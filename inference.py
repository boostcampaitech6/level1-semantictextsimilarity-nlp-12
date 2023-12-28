import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

from data_loaders.data_loaders import Dataloader
from model.model import Model
from model.gru_model import GRUModel

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
            'path':'/data/ephemeral/home/level1-semantictextsimilarity-nlp-12/koelectra_43_0.92.ckpt',
            'model_name':'monologg/koelectra-base-v3-discriminator'
        }, 
        {
            'path': '/data/ephemeral/home/level1-semantictextsimilarity-nlp-12/sts-KR-ELECTRA-discriminator/982fqjhr/checkpoints/sts-epoch=56-val_pearson=0.933.ckpt',
            'model_name':'snunlp/KR-ELECTRA-discriminator'
        },
        {
            'path':'/data/ephemeral/home/level1-semantictextsimilarity-nlp-12/roberta_19_0.936.ckpt',
            'model_name':'xlm-roberta-large'
        },
        # {
        #     'path': '/data/ephemeral/home/level1-semantictextsimilarity-nlp-12/sts-epoch=55-val_pearson=0.933.ckpt',
        #     'model_name':'snunlp/KR-ELECTRA-discriminator'
        #     'add_gru': True
        # },
    ]
    
    OUT_PATH = './output_ensemble_3.csv'
    max_sentence_length = 100
    predictions_list = []
    for config in models_config:
        dataloader = Dataloader(config['model_name'], args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path, max_sentence_length)
        trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1, log_every_n_steps=1)

        if (config.get('add_gru') is not None and config['add_gru'] is True):
            model = GRUModel.load_from_checkpoint(config['path'])
        else:
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
