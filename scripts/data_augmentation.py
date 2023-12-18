from tqdm.auto import tqdm

import random
import pandas as pd

def data_augmentation(path, num_of_samples=1):
    data = pd.read_csv(path)

    aug_rows = []
    for src_idx, src_row in tqdm(data.iterrows(), desc='data_augmentation', total=len(data)):
        aug_rows.append([f'{src_idx}_{src_idx}', 'augmented', src_row['sentence_2'], \
                        src_row['sentence_1'], src_row['label'], src_row['binary-label']])
        for trg_idx, trg_row in data.sample(n=num_of_samples).iterrows():
            if src_idx == trg_idx:
                continue
            aug_rows.append([f'{src_idx}_{trg_idx}', 'augmented', src_row['sentence_1'], \
                            trg_row['sentence_2'], round(random.uniform(0.1, 2.0),1), 0])
            # aug_rows.append([f'{src_idx}_{trg_idx}', 'augmented', src_row['sentence_1'], \
                            #  trg_row['sentence_2'], 0.3, 0])
    
    aug_df = pd.DataFrame(aug_rows, columns=data.columns) 
    data = pd.concat([data, aug_df])
    print('num of augmented data', len(aug_rows))
    data.to_csv('./train_augmented_4.csv')
       

if __name__ == '__main__':
    data_augmentation('../../data/train.csv', 4)