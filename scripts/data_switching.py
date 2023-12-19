import random
import pandas as pd
from tqdm import tqdm

# random seed 설정
random.seed(33)

# csv 파일 이름 설정
num = 1 # 홀수로 입력하기
mode = 'switching'

def random_switch(s):
    s = s.split()
    if len(s) < 2:
        return s
    
    idx1, idx2 = random.sample(range(len(s)), 2)
    s[idx1], s[idx2] = s[idx2], s[idx1]
    switched_sentence = ' '.join(s)

    return switched_sentence


def switching_process(path, dataset_name):
    data = pd.read_csv(path)
    rnd_list = []
    # dev.csv 적용
    if dataset_name == 'dev':
        for _, src_row in tqdm(data.iterrows(), desc='data_switching', total=len(data)):
            rnd_s1 = src_row['sentence_1']
            rnd_s2 = src_row['sentence_2']
            for _ in range(num):
                rnd_s1 = random_switch(''.join(rnd_s1))
                rnd_s2 = random_switch(''.join(rnd_s2))
            rnd_list.append([src_row['id'], src_row['source'], rnd_s1, rnd_s2, src_row['label'], src_row['binary-label']])
        rnd_df = pd.DataFrame(rnd_list, columns=data.columns)
    
    
    # train.csv 적용
    else:
        filtered_data = data[data['label']>=1] # label 1-5 늘리기
        for _, src_row in tqdm(filtered_data.iterrows(), desc='data_switching', total=len(filtered_data)):
            rnd_s1 = src_row['sentence_1']
            rnd_s2 = src_row['sentence_2']
            for _ in range(num):
                rnd_s1 = random_switch(''.join(rnd_s1))
                rnd_s2 = random_switch(''.join(rnd_s2))
            rnd_list.append([src_row['id'], src_row['source'], rnd_s1, rnd_s2, src_row['label'], src_row['binary-label']])
        rnd_df = pd.DataFrame(rnd_list, columns=data.columns)
        rnd_df = pd.concat([data, rnd_df])


    rnd_df.to_csv(f'../../data/{dataset_name}_cleaning_{mode}_{num}.csv', index=False)
    print(f'{dataset_name} 길이: {len(rnd_df)}')
if __name__ == '__main__':
    switching_process('../../data/train.csv', 'train')
    switching_process('../../data/dev.csv', 'dev')