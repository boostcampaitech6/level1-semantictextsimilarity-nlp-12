import pandas as pd
from tqdm import tqdm


# csv 파일 이름 설정
ver = 1
mode = 'reverse'


def reverse_process(path, dataset_name):
    data = pd.read_csv(path)
    rvs_list = []
    # train, dev.csv 적용
    for _, src_row in tqdm(data.iterrows(), desc='data_cleaning', total=len(data)):
        rvs_list.append([src_row['id'], src_row['source'], src_row['sentence_2'], src_row['sentence_1'], src_row['label'], src_row['binary-label']])

    cln_df = pd.DataFrame(rvs_list, columns=data.columns)
    cln_df.to_csv(f'../../data/{dataset_name}_{mode}_{ver}.csv', index=False)

if __name__ == '__main__':
    ### 불러올 데이터 path 수정 필요 ###
    reverse_process('../../data/train.csv', 'train')
    reverse_process('../../data/dev.csv', 'dev')