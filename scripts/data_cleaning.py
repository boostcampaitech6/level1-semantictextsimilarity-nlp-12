import re
import pandas as pd
from tqdm import tqdm



def data_cleaning(s):
    clean_s = re.sub(r'<PERSON>', ' ', s) # Slack에 있는 <PERSON> 제거
    clean_s = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z.,?!]', ' ', clean_s)
    clean_s = re.sub(r'(\w)\1+', r'\1', clean_s)
    clean_s = re.sub(r'(\W)\1+', r'\1', clean_s) # 연속된 특수문자(.,?!) 하나로 통일
    clean_s = ' '.join(clean_s.split()) # 연속된 공백 하나로 바꿈
    return clean_s


def cleaning_process(path, dataset_name):
    data = pd.read_csv(path)
    cln_list = []
    # train, dev.csv 적용
    if dataset_name != 'test':
        for _, src_row in tqdm(data.iterrows(), desc='data_cleaning', total=len(data)):
            cln_s1 = data_cleaning(src_row['sentence_1'])
            cln_s2 = data_cleaning(src_row['sentence_2'])
            # print(cln_s1)
            cln_list.append([src_row['id'], src_row['source'], cln_s1, cln_s2, src_row['label'], src_row['binary-label']])

        cln_df = pd.DataFrame(cln_list, columns=data.columns)
        cln_df.to_csv(f'../../data/{dataset_name}.csv', index=False)
    
    # test.csv 적용
    else:
        for _, src_row in tqdm(data.iterrows(), desc='data_cleaning', total=len(data)):
            cln_s1 = data_cleaning(src_row['sentence_1'])
            cln_s2 = data_cleaning(src_row['sentence_2'])
            # print(cln_s1)
            cln_list.append([src_row['id'], src_row['source'], cln_s1, cln_s2])

        cln_df = pd.DataFrame(cln_list, columns=data.columns)
        cln_df.to_csv(f'../../data/{dataset_name}_.csv', index=False)

if __name__ == '__main__':
    cleaning_process('../../data/train.csv', 'train')
    cleaning_process('../../data/dev.csv', 'dev')