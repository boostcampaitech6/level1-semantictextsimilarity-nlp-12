## Semantic Text Similarity 
- Semantic Text Similarity(STS)는 두 문장쌍이 얼마나 의미적으로 유사한지 판단하는 task로, 두 문장이 서로 동등한 양 방향성을 가진다는 것이 특징이며 qa, 요약, 데이터 증강 등에 응용되고 있다.
- 이번 프로젝트는 두 문장의 유사도를 0부터 5까지의 소수점 한 자리의 연속적인 숫자로 예측하는 모델을 구현하는 것이 목표이다.
- 기본적으로 pretrained model을 finetuing 하는 방식으로 수행한다. 그 외 모델 구조 변형, 앙상블 등의 시도를 적용해본다.
  
## Dataset
- 총 10,974개의 한국어 문장 쌍으로 구성된 데이터를 제공받았다. Train 데이터 9,324개, Dev 데이터 550개, Test 데이터 1,100개로 구성된다.
- 라벨 점수는 0에서 5 사이의 실수로 두 문장 간의 유사도를 나타낸다.
  - ex) 5.0~ : 두 문장의 핵심 내용이 동일하며, 부가적인 내용들도 동일함
  - 3.0 ~ 3.9 : 두 문장의 핵심 내용은 대략적으로 동등하지만, 부가적인 내용에 무시하기 어려운 차이가 있음
  - 1.0 ~ 1.9 : 두 문장의 핵심 내용은 동등하지 않지만, 비슷한 주제를 다루고 있음

## Evaluation Metric
- Pearson correlation coefficient

## Data Augmentation
- 두 문장의 순서 역전
- 오타제거, 기존 문장을 구성하는 단어의 순서 변경 혹은 일부 제거
- 무작위로 서로 다른 두 문장 쌍을 조합해서 새로운 데이터 샘플 생성

## How to Run
`python train.py --config=./configs/KR-ELECTRA-discriminator.yml`

`python train_sweep.py --config=./configs/KR-ELECTRA-discriminator.yml`

`python inference.py`

## Result
<img width="1173" alt="Screenshot 2023-12-28 at 5 21 20 PM" src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-12/assets/76895949/5df7d16e-131c-49d6-bd92-731044d54cf0">

<img width="892" alt="Screenshot 2023-12-28 at 5 21 49 PM" src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-12/assets/76895949/93982022-459d-4296-9f68-ddb80556d30b">
<img width="832" alt="Screenshot 2023-12-28 at 5 21 54 PM" src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-12/assets/76895949/24a2cf96-41bb-4545-bf2e-95262f50c1b9">

- snunlp/KR-ELECTRA-discriminator (val_pearson : 93.6)
- monologg/koelectra-base-v3-discriminator (val_pearson : 92.5)
- xlm-roberta-large(val_pearson : 93.5)
- 위 세 모델을 최종 앙상블하여 prediction 수행

