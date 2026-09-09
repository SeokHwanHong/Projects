# 노르웨이 전력망 손실 예측

2017년 12월부터 2020년 5월까지의 시간 단위 전력망 데이터를 활용해 세 전력망의 손실을 예측한 시계열 프로젝트입니다. 계절 차분 기반 SARIMAX와 파생 변수 기반 XGBoost를 비교했습니다.

## Notebook

- `notebooks/norway-power-grid-forecasting.ipynb`: 데이터 확인, EDA, SARIMAX, XGBoost 학습과 예측 비교

## Data

학습 데이터는 저장소에 포함하지 않습니다. 노트북에서 참조하는 원본 train/test 파일을 `data/`에 준비해야 합니다.

## Main features

- 시간 파생 변수: hour, dayofweek, month
- 상호작용·비율 변수: `load_square`, `ratio`, `tem_prod`
- 계절성: 24시간 주기와 시계열 차분 검토

EDA 그래프와 모델 결과는 노트북 출력으로 함께 보존했습니다.
