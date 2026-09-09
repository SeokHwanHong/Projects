# LG Aimers 6기 — 난임 성공 예측

난임 시술 정보를 바탕으로 임신 성공 여부를 예측한 대회 프로젝트입니다. 범주형 변수·범위형 수치·결측의 의미를 구분하고, CatBoost와 LightGBM 앙상블을 비교했습니다.

## Included

- `outputs/catboost-training.json`: CatBoost 학습 로그
- `notes/feature-engineering.md`: 원본 분석 기록에서 확인한 피처 설계 원칙

## Data

대회 데이터에는 민감한 의료 정보가 포함되어 있어 저장소에 올리지 않습니다. LG Aimers 대회에서 제공한 `train.csv`, `test.csv`와 데이터 명세를 별도로 준비해야 합니다.

## Reproducibility note

공유된 프로젝트 폴더에는 최종 학습 노트북 또는 스크립트가 포함되어 있지 않았습니다. 따라서 이 폴더는 실제로 보존된 학습 로그와 피처 엔지니어링 기록만 담습니다. 원본 학습 코드가 확보되면 `src/`와 `notebooks/`에 추가합니다.
