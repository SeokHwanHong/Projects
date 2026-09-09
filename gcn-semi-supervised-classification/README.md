# GCN 반지도 노드 분류

Facebook GEMSEC 그래프 데이터에서 노드 특성이 없는 조건을 다룬 반지도 노드 분류 프로젝트입니다. LDP 기반 특징을 사용하고, 불균형 클래스 문제를 균형 샘플링으로 조정해 GCN·Propagation 결과를 비교했습니다.

## Notebooks

- `notebooks/GCN.ipynb`: GCN 학습
- `notebooks/GCN-Utils.ipynb`: 데이터·그래프 처리 유틸리티
- `notebooks/GCN-Inference.ipynb`: 추론과 시각화

## Data

`gemsec_facebook_dataset`은 저장소에 포함하지 않습니다. 데이터셋을 `data/`에 준비한 뒤 노트북의 경로 설정을 확인해야 합니다.

## Environment

Python, PyTorch, PyTorch Geometric, NetworkX, scikit-learn, matplotlib 환경을 사용했습니다. 정확한 버전은 사용 환경에 맞춰 별도 고정할 예정입니다.
