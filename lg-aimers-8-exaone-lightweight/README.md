# LG Aimers 8기 — EXAONE 4.0-1.2B 경량화

RAG 기반 질의응답 과제에서 EXAONE 4.0-1.2B에 4-bit QLoRA를 적용한 프로젝트입니다. 제한된 자원에서 배포 가능한 모델 구성을 목표로 했습니다.

## Experiment record

- 데이터 분할: 1,000개 QA를 9:1로 분할
- 학습 방식: 4-bit QLoRA, BF16 연산
- 대상 모듈: q/k/v/o/gate/up/down projection
- LoRA 설정: r=16, alpha=32, dropout=0.1
- 학습 설정: 1 epoch, learning rate 2e-5, cosine scheduler, warmup ratio 0.05
- 점수: 0.5200 → 0.5938

## Reproducibility note

공유된 자료에는 프로젝트 정리 PDF만 있고 학습 코드·데이터·adapter 가중치는 포함되어 있지 않습니다. 원본 코드가 제공되면 `src/`, 데이터 준비 절차, 추론 예제를 추가합니다.
