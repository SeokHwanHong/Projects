# 언어 기반 색채화 및 초해상도

인하 AI Challenge에서 512×512 이미지를 256×256 흑백 이미지로 변환한 뒤, 텍스트 조건 기반 색채화와 2× 초해상도를 순차 적용한 프로젝트입니다.

## Pipeline

1. L-CAD로 텍스트 설명을 조건으로 한 색채화 수행
2. SwinIR로 256×256 결과를 512×512로 복원
3. CLIP 이미지 임베딩을 추출하고 제출 형식으로 압축

## Included code

- `src/swinir.py`: SwinIR 2× 업스케일링 실행
- `src/submission.py`: CLIP 임베딩 생성과 제출용 압축

색채화 단계는 [L-CAD](https://github.com/changzheng123/L-CAD), 초해상도 단계는 [SwinIR](https://github.com/JingyunLiang/SwinIR)를 기반으로 했습니다. 두 외부 프로젝트의 전체 소스와 가중치는 중복 수록하지 않습니다.

## Setup

```bash
pip install -r requirements.txt
```

L-CAD와 SwinIR 원본 저장소를 프로젝트 루트에 내려받고, 각 저장소에서 안내하는 가중치를 준비해야 합니다. 가중치와 대회 이미지·제출 파일은 저장소에 포함하지 않습니다.

## Run

```bash
python src/swinir.py --input_dir <colorized_images> --output_dir <super_resolved_images>
python src/submission.py --test_csv_path <test.csv> --input_img_dir <super_resolved_images>
```
