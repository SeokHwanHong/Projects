# LG Aimers 6기 — 난임 환자 임신 성공 예측

난임 시술 기록으로 `임신 성공 여부`를 예측한 팀 프로젝트다. **팀 성과는 예선 상위 3%·오프라인 본선 진출**이다.

**주요 역할은 모델의 성능 지표와 feature importance를 참고한 파생변수 탐색이다.** 시술 이력과 변수의 의미를 해석하고, 관계를 표현하는 후보를 검토한 경험에 초점을 둔다.

## 개인 역할과 팀 수행

| 구분 | 설명 | 
| --- | --- | 
| 개인 역할 | 모델 평가 지표·중요도를 참고한 파생변수 탐색 |
| 담당 후보 | 과거 시술·임신·출산 비율, 원인 플래그 합계·관측 여부, 수치/범주 표현·상호작용 |
| 팀 수행 | CatBoost·LightGBM·AutoGluon 학습·튜닝·예측 결합 | 
| 팀 성과 | 예선 상위 3%, 오프라인 본선 진출 | 

## 문제·데이터·평가

타깃은 시술 기록의 임신 성공 여부(0/1), 평가는 ROC-AUC다. 입력에는 연령 범주, 시술/임신/출산 이력, 불임 원인, 난자·배아 수량과 경과일 등이 있다. 기존 [집계 기록](outputs/artifact-summary.json)은 원본 train 256,351행·69열, test 90,067행, 양성 비율 약 25.83%를 보고한다.

## 파생변수 탐색의 핵심 접근

**시술 맥락·변수 의미 해석 → 관계를 표현하는 후보 구성 → 성능 지표·중요도로 검토 → 관찰 → 한계 확인.** 

| 후보 영역 | 코드에서 확인한 예 | 읽어야 할 의미 |
| --- | --- | --- |
| 과거 이력 | 총/IVF/DI 시술 대비 임신, 임신 대비 출산 비율 | 과거 기록 간 관계; 개인의 임상 성공확률 아님 |
| 원인·관측 여부 | 원인 플래그 합계, `임신 시도 여부` | 합계는 중증도가 아니며 해당 이름은 실제로 경과 연수의 관측 여부 |
| 수치·범주·조합 | `_num`·`_cate`, `이식 범주 결합`, 수량×경과일 | 같은 정보를 다른 방식으로 표현하는 탐색 |

저장된 중요도 표에서 `이식 범주 결합`은 CatBoost 4위, LightGBM 1위, AutoGluon 2위다. 모델별 중요도 방식·척도가 다르고 AutoGluon에는 학습에 사용한 train_data를 전달했다. [변수별 산식·주의·귀속 상태](notes/feature-engineering.md)와 [셀별 원출력 발췌](outputs/notebook-evidence.json)를 함께 확인할 수 있다.

## 모델 비교와 최종 제출

| 구성 | 확인된 내용 |
| --- | --- |
| CatBoost | 저장 출력 5-fold 평균 AUC **0.7402164844** |
| LightGBM | 저장 출력 5-fold 평균 AUC **0.7395960954** |
| AutoGluon | 내부 validation leaderboard와 train_data 중요도 출력 존재 |
| CatBoost+LightGBM | 두 예측의 동일 가중치 평균 코드 존재 |
| 최종 3모델 결합 | `(p_catboost + p_lightgbm + p_autogluon) / 3` |


별도 보존 CatBoost 로그의 최고 AUC **0.7237084142**는 위 노트북 출력과 다른 실험 기록이다. [분할·평가 데이터·근거와 한계](notes/experiment-evidence.md)를 참고한다.

## 코드 탐색과 실행 범위

| 자료 | 용도 |
| --- | --- |
| [원본 코드 발췌](src/feature-exploration-original.py) | 생성 함수·train/test 적용·CV·중요도 집계·평균 결합 확인 |
| [파생변수 기록표](notes/feature-engineering.md) | 원본 열·산식·자료형·설계 가설·개인 귀속 상태 |
| [실험 근거·코드 문제](notes/experiment-evidence.md) | 저장 점수의 평가 조건 |
| [노트북 집계 출력 발췌](outputs/notebook-evidence.json) | 셀 위치·실행 번호·fold 점수·중요도 표 |
| [기존 발표 설정](notes/presentation-parameters.json) / [기존 로그](outputs/catboost-validation.tsv) | 다른 시점의 보존 기록 |
| [기존 집계](outputs/artifact-summary.json) / `images/` | 이전 정리에서 만든 집계와 그림 |


## 한계와 후속 개선

우선 변수별 개인 담당을 확인하고, 코드의 잘못된 train/test 참조·중복 산식·분모 처리·실행 순서를 별도 정리본에서 점검해야 한다. 원본의 오류와 과거 출력은 수정하지 않았다. 추가 전후 개선 폭은 현재 기록으로 분리할 수 없으며 중요도로 대신 주장하지 않는다. 임상 인과·실서비스 운영·모델 재현 완료를 주장하지 않는다.

## 블로그·관련 참고자료


- [McLernon 외, BMJ 2016](https://www.bmj.com/content/355/bmj.i5735): 누적 출생 예측과 연령·이전 임신 등 맥락. 대회 임신 성공 타깃과 구별.
- [ASRM·SART 2021](https://www.asrm.org/practice-guidance/practice-committee-documents/guidance-on-the-limits-to-the-number-of-embryos-to-transfer-a---committee-opinion-2021/): 이식 배아 수 제한 지침.
- [ESHRE 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10988112/): IVF/ICSI 이식 배아 수 지침. 배아 발달 단계와 데이터 경과일 변수를 동일시하지 않음.
