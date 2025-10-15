# [Toss] CTR

# DeepFM CTR Modeling Project 

  CTR 예측 대회 데이터를 기반으로 DeepFM 모델을 구축하고, 고카디널리티 피처와 cold start 문제를 다루는 과정에서 얻은 실전 지식을 기록한 프로젝트입니다.[https://dacon.io/competitions/official/236575/overview/description]




# 주요 학습 포인트
## 1. Feature Engineering 전략

inventory_id ⊕ l_feat_14 교차 피처 설계 및 Top-N 해시 전략 적용

hash_bucket_fast_str_series를 통한 UNK=0 오프셋 해시 인코딩

시간형 피처(hour, day_of_week)에 주기적 특성 반영(sin/cos)

sequence 피처를 padding 및 vocab clipping으로 처리해 OOV 안정성 확보

Cross Feature coverage 분석 및 permutation 기반 영향도 평가 

순서형 범주형 피처를 dense와 sparse 피처 모두에 추가하여 상호작용 학습 성능을 개선함.



## 2. Cold Start 문제 대응

Top-N 교차 피처 프리해시(pre-hash) 적용 → cold start coverage 확장

기존 대비 AUC +0.006 개선

OOF 기반 성능 평가로 과적합 방지 및 일반화 성능 향상 확인

Top-5 → Top-8으로 확장하며 coverage curve 개선 


## 3. DeepFM 모델 구축 및 최적화

SparseFeat / VarLenSparseFeat 정의를 통해 embedding 구조 설계

FM layer와 DNN layer의 결합 과정을 코드 레벨에서 직접 구현

TF-Keras 충돌 문제 해결(TF_USE_LEGACY_KERAS=1 + shim)

Early stopping 및 Fold 기반 학습 루틴 정비

OOF validation 구조 및 fold 별 AUC/LogLoss 자동 로깅 추가 


## 4. 모델 Calibration 실험

Platt scaling vs Isotonic Regression 비교

log-loss 관점에서 calibration 효과 측정 및 best fold 적용

Platt scaling 적용 후 예측 확률 안정화 실험 완료 


## 5. 전처리 및 서빙 고려

is_train 플래그 기반으로 train / valid / test 분할 구조 확립

test ID 보존 후 제출용 merge 로직 설계

실시간 서빙 환경을 고려한 feature 삭제 기준 및 안정성 검증 

hashed embedding 기반으로 메모리 사용량 최적화 


## 6. 관련 논문 및 이론 학습

DeepFM의 Feature Interaction 구조(FM + Deep layer) 원리 정리

Wide & Deep vs DeepFM 비교

AutoInt, DIN, BST 등 CTR 모델군의 차이점 학습

Embedding cardinality와 unknown(UNK) 정책의 필요성 이해

Platt scaling vs isotonic regression calibration 차이 정리


## 7. 검증 데이터 설계

### 7.1 사전 진단: 6가지 시각화로 분포/누수 리스크 점검

모델 검증 설계를 하기 전에 다음 6가지 시각화로 train/test 간 분포와 잠재적 누수 요인을 점검.

inventory_id 분포 비교 (train vs test)

l_feat_14 분포 비교 (train vs test)

inventory_id ⊗ l_feat_14 교차토큰 상위 N 커버리지 (head coverage curve)

day_of_week 별 CTR/표본수(신뢰구간) 비교

hour 별 CTR/표본수(신뢰구간) 비교

주요 이산/연속 피처의 CTR-빈도 바이닝 차트(정보량 낮은 구간 식별)

관찰 요약

inventory_id와 l_feat_14 각각의 분포 차이보다, 교차 피처(inventory_id × l_feat_14)의 분포 차이가 test 성능에 훨씬 큰 영향을 미침.

test 데이터에서 day_of_week = 7만 존재하는 특이 분포 확인.

별도의 명시적 timestamp 컬럼이 없어서 시간 기반 순서 보장이 불가능 → 잘못 분할 시 시간 누수 위험.

### 7.2 교차 그룹 결성: inventory_id × l_feat_14

위 시각화 결과를 바탕으로, 실제 서빙/대회 환경에서 클릭 확률에 큰 영향을 주는 교차 상호작용을 검증 단계에서 동일하게 반영하기 위해 교차 그룹을 분할 단위로 채택했다.

그룹 정의: group = (inventory_id, l_feat_14)

분할 기준: StratifiedGroupKFold

Stratified(층화): 타깃 비율(CTR)을 fold 간 유사하게 유지하여 안정적인 OOF 메트릭 확보

Group(그룹보존): 동일 교차 그룹은 반드시 동일 fold에만 존재 → 데이터 누수(동일 토큰의 중복 노출) 방지

### 7.3 시간 누수 최소화 전략: day_of_week 1–6 학습, 7 검증

test는 day_of_week = 7만 존재 → 이를 검증에서 재현해야 실제 배포/제출 상황과 일관성이 생긴다.

명시적 timestamp가 없는 환경에서 시간 순서를 강제하기 어렵기 때문에, 요일을 약한 시간 proxy로 사용:

Train: day_of_week ∈ {1,2,3,4,5,6}

Valid (OOF Fold): day_of_week = 7

이렇게 하면 “미래(7)”를 과거(1–6)로 맞히는 구조가 되어, 시간 누수 최소화 + test 분포 시뮬레이션을 동시에 달성한다.

### 7.4 OOF 학습/평가 체계

각 fold에서 Train(1–6) 로 학습 → Valid(7) 로 평가하여 OOF 예측을 취합.
fold별 AUC/LogLoss를 수집하고, 전체 OOF 기반 메트릭을 최종 지표로 사용.

이 구조는 리더보드(private)와의 상관을 높이고, 배포 후 운영 성능과의 괴리를 줄여줌.


### 7.5 설계 선택의 논리적 이유

분포 일치성: test가 day_of_week=7만 가지므로, 검증도 7만으로 구성해 분포 미스매치를 최소화.

누수 방지: timestamp 부재 상황에서 “요일”을 약한 시간 축으로 삼아 미래→과거 누수를 차단.

교차 상호작용 보존: 검증에서 그룹 누수를 막기 위해 inventory_id × l_feat_14 단위로 그룹 분할.

일관된 파이프라인: is_train 플래그로 train/valid/test를 동일 전처리 라인에서 관리 → 온라인 서빙과 완전 호환.

일반화 성능 지향: 단순 랜덤 KFold가 아닌 StratifiedGroupKFold로 target 비율을 맞추고, 그룹 중복을 제거하여 현실적인 난이도를 재현.



### 7.6 장단점

장점:

test와 유사 분포에서 검증 → 리더보드/실운영과 높은 정합성

시간 누수 최소화, 그룹 누수 차단으로 과적합 방지

OOF 기반 메트릭이 안정적이며 모델 선택/튜닝에 신뢰도 제공

유의점:

day_of_week=7에 표본이 적다면 신뢰구간이 넓어질 수 있음 → Top-N cross coverage 확대, pre-hash 전략으로 보완

특정 요일 편향이 큰 데이터셋에서는 추가적인 reweighting 또는 data augmentation을 고려




## 실험 및 성능 결과
| Experiment                  | Description                                     | AUC (OOF) | LogLoss (OOF) |
|-----------------------------|-------------------------------------------------|-----------|--------------|
| Baseline                    | Basic DeepFM (no cross)                         | 0.585     | 0.209        |
| Cross Feature (Top-5)       | `inventory_id × l_feat_14`                      | 0.590     | 0.206        |
| Cross Feature (Top-8)       | + Cold Start Coverage 확대                      | 0.591     | 0.205        |
| + Permutation Analysis      | Cross Feature 영향도 정량화                     | 0.591     | 0.205        |
| Calibration (Platt Scaling) | 확률 보정 적용                                  | **0.592** | **0.204**    |


## 내가 얻은 인사이트 

cross feature 전략이 단순 해시보다 cold start 문제에 효과적이다.
→ permutation test로 영향도를 확인함.

pre-hash + Top-N 조정으로 test coverage를 유의미하게 개선할 수 있다.

calibration은 AUC 향상보다 확률 안정성 향상이라는 실질적 이점을 준다.

데이터 split 시 is_train flag를 활용하면 모델링–제출 파이프라인이 단순화된다.

hashed embedding은 sparse high-cardinality feature 처리에서 메모리 절감에 탁월하다.
