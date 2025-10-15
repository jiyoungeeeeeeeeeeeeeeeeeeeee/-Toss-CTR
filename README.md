# [Toss] CTR

# DeepFM CTR Modeling Project 

  CTR 예측 대회 데이터를 기반으로 DeepFM 모델을 구축하고, 고카디널리티 피처와 cold start 문제를 다루는 과정에서 얻은 실전 지식을 기록한 프로젝트입니다.




# 주요 학습 포인트
## 1. Feature Engineering 전략

inventory_id ⊕ l_feat_14 교차 피처 설계 및 Top-N 해시 전략 적용

hash_bucket_fast_str_series를 통한 UNK=0 오프셋 해시 인코딩

시간형 피처(hour, day_of_week)에 주기적 특성 반영(sin/cos)

sequence 피처를 padding 및 vocab clipping으로 처리해 OOV 안정성 확보

Cross Feature coverage 분석 및 permutation 기반 영향도 평가 



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
