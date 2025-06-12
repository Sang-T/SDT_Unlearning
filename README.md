# Soft Decision Tree 지식 증류 및 클래스 언러닝 프레임워크

본 프로젝트는 복잡한 신경망 모델의 지식을 해석 가능한 **Soft Decision Tree (SBDT)**로 증류(distill)하는 기법을 탐구합니다. 나아가, 학습된 SDT 모델에서 특정 클래스의 정보를 효율적으로 제거하는 **'타겟 클래스 언러닝(Unlearning)'** 알고리즘을 제안하고 그 성능을 검증합니다.

실험은 **MNIST, FashionMNIST, CIFAR-10, SVHN** 데이터셋을 대상으로 진행되었습니다.


## 목차 (Contents)

| 위치 | 내용 설명 |
|---|---|
| [models/tree.py](models/tree.py) | Soft Decision Tree 모델을 `tf.keras`로 구현한 코드입니다. 손실 정규화 등 핵심 로직은 순수 TensorFlow를 사용합니다. |
| [models/convnet.py](models/convnet.py) | 각 데이터셋의 지식 증류를 위한 교사(Teacher) CNN 모델 코드입니다. |
| [models/utils.py](models/utils.py) | TensorFlow 세션 관리 및 모델 파라미터 시각화를 위한 유틸리티 함수입니다. |
| [notebooks/](notebooks/) | **실험 코드:**<br>• `MNIST.ipynb`, `FashionMNIST.ipynb`, `CIFAR-10.ipynb`, `SVHN.ipynb`: 각 데이터셋에 대한 개별 실험(교사 학습, 증류, 언러닝)을 수행합니다.<br>• `Test_code.ipynb`: 모든 데이터셋에 대해 타겟 클래스를 제외하고 모델을 처음부터 재학습(Retrain)하는 비교 실험 코드입니다. |
| [visualize/](visualize/) | **실험을 통해 생성된 최종 결과 시각화 자료 (Pseudo Code, 언러닝 전/후의 트리 분기)가 저장된 폴더입니다.** |
| [assets/](assets/) | **모델 체크포인트:**<br>각 데이터셋별 최적의 모델 가중치를 저장합니다. 폴더에는 (1) 교사(ConvNet) 모델, (2) 학생(SBDT) 모델, (3) 타겟 클래스를 제거하고 재학습한 모델의 체크포인트가 포함되어 있어 실험 재현을 돕습니다. |

## 요구사항 (Requirements)
코드는 **Python 3.7+**, **TensorFlow** 환경에서 테스트되었습니다. 하이퍼파라미터 최적화에는 **Optuna**가 사용되었습니다.

필요한 Python 패키지를 설치하려면 다음을 실행하세요.
```bash
pip install -r requirements.txt
```
GPU 지원을 위해서는 CUDA 및 cuDNN과 호환되는 `tensorflow-gpu` 버전을 설치해야 합니다.

## 사용된 데이터셋
본 연구는 다음 4가지 벤치마크 데이터셋을 사용하여 모델의 성능을 평가했습니다.

*   **MNIST, FashionMNIST, CIFAR-10:** `tf.keras.datasets` API를 통해 불러왔습니다.
    *   **MNIST:** 0-9 손글씨 숫자로 구성된 28x28 흑백 이미지입니다.
    *   **FashionMNIST:** 10종류의 패션 아이템으로 구성된 28x28 흑백 이미지입니다.
    *   **CIFAR-10:** 10개의 사물 클래스로 구성된 32x32 컬러 이미지입니다.
*   **SVHN (Street View House Numbers):** 실제 거리에서 촬영된 집 번호 숫자 이미지입니다. 스탠포드 대학에서 제공하는 `train_32x32.mat`와 `test_32x32.mat` 파일을 직접 로드하여 사용했습니다.

## 실험 과정 (Experimental Process)

본 연구는 3단계의 실험 과정으로 구성됩니다.

### 1. Teacher Model (ConvNet) 학습
각 데이터셋에 대해 높은 성능을 보이는 교사(Teacher) CNN 모델을 학습합니다. 이 모델은 SDT에 지식을 전달하는 역할을 합니다.

### 2. Soft Decision Tree로 지식 증류 (Knowledge Distillation)
교사 모델의 예측 결과인 **소프트 레이블(soft label)**을 사용하여 해석 가능한 SDT를 학습시킵니다.

#### 피처 추출 (Feature Extraction)
-   **흑백 이미지 (MNIST, FashionMNIST):** 이미지를 1차원 벡터로 평탄화(flattening)하여 사용합니다.
-   **컬러 이미지 (CIFAR-10, SVHN):** 교사 CNN 모델의 중간 레이어 출력에 Global Average Pooling(GAP)을 적용하여 고차원 피처 벡터를 추출합니다.


#### 하이퍼파라미터 최적화
`Optuna`를 사용하여 SDT의 성능을 극대화하는 최적의 하이퍼파라미터 조합(`tree_depth`, `penalty_strength`, `penalty_decay`, `ema_win_size`, `inv_temp`, `learning_rate`, `batch_size`)을 탐색했습니다.

*   **Penalty strength:** 트리의 내부 노드가 한쪽으로만 분기하는 것을 방지하는 정규화 강도입니다. 값이 높을수록 좌/우 분기가 균형을 이루도록 유도하여 트리가 더 다양한 구조를 학습하게 합니다.
*   **Penalty decay:** 트리의 깊이가 깊어질수록 페널티의 영향을 점진적으로 줄입니다. 이를 통해 얕은 노드는 일반적인 규칙을, 깊은 노드는 세부적인 특징을 학습하도록 만듭니다.
*   **Ema_win_size:** 미니배치마다 변동하는 노드의 분기 경향을 부드럽게 만들어 학습을 안정화시킵니다. 윈도우 크기가 크면 장기적인 안정성을, 작으면 변화에 대한 빠른 적응을 우선시합니다.
*   **Inverse temperature:** 내부 노드의 시그모이드(Sigmoid) 함수를 더 날카롭게 만들어 분기 결정을 명확하게 합니다. β 값이 높을수록 분기 확률이 0 또는 1에 가까워져 'Hard' Decision Tree처럼 동작하게 됩니다.

### 3. 타겟 클래스 언러닝 (Target Class Unlearning)
학습된 SDT 모델에서 특정 클래스에 대한 정보를 제거하는 언러닝 알고리즘을 적용합니다. 전체 재학습 없이 효율적인 망각을 수행하는 것을 목표로 합니다.
제안하는 언러닝 알고리즘에 대한 자세한 내용은 `visualize` 폴더에 포함된 Pseudo Code를 참고해 주시기 바랍니다.

## 실험 결과 (Results)

### 1. 교사 모델 (ConvNet) 성능

| 데이터셋 | 정확도 (Accuracy) | 정밀도 (Precision) | 재현율 (Recall) | F1-Score |
|---|---|---|---|---|
| MNIST | 99.06% | 99.05% | 99.06% | 99.05% |
| FashionMNIST | 93.34% | 93.33% | 93.34% | 93.30% |
| CiFAR-10 | 90.09% | 89.17% | 89.09% | 88.97% |
| SVHN | 95.90% | 95.57% | 95.55% | 95.56% |

### 2. 증류된 SDT 모델 성능

| 데이터셋 | 정확도 (Accuracy) | 정밀도 (Precision) | 재현율 (Recall) | F1-Score |
|---|---|---|---|---|
| MNIST | 91.53% | 91.55% | 91.42% | 91.44% |
| FashionMNIST | 85.18% | 85.09% | 85.05% | 84.91% |
| CiFAR-10 | 81.27% | 81.34% | 81.25% | 81.09% |
| SVHN | 84.07% | 85.56% | 83.87% | 84.57% |

### 3. 언러닝 성능 비교
제안하는 언러닝 프레임워크와 망각 데이터를 제외하고 모델을 처음부터 재학습(Retrain)하는 방식의 성능 비교입니다.

| 데이터셋 (타겟 클래스) | 방법 | 전체 정확도 | 타겟 정확도 | 비타겟 정확도 | 소요 시간 |
|---|---|---|---|---|---|
| **MNIST** (Class: 3) | **제안 프레임워크** | **80.74%** | **0%** | **89.81%** | **33m** |
| | 재학습 (Retrain) | 80.62% | **0%** | 89.68% | 51m 15s |
| **FashionMNIST** (Class: 0) | **제안 프레임워크** | **75.37%** | **0%** | **83.74%** | **33m 23s** |
| | 재학습 (Retrain) | 68.57% | **0%** | 75.37% | 50m 54s |
| **CIFAR-10** (Class: 0) | **제안 프레임워크** | **71.12%** | **0%** | **79.02%** | **44m 23s** |
| | 재학습 (Retrain) | 71.51% | **0%** | 79.46% | 96m 20s |
| **SVHN** (Class: 0) | **제안 프레임워크** | **78.28%** | **0%** | **83.90%** | **22m 30s** |
| | 재학습 (Retrain) | 71.68% | **0%** | 78.28% | 51m |

제안하는 언러닝 프레임워크는 타겟 클래스에 대한 정보를 완벽히 제거(**타겟 정확도 0%**)하면서도, 재학습 방식보다 **훨씬 빠른 시간** 내에 유사하거나 더 나은 **비타겟 정확도**를 달성했습니다.
