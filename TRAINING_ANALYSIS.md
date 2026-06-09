# MedEfficientSAM Distillation 학습 분석

> Run: `distill_l1_npz_clean_20260514`  
> Run ID: `7e6d65f73c7d45728c33de6e07f675fc`  
> Experiment: `medficientsam` (ID: `986928226268314361`)  
> 기간: 2026-05-14 10:08 ~ 2026-05-17 01:59

---

## 1. 실험 전체 이력

총 8개의 run이 기록되었으며, 데이터셋 구성과 설정을 바꾸며 반복 시도 후 마지막 run에서 성공.

| Run 이름 | 상태 | 시작 | 종료 | 비고 |
|---|---|---|---|---|
| distill_l1_no_extracted | FINISHED | 05-08 09:42 | 05-11 01:20 | 첫 장기 run, 2 epoch만 완료 |
| distill_l1_no_extracted | RUNNING | 05-12 16:48 | - | 미완료 상태로 잔존 |
| distill_l1_no_extracted | FAILED | 05-13 13:54 | 05-13 13:54 | 즉시 실패 |
| distill_l1_no_extracted | FAILED | 05-13 13:55 | 05-13 16:43 | 약 3시간 후 실패 |
| distill_l1_train_npz_all_20260513 | FAILED | 05-13 17:11 | 05-13 17:27 | 약 16분 후 실패 |
| distill_l1_train_npz_all_20260513 | RUNNING | 05-13 17:27 | - | 미완료 상태로 잔존 |
| distill_l1_train_npz_all_20260513 | FAILED | 05-14 09:13 | 05-14 09:58 | 약 45분 후 실패 |
| **distill_l1_npz_clean_20260514** | **FINISHED** | **05-14 10:08** | **05-17 01:59** | **8 epoch 완주 성공** |

---

## 2. 성공 Run 학습 설정

### 모델 구조

| 항목 | 값 |
|---|---|
| Teacher | MedSAM (SAM ViT-B), `medsam_vit_b.pth` — 학습 중 **동결** |
| Student | EfficientViT-SAM L1 (`pretrained=False`) |
| 학습 파라미터 | 43,585,568개 (전체 133,256,480개 중 student만) |
| 동결 파라미터 | 89,670,912개 (teacher) |

### 데이터

| 항목 | 값 |
|---|---|
| 데이터 소스 | `${paths.cvpr2024_medsam_data_dir}/train_npz` |
| 최대 샘플 수 | 400,000 (`limit_sample`) |
| Teacher 입력 해상도 | 1024 × 1024 |
| Student 입력 해상도 | 512 × 512 |
| Data augmentation | True |
| Embedding 사전 추출 | 없음 (`embedding_dir: None`) — 매 step마다 on-the-fly 추론 |
| Batch size | 8 |
| Num workers | 8 |

### 학습 하이퍼파라미터

| 항목 | 값 |
|---|---|
| Epochs | 8 |
| Steps per epoch | 50,000 |
| 총 steps | 400,000 |
| Optimizer | AdamW (lr=0.075, weight_decay=0.0005) |
| LR Scheduler | ExponentialLR (gamma=0.5, epoch 단위 적용) |
| Precision | bf16-mixed |
| Gradient clip | 0.5 |
| Checkpoint 저장 | 10,000 step마다 |
| ckpt_path | `None` (scratch부터 시작) |

---

## 3. Epoch별 학습 결과

| Epoch | Steps | Train Loss | LR | 소요시간 | 누적시간 | Loss 감소율 |
|---|---|---|---|---|---|---|
| 0 | 49,999 | 0.002261 | 0.075000 | 8.09h | 8.09h | — |
| 1 | 99,999 | 0.001504 | 0.037500 | 8.15h | 16.24h | ▼ 33.5% |
| 2 | 149,999 | 0.001336 | 0.018750 | 8.12h | 24.36h | ▼ 11.2% |
| 3 | 199,999 | 0.001245 | 0.009375 | 7.90h | 32.26h | ▼ 6.8% |
| 4 | 249,999 | 0.001186 | 0.004688 | 7.86h | 40.12h | ▼ 4.7% |
| 5 | 299,999 | 0.001147 | 0.002344 | 8.02h | 48.14h | ▼ 3.3% |
| 6 | 349,999 | 0.001123 | 0.001172 | 7.85h | 55.99h | ▼ 2.1% |
| 7 | 399,999 | 0.001110 | 0.000586 | 7.85h | 63.84h | ▼ 1.2% |

- **총 학습 시간**: 63.84시간 (약 2.7일)
- **전체 Loss 감소**: 0.002261 → 0.001110 (**▼ 50.9%**)
- **최종 step loss 범위**: 0.0008 ~ 0.0015 (정상적인 확률적 노이즈)

### 수렴 패턴 해석

- **Epoch 0 → 1**: LR 0.075 상태에서 loss가 33.5% 급락 — global minimum 방향으로 빠르게 진입
- **Epoch 3 이후**: 감소 폭이 6% → 1% 수준으로 줄어들며 수렴 후반부 진입
- **Epoch 7 종료 시점에도 여전히 감소 중** → 추가 epoch 시 소폭 개선 여지 있음

---

## 4. 실패 Run 대비 성공 요인

| 항목 | 실패 runs | 성공 run |
|---|---|---|
| `ckpt_path` | `weights/distilled-l1-mlflow/last.ckpt` (이전 run 이어받기) | `None` (처음부터) |
| `batch_size` | 16 | **8** |
| `num_workers` | 16 | **8** |
| 데이터셋 명칭 | `no_extracted`, `npz_all` | **`npz_clean`** |

**추정 원인**: 이전 run들은 batch_size=16이 GPU VRAM을 초과했거나, 손상된 이전 checkpoint를 이어받아 실패한 것으로 추정. 성공 run은 batch=8로 줄이고 scratch부터 재시작.

---

## 5. 산출물

| 파일 | 크기 | 생성일 | 설명 |
|---|---|---|---|
| `step_010000.ckpt` ~ `step_400000.ckpt` | 842MB × 40개 | 05-14 ~ 05-17 | 10,000 step마다 저장된 전체 모델 |
| `last.ckpt` | 842MB | 05-17 01:59 | 최종 체크포인트 (step_400000과 동일) |
| `student_encoder.pt` | **167MB** | 05-18 11:03 | student image encoder만 추출한 배포용 가중치 |

- **총 디스크 사용량**: 34GB (`weights/distilled-l1-clean-20260514/`)
- `student_encoder.pt`는 학습 완료 후 별도로 추출 — ONNX 변환 및 배포 파이프라인에 바로 사용 가능한 상태

---

## 6. 재현 정보

```
Source: src/train.py
Git commit: 59504938bb37ab7e2832ede358051976e740efe5
Tags: ['distill', 'efficientvit']
User: infinitt
```

체크포인트 경로: `weights/distilled-l1-clean-20260514/`
