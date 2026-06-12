# MedficientSAM 경량화 실험 로그

EfficientViT-SAM L1(student)을 MedSAM ViT-B(teacher)로 knowledge distillation한 실험 기록.  
distillation 기반 fine-tuning은 성능 불량으로 채택되지 않았으며, **배포 모델은 공식 EfficientViT-SAM L1 pretrained 가중치 기반 fine-tuning**을 사용한다.

---

## 배포 파이프라인

```
[1] Fine-tuning  (Ki-67 도메인 특화)
    기반:   EfficientViT-SAM L1 공식 pretrained (GitHub 공식 배포)
    데이터:  Ki-67 IHC + PanNuke + MoNuSeg (augmented)  ※ PanNuke·MoNuSeg: 병리 도메인 보강 목적, Google Sheet 경유 추가
    출력:   weights/finetuned-l1-augmented/best.ckpt

        ↓

[2] Export
    설정:   configs/experiment/export_finetuned_l1_onnx.yaml
    출력:   deployment/onnx/encoder.quantized.onnx (INT8)
             deployment/onnx/decoder.quantized.onnx (INT8)
             deployment/openvino/{encoder,decoder}.{xml,bin} (FP32)

        ↓

[3] 서비스
    진입점: http://10.10.40.194:3000/realtime
    런타임: OpenVINO FP32 (ONNX INT8 대비 e2e 5.1× 빠름)
```

---

## 전체 실험 이력

| 날짜 | Run 이름 | 상태 | 비고 |
|---|---|---|---|
| 2026-04-28 | distill_l1_no_extracted | FAILED (즉시) | albumentations 버전 비호환 |
| 2026-04-29 ~ 05-01 | distill_l1_no_extracted | **FINISHED** | 소규모 데이터 첫 성공 |
| 2026-05-08 ~ 05-11 | distill_l1_no_extracted | **FINISHED** | CVPR2024 full 학습 |
| 2026-05-12 ~ 05-13 | distill_l1_no_extracted | KILLED | 외부 강제 종료 |
| 2026-05-13 (13:55) | distill_l1_no_extracted | FAILED (3h) | Pathology_new 누락 파일 |
| 2026-05-13 (17:27) | distill_l1_train_npz_all | KILLED | 외부 강제 종료 (5h 36m) |
| 2026-05-14 ~ 05-17 | distill_l1_npz_clean | **FINISHED** | 최종 완주, 배포 미채택 |
| 2026-06-12 ~ | distill_l1_bs16_ep8_20260612 | **IN PROGRESS** | Pathology_new 포함 재학습 |

---

## 실험 상세 (날짜순)

> 각 항목은 **데이터 → 설정 → 체크포인트 → 결과** 순으로 기술.

---

### 2026-04-28 | `distill_l1_no_extracted` | FAILED

| 항목 | 내용 |
|---|---|
| 데이터 | — (학습 시작 전 즉시 크래시) |
| 시작점 | scratch |
| batch_size / num_workers | — |
| Epochs / 총 steps | — |
| Optimizer / Scheduler | — |
| 체크포인트 | — |
| 로그 | — |
| **결과** | `AttributeError: module 'albumentations' has no attribute 'TransformType'` — albumentations 버전 비호환으로 즉시 실패. `environment.yaml` 기준 환경 재현으로 해결. |

---

### 2026-04-29 ~ 05-01 | `distill_l1_no_extracted` | FINISHED

| 항목 | 내용 |
|---|---|
| 데이터 | `train_npz/` 부분 다운로드 (Pathology_new **미포함**), 유효 슬라이스 약 **127,936개** |
| 시작점 | scratch |
| batch_size / num_workers | 8 / 16 |
| Epochs / 총 steps | 8 ep / 400,000 steps (50,000 steps/epoch) |
| Optimizer / Scheduler | AdamW (lr=0.075, wd=0.0005) / ExponentialLR (gamma=0.5) |
| 체크포인트 | `experiment_weights/distilled-l1-prev-run/` — step_010000 ~ step_120000 + last.ckpt (13개, 10.9GB) |
| 로그 | — |
| **결과** | 8 epoch 완주, 최종 loss **0.00133**. 데이터 다양성 부족 → CVPR2024 전체 데이터로 재학습 필요. |

---

### 2026-05-08 ~ 05-11 | `distill_l1_no_extracted` | FINISHED

| 항목 | 내용 |
|---|---|
| 데이터 | `train_npz/` CVPR2024 공식 전량 (Pathology_new **미포함**), 70,864 npz, 유효 슬라이스 ~**1,050,000개**, limit_sample=400,000 |
| 시작점 | scratch |
| batch_size / num_workers | 16 / — |
| Epochs / 총 steps | 8 ep / 200,000 steps (25,000 steps/epoch) |
| Optimizer / Scheduler | AdamW (lr=0.075, wd=0.0005) / ExponentialLR (gamma=0.5) |
| 체크포인트 | `logs/train/runs/2026-05-08_09-42-08/checkpoints/` |
| 로그 | — |
| **결과** | 8 epoch 완주, 최종 loss **0.00113**. DSC **0.8588** / NSD **0.8918** (CVPR2024 validation 3,077 cases). batch_size=16은 이후 resume 시 불안정성 유발. |

---

### 2026-05-12 ~ 05-13 — 연속 3회 시도, 모두 실패

| 항목 | ① 05-12~13 KILLED | ② 05-13 13:55 FAILED | ③ 05-13 17:27 KILLED |
|---|---|---|---|
| Run 이름 | distill_l1_no_extracted | distill_l1_no_extracted | distill_l1_train_npz_all |
| 데이터 | Pathology_new **포함** | Pathology_new **포함** | Pathology_new **포함** |
| 시작점 | `distilled-l1-mlflow/last.ckpt` resume | `distilled-l1-mlflow/last.ckpt` resume | scratch |
| batch_size | 16 | 16 | — |
| 종료 시점 | 외부 KILL | Epoch 2, step ~8,745 (약 3h) | Epoch 0, 63% (5h 36m) |
| 체크포인트 | `experiment_weights/distilled-l1-mlflow/` (5.0GB, step_050000) | — | `experiment_weights/distilled-l1-train_npz/` (3.3GB, step_030000) |
| **종료 원인** | 외부 강제 종료 | `Pathology_new/gts_npz_s128/` 누락 파일 (불완전 다운로드) | 외부 강제 종료 |

**핵심 실패 원인 (②)**: seed=42 샘플링 결과 Epoch 0~1은 우연히 정상 파일만 접근했고, Epoch 2에서 처음 누락 파일에 도달해 `FileNotFoundError` 발생.  
**해결 방향**: Pathology_new 제외 정제 데이터로 scratch 재학습.

---

### 2026-05-14 ~ 05-17 | `distill_l1_npz_clean_20260514` | FINISHED

| 항목 | 내용 |
|---|---|
| 데이터 | `train_npz/` (Pathology_new **제외**), limit_sample=400,000 |
| 시작점 | scratch (`ckpt_path=None`) |
| batch_size / num_workers | **8** / 8 |
| Epochs / 총 steps | 8 ep / 400,000 steps (50,000 steps/epoch) |
| Optimizer / Scheduler | AdamW (lr=0.075, wd=0.0005) / ExponentialLR (gamma=0.5) |
| Precision / Gradient clip | bf16-mixed / 0.5 |
| Teacher | `weights/medsam/medsam_vit_b.pth` (SAM ViT-B, 동결) — 입력 1024×1024 |
| Student | EfficientViT-SAM L1 (`pretrained=False`) — 입력 512×512, 학습 파라미터 43.6M |
| 체크포인트 | `experiment_weights/distilled-l1-clean-20260514/` — 10,000 step마다, 40개 (34GB) |
| 로그 | `medficientsam/logs/distill_l1_nohup_20260514.log` |
| **결과** | 총 **63.84h** (약 2.7일) 완주, 최종 loss **0.00111** (▼50.9%). 성능 불량으로 배포 미채택. |

**Epoch별 loss**

| Epoch | Train Loss | LR | 소요 | 누적 |
|---|---|---|---|---|
| 0 | 0.002261 | 0.075000 | 8.09h | 8.09h |
| 1 | 0.001504 | 0.037500 | 8.15h | 16.24h |
| 2 | 0.001336 | 0.018750 | 8.12h | 24.36h |
| 3 | 0.001245 | 0.009375 | 7.90h | 32.26h |
| 4 | 0.001186 | 0.004688 | 7.86h | 40.12h |
| 5 | 0.001147 | 0.002344 | 8.02h | 48.14h |
| 6 | 0.001123 | 0.001172 | 7.85h | 55.99h |
| 7 | 0.001110 | 0.000586 | 7.85h | 63.84h |

---

### 2026-06-12 ~ | `distill_l1_bs16_ep8_20260612` | IN PROGRESS

| 항목 | 내용 |
|---|---|
| 데이터 | `train_npz/` (Pathology_new **포함**, 19,062개 무결성 검증 완료), limit_sample=400,000 |
| 시작점 | scratch (`ckpt_path=None`) |
| batch_size / num_workers | **16** / 16 |
| Epochs / 총 steps | 8 ep / 200,000 steps (25,000 steps/epoch) |
| Optimizer / Scheduler | AdamW (lr=0.075, wd=0.0005) / ExponentialLR (gamma=0.5) |
| Precision / Gradient clip | bf16-mixed / 0.5 |
| Teacher | `weights/medsam/medsam_vit_b.pth` (SAM ViT-B, 동결) |
| Student | EfficientViT-SAM L1 (`pretrained=False`) |
| 체크포인트 | `experiment_weights/distilled-l1-20260612/` — 10,000 step마다 |
| 로그 | `DP_MedificientSAM/logs/distill_l1_20260612.log` |
| MLflow | http://localhost:5001, run: `distill_l1_bs16_ep8_20260612` |
| **결과** | 진행 중 |

**05-14 run 대비 주요 변경**

| 항목 | 05-14 run | 06-12 run |
|---|---|---|
| Pathology_new | 제외 | **포함** (무결성 검증 완료) |
| batch_size | 8 | **16** |
| steps/epoch | 50,000 | **25,000** |
| 총 steps | 400,000 | **200,000** |

---

## 체크포인트 현황

### 공식 드라이브 다운로드 (2026-04-29)

| 경로 | 크기 | 용도 |
|---|---|---|
| `weights/medsam/medsam_vit_b.pth` | 358MB | Teacher 모델 — 학습 중 동결 |
| `weights/medsam/lite_medsam.pth` | 38MB | LiteMedSAM 참고용 |
| `weights/distilled-l1/step_400000.ckpt` | 842MB | 공식 distilled L1 — 출처 불명, 비교 참고용 |
| `weights/distilled-l0/` | — | 공식 distilled L0 |
| `weights/distilled-l2/` | — | 공식 distilled L2 |
| `weights/finetuned-l1-augmented/best.ckpt` | 542MB | **배포 모델** — 공식 pretrained 기반 fine-tuning |
| `weights/finetuned-l1-unaugmented/` | — | augmentation 없는 fine-tuning 비교군 |
| `weights/finetuned-l0-augmented/`, `finetuned-l0-unaugmented/` | — | L0 fine-tuning 비교군 |
| `weights/finetuned-l2-augmented/`, `finetuned-l2-unaugmented/` | — | L2 fine-tuning 비교군 |

### 직접 실험으로 생성된 체크포인트

경로 기준: `/mnt/Disk1/sylee/experiment_weights/` (심볼릭 링크: `DP_MedificientSAM/experiment_weights/`)

| 경로 | 실험 기간 | 크기 | 설명 |
|---|---|---|---|
| `experiment_weights/distilled-l1-prev-run/` | 04-29 ~ 05-01 | 10.9GB | step_010000 ~ step_120000 + last.ckpt (13개), 소규모 데이터 FINISHED |
| `experiment_weights/distilled-l1-mlflow/` | 05-12 ~ 05-13 | 5.0GB | step_050000까지 진행 후 KILL, resume 기점으로 사용됐으나 실패 |
| `experiment_weights/distilled-l1-train_npz/` | 05-13 | 3.3GB | step_030000 진행 후 FileNotFoundError 실패 |
| `experiment_weights/distilled-l1-clean-20260514/` | 05-14 ~ 05-17 | 34GB | 40개 체크포인트, 최종 완주. 성능 불량으로 배포 미채택 |
| `experiment_weights/distilled-l1-20260612/` | 06-12 ~ | 진행 중 | 10,000 step마다 저장 |

---

## 재현 주의사항

1. **데이터 무결성 사전 검증**: `train_npz/Pathology_new/` 내 누락 파일 다수 — **학습 시 제외** 필수.
2. **batch_size=8 고정**: 16으로 설정 시 OOM 또는 불안정성 재발 가능. epoch당 50,000 steps 기준.
3. **ckpt_path=None**: 설정이 다른 run의 checkpoint에서 resume하면 데이터 분포 불일치로 학습 불안정.
4. **albumentations 버전**: `A.TransformType` 속성 존재 버전 필요. `environment.yaml` 기준 환경 재현.

---

*MLflow Experiment ID: `986928226268314361`*  
*최종 Git commit: `59504938bb37ab7e2832ede358051976e740efe5`*
