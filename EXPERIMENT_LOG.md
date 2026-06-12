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

## Distillation 실험 이력

| 날짜 | Run 이름 | 상태 | 원인 |
|---|---|---|---|
| 2026-04-28 | distill_l1_no_extracted | FAILED (즉시) | albumentations 버전 비호환 |
| 2026-04-29 ~ 05-01 | distill_l1_no_extracted | **FINISHED** | — (소규모 데이터 첫 성공) |
| 2026-05-08 ~ 05-11 | distill_l1_no_extracted | **FINISHED** | — (CVPR2024 full 학습) |
| 2026-05-12 ~ 05-13 | distill_l1_no_extracted | KILLED | 외부 강제 종료 |
| 2026-05-13 (13:55) | distill_l1_no_extracted | FAILED (3h) | Pathology_new 누락 파일 |
| 2026-05-13 (17:27) | distill_l1_train_npz_all | KILLED | 외부 강제 종료 (5h 36m) |
| 2026-05-14 ~ 05-17 | distill_l1_npz_clean | **FINISHED** | — (최종 성공, 배포 미채택) |
| 2026-06-12 ~ | distill_l1_bs16_ep8_20260612 | **IN PROGRESS** | — (Pathology_new 포함 재학습) |

---

## 실패 원인 상세

### [1] 2026-04-28 — albumentations 버전 비호환

**증상**: 학습 시작 직후 즉시 크래시

```
AttributeError: module 'albumentations' has no attribute 'TransformType'.
Did you mean: 'Transform3D'?
```

`medsam_dataset.py`에서 `A.TransformType` 타입 힌트를 사용하는데, 설치된 버전에 해당 속성이 없었음.

**해결**: albumentations 버전 업그레이드 (`environment.yaml` 기준으로 환경 재현).

---

### [2] 2026-04-29 ~ 05-01 — 소규모 데이터 첫 성공

- 데이터: `train_npz/` 부분 다운로드 상태 (유효 2D 슬라이스 약 127,936개)
- batch_size: 8, num_workers: 16
- 저장: `weights/distilled-l1/`

**결과**: 8 epoch 완주, 최종 loss **0.00133**  
데이터 다양성 부족 → CVPR2024 전체 데이터로 재학습 필요.

---

### [3] 2026-05-08 ~ 05-11 — CVPR2024 데이터 도입

- 데이터: CVPR2024 MedSAM 공식 `train_npz/` 전량 (70,864 npz, 유효 슬라이스 ~1,050,000개), `limit_sample=400,000`
- batch_size: 16 (epoch당 25,000 steps)
- 저장: `logs/train/runs/2026-05-08_09-42-08/checkpoints/`

**결과**: 8 epoch / 200,000 steps 완주, 최종 loss **0.00113**  
평가: **DSC 0.8588 / NSD 0.8918** (`eval_results/last_ckpt_metrics.csv`, 3,077 cases)

> CVPR2024 MedSAM validation set 기준. Ki-67 병리 이미지 성능과는 무관.

**한계**: batch_size=16은 이후 resume 시 불안정성 유발.

---

### [4] 2026-05-13 (13:55) — Pathology_new 누락 파일 (핵심 실패)

**증상**: Epoch 2, step ~8,745 (약 2시간 47분) 시점 크래시

```
FileNotFoundError: .../train_npz/Pathology_new/gts_npz_s128/
2D_S26-03104,..._r06_c07.npz
```

- `weights/distilled-l1-mlflow/last.ckpt` (batch_size=16, step_050000)에서 resume 시도
- `Pathology_new/gts_npz_s128/` 하위 다수 파일이 디스크에 없음 (불완전한 다운로드)
- seed=42 샘플링 결과 Epoch 2에서 처음 누락 파일 접근 → Epoch 0~1은 우연히 정상 파일만 사용

**해결 방향**: `Pathology_new` 제외 정제 데이터로 scratch 재학습.

---

### [5] 2026-05-13 (17:27) — 외부 강제 종료

- Pathology_new 포함 상태에서 재시도, Epoch 0, 63% 진행 중 외부 KILL
- 학습 자체 문제는 없었음. 다른 프로세스와 충돌 또는 수동 중단으로 추정.

---

## 최종 성공 Run — distill_l1_npz_clean_20260514

### 학습 설정

| 항목 | 값 |
|---|---|
| Teacher | MedSAM (SAM ViT-B), `medsam_vit_b.pth` — 학습 중 동결 |
| Student | EfficientViT-SAM L1 (`pretrained=False`) |
| 학습 파라미터 | 43,585,568개 (전체 133,256,480개 중 student만) |
| 데이터 소스 | `train_npz/` (Pathology_new 제외), limit_sample=400,000 |
| Teacher 입력 해상도 | 1024×1024 / Student 입력 해상도 512×512 |
| Batch size | 8 / Num workers 8 |
| Epochs | 8 (50,000 steps/epoch, 총 400,000 steps) |
| Optimizer | AdamW (lr=0.075, weight_decay=0.0005) |
| LR Scheduler | ExponentialLR (gamma=0.5, epoch 단위) |
| Precision | bf16-mixed / Gradient clip 0.5 |
| Checkpoint 저장 | 10,000 step마다 |

**이전 run과의 주요 차이**:

| 항목 | 이전 | 이번 |
|---|---|---|
| 시작점 | checkpoint resume | `ckpt_path=None` (scratch) |
| batch_size | 16 | **8** |
| 데이터 | Pathology_new 포함 | **Pathology_new 제외** |

**Epoch별 결과**:

| Epoch | Train Loss | LR | 소요시간 | 누적 |
|---|---|---|---|---|
| 0 | 0.002261 | 0.075000 | 8.09h | 8.09h |
| 1 | 0.001504 | 0.037500 | 8.15h | 16.24h |
| 2 | 0.001336 | 0.018750 | 8.12h | 24.36h |
| 3 | 0.001245 | 0.009375 | 7.90h | 32.26h |
| 4 | 0.001186 | 0.004688 | 7.86h | 40.12h |
| 5 | 0.001147 | 0.002344 | 8.02h | 48.14h |
| 6 | 0.001123 | 0.001172 | 7.85h | 55.99h |
| 7 | 0.001110 | 0.000586 | 7.85h | 63.84h |

- 총 학습 시간: **63.84h** (약 2.7일)
- 최종 loss: **0.00111** (초기 대비 ▼50.9%)
- 로그: `medficientsam/logs/distill_l1_nohup_20260514.log`

---

## Distillation 체크포인트 현황

### `weights/distilled-l1/` — 출처 불명 (다운로드 추정)

`step_400000.ckpt` 1개 (842MB, 2026-04-29 09:24).  
step_010000 ~ step_120000은 오늘(2026-06-12) `distilled-l1-prev-run/`으로 이동.

### `weights/distilled-l1-prev-run/` — 2026-04-29 ~ 05-01 (소규모 데이터 FINISHED)

step_010000 ~ step_120000, last.ckpt (13개, 10.9GB).  
소규모 데이터(~127,936 슬라이스), batch_size=8, 8에폭 완주.

### `weights/distilled-l1-mlflow/` (5.0GB)

2026-05-12 ~ 05-13. batch_size=16, step_050000까지 진행 후 외부 KILL.  
이 `last.ckpt`를 resume 기점으로 쓰려다 Pathology_new 누락 문제와 겹쳐 실패.

### `weights/distilled-l1-train_npz/` (3.3GB)

2026-05-13. Pathology_new 포함 전체 데이터로 시도, step_030000 진행 후 FileNotFoundError 실패.

### `weights/distilled-l1-clean-20260514/` (34GB) — **최종 완료 (배포 미채택)**

2026-05-14 ~ 05-17 (63.8h). Pathology_new 제외, batch_size=8, scratch 학습.  
8 epoch / 400,000 steps 완주. 10,000 steps마다 40개 체크포인트 저장.  
`step_400000.ckpt` (842MB) = `last.ckpt` — fine-tuning 시도했으나 성능 불량으로 채택 안 됨.

### `weights/distilled-l1-20260612/` — **진행 중**

2026-06-12 ~. Pathology_new 포함(19,062개 무결성 검증 완료), batch_size=16, num_workers=16, scratch 학습.  
체크포인트: 10,000 step마다. 로그: `DP_MedificientSAM/logs/distill_l1_20260612.log`  
MLflow: http://localhost:5001 (run: `distill_l1_bs16_ep8_20260612`)

---

## 2026-06-12 재학습 — distill_l1_bs16_ep8_20260612

### 학습 설정

| 항목 | 값 |
|---|---|
| Teacher | MedSAM (SAM ViT-B), `medsam_vit_b.pth` — 동결 |
| Student | EfficientViT-SAM L1 (`pretrained=False`) |
| 데이터 소스 | `train_npz/` **Pathology_new 포함** (무결성 검증 완료, 19,062개), limit_sample=400,000 |
| Batch size | **16** / Num workers 16 |
| Epochs | 8 (25,000 steps/epoch, 총 200,000 steps) |
| Optimizer | AdamW (lr=0.075, weight_decay=0.0005) |
| LR Scheduler | ExponentialLR (gamma=0.5, epoch 단위) |
| Precision | bf16-mixed / Gradient clip 0.5 |
| Checkpoint 저장 | `/mnt/Disk1/sylee/weights/distilled-l1-20260612/` (10,000 step마다) |
| 로그 | `DP_MedificientSAM/logs/distill_l1_20260612.log` |
| MLflow | http://localhost:5001, run: `distill_l1_bs16_ep8_20260612` |

**이전 run과의 주요 차이**:

| 항목 | distill_l1_npz_clean (05-14) | 이번 (06-12) |
|---|---|---|
| Pathology_new | **제외** | **포함** (무결성 검증 완료) |
| batch_size | 8 | **16** |
| steps/epoch | 50,000 | **25,000** |
| 총 steps | 400,000 | **200,000** |

---

## 재현 주의사항

1. **데이터 무결성 사전 검증**: `train_npz/Pathology_new/` 내 누락 파일 다수 — **학습 시 제외** 필수.
2. **batch_size=8 고정**: 16으로 설정 시 OOM 또는 불안정성 재발 가능. epoch당 50,000 steps 기준.
3. **ckpt_path=None**: 설정이 다른 run의 checkpoint에서 resume하면 데이터 분포 불일치로 학습 불안정.
4. **albumentations 버전**: `A.TransformType` 속성 존재 버전 필요. `environment.yaml` 기준 환경 재현.

---

*MLflow Experiment ID: `986928226268314361`*  
*최종 Git commit: `59504938bb37ab7e2832ede358051976e740efe5`*
