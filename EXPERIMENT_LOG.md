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

| 날짜 | Run 이름 | 상태 | 원인 |
|---|---|---|---|
| 2026-04-28 | distill_l1_no_extracted | FAILED (즉시) | albumentations 버전 비호환 |
| 2026-04-29 ~ 05-01 | distill_l1_no_extracted | **FINISHED** | — (소규모 데이터 첫 성공) |
| 2026-05-08 ~ 05-11 | distill_l1_no_extracted | **FINISHED** | — (CVPR2024 full 학습) |
| 2026-05-12 ~ 05-13 | distill_l1_no_extracted | KILLED | 외부 강제 종료 |
| 2026-05-13 (13:55) | distill_l1_no_extracted | FAILED (3h) | Pathology_new 누락 파일 |
| 2026-05-13 (17:27) | distill_l1_train_npz_all | KILLED | 외부 강제 종료 (5h 36m) |
| 2026-05-14 ~ 05-17 | distill_l1_npz_clean | **FINISHED** | — (최종 성공, 배포 미채택) |
| 2026-06-12 ~ 06-15 | distill_l1_bs16_ep8_20260612 | **FINISHED** | — (Pathology_new 포함 재학습) |

---

## 실험 상세 (날짜순)

---

### 2026-04-28 — 환경 오류로 즉시 실패

**Run**: `distill_l1_no_extracted` | **상태**: FAILED (즉시)

**증상**: 학습 시작 직후 즉시 크래시

```
AttributeError: module 'albumentations' has no attribute 'TransformType'.
Did you mean: 'Transform3D'?
```

`medsam_dataset.py`에서 `A.TransformType` 타입 힌트를 사용하는데, 설치된 버전에 해당 속성이 없었음.

**해결**: albumentations 버전 업그레이드 (`environment.yaml` 기준으로 환경 재현).

---

### 2026-04-29 ~ 05-01 — 소규모 데이터 첫 성공

**Run**: `distill_l1_no_extracted` | **상태**: FINISHED

- 데이터: `train_npz/` 부분 다운로드 상태 (유효 2D 슬라이스 약 127,936개)
- batch_size: 8 / num_workers: 16
- 체크포인트: `weights/distilled-l1-prev-run/` (step_010000 ~ step_120000, last.ckpt, 13개, 10.9GB)

**결과**: 8 epoch 완주, 최종 loss **0.00133**  
데이터 다양성 부족 → CVPR2024 전체 데이터로 재학습 필요.

---

### 2026-05-08 ~ 05-11 — CVPR2024 전체 데이터 학습

**Run**: `distill_l1_no_extracted` | **상태**: FINISHED

- 데이터: CVPR2024 MedSAM 공식 `train_npz/` 전량 (70,864 npz, 유효 슬라이스 ~1,050,000개), `limit_sample=400,000`
- batch_size: 16 (epoch당 25,000 steps)
- 체크포인트: `logs/train/runs/2026-05-08_09-42-08/checkpoints/`

**결과**: 8 epoch / 200,000 steps 완주, 최종 loss **0.00113**  
평가: **DSC 0.8588 / NSD 0.8918** (`eval_results/last_ckpt_metrics.csv`, 3,077 cases)

> CVPR2024 MedSAM validation set 기준. Ki-67 병리 이미지 성능과는 무관.

**한계**: batch_size=16은 이후 resume 시 불안정성 유발.

---

### 2026-05-12 ~ 05-13 — 연속 3회 시도, 모두 실패

같은 기간 동안 3번의 run이 연속으로 시도됐으며 모두 완주에 실패했다.

#### 3개 Run 비교

| 항목 | ① 05-12~13 KILLED | ② 05-13 13:55 FAILED | ③ 05-13 17:27 KILLED |
|---|---|---|---|
| Run 이름 | distill_l1_no_extracted | distill_l1_no_extracted | distill_l1_train_npz_all |
| 시작점 | distilled-l1-mlflow last.ckpt (resume) | distilled-l1-mlflow last.ckpt (resume) | scratch |
| Pathology_new | 포함 | 포함 | 포함 |
| batch_size | 16 | 16 | — |
| 종료 시점 | 외부 KILL | Epoch 2, step ~8,745 (약 3h) | Epoch 0, 63% 진행 중 (5h 36m) |
| 종료 원인 | 외부 강제 종료 | Pathology_new 누락 파일 | 외부 강제 종료 |
| 체크포인트 | `weights/distilled-l1-mlflow/` (5.0GB, step_050000) | — | `weights/distilled-l1-train_npz/` (3.3GB, step_030000) |

#### ① 2026-05-12~13 — KILLED

- `weights/distilled-l1-mlflow/last.ckpt` 기점으로 resume, step_050000까지 진행 후 외부 KILL.
- 이 `last.ckpt`를 ②의 resume 기점으로 재사용했으나 Pathology_new 문제와 겹쳐 실패.

#### ② 2026-05-13 13:55 — Pathology_new 누락 파일 (핵심 실패)

**증상**: Epoch 2, step ~8,745 (약 2시간 47분) 시점 크래시

```
FileNotFoundError: .../train_npz/Pathology_new/gts_npz_s128/
2D_S26-03104,..._r06_c07.npz
```

- `Pathology_new/gts_npz_s128/` 하위 다수 파일이 디스크에 없음 (불완전한 다운로드)
- seed=42 샘플링 결과 Epoch 2에서 처음 누락 파일 접근 → Epoch 0~1은 우연히 정상 파일만 사용

**해결 방향**: `Pathology_new` 제외 정제 데이터로 scratch 재학습.

#### ③ 2026-05-13 17:27 — KILLED

- Pathology_new 포함 상태에서 재시도, Epoch 0, 63% 진행 중 외부 KILL.
- 학습 자체 문제는 없었음. 다른 프로세스와 충돌 또는 수동 중단으로 추정.

---

### 2026-05-14 ~ 05-17 — Distillation 최종 완주 (배포 미채택)

**Run**: `distill_l1_npz_clean_20260514` | **상태**: FINISHED

#### 학습 설정

| 항목 | 값 |
|---|---|
| Teacher | MedSAM (SAM ViT-B), `medsam_vit_b.pth` — 학습 중 동결 |
| Student | EfficientViT-SAM L1 (`pretrained=False`) |
| 학습 파라미터 | 43,585,568개 (전체 133,256,480개 중 student만) |
| 데이터 소스 | `train_npz/` (Pathology_new **제외**), limit_sample=400,000 |
| Teacher 입력 해상도 | 1024×1024 / Student 입력 해상도 512×512 |
| Batch size | **8** / Num workers 8 |
| Epochs | 8 (50,000 steps/epoch, 총 400,000 steps) |
| Optimizer | AdamW (lr=0.075, weight_decay=0.0005) |
| LR Scheduler | ExponentialLR (gamma=0.5, epoch 단위) |
| Precision | bf16-mixed / Gradient clip 0.5 |
| 체크포인트 | `weights/distilled-l1-clean-20260514/` (10,000 step마다, 34GB) |
| 로그 | `medficientsam/logs/distill_l1_nohup_20260514.log` |

#### Epoch별 결과

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

---

### 2026-06-12 ~ 06-15 — Pathology_new 포함 재학습 (완료)

**Run**: `distill_l1_bs16_ep8_20260612` | **상태**: FINISHED

#### 05-14 run과의 비교

| 항목 | distill_l1_npz_clean (05-14) | distill_l1_bs16_ep8 (06-12) |
|---|---|---|
| Pathology_new | **제외** | **포함** (19,062개 무결성 검증 완료) |
| 시작점 | scratch | scratch |
| batch_size | 8 | **16** |
| steps/epoch | 50,000 | **25,000** |
| 총 steps | 400,000 | **200,000** |
| num_workers | 8 | 16 |

#### 학습 설정

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
| 체크포인트 | `experiment_weights/distilled-l1-20260612/` (10,000 step마다, step_200000.ckpt 최종) |
| 로그 | `DP_MedificientSAM/logs/distill_l1_20260612.log` |
| MLflow | http://localhost:5001, run: `distill_l1_bs16_ep8_20260612` |

#### Epoch별 결과

| Epoch | Train Loss | LR | 소요시간 | 누적 |
|---|---|---|---|---|
| 0 | 0.002487 | 0.075000 | 8.17h | 8.17h |
| 1 | 0.001490 | 0.037500 | 8.28h | 16.45h |
| 2 | 0.001340 | 0.018750 | 8.23h | 24.68h |
| 3 | 0.001262 | 0.009375 | 8.24h | 32.92h |
| 4 | 0.001215 | 0.004688 | 8.18h | 41.10h |
| 5 | 0.001187 | 0.002344 | 8.20h | 49.30h |
| 6 | 0.001171 | 0.001172 | 8.17h | 57.47h |
| 7 | 0.001162 | 0.000586 | 8.20h | 65.67h |

- 총 학습 시간: **65.67h** (약 2.7일, 2026-06-12 ~ 06-15)
- 최종 loss: **0.001162** (초기 대비 ▼53.3%)
- 에러/데이터 누락: **없음** (8에폭 정상 완주)

---

## 두 완주 Run 상세 비교: 05-14 vs 06-12

> `step_400000.ckpt` vs `step_200000.ckpt` — 동일 아키텍처, 동일 하이퍼파라미터, 데이터·배치 구성 차이

### 학습 설정 비교

| 항목 | distill-l1-clean-20260514 | distill-l1-20260612 |
|---|---|---|
| 체크포인트 | `experiment_weights/distilled-l1-clean-20260514/step_400000.ckpt` | `experiment_weights/distilled-l1-20260612/step_200000.ckpt` |
| 데이터 pool | 51,802개 npz (Pathology_new **제외**) | 70,864개 npz (Pathology_new **포함**, 19,062개 추가) |
| limit_sample | 400,000 | 400,000 |
| batch_size | **8** | **16** |
| steps / epoch | **50,000** | **25,000** |
| 총 gradient update 횟수 | **400,000** | **200,000** |
| epoch당 처리 샘플 수 | 50,000 × 8 = **400,000** | 25,000 × 16 = **400,000** |
| Optimizer | AdamW (lr=0.075, wd=0.0005) | AdamW (lr=0.075, wd=0.0005) |
| LR Scheduler | ExponentialLR (gamma=0.5, epoch 단위) | ExponentialLR (gamma=0.5, epoch 단위) |
| LR 감소 주기 | 50,000 step마다 | 25,000 step마다 |
| Precision | bf16-mixed | bf16-mixed |
| 총 학습 시간 | 63.84h | 65.68h |

### Epoch별 Loss 비교 (MLflow 기반)

| Epoch | LR | 05-14 loss | 06-12 loss | 차이 (06-12 − 05-14) |
|---|---|---|---|---|
| 0 | 0.075000 | 0.002261 | 0.002487 | +0.000226 (+10.0%) |
| 1 | 0.037500 | 0.001504 | 0.001490 | −0.000014 (−0.9%) |
| 2 | 0.018750 | 0.001336 | 0.001340 | +0.000004 (+0.3%) |
| 3 | 0.009375 | 0.001245 | 0.001262 | +0.000017 (+1.4%) |
| 4 | 0.004688 | 0.001186 | 0.001215 | +0.000029 (+2.4%) |
| 5 | 0.002344 | 0.001147 | 0.001187 | +0.000040 (+3.5%) |
| 6 | 0.001172 | 0.001123 | 0.001171 | +0.000048 (+4.3%) |
| 7 | 0.000586 | **0.001110** | **0.001162** | +0.000052 (+4.7%) |

- **초기 loss 감소율**: 05-14 ▼50.9% / 06-12 ▼53.3%
- **최종 loss**: 05-14가 0.00116 − 0.00111 = **0.000052 낮음 (약 4.5% 차이)**

### 분석 및 해석

**① Epoch 0 초기 loss 차이 (+10%)**

06-12 run의 Epoch 0 loss가 0.002487로 05-14(0.002261)보다 10% 높다. Pathology_new 19,062개가 추가되어 데이터 pool이 다양해졌기 때문으로 해석된다. 병리 이미지(PanNuke, MoNuSeg)는 CT·XRay 등 다른 모달리티와 시각적 특성이 달라 초기 student 인코더가 더 높은 손실을 보이는 것이 자연스럽다.

**② Epoch 1부터 loss 궤적 역전**

Epoch 1에서 06-12(0.001490)와 05-14(0.001504)가 사실상 동등해지고, Epoch 2부터는 미세하게 05-14가 낮아진다. 이는 LR이 0.075 → 0.0375로 감소하면서 gradient update가 안정화되는 시점이 일치하기 때문으로 보인다.

**③ 최종 loss 차이(0.000052)의 원인: gradient update 횟수**

05-14는 총 400,000 gradient update, 06-12는 200,000 update. 동일한 limit_sample=400,000에서 epoch당 동일한 수의 샘플을 처리하지만, 05-14는 파라미터를 2배 더 업데이트했다. 최종 loss 차이 4.7%는 대부분 이 update 횟수 차이에서 기인한다.

**④ batch_size 영향**

batch_size=16이 8 대비 더 안정적인 gradient 추정을 제공하지만, update 횟수가 절반으로 줄어든 trade-off가 있다. 두 run 모두 동일 epoch당 총 샘플 수(400,000)를 처리하므로, loss 곡선 형태는 유사하다.

**⑤ 병리 도메인 성능 차이 가능성**

05-14는 Pathology_new를 전혀 학습하지 않았으므로, Ki-67 IHC 등 병리 이미지에 대해서는 06-12가 유리할 수 있다. 최종 loss는 MSE on image embeddings (전 모달리티 평균) 이므로 도메인별 성능은 별도 평가 필요.

**⑥ 배포 채택 여부**

두 run 모두 배포 모델(공식 pretrained 기반 fine-tuning)로는 채택되지 않음. 향후 distillation 기반 접근 재검토 시 06-12 체크포인트가 기준점.

---

## 코드 수정 이력 (이 프로젝트에서 변경한 사항)

| 파일 | 수정 내용 | 이유 |
|---|---|---|
| `src/data/components/medsam_dataset.py:68` | `A.TransformType` → `A.BasicTransform` | albumentations 2.0.8에서 `TransformType` 제거됨 |
| `configs/callbacks/distill.yaml` | `save_top_k: -1`, `monitor: null` 추가 | Lightning 2.2.5에서 monitor 없이 save_top_k>0 사용 불가 |
| `configs/callbacks/distill.yaml` | `rich_progress_bar` 기본값 제거 | nohup 환경(TTY 없음)에서 IndexError 발생 |
| `configs/logger/mlflow.yaml` | `experiment_name`, `run_name` 필드 명시 추가 | Hydra struct 외 필드 override 시 오류 방지 |
| `.env` | `CVPR2024_MEDSAM_DATA_DIR="/mnt/Disk1/sylee"` 설정 | 학습 데이터 경로 환경변수 |

---

## 재현 주의사항

1. **Pathology_new 사용 전 무결성 검증
 필수**: 2026-05-13 당시 `train_npz/Pathology_new/gts_npz_s128/` 내 다수 파일 누락으로 Epoch 2에서 FileNotFoundError 발생. 2026-06-12 재학습 시 19,062개 전량 검증 완료 후 포함. 재현 시 `numpy.load` + key 확인으로 사전 검증 권장.
2. **batch_size 선택**: 05-14 run은 batch_size=8 (50,000 steps/epoch), 06-12 run은 batch_size=16 (25,000 steps/epoch) — 둘 다 성공. GPU 메모리(H100 80GB)에서 batch_size=16 정상 동작 확인됨.
3. **ckpt_path=None**: 설정이 다른 run의 checkpoint에서 resume하면 데이터 분포 불일치로 학습 불안정. 항상 scratch로 시작 권장.
4. **albumentations 버전**: 2.0.8에서 `A.TransformType` 제거됨. `A.BasicTransform`으로 교체 완료 (`medsam_dataset.py:68`).

---

*MLflow Experiment ID: `832243143508473923`* (tracking URI: `DP_MedificientSAM/logs/mlflow/mlruns/`, port 5001)  
*DP_MedificientSAM 서브모듈 commit (06-12 run 기준): `b804e79d30dd87abda4cde777e9671ae72630efc`* (브랜치: `feat/distillation-config`, infinittAI/DP_MedificientSAM에 PR 머지 대기 중)
