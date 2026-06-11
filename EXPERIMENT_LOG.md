# MedficientSAM 경량화 실험 로그

EfficientViT-SAM L1(student)을 MedSAM ViT-B(teacher)로 knowledge distillation한 실험 기록.  
distillation 기반 fine-tuning은 성능 불량으로 채택되지 않았으며, **배포 모델은 공식 EfficientViT-SAM L1 pretrained 가중치 기반 fine-tuning**을 사용한다.

---

## 배포 파이프라인

```
[1] Fine-tuning  (Ki-67 도메인 특화)
    기반:   EfficientViT-SAM L1 공식 pretrained (GitHub 공식 배포)
    데이터:  Ki-67 IHC + PanNuke + MoNuSeg (augmented)
    출력:   weights/finetuned-l1-augmented/best.ckpt

        ↓

[2] Export
    설정:   configs/experiment/export_finetuned_l1_onnx.yaml
    출력:   Ki-67_service/deployment/encoder.quantized.onnx (INT8)
             Ki-67_service/deployment/decoder.quantized.onnx (INT8)
             Ki-67_service/deployment/openvino/{encoder,decoder}.{xml,bin} (FP32)

        ↓

[3] 서비스
    진입점: http://10.10.40.194:3000/realtime
    런타임: OpenVINO FP32 (ONNX INT8 대비 e2e 5.1× 빠름)
```

---

## Distillation 실험 요약

EfficientViT-SAM L1을 MedSAM ViT-B로 distillation하는 실험을 진행했으나,  
distillation 가중치(`distilled-l1-clean-20260514/step_400000.ckpt`) 기반 fine-tuning은  
공식 pretrained 기반 대비 성능이 낮아 최종 배포에서 제외되었다.

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
| 2026-05-14 ~ 05-17 | distill_l1_npz_clean | **FINISHED** | — (최종 성공) |

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

**설정**:
- 데이터: CVPR2024 도입 이전 소규모 데이터셋 (`train_npz/` 전체), 유효 2D 슬라이스 약 127,936개
  - 모달리티 구성: 현재 CVPR2024 기준과 동일하게 CT·MR·PET(3D 볼륨) + XRay·Endoscopy·Pathology_new 등 2D 12개 서브디렉토리였으나, 파일 수가 훨씬 적은 부분 다운로드 상태
  - CT/MR/PET 3D 볼륨의 유효 비공백 슬라이스 포함 총합이 127,936개 (limit_sample=None, 전량 사용)
- batch_size: 8 (configs/data/distill_medsam.yaml 기본값; 127,936 / 8 = 15,992 steps/epoch과 일치), num_workers: 16
- 저장: `weights/distilled-l1/`

**결과**: 8 epoch 완주, 최종 loss **0.00133**  
데이터 다양성 부족으로 성능 한계 있음 → CVPR2024 MedSAM 전체 데이터로 재학습 필요.

---

### [3] 2026-05-08 ~ 05-11 — CVPR2024 데이터 도입

**설정**:
- 데이터: CVPR2024 MedSAM 공식 `train_npz/` 전량 (12개 모달리티, 70,864 npz 파일, 유효 슬라이스 ~1,050,000개)
  - 3D: CT 3,102개(평균 228 슬라이스/파일), MR 4,881개(평균 55 슬라이스/파일), PET 345개(평균 44 슬라이스/파일)
  - 2D: XRay 22,178개, Endoscopy 12,924개, Pathology_new 19,062개, US 1,646개, Dermoscopy 2,000개, OCT 1,436개, Microscopy 1,000개, Fundus 1,057개, Mammography 1,233개
  - `limit_sample=400,000` 으로 seed=42 랜덤 샘플링 → 실제 사용 샘플 40만 개
- batch_size: 16 (epoch당 25,000 steps), ckpt_path: None
- 저장: `logs/train/runs/2026-05-08_09-42-08/checkpoints/`

**결과**: 8 epoch / 200,000 steps 완주, 최종 loss **0.00113**  
평가: **DSC 0.8588 / NSD 0.8918** (`eval_results/last_ckpt_metrics.csv`, 3,077 cases)

> 이 DSC/NSD 수치는 CVPR2024 MedSAM validation set 기준이며, Ki-67 병리 이미지 성능과는 무관하다.

**한계**: batch_size=16은 이후 resume 시 불안정성 유발.

---

### [4] 2026-05-13 (13:55) — Pathology_new 누락 파일 (핵심 실패)

**증상**: Epoch 2, step ~8,745 (약 2시간 47분) 시점에 크래시

```
FileNotFoundError: .../train_npz/Pathology_new/gts_npz_s128/
2D_S26-03104,..._r06_c07.npz
```

- 데이터: run [3]과 동일한 CVPR2024 `train_npz/` (Pathology_new 포함 전체), limit_sample=400,000
- `weights/distilled-l1-mlflow/last.ckpt` (batch_size=16, step_050000 시점)에서 resume 시도
- `Pathology_new/` 서브셋은 19,062개 npz 경로가 glob에 잡히지만, `gts_npz_s128/` 하위 다수 파일이 실제 디스크에 없음 (불완전한 다운로드 추정)
- seed=42 샘플링 결과 Epoch 2에서 해당 누락 파일이 처음 접근됨 → Epoch 0~1은 우연히 정상 파일만 사용

**해결 방향**: `Pathology_new` 제외 정제 데이터(`npz_clean`)로 scratch 재학습.

---

### [5] 2026-05-13 (17:27) — 외부 강제 종료

- 데이터: run [4]와 동일한 CVPR2024 `train_npz/` (Pathology_new 포함), limit_sample=400,000  
  (Pathology_new 누락 문제는 아직 제거 전, 같은 데이터로 재시도한 run)

**상황**: Epoch 0, 63% 진행 중 (1.57 it/s — 정상 속도) 외부 KILL.  
학습 자체 문제는 없었음. 다른 프로세스와 충돌 또는 수동 중단으로 추정.

---

## 최종 성공 Run — distill_l1_npz_clean_20260514

**핵심 변경사항**:

| 항목 | 이전 | 이번 |
|---|---|---|
| 시작점 | 이전 checkpoint resume | `ckpt_path=None` (scratch) |
| batch_size | 16 | **8** (epoch당 50,000 steps) |
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

### `weights/distilled-l1-mlflow/` (5.0GB)

2026-05-12 ~ 05-13. batch_size=16, step_050000까지 진행 후 외부 KILL.  
이후 이 `last.ckpt`를 resume 시작점으로 쓰려다 Pathology_new 누락 문제와 겹쳐 실패한 이력 있음.

### `weights/distilled-l1-train_npz/` (3.3GB)

2026-05-13. Pathology_new 포함 전체 데이터로 시도, step_030000 진행 후 FileNotFoundError로 실패.

### `weights/distilled-l1-clean-20260514/` (34GB) — **최종 완료 (배포 미채택)**

2026-05-14 ~ 05-17 (63.8h). Pathology_new 제외, batch_size=8, scratch 학습.  
8 epoch / 400,000 steps 완주. 10,000 steps마다 40개 체크포인트 저장.  
`step_400000.ckpt` (842MB) = `last.ckpt` — fine-tuning 시도했으나 성능 불량으로 채택 안 됨.

---

## 추론 성능 벤치마크

Ki-67 병리 이미지 256×256 px 5장 기준 (`Ki-67_service/benchmark/`).  
하드웨어: Intel CPU.

| 런타임 | 모델 정밀도 | encode (ms) | decode ×4 (ms) | e2e (ms) |
|---|---|---|---|---|
| ONNX Runtime 1.20.1 | INT8 | 544.9 | 92.3 | 637.1 |
| OpenVINO 2026.2.0 | FP32 | 75.2 | 49.8 | **125.0** |
| **배율** | | **7.2× 빠름** | **1.9× 빠름** | **5.1× 빠름** |

ONNX INT8 양자화에도 불구하고, OpenVINO의 Intel CPU 전용 oneDNN fused kernel이 압도적으로 빠름.  
encode가 전체 e2e의 85~88%를 차지하므로 encoder 최적화가 가장 중요.  
→ **서비스는 OpenVINO FP32 채택** (`Ki-67_service/deployment/openvino/`).

---

## 재현 주의사항

1. **데이터 무결성 사전 검증**: `train_npz/Pathology_new/` 내 누락 파일 다수 — **학습 시 제외** 필수.
2. **batch_size=8 고정**: 16으로 설정 시 OOM 또는 불안정성 재발 가능. epoch당 50,000 steps 기준.
3. **ckpt_path=None**: 설정이 다른 run의 checkpoint에서 resume하면 데이터 분포 불일치로 학습 불안정.
4. **albumentations 버전**: `A.TransformType` 속성 존재 버전 필요. `environment.yaml` 기준 환경 재현.

---

*분석 기준일: 2026-06-08*  
*MLflow Experiment ID: `986928226268314361`*  
*최종 Git commit: `59504938bb37ab7e2832ede358051976e740efe5`*
