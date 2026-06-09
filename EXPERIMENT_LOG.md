# MedEfficientSAM Distillation — 실험 로그

EfficientViT-SAM L1을 teacher MedSAM(SAM ViT-B)으로 knowledge distillation한 전체 실험 기록.  
성능 지표는 CVPR2024 MedSAM validation set 기준 DSC / NSD.

---

## 실험 이력 요약

| 날짜 | Run 이름 | 상태 | 실패 원인 |
|---|---|---|---|
| 2026-04-28 | distill_l1_no_extracted | FAILED (즉시) | albumentations 버전 비호환 |
| 2026-04-29 | distill_l1_no_extracted | FAILED (즉시 ×여러번) | 설정 오류 (빠른 실패) |
| 2026-04-29 ~ 05-01 | distill_l1_no_extracted | **FINISHED** | — (소규모 데이터로 첫 성공) |
| 2026-05-08 ~ 05-11 | distill_l1_no_extracted | **FINISHED** | — (CVPR2024 데이터 full 학습) |
| 2026-05-12 ~ 05-13 | distill_l1_no_extracted | KILLED | 외부 강제 종료 |
| 2026-05-13 (13:54) | distill_l1_no_extracted | FAILED (즉시) | 설정 오류 |
| 2026-05-13 (13:55) | distill_l1_no_extracted | FAILED (3h) | **데이터셋 누락 파일** (FileNotFoundError) |
| 2026-05-13 (17:11) | distill_l1_train_npz_all | FAILED (16min) | 미상 (빠른 실패) |
| 2026-05-13 (17:27) | distill_l1_train_npz_all | KILLED | 외부 강제 종료 (5h36m) |
| 2026-05-14 (09:13) | distill_l1_train_npz_all | FAILED (45min) | 미상 |
| 2026-05-14 ~ 05-17 | distill_l1_npz_clean | **FINISHED** | — (최종 성공) |

---

## 실패 원인 상세

### [1] 2026-04-28 — albumentations 버전 비호환

**증상**: 학습 시작 직후 즉시 크래시

**에러 메시지**:
```
AttributeError: module 'albumentations' has no attribute 'TransformType'.
Did you mean: 'Transform3D'?

ImportError: Error loading 'src.data.components.medsam_dataset.MedSAMDistillDataset':
    AttributeError("module 'albumentations' has no attribute 'TransformType'")
```

**원인**: `medsam_dataset.py`에서 `A.TransformType` 타입 힌트를 사용하는데, 설치된 albumentations 버전에는 해당 속성이 없었음. `TransformType`은 비교적 최신 버전에서 추가된 타입으로, 구버전에서는 `Transform3D` 등 다른 API를 사용.

**해결**: albumentations 버전 업그레이드 후 다음 실험 진행.

---

### [2] 2026-04-29 ~ 05-01 — 첫 번째 성공 (소규모 데이터)

**설정**:
- 데이터: `/mnt/Disk1/sylee/train_npz` (limit_sample 없음, 전체 사용)
- 실제 샘플 수: ~127,936개 (epoch당 ~15,992 steps × batch 8)
- batch_size: 16, num_workers: 16
- 체크포인트 저장: `weights/distilled-l1/`

**결과**:
- 8 epochs 완주, 최종 loss: **0.00133**
- 데이터셋이 작아 학습 데이터 다양성 부족 → 성능 한계 있음

---

### [3] 2026-05-08 ~ 05-11 — CVPR2024 데이터 도입 (batch=16)

**설정**:
- 데이터: CVPR2024 MedSAM `train_npz`, limit_sample=400,000
- batch_size: **16**, num_workers: 16 → epoch당 25,000 steps
- ckpt_path: None (scratch부터)
- 체크포인트 저장: `logs/train/runs/2026-05-08_09-42-08/checkpoints/`

**결과**:
- 8 epochs 완주, 총 200,000 steps, 최종 loss: **0.00113**
- 이후 평가에서 **DSC 0.8588 / NSD 0.8918** 달성 (→ `last_ckpt_metrics.csv`)

**한계**: batch_size=16은 이후 resume 시도에서 문제를 일으킴.

---

### [4] 2026-05-13 (13:55) — 데이터셋 누락 파일 (핵심 실패)

**증상**: Epoch 2, step ~8,745 (약 2시간 47분 후) 크래시

**에러 메시지**:
```
FileNotFoundError: [Errno 2] No such file or directory:
'/mnt/Disk1/sylee/train_npz/Pathology_new/gts_npz_s128/
2D_S26-03104,0,FDC00099,37298_0116_1_10251_26090_1238_1168_r06_c07.npz'
```

**설정**: 이전 run (`weights/distilled-l1-mlflow/last.ckpt`)을 이어받아 시작

**원인**: `train_npz` 디렉토리 내 `Pathology_new` 서브셋의 일부 `.npz` 파일이 실제로 존재하지 않음. 데이터셋 인덱스에는 등록되어 있으나 파일 자체가 누락됨 (불완전한 데이터 다운로드 또는 삭제로 추정).

**경위**: 이 run은 `ckpt_path=weights/distilled-l1-mlflow/last.ckpt`로 이전 run을 이어받으려 했으며, 데이터 파일 누락 + resume 방식 복합 문제.

**해결 방향**: 누락 파일을 포함하는 `Pathology_new` 데이터 서브셋을 제외하고 (`npz_clean`), 완전히 새로 학습.

---

### [5] 2026-05-13 (17:27) — 학습 속도 저하 후 강제 종료

**증상**: Epoch 0, step 31,624 (약 5시간 36분) 시점에 KILL

**로그**:
```
Epoch 0:  63%|██ | 31624/50000 [5:36:03<3:15:16, 1.57 it/s, ...]
```

**원인**: 학습은 정상 진행 중이었으나 외부에서 강제 종료. 이 시점의 it/s는 1.57로 정상 범위. 디스크 공간, 다른 프로세스와의 충돌, 또는 수동 중단으로 추정.

---

## 최종 성공 Run

### 2026-05-14 ~ 05-17 — distill_l1_npz_clean_20260514

**핵심 변경사항**:
- `ckpt_path: None` — 이전 checkpoint 없이 처음부터
- `batch_size: 8` (기존 16에서 절반으로 감소) → epoch당 50,000 steps
- 데이터: `Pathology_new` 누락 파일 제외한 정제 데이터 (`npz_clean`)

**Epoch별 결과**:

| Epoch | Train Loss | LR | 소요시간 | 누적시간 |
|---|---|---|---|---|
| 0 | 0.002261 | 0.075000 | 8.09h | 8.09h |
| 1 | 0.001504 | 0.037500 | 8.15h | 16.24h |
| 2 | 0.001336 | 0.018750 | 8.12h | 24.36h |
| 3 | 0.001245 | 0.009375 | 7.90h | 32.26h |
| 4 | 0.001186 | 0.004688 | 7.86h | 40.12h |
| 5 | 0.001147 | 0.002344 | 8.02h | 48.14h |
| 6 | 0.001123 | 0.001172 | 7.85h | 55.99h |
| 7 | 0.001110 | 0.000586 | 7.85h | 63.84h |

- **총 학습 시간**: 63.84시간 (약 2.7일)
- **최종 loss**: 0.00111 (전체 ▼ 50.9%)

---

## 실험 체크포인트 현황

공식 배포 가중치(distilled-l0/l1/l2, finetuned-l{0,1,2}-{augmented,unaugmented})를 제외한, 직접 실험한 체크포인트 목록이다.

### distilled-l1-mlflow/ (5.0GB)

**생성 시기**: 2026-05-12 ~ 05-13

CVPR2024 대용량 데이터로 처음 시도한 run. `batch_size=16`, 공식 `distilled-l1/step_400000.ckpt`에서 resume하려 했으나 외부 강제 종료(KILL)로 중단. `step_050000`까지 진행됐고, 이후 `last.ckpt`를 다음 run의 시작점으로 쓰려다 실패한 이력 있음.

### distilled-l1-train_npz/ (3.3GB)

**생성 시기**: 2026-05-13

`Pathology_new`를 포함한 전체 train_npz 데이터로 시도한 run. `step_030000`(약 3시간) 진행 후 `Pathology_new` 서브셋의 누락 파일(`FileNotFoundError`)로 실패.

### distilled-l1-clean-20260514/ (34GB)

**생성 시기**: 2026-05-14 ~ 05-17 (약 63.8시간)

누락 파일 문제를 해결하기 위해 `Pathology_new`를 제외한 정제 데이터로, `batch_size=8`, `ckpt_path=None`(처음부터)으로 재학습한 최종 성공 run. 8 epoch / 400,000 step 완주. 10,000 step마다 총 40개의 체크포인트가 저장되어 있으며, `last.ckpt`는 `step_400000`과 동일. `student_encoder.pt`는 학습 완료 다음날 수동 추출한 미사용 파일.

**최종 배포 대상**: `step_400000.ckpt` (842MB)

---

## 최종 산출물

| 파일 | 경로 | 설명 |
|---|---|---|
| **최종 체크포인트** | `weights/distilled-l1-clean-20260514/step_400000.ckpt` | 842MB, 8 epoch / 400,000 step 완주 결과 |

> **참고**: 실제 서비스 중인 ONNX(`Ki-67_service/deployment/*.quantized.onnx`)는 이 distillation 체크포인트가 아니라 `weights/finetuned-l1-augmented/best.ckpt`(Ki-67 fine-tuning 결과)로 export한 것이다.

---

## 교훈 및 재현 주의사항

1. **데이터 무결성 검증 필수**: `train_npz` 내 모든 파일이 실제 존재하는지 사전 확인 후 학습 시작. `Pathology_new` 서브셋에 누락 파일 있음.
2. **batch_size는 8로 고정**: 16에서 OOM 또는 불안정성 문제 재발 가능성. 에폭당 step 수는 50,000 기준으로 설정.
3. **ckpt_path=None**: 다른 설정의 run에서 이어받으면 데이터 분포 불일치로 학습이 불안정해질 수 있음.
4. **albumentations 버전**: `A.TransformType` 속성이 존재하는 버전 필요 (`environment.yaml` 기준으로 환경 재현).

---

*분석 기준일: 2026-06-08*  
*MLflow Experiment ID: `986928226268314361`*  
*최종 Git commit: `59504938bb37ab7e2832ede358051976e740efe5`*
