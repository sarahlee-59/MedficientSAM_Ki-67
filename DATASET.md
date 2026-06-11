# 데이터셋 설명

## 개요

| 역할 | 데이터셋 | 용도 |
|------|---------|------|
| Distillation 학습 | MedSAM 2024 Challenge 공개 데이터 | Student 인코더가 Teacher(MedSAM) 임베딩을 모방하도록 학습 |
| Fine-tuning / 평가 | Ki-67 병리 슬라이드 (내부) + MoNuSeg / PanNuke (공개) | Ki-67 H&E 이미지에서 세포 instance segmentation |

---

## 1. MedSAM 2024 Challenge 데이터 (Distillation 학습용)

### 경로

```
/mnt/Disk1/sylee/train_npz/
```

### 다운로드 출처

논문(MedficientSAM) 기준 11개 모달리티를 구성했으나, **공식 Google Drive에 CT·Dermoscopy·PanNuke·MoNuSeg 2018이 누락**되어 챌린지 공식 홈페이지([Codabench](https://www.codabench.org/competitions/1847/)) 추가 Google Sheet에서 별도 확보했습니다.

| 경로 | 포함 데이터 | 비고 |
|------|------------|------|
| [공식 Google Drive](https://drive.google.com/drive/folders/1khEIdkO0MC_gG5EkQ7COdDS1jge5_XQs) | Endoscopy, Fundus, Mammography, Microscopy, MR, OCT, PET, US, XRay | 챌린지 주최 측이 NPZ로 변환하여 배포 |
| [공식 홈페이지 추가 Google Sheet](https://docs.google.com/spreadsheets/d/1QxjFs41eU6JG5KNhP576fc8MotrJ58KCrqH83HG-__E/edit?gid=2057737934#gid=2057737934) | CT 전체, Dermoscopy (ISIC-2017), PanNuke, MoNuSeg 2018 | 원본 데이터 링크 제공 — **CT는 NPZ로 다운로드, 나머지는 원본 이미지를 내부 전처리하여 NPZ 생성** |

### 구성

| Modality | Sub-dataset | 파일 수 | 비고 |
|----------|-------------|---------|------|
| CT | AbdomenCT-1K | 1,000 | 스프레드시트 경유 |
| CT | AMOS22 | 240 | 스프레드시트 경유 |
| CT | COVID-19-20 | 199 | 스프레드시트 경유 |
| CT | KiTS23 | 489 | 스프레드시트 경유 |
| CT | TotalSegmentator | 1,174 | 스프레드시트 경유. 전신 117개 부위 커버 |
| Dermoscopy | ISIC-2017 | 2,000 | Sheet 원본 → stride 128, crop 256×256 tiling → NPZ |
| Endoscopy | CholecSeg8k | 10,117 | |
| Endoscopy | Kvasir-SEG | 1,000 | |
| Endoscopy | m2caiSeg | 1,807 | |
| Fundus | IDRiD | 81 | |
| Fundus | PAPILA | 976 | |
| Mammography | CDD-CESM | 1,233 | |
| Microscopy | NeurIPS22CellSeg | 1,000 | |
| MR | AMOS MR, BraTS, CervicalCancer 등 13종 | 4,881 | |
| OCT | Intraretinal-Cystoid-Fluid | 1,436 | |
| Pathology | Ki-67 (gts_npz_s128) | 16,376 | |
| Pathology | PanNuke | 2,538 | |
| Pathology | MoNuSeg2018 | 148 | |
| PET | autoPET | 345 | |
| US | Breast-Ultrasound, hc18 | 1,646 | |
| XRay | Chest X-ray, COVID 관련 4종 | 22,178 | |
| **합계** | | **70,864** | |

### NPZ 파일 형식

```python
data = np.load("sample.npz")
data["imgs"]  # shape: (H, W, 3) uint8  — RGB 이미지 패치 [0, 255]
data["gts"]   # shape: (H, W) int32     — instance segmentation 마스크
              # 3D 볼륨: imgs=(D,H,W,3), gts=(D,H,W)
```

- `2D_` prefix: 2D 이미지 슬라이스 (H×W×3)
- `3D_` prefix: 3D 볼륨 (D×H×W×3), 훈련 시 슬라이스 단위로 처리
- `gts`: 0=background, 양수=각 instance 번호

---

## 2. Ki-67 병리 데이터셋 및 Fine-tuning

### 경로

```
/mnt/Disk1/sylee/train_npz/Pathology_new/gts_npz_s128/   # 학습·평가용 NPZ 패치 (Ki-67)
/mnt/Disk1/sylee/Pathology/                               # 원본 NPZ (절대 좌표 기반)
/mnt/Disk1/sylee/Pathology_new/                           # 원본 PNG 시각화
```

### Ki-67 슬라이드 데이터 규모

부산 백병원 Ki-67 IHC 슬라이드 89장을 256×256 패치로 분할한 데이터셋.

| 항목 | 값 |
|------|----|
| 원본 슬라이드 수 | 89장 (부산 백병원) |
| 패치 추출 방식 | crop 256×256, stride 128 (50% overlap) |
| 총 패치 수 | 16,376 |
| 파일명 형식 | `2D_<슬라이드ID>_r<행>_c<열>.npz` |

```python
data = np.load("2D_<slide>_r00_c00.npz")
data["imgs"]  # (256, 256, 3) uint8  — IHC 패치 RGB
data["gts"]   # (256, 256) int32    — 세포 instance 마스크 (0=bg, 1,2,...=각 세포)
```

### 공개 병리 보조 데이터

| 데이터셋 | 설명 | 패치 수 | 전처리 |
|----------|------|---------|--------|
| PanNuke | Pan-Cancer 세포 핵 segmentation | 2,538 | Google Sheet 원본 → tiling 없이 256×256 resize → NPZ |
| MoNuSeg 2018 | H&E 조직 multi-organ 세포 segmentation | 148 | Google Sheet 원본 → 1000×1000 → 2×2 crop → 256×256 resize → NPZ |

### Fine-tuning 데이터 구성

| 실험 | 데이터 | 총 패치 수 |
|------|--------|-----------|
| 조합 학습 | Ki-67 + PanNuke + MoNuSeg | 19,062 |
| Ki-67 단독 | 백병원 Ki-67만 | 16,376 |

**Ki-67 단독 실험 분할 (90/5/5)**

| 분할 | 패치 수 |
|------|---------|
| Train (90%) | 14,738 |
| Val (5%) | 819 |
| Test (5%) | 819 |
| **합계** | **16,376** |

총 학습 데이터 (augmentation 포함): 14,738 × 16 epochs ≈ **235K patches**

### 데이터 증강 파이프라인 (Fine-tuning)

| 카테고리 | 기법 |
|----------|------|
| Geometric | HorizontalFlip, VerticalFlip, RandomRotate90 |
| IHC stain | **HEDJitter** (H&E/IHC 염색 변이 모사) |
| Scanner / Quality | GaussianBlur / MotionBlur / Defocus 중 1개, ImageCompression (quality 60~100), Downscale |
| Photometric | RandomBrightnessContrast |

---

## 3. NPZ 전처리 코드 출처

| 데이터 | 전처리 주체 | 방법 | 코드 위치 |
|--------|------------|------|----------|
| Endoscopy, Fundus, Mammography, Microscopy, MR, OCT, PET, US, XRay | 챌린지 주최 측 | 이미 NPZ 형태로 배포 | [MedSAM/LiteMedSAM 공식 레포](https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM) (`pre_grey_rgb.py`) |
| CT (AbdomenCT-1K, AMOS22, COVID-19-20, KiTS23, TotalSegmentator) | 챌린지 주최 측 (Google Sheet 경유) | 이미 NPZ 형태로 제공 | 위 MedSAM 레포 (`pre_CT_MR.py`) |
| Dermoscopy — ISIC-2017 | 내부 전처리 | stride 128, crop 256×256 sliding window tiling → NPZ | 별도 스크립트 (이 레포 미포함) |
| PanNuke | 내부 전처리 (Google Sheet 경유) | 256×256 resize → NPZ | 별도 스크립트 (이 레포 미포함) |
| MoNuSeg 2018 | 내부 전처리 (Google Sheet 경유) | 1000×1000 → 2×2 crop → 256×256 resize → NPZ | 별도 스크립트 (이 레포 미포함) |
| Ki-67 IHC 슬라이드 | 자체 병리 전처리 파이프라인 | 256×256 패치, stride 128 | `/mnt/Disk1/DP_IHC/Ki67_pytorchlightning/` (별도 프로젝트) |

### 주최 측 전처리 상세

#### 2D 모달리티 — `pre_grey_rgb.py`

- 픽셀 최대값 > 255이면 min-max 스케일링 → uint8, 그 외 원본 유지
- 그레이스케일 → R=G=B 3채널 복제
- 채널 > 3이면 앞 3채널만 사용
- 리사이즈·패치 추출 없이 **원본 해상도 그대로** NPZ 저장

#### 3D 모달리티 — `pre_CT_MR.py`

| 단계 | CT | MR |
|------|----|----|
| 파일 읽기 | NIfTI (`.nii.gz`) — SimpleITK | 동일 |
| 픽셀값 정규화 | **HU Windowing**: level=40, width=400 → [-160, 240] HU clip → uint8 | **퍼센타일 클리핑**: 0.5~99.5 퍼센타일 clip → uint8 |
| 그레이→RGB | 단채널 → 3채널 복제 | 동일 |
| 노이즈 제거 | 3D 100 voxel 미만 제거, 슬라이스별 2D 10 pixel 미만 제거 | 동일 |
| NPZ 내용 | `imgs` + `gts` + `spacing` | 동일 |

#### 모달리티별 요약

| 모달리티 | 픽셀값 정규화 | 그레이→RGB | 리사이즈 | 작은 객체 제거 |
|---------|------------|----------|---------|------------|
| CT | HU windowing [-160, 240] → uint8 | O | 없음 | 3D 100 voxel, 2D 10 pixel |
| MR | 0.5~99.5 퍼센타일 클리핑 → uint8 | O | 없음 | 3D 100 voxel, 2D 10 pixel |
| 나머지 2D | 픽셀 > 255이면 min-max, 아니면 원본 유지 | O (그레이스케일만) | 없음 | 없음 |

---

## 4. 데이터 전처리 (학습 시 실시간)

### Distillation — `MedSAMDistillDataset`

1. NPZ에서 `imgs` 로드 (2D/3D 자동 분기)
2. 그레이스케일 → 3채널 복제
3. Albumentations augmentation (HorizontalFlip, VerticalFlip p=0.5)
4. Student 입력: longest-side resize → 512×512, min-max 스케일 [0, 1]
5. Teacher 입력: longest-side resize → 1024×1024, min-max 스케일 [0, 1]

### Fine-tuning — `MedSAMTrainDataset`

1. NPZ에서 `imgs`, `gts` 로드
2. 랜덤 label 샘플링 (`mask_num=5`)
3. Augmentation (HorizontalFlip, VerticalFlip, ShiftScaleRotate)
4. 마스크 → bounding box 추출 (random shift ±5 px)
5. 이미지: longest-side resize → 512×512, min-max 스케일
6. Box: 1024×1024 prompt encoder 좌표계로 변환

### 관련 코드

- `src/data/components/medsam_dataset.py` — Dataset 클래스
- `src/data/medsam_datamodule.py` — LightningDataModule
- `src/utils/transforms.py` — ResizeLongestSide, get_bbox, transform_gt
- `configs/data/distill_medsam.yaml` — Distillation 데이터 설정
- `configs/data/finetune_medsam.yaml` — Fine-tuning 데이터 설정
