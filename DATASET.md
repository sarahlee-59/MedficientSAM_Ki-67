# 데이터셋 설명

## 한눈에 보기

이 프로젝트는 모델을 두 단계에 걸쳐 학습시킵니다.

| 단계 | 무엇을 하나 | 어떤 데이터를 쓰나 |
|------|------------|-------------------|
| **1. Distillation** | 작은 Student 모델이 큰 Teacher(MedSAM)의 임베딩을 따라 하도록 학습 | MedSAM 2024 Challenge 공개 데이터 (11개 모달리티, 약 7만 장) |
| **2. Fine-tuning / 평가** | Ki-67 H&E 이미지에서 세포 하나하나를 구분(instance segmentation) | Ki-67 슬라이드(내부) + MoNuSeg / PanNuke(공개) |

---

## 1. MedSAM 2024 Challenge 데이터 (Distillation 학습용)

### 저장 위치

```
/mnt/Disk1/sylee/train_npz/
```

### 데이터를 어디서 모았나

MedficientSAM 논문의 11개 모달리티 기준. 두 곳에서 수집했다.

| 출처 | 받은 데이터 | 형태 |
|------|------------|------|
| [공식 Google Drive](https://drive.google.com/drive/folders/1khEIdkO0MC_gG5EkQ7COdDS1jge5_XQs) | Endoscopy, Fundus, Mammography, Microscopy, MR, OCT, PET, US, XRay | 이미 NPZ로 변환되어 배포됨 |
| [추가 Google Sheet](https://docs.google.com/spreadsheets/d/1QxjFs41eU6JG5KNhP576fc8MotrJ58KCrqH83HG-__E/edit?gid=2057737934#gid=2057737934) | CT 전체, Dermoscopy(ISIC-2017) | 원본 링크 제공 — CT는 NPZ로 받고, Dermoscopy는 원본을 내부 전처리 |
| [추가 Google Sheet](https://docs.google.com/spreadsheets/d/1QxjFs41eU6JG5KNhP576fc8MotrJ58KCrqH83HG-__E/edit?gid=2057737934#gid=2057737934) | PanNuke, MoNuSeg 2018 | pathology 보강용으로 추가 — 원본을 내부 전처리 |

### 모달리티별 구성

| Modality | Sub-dataset | 파일 수 | 비고 |
|----------|-------------|---------|------|
| CT | AbdomenCT-1K | 1,000 | Sheet 경유 |
| CT | AMOS22 | 200 | Sheet 경유, Tr 분할 CT만 (case_id ≤ 500) |
| CT | COVID-19-20 | 199 | Sheet 경유 |
| CT | KiTS23 | 489 | Sheet 경유 |
| CT | TotalSegmentator | 1,174 | Sheet 경유, 전신 117개 부위 |
| Dermoscopy | ISIC-2017 | 2,000 | Sheet 경유, stride 128 tiling |
| Endoscopy | CholecSeg8k | 10,117 | |
| Endoscopy | Kvasir-SEG | 1,000 | |
| Endoscopy | m2caiSeg | 1,807 | |
| Fundus | IDRiD | 81 | |
| Fundus | PAPILA | 976 | |
| Mammography | CDD-CESM | 1,233 | |
| Microscopy | NeurIPS22CellSeg | 1,000 | |
| MR | AMOS MR, BraTS, CervicalCancer 등 13종 | 4,881 | |
| OCT | Intraretinal-Cystoid-Fluid | 1,436 | |
| Pathology | Ki-67 (gts_npz_s128) | 16,376 | 핵심 타깃 |
| Pathology | PanNuke | 2,538 | Sheet 경유, 보강 |
| Pathology | MoNuSeg2018 | 148 | Sheet 경유, 보강 |
| PET | autoPET | 345 | |
| US | Breast-Ultrasound, hc18 | 1,646 | |
| XRay | Chest X-ray, COVID 관련 4종 | 22,178 | |
| **합계** | | **70,824** | |

### NPZ 파일은 어떻게 생겼나

모든 데이터는 `.npz` 파일 하나에 **이미지와 마스크를 함께** 담습니다.

```python
data = np.load("sample.npz")
data["imgs"]  # (H, W, 3) uint8  — RGB 이미지 [0, 255]
data["gts"]   # (H, W)   int32   — 세포별 마스크
              # 3D 볼륨이면: imgs=(D,H,W,3), gts=(D,H,W)
```

파일을 구분하고 해석하는 규칙은 다음과 같습니다.

- 파일명 앞에 **`2D_`** → 2D 이미지 한 장 (H×W×3)
- 파일명 앞에 **`3D_`** → 3D 볼륨 (D×H×W×3). 학습할 때는 한 장씩(슬라이스 단위) 잘라서 처리합니다.
- `gts`(정답 마스크) 값의 의미: **0 = 배경**, 나머지 **양수 = 각각의 세포 번호**

---

## 2. Ki-67 Pathology 데이터셋과 Fine-tuning

### 저장 위치

```
/mnt/Disk1/sylee/train_npz/Pathology_new/gts_npz_s128/   # Ki-67 학습·평가용 NPZ 패치
```

### Ki-67 슬라이드 규모

부산 백병원의 Ki-67 IHC 슬라이드 89장을 작은 패치로 잘라 만들었습니다.

| 항목 | 값 |
|------|----|
| 원본 슬라이드 | 89장 (부산 백병원) |
| 자르는 방식 | 256×256 크기, stride 128 (절반씩 겹쳐 가며 자름) |
| 총 패치 수 | 16,376 |
| 파일명 형식 | `2D_<슬라이드ID>_r<행>_c<열>.npz` |

### 핵 분할 보강 공개 데이터

| 데이터셋 | 무엇인가 | 패치 수 | 전처리 |
|----------|---------|---------|--------|
| PanNuke | 여러 암종의 세포 핵 분할 데이터 | 2,538 | 256×256으로 resize → NPZ |
| MoNuSeg 2018 | H&E 조직, 여러 장기의 세포 분할 데이터 | 148 | 1000×1000 → 2×2로 자름 → 256×256 resize → NPZ |

### Fine-tuning 데이터 묶음

| 실험 | 데이터 구성 | 총 패치 수 |
|------|------------|-----------|
| 조합 학습 | Ki-67 + PanNuke + MoNuSeg | 19,062 |
| Ki-67 단독 | 백병원 Ki-67만 | 16,376 |

Ki-67 단독은 90 / 5 / 5 (train / val / test) 분할.

| 분할 | 패치 수 |
|------|---------|
| Train (90%) | 14,738 |
| Val (5%) | 819 |
| Test (5%) | 819 |
| **합계** | **16,376** |

augmentation 포함 시 학습량: 14,738 × 16 epochs ≈ 235K 패치.

### 데이터 증강 파이프라인

| 카테고리 | 기법 | 목적 |
|----------|------|------|
| Geometric | HorizontalFlip, VerticalFlip, RandomRotate90 | 방향·위치 변화에 강하게 |
| IHC 염색 | **HEDJitter** | H&E/IHC 염색 색 변이 흉내 |
| 스캐너 / 품질 | GaussianBlur·MotionBlur·Defocus 중 1개, ImageCompression(60~100), Downscale | 흐림·압축·저해상도 상황 대응 |
| Photometric | RandomBrightnessContrast | 밝기·대비 변화 대응 |

---

## 3. NPZ 준비 방법

| 방식 | 대상 | 처리 |
|------|------|------|
| **A. 다운로드만** | 9개 모달리티 | 주최 측 NPZ 그대로 사용 |
| **B. NIfTI → NPZ** | CT 5종 | CT windowing → uint8 정규화 → 유효 슬라이스 추출 |
| **C. 내부 전처리** | Ki-67, Dermoscopy, PanNuke, MoNuSeg | crop/resize → NPZ 생성 |

### A. 다운로드만 (변환 없음)

출처: [공식 Google Drive](https://drive.google.com/drive/folders/1khEIdkO0MC_gG5EkQ7COdDS1jge5_XQs)

| 모달리티 | 데이터셋 |
|---------|---------|
| Endoscopy | CholecSeg8k, Kvasir-SEG, m2caiSeg |
| Fundus | IDRiD, PAPILA |
| Mammography | CDD-CESM |
| Microscopy | NeurIPS22CellSeg |
| MR | AMOS MR, BraTS 등 13종 |
| OCT | Intraretinal-Cystoid-Fluid |
| PET | autoPET |
| US | Breast-Ultrasound, hc18 |
| XRay | Chest-Xray, COVID-19-Radiography 등 |

### B. CT 5종 — NIfTI → NPZ 변환

CT 원본(`.nii.gz`)을 NPZ로 변환. 상세 파이프라인(데이터셋별 원본 구조, 스크립트 인자, 실행 방법): [`CT/PREPROCESSING_PIPELINE.md`](CT/PREPROCESSING_PIPELINE.md)

| 데이터셋 | 케이스 수 | CT Window | 비고 |
|---------|----------|-----------|------|
| AbdomenCT-1K | 1,000 | WL=40, WW=400 | 십이지장(label 12) 제거 |
| AMOS22 | 200 | WL=40, WW=400 | Tr CT만 (case_id ≤ 500) — MRI 40개 수동 삭제 완료 |
| COVID-19-20 | 199 | WL=−500, WW=1500 | 폐 window |
| KiTS23 | 489 | WL=100, WW=400 | 신장 window |
| TotalSegmentator | 1,174 | WL=40, WW=400 | 장기별 binary 마스크 17개 → multi-label 병합, 1,228건 중 54건 GT 없어 제외 |

### C. 내부 전처리

| 데이터셋 | 원본 | 처리 | 결과 |
|----------|------|------|------|
| Ki-67 (gts_npz_s128) | IHC 슬라이드 89장 | 256×256 crop, stride 128 | 16,376 패치 |
| Dermoscopy / ISIC-2017 | ISIC-2017 원본 | 256×256 crop, stride 128 | 2,000 패치 |
| PanNuke | 세포 핵 이미지 | 256×256 resize | 2,538 패치 |
| MoNuSeg 2018 | 1000×1000 H&E 이미지 | 2×2 crop → 256×256 resize | 148 패치 |

---

## 4. 학습 시 실시간 전처리

### Distillation — `MedSAMDistillDataset`

```
[NPZ imgs] ──▶ 2D/3D 분기 ──▶ 3채널 복제 ──▶ 증강 ──┬──▶ Student 입력: 512×512, [0,1]
                                                    └──▶ Teacher 입력: 1024×1024, [0,1]
```

1. NPZ에서 `imgs` 로드 (2D/3D 자동 구분)
2. 그레이스케일이면 3채널로 복제
3. Augmentation (HorizontalFlip, VerticalFlip, p=0.5)
4. **Student 입력**: 긴 변 기준 512×512로 resize, [0,1] 스케일
5. **Teacher 입력**: 긴 변 기준 1024×1024로 resize, [0,1] 스케일

> Student와 Teacher가 해상도만 다르고 같은 이미지를 보도록 맞춥니다.

### Fine-tuning — `MedSAMTrainDataset`

```
[NPZ imgs+gts] ──▶ 라벨 샘플링(5개) ──▶ 증강 ──┬──▶ 이미지: 512×512, min-max
                                              └──▶ 마스크 → bbox → 1024×1024 좌표계
```

1. NPZ에서 `imgs`, `gts` 로드
2. 라벨을 랜덤으로 샘플링 (`mask_num=5`)
3. Augmentation (HorizontalFlip, VerticalFlip, ShiftScaleRotate)
4. 마스크에서 bounding box 추출 (±5px 랜덤 이동)
5. 이미지: 긴 변 기준 512×512로 resize, min-max 스케일
6. Box: 1024×1024 prompt encoder 좌표계로 변환

### 관련 코드

| 파일 | 역할 |
|------|------|
| `src/data/components/medsam_dataset.py` | Dataset 클래스 |
| `src/data/medsam_datamodule.py` | LightningDataModule |
| `src/utils/transforms.py` | ResizeLongestSide, get_bbox, transform_gt |
| `configs/data/distill_medsam.yaml` | Distillation 데이터 설정 |
| `configs/data/finetune_medsam.yaml` | Fine-tuning 데이터 설정 |
