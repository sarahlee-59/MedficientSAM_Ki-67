# 데이터셋 설명

## 개요

| 역할 | 데이터셋 | 용도 |
|------|---------|------|
| Distillation 학습 | MedSAM 2024 Challenge 공개 데이터 | Student 인코더가 Teacher(MedSAM) 임베딩을 모방하도록 학습 |
| Fine-tuning / 평가 | Ki-67 pathology 슬라이드 (내부) + MoNuSeg / PanNuke (공개) | Ki-67 H&E 이미지에서 세포 instance segmentation |

---

## 1. MedSAM 2024 Challenge 데이터 (Distillation 학습용)

### 경로

```
/mnt/Disk1/sylee/train_npz/
```

### 다운로드 출처

논문(MedficientSAM) 기준 11개 모달리티를 구성했으나, **공식 Google Drive에 CT·Dermoscopy가 누락**되어 챌린지 공식 홈페이지([Codabench](https://www.codabench.org/competitions/1847/)) 추가 Google Sheet에서 별도 확보했습니다. **PanNuke·MoNuSeg 2018은 pathology domain 보강 목적으로 Google Sheet를 경유하여 별도 추가한 공개 데이터셋입니다.**

| 경로 | 포함 데이터 | 비고 |
|------|------------|------|
| [공식 Google Drive](https://drive.google.com/drive/folders/1khEIdkO0MC_gG5EkQ7COdDS1jge5_XQs) | Endoscopy, Fundus, Mammography, Microscopy, MR, OCT, PET, US, XRay | 챌린지 주최 측이 NPZ로 변환하여 배포 |
| [공식 홈페이지 추가 Google Sheet](https://docs.google.com/spreadsheets/d/1QxjFs41eU6JG5KNhP576fc8MotrJ58KCrqH83HG-__E/edit?gid=2057737934#gid=2057737934) | CT 전체, Dermoscopy (ISIC-2017) | 원본 데이터 링크 제공 — CT는 NPZ로 다운로드, Dermoscopy는 원본 이미지를 내부 전처리하여 NPZ 생성 |
| [공식 홈페이지 추가 Google Sheet](https://docs.google.com/spreadsheets/d/1QxjFs41eU6JG5KNhP576fc8MotrJ58KCrqH83HG-__E/edit?gid=2057737934#gid=2057737934) | PanNuke, MoNuSeg 2018 | pathology domain 보강 목적 추가 — 원본 링크 경유, 내부 전처리하여 NPZ 생성 |

### 구성

| Modality | Sub-dataset | 파일 수 | 비고 |
|----------|-------------|---------|------|
| CT | AbdomenCT-1K | 1,000 | 스프레드시트 경유 |
| CT | AMOS22 | 200 | 스프레드시트 경유, Tr 분할 CT only (case_id ≤ 500) |
| CT | COVID-19-20 | 199 | 스프레드시트 경유 |
| CT | KiTS23 | 489 | 스프레드시트 경유 |
| CT | TotalSegmentator | 1,174 | 스프레드시트 경유, 전신 117개 부위 커버 |
| Dermoscopy | ISIC-2017 | 2,000 | 스프레드시트 경유, stride 128 tiling → NPZ |
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
| Pathology | PanNuke | 2,538 | 스프레드시트 경유, 보강 추가 |
| Pathology | MoNuSeg2018 | 148 | 스프레드시트 경유, 보강 추가 |
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

## 2. Ki-67 Pathology 데이터셋 및 Fine-tuning

### 경로

```
/mnt/Disk1/sylee/train_npz/Pathology_new/gts_npz_s128/   # 학습·평가용 NPZ 패치 (Ki-67)
```

### Ki-67 슬라이드 데이터 규모

부산 백병원 Ki-67 IHC 슬라이드 89장을 256×256 패치로 분할한 데이터셋.

| 항목 | 값 |
|------|----|
| 원본 슬라이드 수 | 89장 (부산 백병원) |
| 패치 추출 방식 | crop 256×256, stride 128 (50% overlap) |
| 총 패치 수 | 16,376 |
| 파일명 형식 | `2D_<슬라이드ID>_r<행>_c<열>.npz` |

### 공개 Pathology 보조 데이터

| 데이터셋 | 설명 | 패치 수 | 전처리 |
|----------|------|---------|--------|
| PanNuke | Pan-Cancer 세포 핵 segmentation | 2,538 | Google Sheet 경유 → tiling 없이 256×256 resize → NPZ |
| MoNuSeg 2018 | H&E 조직 multi-organ 세포 segmentation | 148 | Google Sheet 경유 → 1000×1000 → 2×2 crop → 256×256 resize → NPZ |

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

## 3. NPZ 전처리 명령어

`train_npz/` 데이터는 출처에 따라 세 가지 방식으로 준비했습니다.

### A. 다운로드만 한 것 (변환 없음)

챌린지 주최 측이 이미 NPZ로 만들어서 배포한 데이터입니다. 그대로 다운로드해서 썼습니다.

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

다운로드 출처: [공식 Google Drive](https://drive.google.com/drive/folders/1khEIdkO0MC_gG5EkQ7COdDS1jge5_XQs)

---

### B. 직접 변환한 것 — CT 5종

CT 원본 데이터는 NIfTI(`.nii.gz`) 형식이라 `CT/pre_CT_MR.py`로 NPZ·NPY로 변환했습니다.
원본은 [bowang-lab/MedSAM LiteMedSAM 브랜치](https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM)입니다.

#### 의존성 설치

```bash
pip install connected-components-3d SimpleITK scikit-image tqdm
```

#### 스크립트 사용법

CLI 인자가 없으므로 파일 상단 변수를 직접 수정한 뒤 실행합니다.

```python
# pre_CT_MR.py 상단 설정 변수
modality = "CT"
anatomy  = "AbdomenCT-1K"       # 데이터셋명 — 출력 prefix에 사용됨
img_name_suffix = "_0000.nii.gz" # 이미지 파일 suffix (데이터셋마다 다름)
gt_name_suffix  = ".nii.gz"      # 라벨 파일 suffix

nii_path = "data/FLARE22Train/images"  # 이미지 폴더 경로
gt_path  = "data/FLARE22Train/labels"  # 라벨 폴더 경로
# npy_path는 자동 설정: "data/npy/CT_<anatomy>"

# CT window 설정
WINDOW_LEVEL = 40   # 복부 기본값
WINDOW_WIDTH  = 400

# 제외할 label id (기본: 12=십이지장, bounding box로 지정하기 어려워 제외)
remove_label_ids = [12]
```

```bash
python3 pre_CT_MR.py
```

#### 출력 파일 구조

```
data/npy/CT_<anatomy>/
├── CT_<anatomy>_<케이스ID>.npz            # 볼륨 단위 (imgs, gts, spacing)
├── CT_<anatomy>_<케이스ID>_img.nii.gz     # 산티체크용 — 확인 후 삭제 가능
├── CT_<anatomy>_<케이스ID>_gt.nii.gz      # 산티체크용 — 확인 후 삭제 가능
├── imgs/
│   └── CT_<anatomy>_<케이스ID>-000.npy   # 슬라이스별 (H, W, 3) float64 [0, 1]
└── gts/
    └── CT_<anatomy>_<케이스ID>-000.npy   # 슬라이스별 (H, W) uint8
```

NPZ 키:

| 키 | shape | dtype | 설명 |
|----|-------|-------|------|
| `imgs` | (D, H, W) | uint8 | CT windowing + [0, 255] 정규화, 비제로 슬라이스만 |
| `gts` | (D, H, W) | uint8 | instance label, 비제로 슬라이스만 |
| `spacing` | tuple | float | voxel spacing (sitk 기준) |

변환된 결과는 `data/npy/CT_<anatomy>/` 에 저장되며, 이후 `train_npz/CT/<데이터셋명>/` 으로 이동했습니다.

---

#### CT / AbdomenCT-1K (1,000개)

원본이 `AbdomenCT-1K-ImagePart1~3/` 세 파트로 분할되어 있고, 각 파트 내부에 중첩 디렉토리 구조(`PartN/PartN/*.nii.gz`)를 가짐. GT 마스크는 별도 `Mask/` 폴더에 통합 위치.

```python
anatomy         = "AbdomenCT-1K"
img_name_suffix = "_0000.nii.gz"
gt_name_suffix  = ".nii.gz"
nii_path = "<AbdomenCT-1K 이미지 폴더>"  # Part1~3 통합 후 경로
gt_path  = "<AbdomenCT-1K 라벨 폴더>"   # Mask/ 폴더
# remove_label_ids = [12]  # duodenum 제거 (GT 품질 낮음)
```

#### CT / AMOS22 (200개)

CT(case_id ≤ 500)와 MRI(case_id > 500)가 혼합된 데이터셋. **Tr 분할 CT만 사용** (Va 분할은 GT 비공개).
- CT: case_id 1~410 (결번 있음) → 200개
- MRI: case_id 507~600 (결번 있음, 501~506 없음) → 40개

> **이력**: 구버전 방식(상단 변수 직접 수정)으로 전처리 시 MRI 필터링이 누락되어 `train_npz/CT/AMOS22/`에 CT 200개 + MRI 40개 = 240개가 생성됨. MRI 케이스(`amos_0507`~`amos_0600`, 40개) 수동 삭제 완료 → 현재 200개.

```python
anatomy         = "AMOS22"
img_name_suffix = ".nii.gz"
gt_name_suffix  = ".nii.gz"
nii_path = "<AMOS22 imagesTr 폴더>"  # CT only (case_id <= 500)
gt_path  = "<AMOS22 labelsTr 폴더>"
```

#### CT / COVID-19-20 (199개)

원본이 `파일명_ct.nii.gz` / `파일명_seg.nii.gz` 형태로 섞여 있어서 먼저 정리했습니다.

```bash
# 1단계: 이미지·라벨을 별도 폴더로 분리
mkdir -p /mnt/Disk1/sylee/COVID-19-20_organized/{images,labels}

cd /mnt/Disk1/sylee/COVID-19-20_v2/Train
for f in *_ct.nii.gz;  do ln -sf "$(realpath $f)" "/mnt/Disk1/sylee/COVID-19-20_organized/images/${f%_ct.nii.gz}.nii.gz";  done
for f in *_seg.nii.gz; do ln -sf "$(realpath $f)" "/mnt/Disk1/sylee/COVID-19-20_organized/labels/${f%_seg.nii.gz}.nii.gz"; done

cd /mnt/Disk1/sylee/COVID-19-20_v2/Validation
for f in *_ct.nii.gz;  do ln -sf "$(realpath $f)" "/mnt/Disk1/sylee/COVID-19-20_organized/images/${f%_ct.nii.gz}.nii.gz";  done
for f in *_seg.nii.gz; do ln -sf "$(realpath $f)" "/mnt/Disk1/sylee/COVID-19-20_organized/labels/${f%_seg.nii.gz}.nii.gz"; done
```

```python
# 2단계: 변수 설정 후 실행
anatomy         = "COVID-19-20"
img_name_suffix = ".nii.gz"
gt_name_suffix  = ".nii.gz"
nii_path = "/mnt/Disk1/sylee/COVID-19-20_organized/images"
gt_path  = "/mnt/Disk1/sylee/COVID-19-20_organized/labels"
# CT window: WINDOW_LEVEL = -500, WINDOW_WIDTH = 1500  (lung window)
# 폐 조직 HU 범위(-1000~-500)를 포함하도록 넓은 폭 설정
```

#### CT / KiTS23 (489개)

> bash 히스토리 확인됨

원본이 `case_00001/imaging.nii.gz` 형태의 케이스별 폴더라 먼저 정리했습니다.

```bash
# 1단계: case 폴더 → images/labels 분리
mkdir -p /mnt/Disk1/sylee/kits23_organized/{images,labels}
for case_dir in /mnt/Disk1/sylee/kits23/dataset/case_*/; do
  n=$(basename "$case_dir")
  [ -f "${case_dir}imaging.nii.gz" ]      && ln -sf "$(realpath ${case_dir}imaging.nii.gz)"      "kits23_organized/images/${n}.nii.gz"
  [ -f "${case_dir}segmentation.nii.gz" ] && ln -sf "$(realpath ${case_dir}segmentation.nii.gz)" "kits23_organized/labels/${n}.nii.gz"
done
```

```python
# 2단계: pre_CT_MR.py 상단 변수 수정
anatomy         = "KiTS23"
img_name_suffix = ".nii.gz"
gt_name_suffix  = ".nii.gz"
nii_path = "/mnt/Disk1/sylee/kits23_organized/images"
gt_path  = "/mnt/Disk1/sylee/kits23_organized/labels"
# CT window: WINDOW_LEVEL = 100, WINDOW_WIDTH = 400  (kidney window)
# 신장 실질 HU 범위(20~80)와 종양 조영 증강 최적화
```

```bash
python3 pre_CT_MR.py
```

#### CT / TotalSegmentator (1,174개)

원본이 장기별 개별 binary 마스크 파일(`segmentations/spleen.nii.gz` 등 17개)로 분리 저장되어 있어, 전처리 전에 **단일 multi-label 파일로 병합**이 필요하다.

| label ID | 장기 | label ID | 장기 |
|----------|------|----------|------|
| 1 | spleen | 2 | kidney_right |
| 3 | kidney_left | 4 | gallbladder |
| 5 | liver | 6 | stomach |
| 7 | aorta | 8 | inferior_vena_cava |
| 9 | pancreas | 10 | adrenal_gland_right |
| 11 | adrenal_gland_left | 12 | duodenum |
| 13 | colon | 14 | small_bowel |
| 15 | urinary_bladder | 16 | portal_vein_and_splenic_vein |
| 17 | esophagus | | |

전체 1,228 케이스 중 54건은 GT가 전부 0(유효 장기 없음)으로 NPZ 생성 시 자동 제외 → 유효 1,174개.

```python
anatomy         = "TotalSegmentator"
img_name_suffix = ".nii.gz"
gt_name_suffix  = ".nii.gz"
nii_path = "<TotalSegmentator 이미지 폴더>"  # 병합 후 labels/ 경로
gt_path  = "<TotalSegmentator 라벨 폴더>"
```

#### CT 전처리 재현 파이프라인 (setup_datasets.py + run_preprocessing.sh)

위 각 데이터셋 섹션의 방법은 개별 실행 기록 기준이다. 전체를 일괄 재현하려면 아래 파이프라인을 사용한다. 관련 스크립트는 모두 `CT/` 폴더에 위치한다.

```
raw_ver/          원본 NIfTI (데이터셋마다 폴더 구조 상이)
     │
     │  setup_datasets.py
     │  ① 데이터셋별 symlink 생성 → refine_ver/ (통일된 images/labels 구조)
     │  ② TotalSegmentator 장기별 binary 마스크 → single multi-label 병합
     │  ③ 누락 NPZ 증분 생성
     ▼
refine_ver/       images/ + labels/ 통일 구조
     │
     │  run_preprocessing.sh
     │  └─ pre_CT_MR.py (데이터셋별 CT window 파라미터로 순차 호출)
     ▼
train_npz/CT/     학습용 NPZ
```

```bash
# 1단계: 원본 데이터 경로
# RAW = /mnt/Disk1/sylee/CT/raw_ver/
# 구조: AbdomenCT-1K-ImagePart1~3/, Mask/, amos22/, COVID-19-20_v2/,
#       kits23/, Totalsegmentator_dataset_v201/

cd /mnt/Disk1/sylee/CT
python3 setup_datasets.py   # symlink 정리 + TotalSegmentator 병합 + 증분 NPZ
bash run_preprocessing.sh   # 전체 5종 NPZ 재생성 (필요 시)
```

| 데이터셋 | CT Window | 비고 |
|---------|-----------|------|
| AbdomenCT-1K | WL=40, WW=400 (soft tissue) | duodenum(12) 제거 |
| AMOS22 | WL=40, WW=400 (soft tissue) | Tr CT only (case_id ≤ 500), 200개 |
| COVID-19 | WL=−500, WW=1500 (lung) | 폐 감염 병변 강조 |
| KiTS23 | WL=100, WW=400 (kidney) | 신장 실질/종양 최적화 |
| TotalSegmentator | WL=40, WW=400 (soft tissue) | 17개 장기 multi-label |

---

### C. 내부 전처리 후 다운로드한 것

#### Pathology / Ki-67 (gts_npz_s128)

```bash
wget -O gts_npz_s128.zip "http://10.0.30.191:5000/sharing/4kCrJ5jh6"
unzip gts_npz_s128.zip
# → train_npz/Pathology_new/gts_npz_s128/ 에 배치
```

| 항목 | 내용 |
|------|------|
| 원본 | 부산 백병원 Ki-67 IHC 슬라이드 89장 |
| 변환 방법 | 256×256 패치, stride 128 sliding window |

#### Dermoscopy / ISIC-2017

| 항목 | 내용 |
|------|------|
| 원본 | ISIC-2017 원본 이미지 (Google Sheet 경유 다운로드) |
| 변환 방법 | stride 128, crop 256×256 sliding window tiling → NPZ |

#### Pathology 보강 데이터 (PanNuke, MoNuSeg 2018)

| 데이터셋 | 변환 방법 |
|----------|----------|
| PanNuke | 원본 이미지 → 256×256 resize → NPZ |
| MoNuSeg 2018 | 1000×1000 원본 → 2×2 crop → 256×256 resize → NPZ |

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
