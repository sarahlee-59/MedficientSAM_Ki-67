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
| CT | AMOS22 | 240 | 스프레드시트 경유 |
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

| 방식 | 대상 | 요약 |
|------|------|------|
| A | Endoscopy, Fundus, Mammography, Microscopy, MR, OCT, PET, US, XRay | 챌린지에서 NPZ로 배포 → 그냥 다운로드 |
| B | CT 5종 | 원본 NIfTI를 변환 스크립트로 직접 NPZ 생성 |
| C | Pathology, Dermoscopy | 사수님이 변환 후 내부 서버에서 다운로드 |

---

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

CT 원본 데이터는 NIfTI(`.nii.gz`) 형식이라 `pre_CT_MR.py`로 직접 NPZ로 변환했습니다.
이 레포에 `pre_CT_MR.py`가 포함되어 있으며, 원본은 [bowang-lab/MedSAM LiteMedSAM 브랜치](https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM)입니다.

#### 전처리 과정 (스크립트 내부 동작)

```
① 3D 전체에서 1,000 voxel 미만 객체 제거 (너무 작아서 학습에 불필요)
② 각 2D 슬라이스에서 100 pixel 미만 객체 제거
③ 마스크가 비어있는 슬라이스 제거 → 유효 슬라이스만 추출
④ HU 윈도우 적용: Level=40, Width=400 → 유효 범위 -160~240 HU
   → 0~255 (uint8) 로 정규화
⑤ NPZ로 저장: imgs(이미지 볼륨), gts(마스크 볼륨), spacing(voxel 간격)
```

출력 파일명 형식: `CT_<데이터셋명>_<케이스ID>.npz`

#### 의존성 설치

```bash
pip install connected-components-3d SimpleITK
```

---

#### CT / AbdomenCT-1K (1,000개)

> bash 히스토리 미확인 — 파일명 패턴(`CT_AbdomenCT-1K_Case_00001.npz`)으로 아래 명령 추정

```bash
python3 pre_CT_MR.py \
  -modality CT -anatomy AbdomenCT-1K \
  -img_name_suffix .nii.gz -gt_name_suffix .nii.gz \
  -img_path <AbdomenCT-1K 이미지 폴더> \
  -gt_path  <AbdomenCT-1K 라벨 폴더> \
  -output_path /mnt/Disk1/sylee/npz_output \
  -num_workers 8
```

#### CT / AMOS22 (240개)

> bash 히스토리 미확인 — 파일명 패턴(`CT_AMOS22_amos_0001.npz`)으로 아래 명령 추정

```bash
python3 pre_CT_MR.py \
  -modality CT -anatomy AMOS22 \
  -img_name_suffix .nii.gz -gt_name_suffix .nii.gz \
  -img_path <AMOS22 이미지 폴더> \
  -gt_path  <AMOS22 라벨 폴더> \
  -output_path /mnt/Disk1/sylee/npz_output \
  -num_workers 8
```

#### CT / COVID-19-20 (199개)

> bash 히스토리 확인됨

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

# 2단계: NPZ 변환
cd /mnt/Disk1/sylee
python3 pre_CT_MR.py \
  -modality CT -anatomy COVID-19-20 \
  -img_name_suffix .nii.gz -gt_name_suffix .nii.gz \
  -img_path /mnt/Disk1/sylee/COVID-19-20_organized/images \
  -gt_path  /mnt/Disk1/sylee/COVID-19-20_organized/labels \
  -output_path /mnt/Disk1/sylee/npz_output \
  -num_workers 8
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

# 2단계: NPZ 변환
cd /mnt/Disk1/sylee
python3 pre_CT_MR.py \
  -modality CT -anatomy KiTS23 \
  -img_name_suffix .nii.gz -gt_name_suffix .nii.gz \
  -img_path /mnt/Disk1/sylee/kits23_organized/images \
  -gt_path  /mnt/Disk1/sylee/kits23_organized/labels \
  -output_path /mnt/Disk1/sylee/npz_output \
  -num_workers 8
```

#### CT / TotalSegmentator (1,174개)

> bash 히스토리 미확인 — 파일명 패턴(`CT_TotalSegmentator_s0000.npz`)으로 아래 명령 추정

```bash
python3 pre_CT_MR.py \
  -modality CT -anatomy TotalSegmentator \
  -img_name_suffix .nii.gz -gt_name_suffix .nii.gz \
  -img_path <TotalSegmentator 이미지 폴더> \
  -gt_path  <TotalSegmentator 라벨 폴더> \
  -output_path /mnt/Disk1/sylee/npz_output \
  -num_workers 8
```

변환된 NPZ는 `npz_output/MedSAM_train/CT_<데이터셋명>/` 에 저장되며, 이후 `train_npz/CT/<데이터셋명>/` 으로 이동했습니다.

---

### C. 사수님이 변환 후 제공한 것

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
| 변환 주체 | 사수님 (`/mnt/Disk1/DP_IHC/Ki67_pytorchlightning/`) |

#### Dermoscopy / ISIC-2017

| 항목 | 내용 |
|------|------|
| 원본 | ISIC-2017 원본 이미지 (Google Sheet 경유 다운로드) |
| 변환 방법 | stride 128, crop 256×256 sliding window tiling → NPZ |
| 변환 주체 | 사수님 (별도 스크립트, 이 레포 미포함) |

#### Pathology 보강 데이터 (PanNuke, MoNuSeg 2018)

| 데이터셋 | 변환 방법 |
|----------|----------|
| PanNuke | 원본 이미지 → 256×256 resize → NPZ |
| MoNuSeg 2018 | 1000×1000 원본 → 2×2 crop → 256×256 resize → NPZ |

두 데이터셋 모두 사수님이 변환하여 제공 (별도 스크립트, 이 레포 미포함)


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
