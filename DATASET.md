# 데이터셋 설명

## 한눈에 보기

이 프로젝트는 모델을 두 단계에 걸쳐 학습시킵니다.

| 단계 | 무엇을 하나 | 어떤 데이터를 쓰나 |
|------|------------|-------------------|
| **1. Distillation** | 작은 Student 모델이 큰 Teacher(MedSAM)의 임베딩을 따라 하도록 학습 | MedSAM 2024 Challenge 공개 데이터 (11개 모달리티, 약 7만 장) |
| **2. Fine-tuning / 평가** | Ki-67 H&E 이미지에서 세포 하나하나를 구분(instance segmentation) | Ki-67 슬라이드(내부) + MoNuSeg / PanNuke(공개) |

쉽게 말해, **1단계에서 의료영상 전반을 폭넓게 보는 눈을 키우고, 2단계에서 Ki-67 세포 분할에 특화**시키는 구조입니다.

> **용어 한 줄 정리**
> - **모달리티(modality):** CT, MRI, X-ray처럼 영상을 찍는 방식의 종류.
> - **임베딩(embedding):** 모델이 이미지를 보고 뽑아낸 "특징 요약" 숫자 묶음.
> - **instance segmentation:** 같은 종류(세포)라도 개체를 하나씩 따로 떼어 구분하는 것.

---

## 1. MedSAM 2024 Challenge 데이터 (Distillation 학습용)

### 저장 위치

```
/mnt/Disk1/sylee/train_npz/
```

### 데이터를 어디서 모았나

기준은 MedficientSAM 논문의 **11개 모달리티**입니다. 다만 데이터가 한곳에 다 있지는 않아서, **두 군데**에서 나눠 받았습니다.

- **첫째, 공식 Google Drive**: 대부분의 모달리티를 주최 측이 미리 NPZ로 변환해 배포 → 그대로 다운로드
- **둘째, 공식 홈페이지([Codabench](https://www.codabench.org/competitions/1847/))의 추가 Google Sheet**: Drive에 **빠져 있던 CT·Dermoscopy**, 그리고 **pathology를 보강하려고 추가한 PanNuke·MoNuSeg 2018**을 원본 링크로 받아 직접 전처리.

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
| **합계** | | **70,864** | |

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

이 프로젝트의 **진짜 목표**인 Ki-67 데이터입니다.

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

> **왜 겹쳐서 자르나?** stride(이동 간격)를 패치 크기(256)의 절반인 128로 두면 조각들이 50%씩 겹칩니다. 이렇게 하면 세포가 패치 경계에 걸려 잘려도, 옆 패치에서는 온전히 보이게 됩니다.

### Ki-67을 도와주는 공개 데이터

Ki-67만으로는 부족해서, **세포 핵 분할용 공개 데이터**를 함께 씁니다.

| 데이터셋 | 무엇인가 | 패치 수 | 전처리 |
|----------|---------|---------|--------|
| PanNuke | 여러 암종의 세포 핵 분할 데이터 | 2,538 | 256×256으로 resize → NPZ |
| MoNuSeg 2018 | H&E 조직, 여러 장기의 세포 분할 데이터 | 148 | 1000×1000 → 2×2로 자름 → 256×256 resize → NPZ |

### Fine-tuning 데이터 묶음

성능을 비교하기 위해 **두 가지 실험**을 돌립니다.

- **첫째, 조합 학습:** Ki-67 + PanNuke + MoNuSeg를 모두 합쳐 학습 → 총 19,062 패치
- **둘째, Ki-67 단독:** 백병원 Ki-67만으로 학습 → 총 16,376 패치

| 실험 | 데이터 구성 | 총 패치 수 |
|------|------------|-----------|
| 조합 학습 | Ki-67 + PanNuke + MoNuSeg | 19,062 |
| Ki-67 단독 | 백병원 Ki-67만 | 16,376 |

**Ki-67 단독 실험은 데이터를 90 / 5 / 5 비율로 나눕니다.** (학습용 / 검증용 / 시험용)

| 분할 | 패치 수 |
|------|---------|
| Train (90%) | 14,738 |
| Val (5%) | 819 |
| Test (5%) | 819 |
| **합계** | **16,376** |

증강(augmentation)까지 포함하면 실제로 보는 학습량은 **14,738 × 16 epochs ≈ 약 235K 패치**가 됩니다.

### 데이터 증강 파이프라인

병리 이미지는 염색 상태와 스캐너 종류에 따라 색과 품질이 들쭉날쭉합니다. 그래서 **일부러 다양한 변형을 섞어** 넣어, 어떤 조건에서도 잘 동작하는 튼튼한 모델을 만듭니다.

| 카테고리 | 기법 | 목적 |
|----------|------|------|
| Geometric | HorizontalFlip, VerticalFlip, RandomRotate90 | 방향·위치 변화에 강하게 |
| IHC 염색 | **HEDJitter** | H&E/IHC 염색 색 변이 흉내 |
| 스캐너 / 품질 | GaussianBlur·MotionBlur·Defocus 중 1개, ImageCompression(60~100), Downscale | 흐림·압축·저해상도 상황 대응 |
| Photometric | RandomBrightnessContrast | 밝기·대비 변화 대응 |

---

## 3. NPZ는 어떻게 준비했나

`train_npz/` 데이터는 출처에 따라 **세 가지 방식**으로 준비했습니다.

- **첫째 방식 (A) — 다운로드만:** 주최 측이 이미 NPZ로 만들어 둔 것 → 받기만 함. (9개 모달리티)
- **둘째 방식 (B) — 직접 변환:** NIfTI 원본을 NPZ로 바꿈. (CT 5종)
- **셋째 방식 (C) — 내부 전처리:** 원본 이미지를 잘라/리사이즈해 NPZ로 만듦. (Ki-67, Dermoscopy, PanNuke, MoNuSeg)

| 방식 | 대상 | 한 일 |
|------|------|-------|
| **A. 다운로드만** | 9개 모달리티 | 주최 측이 NPZ로 배포 → 그대로 사용 |
| **B. 직접 변환** | CT 5종 | NIfTI 원본을 NPZ로 변환 |
| **C. 내부 전처리** | Ki-67, Dermoscopy, PanNuke, MoNuSeg | 원본 이미지를 잘라/리사이즈해 NPZ 생성 |

---

### A. 그냥 다운로드만 (변환 없음)

주최 측이 이미 NPZ로 만들어 둔 데이터라 받기만 하면 됩니다.

```
[원본] 주최 측 NPZ  ──(다운로드)──▶  [결과] 그대로 train_npz/ 에 배치
```

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

출처: [공식 Google Drive](https://drive.google.com/drive/folders/1khEIdkO0MC_gG5EkQ7COdDS1jge5_XQs)

---

### B. 직접 변환 — CT 5종

CT 원본은 NIfTI(`.nii.gz`) 형식이라 NPZ로 바꿔야 합니다. `CT/pre_CT_MR.py`로 변환했고, 스크립트 원본은 [bowang-lab/MedSAM (LiteMedSAM 브랜치)](https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM)입니다.

#### 변환 한눈에 보기

변환은 다음 **네 단계**를 거칩니다.

```
[원본]                          [변환 과정]                        [결과]
.nii.gz 3D 볼륨        ──▶   ① CT windowing (조직별 HU 강조)   ──▶   .npz 1개 (볼륨 단위)
(HU 값, 장기별 라벨)         ② [0,255] uint8 정규화                  imgs (D,H,W) uint8
                            ③ 빈 슬라이스 제거                       gts  (D,H,W) uint8
                            ④ instance label 정리                   spacing
```

- **첫째,** CT windowing — 보고 싶은 조직의 HU(밝기) 범위만 강조합니다.
- **둘째,** [0, 255] 범위의 uint8로 정규화합니다.
- **셋째,** 정답이 하나도 없는 빈 슬라이스를 제거합니다.
- **넷째,** instance label(개체별 번호)을 정리합니다.

| 구분 | 원본 (Before) | 결과 (After) |
|------|--------------|--------------|
| 파일 형식 | NIfTI `.nii.gz` | NumPy `.npz` |
| 데이터 차원 | 3D 볼륨 (D, H, W) | 동일하나 빈 슬라이스 제거됨 |
| 픽셀 값 | 원본 HU 값 | CT window 적용 후 [0, 255] uint8 |
| 라벨 | 장기별 정수 라벨 | instance label로 정리, 일부 label 제외 |
| 폴더 구조 | 데이터셋마다 제각각 | `train_npz/CT/<데이터셋명>/` 으로 통일 |

> **용어 한 줄 정리**
> - **HU(Hounsfield Unit):** CT에서 조직의 밀도를 나타내는 값. 공기는 낮고 뼈는 높습니다.
> - **CT window:** 그중 보고 싶은 밝기 구간만 골라 또렷하게 보는 설정(WL=중심, WW=폭).

#### 의존성 설치

```bash
pip install connected-components-3d SimpleITK scikit-image tqdm
```

#### 사용법

이 스크립트는 CLI 인자가 없습니다. 파일 상단 변수를 직접 고친 뒤 실행하세요.

```python
# pre_CT_MR.py 상단 설정
modality = "CT"
anatomy  = "AbdomenCT-1K"        # 데이터셋명 — 출력 파일 prefix에 쓰임
img_name_suffix = "_0000.nii.gz" # 이미지 파일 suffix (데이터셋마다 다름)
gt_name_suffix  = ".nii.gz"      # 라벨 파일 suffix

nii_path = "data/FLARE22Train/images"  # 이미지 폴더
gt_path  = "data/FLARE22Train/labels"  # 라벨 폴더
# npy_path는 자동: "data/npy/CT_<anatomy>"

# CT window (보고 싶은 조직에 따라 조절)
WINDOW_LEVEL = 40   # 복부 기본값
WINDOW_WIDTH  = 400

# 제외할 label id (12=십이지장: bounding box 잡기 어려워 제외)
remove_label_ids = [12]
```

```bash
python3 pre_CT_MR.py
```

#### 출력 구조

```
data/npy/CT_<anatomy>/
├── CT_<anatomy>_<케이스ID>.npz            # 볼륨 단위 (imgs, gts, spacing)
├── CT_<anatomy>_<케이스ID>_img.nii.gz     # 확인용 — 점검 후 삭제 가능
├── CT_<anatomy>_<케이스ID>_gt.nii.gz      # 확인용 — 점검 후 삭제 가능
├── imgs/
│   └── ..._<케이스ID>-000.npy   # 슬라이스별 (H, W, 3) float64 [0, 1]
└── gts/
    └── ..._<케이스ID>-000.npy   # 슬라이스별 (H, W) uint8
```

NPZ 키 설명:

| 키 | shape | dtype | 설명 |
|----|-------|-------|------|
| `imgs` | (D, H, W) | uint8 | CT windowing 후 [0,255] 정규화, 비어 있지 않은 슬라이스만 |
| `gts` | (D, H, W) | uint8 | instance label, 비어 있지 않은 슬라이스만 |
| `spacing` | tuple | float | voxel spacing (SimpleITK 기준) |

결과는 `data/npy/CT_<anatomy>/`에 생기고, 이후 `train_npz/CT/<데이터셋명>/`으로 옮겼습니다.

CT는 데이터셋마다 폴더 구조가 달라서, 변환 전에 정리가 필요했습니다. 아래는 데이터셋별로 **원본이 어떻게 생겼고 → 어떻게 정리했는지**를 정리한 부분입니다.

---

#### CT / AbdomenCT-1K (1,000개)

**원본 구조** — `Part1~3` 세 덩어리로 나뉘고, 각 Part 안에 또 폴더가 중첩(`PartN/PartN/*.nii.gz`)됩니다. GT 마스크는 별도 `Mask/` 폴더에 모여 있습니다.

**정리** — Part1~3 이미지를 한 폴더로 통합하고, GT는 `Mask/` 폴더를 그대로 지정.

```python
anatomy         = "AbdomenCT-1K"
img_name_suffix = "_0000.nii.gz"
gt_name_suffix  = ".nii.gz"
nii_path = "<AbdomenCT-1K 이미지 폴더>"  # Part1~3 통합 후 경로
gt_path  = "<AbdomenCT-1K 라벨 폴더>"   # Mask/ 폴더
# remove_label_ids = [12]  # 십이지장 제거 (GT 품질 낮음)
```

#### CT / AMOS22 (200개)

**원본 구조** — CT와 MRI가 섞인 데이터셋. **Tr 분할의 CT만** 사용 (Va 분할은 GT 비공개).
- CT: case_id 1~410 (결번 있음) → 200개
- MRI: case_id 507~600 → 40개 (사용 안 함)

> **이력 메모**: 구버전 방식으로 전처리할 때 MRI 필터링이 빠져서 CT 200개 + MRI 40개 = **240개**가 만들어졌습니다. 이후 MRI 케이스(`amos_0507`~`amos_0600`) 40개를 수동 삭제해 **현재 200개**입니다.

```python
anatomy         = "AMOS22"
img_name_suffix = ".nii.gz"
gt_name_suffix  = ".nii.gz"
nii_path = "<AMOS22 imagesTr 폴더>"  # CT only (case_id <= 500)
gt_path  = "<AMOS22 labelsTr 폴더>"
```

#### CT / COVID-19-20 (199개)

**원본 구조** — `파일명_ct.nii.gz`(이미지)와 `파일명_seg.nii.gz`(라벨)가 한 폴더에 뒤섞여 있습니다.

**정리** — 먼저 이미지와 라벨을 각각 별도 폴더로 분리한 뒤 변환.

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
# lung window: WINDOW_LEVEL = -500, WINDOW_WIDTH = 1500
# 폐 조직 HU 범위(-1000~-500)를 담도록 폭을 넓게
```

#### CT / KiTS23 (489개)

**원본 구조** — `case_00001/imaging.nii.gz`, `case_00001/segmentation.nii.gz` 처럼 케이스별 폴더 안에 이미지와 라벨이 들어 있습니다.

**정리** — 케이스 폴더를 훑어 이미지/라벨을 각각 모은 뒤 변환.

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
# 2단계: 변수 수정 후 실행
anatomy         = "KiTS23"
img_name_suffix = ".nii.gz"
gt_name_suffix  = ".nii.gz"
nii_path = "/mnt/Disk1/sylee/kits23_organized/images"
gt_path  = "/mnt/Disk1/sylee/kits23_organized/labels"
# kidney window: WINDOW_LEVEL = 100, WINDOW_WIDTH = 400
# 신장 실질(HU 20~80)과 종양 조영 증강에 맞춤
```

```bash
python3 pre_CT_MR.py
```

#### CT / TotalSegmentator (1,174개)

**원본 구조** — 장기마다 별도 binary 마스크(`segmentations/spleen.nii.gz` 등 17개)로 쪼개져 있습니다.

**정리** — 전처리 전에 17개 마스크를 **하나의 multi-label 파일로 병합**해야 합니다.

```
[원본] spleen.nii.gz, liver.nii.gz, ... (장기별 0/1 마스크 17개)
                          │  병합
                          ▼
[결과] segmentation.nii.gz (1개, 픽셀 값 = 장기 ID 1~17)
```

| ID | 장기 | ID | 장기 |
|----|------|----|------|
| 1 | spleen | 2 | kidney_right |
| 3 | kidney_left | 4 | gallbladder |
| 5 | liver | 6 | stomach |
| 7 | aorta | 8 | inferior_vena_cava |
| 9 | pancreas | 10 | adrenal_gland_right |
| 11 | adrenal_gland_left | 12 | duodenum |
| 13 | colon | 14 | small_bowel |
| 15 | urinary_bladder | 16 | portal_vein_and_splenic_vein |
| 17 | esophagus | | |

전체 1,228 케이스 중 54건은 GT가 전부 0(유효 장기 없음)이라 NPZ 생성 시 자동 제외됩니다 → **유효 1,174개**.

> **병합 우선순위** — `LABEL_MAP` 순서(spleen=1 → … → esophagus=17)로 처리하므로, 겹치는 voxel은 **나중에 칠해지는 후순위 라벨로 덮어씌워집니다.**

**NPZ는 다른 데이터셋과 경로가 다릅니다.** TotalSegmentator는 `run_preprocessing.sh` + `pre_CT_MR.py`를 거치지 않습니다. `setup_datasets.py` 안의 `fill_missing_npz` → `make_npz`가 마스크 병합 직후 곧바로 NPZ를 만들기 때문에, **이 데이터셋은 `setup_datasets.py` 한 번 실행으로 전처리가 끝납니다.** 파일명 prefix가 `CT_TotalSegmentator_`인 것도 이 때문입니다 (`run_preprocessing.sh`로 만들면 `CT_Abd_TotalSeg_`가 됨).

**raw → NPZ 전체 흐름**

```
raw_ver/Totalsegmentator_dataset_v201/sXXXX/
├── ct.nii.gz
└── segmentations/{spleen, liver, ...}.nii.gz   ← 장기별 binary 마스크 17개
       │
       │  [setup_datasets.py — 마스크 병합]
       ▼
refine_ver/TotalSegmentator/
├── images/sXXXX_0000.nii.gz   (symlink → ct.nii.gz)
└── labels/sXXXX.nii.gz        (병합된 multi-label, label 1~17)
       │
       │  [setup_datasets.py — make_npz]
       │   WL=40 / WW=400 → clip [-160, 240] HU → uint8 정규화
       ▼
train_npz/CT/TotalSegmentator/CT_TotalSegmentator_sXXXX.npz
```

**최종 NPZ 구조**

```python
npz = np.load("CT_TotalSegmentator_s0001.npz")
npz["imgs"]    # (N_valid_slices, H, W) uint8,  0~255
npz["gts"]     # (N_valid_slices, H, W) uint8,  0~17 (17개 장기 라벨)
npz["spacing"] # (x_mm, y_mm, z_mm)
```

`N_valid_slices`는 GT가 nonzero인 슬라이스 수라 원본 3D 볼륨의 z 길이보다 작습니다. 54건은 유효 슬라이스가 0이라 NPZ 자체가 생성되지 않아 1,228 → 1,174건이 됩니다.

```python
# setup_datasets.py 사용 시 (권장)
anatomy         = "TotalSegmentator"
img_name_suffix = ".nii.gz"
gt_name_suffix  = ".nii.gz"
nii_path = "<TotalSegmentator 이미지 폴더>"
gt_path  = "<TotalSegmentator 라벨 폴더>"  # 병합 후 labels/ 경로
```

---

#### CT 전체를 한 번에 재현하기 (setup_datasets.py + run_preprocessing.sh)

위 섹션들은 데이터셋을 하나씩 처리한 기록입니다. **전체를 일괄 재현**하려면 아래 파이프라인을 쓰세요. 관련 스크립트는 모두 `CT/` 폴더에 있습니다. 순서는 두 단계입니다.

- **첫째,** `setup_datasets.py` — 데이터셋별 폴더 구조를 통일하고, TotalSegmentator 마스크를 병합하고, 누락된 NPZ를 채워 만듭니다. **현재 `train_npz/CT/`의 NPZ는 모두 이 단계에서 만들어졌습니다.**
- **둘째,** `run_preprocessing.sh` — 재실행하거나 새 데이터셋을 추가할 때 쓰는 대안 경로입니다. 내부적으로 `pre_CT_MR.py`를 데이터셋별 CT window로 순차 호출합니다.

```
[원본]  raw_ver/              데이터셋마다 구조가 제각각인 NIfTI
            │
            │  ① setup_datasets.py        ◀ 현재 train_npz/CT/ 의 실제 생성 경로
            │     · 데이터셋별 symlink 생성 → images/labels 구조 통일
            │     · TotalSegmentator 장기별 마스크 → 하나로 병합
            │     · 누락된 NPZ 증분 생성 (make_npz)
            │         GT 소형 세그먼트 제거 → 유효 슬라이스 추출
            │         CT 윈도잉 → 0~255 정규화 → NPZ 저장
            ├──────────────────────────────────────────▶  [결과] train_npz/CT/  학습용 NPZ
            ▼
[중간]  refine_ver/           images/ + labels/ 로 통일된 구조
            │
            │  ② run_preprocessing.sh     ◀ 재실행 / 새 데이터셋 추가용 대안 경로
            │     └─ pre_CT_MR.py (데이터셋별 CT window로 순차 호출)
            ▼
[결과]  train_npz/CT/         학습용 NPZ 완성
```

> **핵심** — 현재 `train_npz/CT/`에 있는 NPZ는 모두 `setup_datasets.py`의 `make_npz`로 생성된 것입니다. `run_preprocessing.sh` + `pre_CT_MR.py`는 재생성·신규 데이터셋 추가용 대안 경로이며, 이 경로로 만들면 파일명 prefix가 달라집니다 (`CT_Abd_TotalSeg_` vs 현재 `CT_TotalSegmentator_`).

```bash
# 원본: /mnt/Disk1/sylee/CT/raw_ver/
#   구조: AbdomenCT-1K-ImagePart1~3/, Mask/, amos22/, COVID-19-20_v2/,
#         kits23/, Totalsegmentator_dataset_v201/

cd /mnt/Disk1/sylee/CT
python3 setup_datasets.py   # symlink 정리 + TotalSegmentator 병합 + 증분 NPZ
bash run_preprocessing.sh   # 전체 5종 NPZ 재생성 (필요 시)
```

#### make_npz() — 증분 NPZ 생성의 전처리 상세

`setup_datasets.py`의 `fill_missing_npz`가 호출하는 `make_npz`가 실제 전처리를 담당합니다. `pre_CT_MR.py`의 `preprocess`와 동일한 로직이라, `setup_datasets.py` 하나로 symlink 구성부터 NPZ 생성까지 끝납니다. 내부 처리는 **다섯 단계**입니다.

```
입력: images/{stem}{img_suffix}  +  labels/{stem}{gt_suffix}
       ↓
[1] GT 소형 세그먼트 제거 (cc3d.dust)
      · 3D: 100 voxel 미만 제거 (connectivity=26)
      · 2D: slice별 10 pixel 미만 제거 (connectivity=8)
       ↓
[2] 유효 슬라이스 추출
      · GT가 nonzero인 z 인덱스만 선택
      · 전부 0이면 스킵 (TotalSegmentator: 1,228건 중 54건 → 1,174건)
       ↓
[3] CT 윈도잉 + 정규화
      · clip: [WL − WW/2,  WL + WW/2]  (HU 단위)
      · (clip − min) / (max − min) × 255  →  uint8
       ↓
[4] 이미지 / GT를 같은 z 인덱스로 크롭
       ↓
[5] NPZ 저장
      · 파일명: {prefix}{stem}.npz
      · keys: imgs (uint8), gts (uint8), spacing (mm tuple)
```

- **첫째,** GT의 너무 작은 조각(노이즈)을 제거합니다.
- **둘째,** 정답이 있는 슬라이스만 골라냅니다.
- **셋째,** CT 윈도잉 후 0~255로 정규화합니다.
- **넷째,** 이미지와 GT를 같은 슬라이스 위치로 잘라 맞춥니다.
- **다섯째,** NPZ로 저장합니다.

이미 NPZ가 있는 케이스는 `make_npz` 진입 즉시 리턴합니다(`if os.path.exists(out_path): return`). 그래서 **중단 후 재시작해도 안전**하고, 이미 만든 파일은 건너뜁니다.

데이터셋별 CT window 정리:

| 데이터셋 | CT Window | 비고 |
|---------|-----------|------|
| AbdomenCT-1K | WL=40, WW=400 (연부조직) | 십이지장(12) 제거 |
| AMOS22 | WL=40, WW=400 (연부조직) | Tr CT만 (case_id ≤ 500), 200개 |
| COVID-19 | WL=−500, WW=1500 (폐) | 폐 감염 병변 강조 |
| KiTS23 | WL=100, WW=400 (신장) | 신장 실질/종양 최적화 |
| TotalSegmentator | WL=40, WW=400 (연부조직) | 17개 장기 multi-label |

---

### C. 내부 전처리 후 받은 것

원본 이미지를 직접 잘라서(crop/tiling) 또는 리사이즈해서 NPZ로 만든 데이터입니다. 각 데이터의 **원본 → 결과** 변화는 다음과 같습니다.

#### Pathology / Ki-67 (gts_npz_s128)

```
[원본] IHC 슬라이드 89장 (큰 전체 이미지)
              │  256×256 crop, stride 128 (50% 겹침) 으로 sliding window
              ▼
[결과] 256×256 패치 16,376장 (.npz)
```

```bash
wget -O gts_npz_s128.zip "http://10.0.30.191:5000/sharing/4kCrJ5jh6"
unzip gts_npz_s128.zip
# → train_npz/Pathology_new/gts_npz_s128/ 에 배치
```

| 항목 | 원본 (Before) | 결과 (After) |
|------|--------------|--------------|
| 데이터 | 부산 백병원 Ki-67 IHC 슬라이드 89장 | 256×256 패치 16,376장 |
| 처리 | — | 256×256 crop, stride 128 sliding window |

#### Dermoscopy / ISIC-2017

```
[원본] ISIC-2017 원본 이미지 (Google Sheet 경유 다운로드)
              │  256×256 crop, stride 128 sliding window tiling
              ▼
[결과] 256×256 패치 2,000장 (.npz)
```

| 항목 | 원본 (Before) | 결과 (After) |
|------|--------------|--------------|
| 데이터 | ISIC-2017 원본 이미지 | 256×256 패치 2,000장 |
| 처리 | — | stride 128, 256×256 sliding window tiling |

#### Pathology 보강 데이터 (PanNuke, MoNuSeg 2018)

| 데이터셋 | 원본 (Before) | 처리 | 결과 (After) |
|----------|--------------|------|--------------|
| PanNuke | 원본 세포 핵 이미지 | 256×256으로 resize | 256×256 패치 2,538장 |
| MoNuSeg 2018 | 1000×1000 H&E 이미지 | 2×2로 자른 뒤 각각 256×256 resize | 256×256 패치 148장 |

```
[MoNuSeg 변환 흐름]
1000×1000 원본 1장  ──(2×2 crop)──▶  500×500 4장  ──(resize)──▶  256×256 4장
```

---

## 4. 학습할 때 실시간으로 일어나는 전처리

여기까지는 디스크에 NPZ를 만들어 두는 과정이었습니다. 이제부터는 그 NPZ를 **모델에 넣기 직전**, 매 배치마다 자동으로 일어나는 변환입니다.

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
