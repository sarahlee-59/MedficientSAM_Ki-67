# CT 데이터 전처리 파이프라인

## 목차
1. [개요](#개요)
2. [raw_ver 폴더 구조 (원본 데이터)](#raw_ver-폴더-구조-원본-데이터)
3. [디렉토리 구조](#디렉토리-구조)
4. [전체 파이프라인 흐름](#전체-파이프라인-흐름)
5. [Step 1: 데이터셋 정리 (setup_datasets.py)](#step-1-데이터셋-정리-setup_datasetspy)
6. [Step 2: 전처리 실행 (run_preprocessing.sh)](#step-2-전처리-실행-run_preprocessingsh)
7. [Step 3: 핵심 전처리 엔진 (pre_CT_MR.py)](#step-3-핵심-전처리-엔진-pre_ct_mrpy)
8. [데이터셋별 상세 설정](#데이터셋별-상세-설정)
8. [전체 데이터 규모](#전체-데이터-규모)
9. [실행 방법](#실행-방법)
10. [출력 파일 형식](#출력-파일-형식)

---

## 개요

본 파이프라인은 **MedSAM (Medical Segment Anything Model)** 학습을 위해 5개의 공개 CT 데이터셋을 통일된 형식의 NPZ 파일로 변환한다.

- **입력**: 각 데이터셋 고유 구조의 NIfTI (`.nii.gz`) 파일
- **출력**: `{imgs, gts, spacing}`을 담은 압축 NPZ 파일
- **총 케이스 수**: 약 3,062건 (유효 GT 기준)

---

## raw_ver 폴더 구조 (원본 데이터)

실제 확인된 `/mnt/Disk1/sylee/CT/raw_ver/` (Windows: `Desktop\prepare\CT\raw_ver\`) 의 구조.
각 데이터셋은 다운로드 방식이 달라 **폴더 구조가 모두 상이**하다.

```
raw_ver/
│
├── meta.csv                                    ← TotalSegmentator 메타데이터
│                                                 (image_id, age, gender, scanner, pathology 등)
│
├── AbdomenCT-1K-ImagePart1/
│   └── AbdomenCT-1K-ImagePart1/               ← 중첩 디렉토리 (압축 해제 아티팩트)
│       ├── Case_00001_0000.nii.gz
│       ├── Case_00002_0000.nii.gz
│       └── ...                                ← CT 이미지 (suffix: _0000.nii.gz)
│
├── AbdomenCT-1K-ImagePart2/
│   └── AbdomenCT-1K-ImagePart2/               ← 동일한 중첩 구조
│       ├── Case_00XXX_0000.nii.gz
│       └── ...
│
├── AbdomenCT-1K-ImagePart3/
│   └── AbdomenCT-1K-ImagePart3/               ← 동일한 중첩 구조
│       ├── Case_00XXX_0000.nii.gz
│       └── ...
│
├── Mask/                                       ← AbdomenCT-1K GT 마스크 (Part 구분 없이 통합)
│   ├── Case_00001.nii.gz
│   ├── Case_00002.nii.gz
│   └── ...                                    ← 총 1,000개 (suffix: .nii.gz)
│
├── amos22/
│   ├── __MACOSX/                              ← macOS 압축 해제 아티팩트 (무시)
│   └── amos22/                                ← 중첩 디렉토리
│       ├── imagesTr/
│       │   ├── amos_0001.nii.gz               ← CT (case_id 1~410, 결번 있음)
│       │   ├── amos_0507.nii.gz               ← MRI (case_id 507~600, 결번 있음) ← 파이프라인에서 제외
│       │   └── ...
│       ├── labelsTr/
│       │   ├── amos_0001.nii.gz
│       │   └── ...
│       ├── imagesVa/                          ← Validation 이미지 (GT 비공개 → 제외)
│       └── labelsVa/                          ← Validation 라벨 (비공개)
│
├── COVID-19-20_v2/
│   ├── Train/                                 ← 학습에 사용
│   │   ├── volume-covid19-A-0001_ct.nii.gz    ← CT 이미지 (suffix: _ct.nii.gz)
│   │   ├── volume-covid19-A-0001_seg.nii.gz   ← GT 마스크 (suffix: _seg.nii.gz)
│   │   └── ...                                ← 총 199개 쌍
│   ├── Validation/                            ← GT 비공개 → 파이프라인에서 제외
│   ├── checksum.md5
│   ├── COVID-19-20_TrainValidation.xlsx       ← 케이스 메타데이터
│   ├── license.txt
│   └── README.md
│
├── kits23/                                    ← GitHub 클론 구조 그대로 존재
│   ├── dataset/
│   │   ├── case_00000/
│   │   │   ├── imaging.nii.gz                 ← CT 이미지
│   │   │   └── segmentation.nii.gz            ← GT 마스크 (1=kidney, 2=tumor, 3=cyst)
│   │   ├── case_00001/
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   └── ...                                ← 총 489개 케이스
│   ├── kits23/                                ← Python 패키지 소스
│   ├── kits23.egg-info/
│   ├── tests/
│   ├── .git/
│   ├── README.md
│   ├── setup.py
│   └── kits23_download.log                    ← 다운로드 이력 로그
│
└── Totalsegmentator_dataset_v201/             ← 총 1,228개 케이스 폴더
    ├── s0000/
    │   ├── ct.nii.gz                          ← CT 이미지
    │   └── segmentations/                     ← 장기별 개별 binary 마스크
    │       ├── spleen.nii.gz
    │       ├── kidney_right.nii.gz
    │       ├── kidney_left.nii.gz
    │       ├── gallbladder.nii.gz
    │       ├── liver.nii.gz
    │       ├── stomach.nii.gz
    │       ├── aorta.nii.gz
    │       ├── inferior_vena_cava.nii.gz
    │       ├── pancreas.nii.gz
    │       ├── adrenal_gland_right.nii.gz
    │       ├── adrenal_gland_left.nii.gz
    │       ├── duodenum.nii.gz
    │       ├── colon.nii.gz
    │       ├── small_bowel.nii.gz
    │       ├── urinary_bladder.nii.gz
    │       ├── portal_vein_and_splenic_vein.nii.gz
    │       └── esophagus.nii.gz               ← 총 17개 장기
    ├── s0001/
    │   ├── ct.nii.gz
    │   └── segmentations/
    │       └── ...
    └── ...                                    ← s0000 ~ s1582 (일부 번호 결번)
```

### 데이터셋별 구조 특이사항 요약

| 데이터셋 | 구조 특이사항 | 파이프라인 처리 방식 |
|---------|------------|-----------------|
| AbdomenCT-1K | 이미지가 Part1~3으로 분할, 중첩 폴더 구조 | 3개 part_dir 순회하여 symlink 통합 |
| AMOS22 | CT+MRI 혼합, `__MACOSX` 잡폴더 포함, 중첩 폴더 | case_id ≤ 500 필터링, Tr 분할만 사용 |
| COVID-19 | 이미지/GT가 같은 Train 폴더에 `_ct`/`_seg` suffix로 공존 | suffix 패턴으로 이미지/GT 분리 |
| KiTS23 | GitHub repo 구조 그대로, case별 디렉토리 | 케이스 폴더 순회 후 파일명 재명명 |
| TotalSegmentator | 장기별 binary 마스크 개별 파일로 분리 저장 | 17개 마스크를 single multi-label로 병합 |

---

## 디렉토리 구조

### 입력 (raw_ver)
```
/mnt/Disk1/sylee/CT/raw_ver/
├── AbdomenCT-1K-ImagePart1/
│   └── AbdomenCT-1K-ImagePart1/
│       └── *.nii.gz                      ← 이미지 (중첩 디렉토리 구조)
├── AbdomenCT-1K-ImagePart2/
│   └── AbdomenCT-1K-ImagePart2/
│       └── *.nii.gz
├── AbdomenCT-1K-ImagePart3/
│   └── AbdomenCT-1K-ImagePart3/
│       └── *.nii.gz
├── Mask/
│   └── *.nii.gz                          ← AbdomenCT-1K GT 마스크
├── amos22/amos22/
│   ├── imagesTr/
│   │   └── amos_XXXX.nii.gz
│   └── labelsTr/
│       └── amos_XXXX.nii.gz
├── COVID-19-20_v2/Train/
│   ├── *_ct.nii.gz                       ← 이미지
│   └── *_seg.nii.gz                      ← GT 마스크
├── kits23/dataset/
│   └── case_XXXXX/
│       ├── imaging.nii.gz
│       └── segmentation.nii.gz
└── Totalsegmentator_dataset_v201/
    └── sXXXX/
        ├── ct.nii.gz
        └── segmentations/
            ├── spleen.nii.gz
            ├── liver.nii.gz
            ├── kidney_right.nii.gz
            └── ...                       ← 장기별 binary 마스크
```

### 중간 (refine_ver) — symlink 기반 통일 구조
```
/mnt/Disk1/sylee/CT/refine_ver/
├── AbdomenCT-1K/
│   ├── images/   ← *_0000.nii.gz
│   └── labels/   ← *.nii.gz
├── AMOS22/
│   ├── images/   ← amos_XXXX.nii.gz  (CT만, case_id ≤ 500)
│   └── labels/   ← amos_XXXX.nii.gz
├── COVID-19/
│   ├── images/   ← *_ct.nii.gz
│   └── labels/   ← *_seg.nii.gz
├── KiTS23/
│   ├── images/   ← case_XXXXX_0000.nii.gz
│   └── labels/   ← case_XXXXX.nii.gz
└── TotalSegmentator/
    ├── images/   ← sXXXX_0000.nii.gz
    └── labels/   ← sXXXX.nii.gz  (병합된 multi-label)
```

### 출력 (train_npz)
```
/mnt/Disk1/sylee/train_npz/CT/
├── AbdomenCT-1K/
│   └── CT_AbdomenCT-1K_<stem>.npz
├── AMOS22/
│   └── CT_AMOS22_<stem>.npz
├── COVID-19-20/
│   └── CT_COVID-19-20_<stem>.npz
├── KiTS23/
│   └── CT_KiTS23_<stem>.npz
└── TotalSegmentator/
    └── CT_TotalSegmentator_<stem>.npz
```

---

## 전체 파이프라인 흐름

```
[raw_ver/]  원본 데이터 (데이터셋마다 다른 구조)
     │
     │  setup_datasets.py
     │  ① 데이터셋별 symlink 생성 (refine_ver/)
     │  ② TotalSegmentator 다중 organ mask → 단일 multi-label 병합
     │  ③ 누락된 NPZ 증분 생성
     ▼
[refine_ver/]  통일된 images/ + labels/ 구조
     │
     │  run_preprocessing.sh
     │  └─ pre_CT_MR.py (데이터셋별 파라미터로 반복 호출)
     │     ① GT 마스크 노이즈 제거
     │     ② CT 윈도윙 → 0~255 정규화
     │     ③ 유효 슬라이스만 추출
     │     ④ NPZ 저장
     ▼
[train_npz/CT/]  학습용 NPZ 파일
```

---

## Step 1: 데이터셋 정리 (setup_datasets.py)

### 목적
데이터셋마다 상이한 원본 디렉토리 구조를 `refine_ver/`의 통일된 구조로 재구성한다.
원본 파일을 복사하지 않고 **심볼릭 링크(symlink)** 를 사용해 디스크 공간을 절약한다.

### symlink 안전 처리

```python
def symlink(src, dst):
    if os.path.lexists(dst):
        if not os.path.exists(dst):  # 깨진 링크 감지
            os.unlink(dst)           # 깨진 링크 제거 후 재생성
        else:
            return                   # 정상 링크 → 건너뜀
    os.symlink(src, dst)
```

- `lexists`: 깨진 심볼릭 링크도 `True` 반환
- `exists`: 깨진 링크는 `False` 반환
- 위 두 함수의 차이를 이용해 깨진 링크를 안전하게 교체

### 데이터셋별 심볼릭 링크 생성

#### AbdomenCT-1K
- Part1, Part2, Part3 모두 동일한 **중첩 디렉토리 구조** (`PartN/PartN/*.nii.gz`)
- GT 마스크는 별도 `Mask/` 디렉토리에 통합 위치
- 이미지 파일명 형식: `<stem>_0000.nii.gz`

#### AMOS22
- CT(case_id ≤ 500)와 MRI(case_id > 500)가 혼합된 데이터셋
- 정규식으로 case ID 파싱: `re.search(r"amos_(\d+)", filename)`
- **Tr 분할만 사용** (Va 분할은 GT 비공개)
- CT 케이스: 200건

#### COVID-19
- Train 폴더만 사용 (Validation GT 비공개)
- 이미지: `*_ct.nii.gz` / GT: `*_seg.nii.gz` 쌍으로 구성

#### KiTS23
- 원본 구조: `case_XXXXX/imaging.nii.gz` + `case_XXXXX/segmentation.nii.gz`
- 변환 후: `case_XXXXX_0000.nii.gz` + `case_XXXXX.nii.gz`
- 폴더 기준으로 순회하여 이미지-GT 쌍 일치 확인 후 링크 생성

#### TotalSegmentator — 핵심: Multi-label 병합

원본은 장기별 개별 binary 마스크 파일로 구성되어 있어 **하나의 multi-label 파일로 병합**이 필요하다.

```python
LABEL_MAP = {
    "spleen":                        1,
    "kidney_right":                  2,
    "kidney_left":                   3,
    "gallbladder":                   4,
    "liver":                         5,
    "stomach":                       6,
    "aorta":                         7,
    "inferior_vena_cava":            8,
    "pancreas":                      9,
    "adrenal_gland_right":          10,
    "adrenal_gland_left":           11,
    "duodenum":                     12,
    "colon":                        13,
    "small_bowel":                  14,
    "urinary_bladder":              15,
    "portal_vein_and_splenic_vein": 16,
    "esophagus":                    17,
}
```

**병합 과정:**
1. CT 이미지(`ct.nii.gz`)의 shape, affine, header 참조
2. `np.zeros(shape, dtype=np.uint8)`로 빈 배열 초기화
3. 각 장기 마스크를 순서대로 읽어 해당 label ID로 덮어쓰기
4. `nib.Nifti1Image`로 저장 (원본 affine/header 유지)

> **주의:** `^s\d{4}$` 패턴에 맞는 폴더만 처리하여 `__MACOSX` 등 불필요한 폴더 제외

### Sanity Check

```python
def report(name, expected, img_dir, gt_dir):
    n_img = len(os.listdir(img_dir))
    n_gt  = len(os.listdir(gt_dir))
    status = "OK" if n_img == expected and n_gt == expected else "MISMATCH"
    print(f"  [{status}] {name}: images={n_img}, labels={n_gt} (expected={expected})")
```

### 증분 NPZ 생성 (fill_missing_npz)

```
① gt_dir에서 유효한 GT 파일 목록 수집 (이미지 파일 존재 여부 확인)
② npz_dir에서 기존 NPZ 파일의 stem 집합 추출
③ missing = GT 목록 - 기존 NPZ stem 집합
④ missing이 없으면 상태만 출력하고 종료
⑤ missing이 있으면 mp.Pool로 병렬 처리
```

이미 생성된 파일은 건너뛰므로 **중단 후 재시작이 안전**하다.

---

## Step 2: 전처리 실행 (run_preprocessing.sh)

5개 데이터셋에 대해 `pre_CT_MR.py`를 순차적으로 호출하는 셸 스크립트.

```bash
set -e          # 오류 발생 시 즉시 중단
WORKERS=8       # 멀티프로세싱 워커 수
```

### 데이터셋별 호출 파라미터

| 순서 | 데이터셋 | anatomy 태그 | img suffix | gt suffix | WL | WW | remove_label_ids |
|------|---------|-------------|-----------|-----------|----|----|-----------------|
| 1/5 | AbdomenCT-1K | Abd_AbdomenCT1K | `_0000.nii.gz` | `.nii.gz` | 40 | 400 | 12 (duodenum) |
| 2/5 | AMOS22 | Abd_AMOS22 | `.nii.gz` | `.nii.gz` | 40 | 400 | 없음 |
| 3/5 | COVID-19 | Lung_COVID19 | `_ct.nii.gz` | `_seg.nii.gz` | -500 | 1500 | 없음 |
| 4/5 | KiTS23 | Abd_KiTS23 | `_0000.nii.gz` | `.nii.gz` | 100 | 400 | 없음 |
| 5/5 | TotalSegmentator | Abd_TotalSeg | `_0000.nii.gz` | `.nii.gz` | 40 | 400 | 없음 |

### CT 윈도우 설정 근거

CT 윈도윙은 특정 조직을 강조하기 위해 HU(Hounsfield Unit) 범위를 조정한다.

| 조직/용도 | WL (중심) | WW (폭) | HU 범위 |
|----------|----------|---------|---------|
| 복부 장기 (soft tissue) | 40 | 400 | -160 ~ +240 |
| 폐 (lung) | -500 | 1500 | -1250 ~ +250 |
| 신장 (kidney) | 100 | 400 | -100 ~ +300 |

> 참고: [Radiopaedia — Windowing (CT)](https://radiopaedia.org/articles/windowing-ct)

---

## Step 3: 핵심 전처리 엔진 (pre_CT_MR.py)

### 인자 목록

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `-modality` | str | CT | CT 또는 MR |
| `-anatomy` | str | Abd | 해부학적 부위 + 데이터셋 이름 |
| `-img_name_suffix` | str | `_0000.nii.gz` | 이미지 파일 접미사 |
| `-gt_name_suffix` | str | `.nii.gz` | GT 파일 접미사 |
| `-img_path` | str | - | 이미지 디렉토리 경로 |
| `-gt_path` | str | - | GT 디렉토리 경로 |
| `-output_path` | str | `data/npz` | NPZ 출력 경로 |
| `-num_workers` | int | 4 | 병렬 처리 워커 수 |
| `-window_level` | int | 40 | CT 윈도우 레벨 (HU) |
| `-window_width` | int | 400 | CT 윈도우 폭 (HU) |
| `-remove_label_ids` | str | "12" | 제거할 라벨 ID (쉼표 구분) |
| `--save_nii` | flag | False | 검증용 NIfTI 저장 여부 |

### 출력 디렉토리 구조

```
output_path/
├── MedSAM_train/
│   └── CT_<anatomy>/     ← 학습 데이터 NPZ
└── MedSAM_test/
    └── CT_<anatomy>/     ← 테스트 데이터 NPZ (현재는 빈 상태)
```

파일명 형식: `CT_<anatomy>_<stem>.npz`

### preprocess() 함수 상세 흐름

```
입력: name (GT 파일명), npz_path (저장 경로)
```

#### 1단계: 파일 읽기

```python
def _read_as_array(path):
    try:
        s = sitk.ReadImage(path)
        return sitk.GetArrayFromImage(s), s.GetSpacing()
    except RuntimeError:
        n = nib.load(path)
        arr = np.asarray(n.dataobj)        # nibabel 기본: (X, Y, Z)
        arr = arr.transpose(2, 1, 0)       # → (Z, Y, X) — SimpleITK 표준 순서
        zooms = n.header.get_zooms()[:3]
        return arr, tuple(float(z) for z in zooms)
```

- SimpleITK로 우선 시도, 실패 시 nibabel로 폴백
- nibabel은 (X, Y, Z) 순서이므로 `transpose(2, 1, 0)`으로 (Z, Y, X)로 변환하여 축 순서 통일

#### 2단계: GT 마스크 전처리

**2-1. 라벨 제거**
```python
for remove_label_id in remove_label_ids:
    gt_data_ori[gt_data_ori == remove_label_id] = 0
```
AbdomenCT-1K에서 duodenum(12)을 제거하는 이유: 십이지장은 경계가 불분명하여 GT 품질이 낮음

**2-2. 종양 인스턴스 분리 (선택적)**
```python
if tumor_id is not None:
    tumor_bw = np.uint8(gt_data_ori == tumor_id)
    gt_data_ori[tumor_bw > 0] = 0
    tumor_inst, tumor_n = cc3d.connected_components(
        tumor_bw, connectivity=26, return_N=True
    )
    gt_data_ori[tumor_inst > 0] = tumor_inst[tumor_inst > 0] + np.max(gt_data_ori)
```
- 여러 개의 종양이 같은 label ID를 공유할 때, 각각을 별도 인스턴스로 분리
- 현재 코드에서는 `tumor_id = None`으로 비활성화 상태

**2-3. 3D 노이즈 제거**
```python
gt_data_ori = cc3d.dust(
    gt_data_ori, threshold=100, connectivity=26, in_place=True
)
```
- 3D 공간에서 연결된 voxel 수가 100개 미만인 독립 컴포넌트를 제거
- `connectivity=26`: 26방향 이웃 (면 + 모서리 + 꼭짓점) 연결성

**2-4. 2D 슬라이스별 노이즈 제거**
```python
for slice_i in range(gt_data_ori.shape[0]):
    gt_i = gt_data_ori[slice_i, :, :]
    gt_data_ori[slice_i, :, :] = cc3d.dust(
        gt_i, threshold=10, connectivity=8, in_place=True
    )
```
- 각 슬라이스에서 pixel 수가 10개 미만인 작은 객체를 제거
- `connectivity=8`: 2D 8방향 이웃 연결성
- 목적: 검출보다 분할이 핵심인 MedSAM 특성상, 너무 작아 의미 없는 영역 제거

#### 3단계: 유효 슬라이스 추출

```python
z_index, _, _ = np.where(gt_data_ori > 0)
z_index = np.unique(z_index)

if len(z_index) > 0:
    gt_roi  = gt_data_ori[z_index, :, :]    # GT가 있는 슬라이스만
    img_roi = image_data_pre[z_index, :, :] # 대응되는 이미지 슬라이스
```

GT가 전혀 없는 슬라이스(배경만 있는 위아래 슬라이스)를 제거해 **용량 절약 + 학습 효율 향상**

#### 4단계: 이미지 정규화

**CT 모드:**
```python
lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
image_data_pre = np.clip(image_data, lower_bound, upper_bound)
image_data_pre = (image_data_pre - np.min(image_data_pre)) \
               / (np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
```
1. HU 값을 윈도우 범위로 클리핑
2. Min-Max 정규화 → [0, 255] 범위

**MR 모드:**
```python
lower_bound = np.percentile(image_data[image_data > 0], 0.5)
upper_bound = np.percentile(image_data[image_data > 0], 99.5)
image_data_pre = np.clip(image_data, lower_bound, upper_bound)
image_data_pre = (image_data_pre - np.min(image_data_pre)) \
               / (np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
image_data_pre[image_data == 0] = 0    # 배경 픽셀 보존
```
1. 배경(0) 제외 후 0.5%~99.5% percentile로 이상값(outlier) 제거
2. Min-Max 정규화 → [0, 255] 범위
3. 원본에서 0이었던 픽셀은 0으로 복원

#### 5단계: NPZ 저장

```python
np.savez_compressed(
    join(npz_path, prefix + stem + '.npz'),
    imgs    = img_roi,   # shape: (N, H, W), dtype: uint8
    gts     = gt_roi,    # shape: (N, H, W), dtype: uint8
    spacing = spacing    # (x_spacing, y_spacing, z_spacing) in mm
)
```

| 키 | dtype | shape | 설명 |
|----|-------|-------|------|
| `imgs` | uint8 | (N, H, W) | 정규화된 CT 이미지 슬라이스 |
| `gts` | uint8 | (N, H, W) | 라벨 마스크 슬라이스 |
| `spacing` | tuple(float) | (3,) | mm 단위 voxel 간격 (x, y, z) |

#### 6단계: NIfTI 저장 (선택적, 검증용)

`--save_nii` 플래그 사용 시 전처리된 이미지와 GT를 NIfTI로도 저장.
육안으로 결과를 확인(sanity check)하는 용도이며, 확인 후 삭제 가능.

```
<npz_path>/CT_<anatomy>_<stem>_img.nii.gz
<npz_path>/CT_<anatomy>_<stem>_gt.nii.gz
```

### 멀티프로세싱 구조

```python
preprocess_tr = partial(preprocess, npz_path=npz_tr_path)

with mp.Pool(num_workers) as p:
    for i, _ in tqdm(enumerate(p.imap_unordered(preprocess_tr, tr_names))):
        pbar.update()
```

- `partial()`: npz_path를 고정하여 단일 인자 함수로 변환
- `imap_unordered()`: 완료 순서에 관계없이 결과를 즉시 반환 (순서 불필요한 전처리에 적합)

---

## 데이터셋별 상세 설정

### 1. AbdomenCT-1K

| 항목 | 값 |
|------|-----|
| 케이스 수 | 1,000 (이미지 1,062, 라벨 1,000) |
| 라벨 수 | 12개 → 11개 (duodenum 제거) |
| 윈도우 | WL=40, WW=400 (soft tissue) |
| 이미지 suffix | `_0000.nii.gz` |
| 파일 출처 | Part1 + Part2 + Part3 (각각 중첩 디렉토리) |

**제거 라벨:**
- `12`: duodenum (십이지장) — GT 품질이 낮아 학습에 불리

### 2. AMOS22

| 항목 | 값 |
|------|-----|
| 케이스 수 | **200** (imagesTr CT only, case_id ≤ 500) |
| 라벨 수 | 15개 (전체 유지) |
| 윈도우 | WL=40, WW=400 (soft tissue) |
| CT/MRI 구분 | case_id 1~410 (결번 있음) → CT 200개 / case_id 507~600 (결번 있음) → MRI 40개 |

**데이터 분할:**
- imagesTr만 사용 (imagesVa는 GT 비공개로 제외)
- imagesTr 내 CT(200개) + MRI(40개) = 240개 혼합 → **CT만 필터링(case_id ≤ 500)**
  - CT: case_id 1~410 범위, 결번 있음 (연속적이지 않음)
  - MRI: case_id 507~600 범위, 결번 있음 (501~506 없음)

**실제 NPZ 이력 및 정리:**
- 구버전 방식(상단 변수 직접 수정)으로 pre_CT_MR.py 실행 시 MRI 필터링 누락
- 결과적으로 `train_npz/CT/AMOS22/`에 CT 200개 + MRI 40개 = 240개 NPZ 생성됨
- MRI 케이스(`amos_0507` ~ `amos_0600`, 40개) 수동 삭제 완료 → **현재 200개**

### 3. COVID-19 CT

| 항목 | 값 |
|------|-----|
| 케이스 수 | 199 (Train only) |
| 라벨 수 | 1개 (binary: 0=배경, 1=감염 병변) |
| 윈도우 | WL=-500, WW=1500 (lung window) |
| 이미지/GT 구분 | `*_ct.nii.gz` / `*_seg.nii.gz` |

**폐 윈도우 적용 이유:**
폐 조직의 HU 범위(-1000~-500)를 포함하도록 넓은 폭 설정. 감염 병변(폐렴 음영)과 정상 폐 조직의 대비를 최대화.

### 4. KiTS23

| 항목 | 값 |
|------|-----|
| 케이스 수 | 489 |
| 라벨 수 | 3개 (1=kidney, 2=tumor, 3=cyst) |
| 윈도우 | WL=100, WW=400 (kidney window) |
| 원본 구조 | `case_XXXXX/imaging.nii.gz` |

**신장 윈도우 적용 이유:**
신장 실질(parenchyma)의 HU 범위(20~80)와 종양의 조영 증강을 최적화하기 위해 soft tissue window보다 높은 WL 적용.

### 5. TotalSegmentator

| 항목 | 값 |
|------|-----|
| 전체 케이스 수 | 1,228 |
| 유효 케이스 수 | 1,174 (54건은 GT 전체 0) |
| 라벨 수 | 17개 복부 장기 |
| 윈도우 | WL=40, WW=400 (soft tissue) |
| 핵심 처리 | 장기별 binary 마스크 → single multi-label 병합 |

**병합 우선순위:**
나중에 덮어씌워지는 label이 우선순위가 낮아짐. LABEL_MAP 순서(spleen=1 → esophagus=17)로 처리되므로 겹치는 voxel은 후순위 라벨로 덮어씌워짐.

---

## 전체 데이터 규모

| 데이터셋 | 케이스 수 | 라벨 수 | 해부학 부위 | NPZ prefix |
|---------|----------|---------|-----------|-----------|
| AbdomenCT-1K | 1,000 | 11개 | 복부 다장기 | `CT_AbdomenCT-1K_` |
| AMOS22 | 200 (imagesTr CT only, case_id ≤ 500 / MRI 40개 삭제 완료) | 15개 | 복부 다장기 | `CT_AMOS22_` |
| COVID-19 | 199 | 1개 | 폐 감염 | `CT_COVID-19-20_` |
| KiTS23 | 489 | 3개 | 신장/종양/낭종 | `CT_KiTS23_` |
| TotalSegmentator | 1,174 | 17개 | 복부 다장기 | `CT_TotalSegmentator_` |
| **합계** | **3,062** | | | |

---

## 실행 방법

### 전체 파이프라인 (권장)

```bash
# 1단계: 데이터셋 정리 + 심볼릭 링크 + 초기 NPZ 생성
python3 setup_datasets.py

# 2단계: run_preprocessing.sh로 데이터셋별 NPZ 재생성 (필요 시)
bash run_preprocessing.sh
```

### 개별 데이터셋 처리 (pre_CT_MR.py 직접 실행)

```bash
# 복부 CT (soft tissue window)
python3 pre_CT_MR.py \
  -modality CT \
  -anatomy Abd_MyDataset \
  -img_path /path/to/images \
  -gt_path  /path/to/labels \
  -output_path /path/to/output \
  -window_level 40 \
  -window_width 400 \
  -remove_label_ids "" \
  -num_workers 8

# 폐 CT (lung window)
python3 pre_CT_MR.py \
  -modality CT \
  -anatomy Lung_MyDataset \
  -img_path /path/to/images \
  -gt_path  /path/to/labels \
  -output_path /path/to/output \
  -window_level -500 \
  -window_width 1500 \
  -remove_label_ids "" \
  -num_workers 8

# 검증용 NIfTI 함께 저장
python3 pre_CT_MR.py ... --save_nii
```

### 필수 패키지

```bash
pip install nibabel SimpleITK numpy tqdm connected-components-3d
```

---

## 출력 파일 형식

### NPZ 파일 구조

```python
import numpy as np

data = np.load("CT_Abd_case001.npz")

data["imgs"]    # shape: (N, H, W), dtype: uint8, range: [0, 255]
data["gts"]     # shape: (N, H, W), dtype: uint8, range: [0, max_label]
data["spacing"] # tuple: (x_mm, y_mm, z_mm)  — voxel 크기 (mm 단위)
```

- `N`: GT가 존재하는 슬라이스 수
- `H`, `W`: 이미지 높이, 너비
- `imgs`와 `gts`는 동일한 슬라이스 인덱스에 대응

### 파일명 규칙

```
{modality}_{anatomy}_{original_stem}.npz

예시:
  CT_Abd_AbdomenCT1K_case_00001.npz
  CT_Lung_COVID19_volume-covid19-A-0001.npz
  CT_Abd_KiTS23_case_00100.npz
```
