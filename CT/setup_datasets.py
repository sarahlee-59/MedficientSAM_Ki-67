#!/usr/bin/env python3
"""
Organizes all CT datasets under /mnt/Disk1/sylee/CT/refine_ver/ using symlinks,
merges TotalSegmentator per-organ binary masks into a single multi-label file,
and fills any missing NPZ files under /mnt/Disk1/sylee/train_npz/CT/.

Reads raw data from:   /mnt/Disk1/sylee/CT/raw_ver/
Writes symlinks to:    /mnt/Disk1/sylee/CT/refine_ver/
Writes NPZ files to:   /mnt/Disk1/sylee/train_npz/CT/

Output structure:
  refine_ver/
  ├── AbdomenCT-1K/images/  ← Part1+2+3 (_0000.nii.gz)
  ├── AbdomenCT-1K/labels/  ← Mask/ (.nii.gz, same stem)
  ├── AMOS22/images/        ← imagesTr only, CT only (case_id <= 500)
  ├── AMOS22/labels/        ← labelsTr only, CT only (case_id <= 500)
  ├── COVID-19/images/      ← Train *_ct.nii.gz only
  ├── COVID-19/labels/      ← Train *_seg.nii.gz only
  ├── KiTS23/images/        ← case_XXXXX_0000.nii.gz → .../imaging.nii.gz
  ├── KiTS23/labels/        ← case_XXXXX.nii.gz → .../segmentation.nii.gz
  ├── TotalSegmentator/images/ ← sXXXX_0000.nii.gz → .../ct.nii.gz
  └── TotalSegmentator/labels/ ← sXXXX.nii.gz (merged multi-label)
"""

import os
import re
import glob
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import cc3d
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
RAW     = "/mnt/Disk1/sylee/CT/raw_ver"
OUT     = "/mnt/Disk1/sylee/CT/refine_ver"
NPZ_ROOT = "/mnt/Disk1/sylee/train_npz/CT"

# ── NPZ preprocessing config ─────────────────────────────────────────────────
NPZ_WORKERS  = 8
VOXEL_THR_2D = 10
VOXEL_THR_3D = 100


# ── Helpers ──────────────────────────────────────────────────────────────────
def symlink(src, dst):
    """
    Fix #6: broken symlink 안전 처리
    - lexists: 깨진 링크도 True 반환
    - exists: 깨진 링크는 False
    -> 깨진 링크면 지우고 새로 만들고, 정상이면 건너뜀
    """
    if os.path.lexists(dst):
        if not os.path.exists(dst):  # broken link
            os.unlink(dst)
        else:
            return
    os.symlink(src, dst)


def report(name, expected, img_dir, gt_dir):
    """Fix #7: 데이터셋별 sanity check 로깅"""
    n_img = len(os.listdir(img_dir))
    n_gt = len(os.listdir(gt_dir))
    status = "OK" if n_img == expected and n_gt == expected else "MISMATCH"
    print(f"  [{status}] {name}: images={n_img}, labels={n_gt} (expected={expected})")


def parse_amos_id(filename):
    """AMOS22 파일명에서 case ID 추출. ex: amos_0001.nii.gz -> 1"""
    m = re.search(r"amos_(\d+)", os.path.basename(filename))
    return int(m.group(1)) if m else -1


def _read(path):
    """SimpleITK 실패 시 nibabel로 폴백. 반환: (array ZYX, spacing, sitk_obj|None)"""
    try:
        s = sitk.ReadImage(path)
        return sitk.GetArrayFromImage(s), s.GetSpacing(), s
    except RuntimeError:
        n = nib.load(path)
        arr = np.asarray(n.dataobj).transpose(2, 1, 0)
        zooms = tuple(float(z) for z in n.header.get_zooms()[:3])
        return arr, zooms, None


def make_npz(gt_name, img_dir, gt_dir, npz_dir, prefix,
             img_suffix, gt_suffix, wl, ww, remove_ids):
    """GT 파일 하나를 전처리해 NPZ로 저장. 이미 있으면 스킵."""
    stem = gt_name[: -len(gt_suffix)]
    out_path = os.path.join(npz_dir, prefix + stem + ".npz")
    if os.path.exists(out_path):
        return

    try:
        gt_arr, _, _ = _read(os.path.join(gt_dir, gt_name))
        gt = np.uint8(gt_arr)
        for rid in remove_ids:
            gt[gt == rid] = 0
        gt = cc3d.dust(gt, threshold=VOXEL_THR_3D, connectivity=26, in_place=True)
        for z in range(gt.shape[0]):
            gt[z] = cc3d.dust(gt[z], threshold=VOXEL_THR_2D, connectivity=8, in_place=True)

        z_index = np.unique(np.where(gt > 0)[0])
        if len(z_index) == 0:
            return  # empty GT — skip silently

        gt_roi = gt[z_index]

        img_arr, spacing, img_sitk = _read(os.path.join(img_dir, stem + img_suffix))
        if img_sitk is not None:
            spacing = img_sitk.GetSpacing()

        lo, hi = wl - ww / 2, wl + ww / 2
        img_pre = np.clip(img_arr, lo, hi)
        img_pre = (img_pre - img_pre.min()) / (img_pre.max() - img_pre.min()) * 255.0
        img_roi = np.uint8(img_pre)[z_index]

        np.savez_compressed(out_path, imgs=img_roi, gts=gt_roi, spacing=spacing)
    except Exception as e:
        print(f"  Warning: {stem} skipped — {e}")


def fill_missing_npz(dataset_name, img_dir, gt_dir, npz_dir, prefix,
                     img_suffix, gt_suffix, wl, ww, remove_ids, expected):
    """NPZ가 없는 케이스만 골라 멀티프로세싱으로 생성."""
    os.makedirs(npz_dir, exist_ok=True)

    gt_names = sorted(
        f for f in os.listdir(gt_dir)
        if f.endswith(gt_suffix)
        and os.path.exists(os.path.join(img_dir, f[: -len(gt_suffix)] + img_suffix))
    )
    existing_stems = {
        f[len(prefix): -len(".npz")]
        for f in os.listdir(npz_dir)
        if f.startswith(prefix) and f.endswith(".npz")
    }
    missing = [f for f in gt_names if f[: -len(gt_suffix)] not in existing_stems]

    if not missing:
        n = len([f for f in os.listdir(npz_dir) if f.endswith(".npz")])
        status = "OK" if n >= expected else "MISMATCH"
        print(f"  [{status}] {dataset_name} NPZ: {n} (expected={expected}), nothing to do")
        return

    print(f"  {dataset_name} NPZ: {len(missing)} missing → processing with {NPZ_WORKERS} workers")
    fn = partial(
        make_npz,
        img_dir=img_dir, gt_dir=gt_dir, npz_dir=npz_dir,
        prefix=prefix, img_suffix=img_suffix, gt_suffix=gt_suffix,
        wl=wl, ww=ww, remove_ids=remove_ids,
    )
    with mp.Pool(NPZ_WORKERS) as p:
        list(tqdm(p.imap_unordered(fn, missing), total=len(missing), desc=f"  {dataset_name}"))

    done = len([f for f in os.listdir(npz_dir) if f.endswith(".npz")])
    status = "OK" if done >= expected else "MISMATCH"
    print(f"  [{status}] {dataset_name} NPZ: {done} (expected={expected})")


# ── Sanity check: raw 경로 존재 확인 ─────────────────────────────────────────
if not os.path.isdir(RAW):
    raise FileNotFoundError(
        f"Raw data directory not found: {RAW}\n"
        f"Please update the RAW path at the top of this script."
    )
os.makedirs(OUT, exist_ok=True)


# ── 1. AbdomenCT-1K ──────────────────────────────────────────────────────────
print("\n[1/5] AbdomenCT-1K")
os.makedirs(f"{OUT}/AbdomenCT-1K/images", exist_ok=True)
os.makedirs(f"{OUT}/AbdomenCT-1K/labels", exist_ok=True)

# Fix #2: Part3도 nested 구조 (Part1/Part2와 동일하게)
part_dirs = [
    f"{RAW}/AbdomenCT-1K-ImagePart1/AbdomenCT-1K-ImagePart1",
    f"{RAW}/AbdomenCT-1K-ImagePart2/AbdomenCT-1K-ImagePart2",
    f"{RAW}/AbdomenCT-1K-ImagePart3/AbdomenCT-1K-ImagePart3",
]
for part_dir in part_dirs:
    if not os.path.isdir(part_dir):
        print(f"  WARNING: part dir missing: {part_dir}")
        continue
    for f in sorted(glob.glob(f"{part_dir}/*.nii.gz")):
        symlink(f, f"{OUT}/AbdomenCT-1K/images/{os.path.basename(f)}")

for f in sorted(glob.glob(f"{RAW}/Mask/*.nii.gz")):
    symlink(f, f"{OUT}/AbdomenCT-1K/labels/{os.path.basename(f)}")

report("AbdomenCT-1K", 1000, f"{OUT}/AbdomenCT-1K/images", f"{OUT}/AbdomenCT-1K/labels")
fill_missing_npz(
    "AbdomenCT-1K",
    img_dir    = f"{OUT}/AbdomenCT-1K/images",
    gt_dir     = f"{OUT}/AbdomenCT-1K/labels",
    npz_dir    = f"{NPZ_ROOT}/AbdomenCT-1K",
    prefix     = "CT_AbdomenCT-1K_",
    img_suffix = "_0000.nii.gz",
    gt_suffix  = ".nii.gz",
    wl=40, ww=400,
    remove_ids = [],
    expected   = 1000,
)


# ── 2. AMOS22 ────────────────────────────────────────────────────────────────
print("\n[2/5] AMOS22 (Tr only, CT only)")
os.makedirs(f"{OUT}/AMOS22/images", exist_ok=True)
os.makedirs(f"{OUT}/AMOS22/labels", exist_ok=True)

# Fix #3: Tr만 + CT만 (case_id <= 500)
# AMOS22 imagesTr = CT 200개 + MRI 40개 혼합 (총 240개)
#   CT:  case_id 1~410   (결번 있음, 연속적이지 않음) → 200개
#   MRI: case_id 507~600 (결번 있음, 501~506 없음)   → 40개
# case_id <= 500 조건으로 CT만 필터링
for f in sorted(glob.glob(f"{RAW}/amos22/amos22/imagesTr/*.nii.gz")):
    case_id = parse_amos_id(f)
    if case_id <= 0 or case_id > 500:
        continue  # MRI(507~600) 또는 파싱 실패 케이스 제외
    symlink(f, f"{OUT}/AMOS22/images/{os.path.basename(f)}")

for f in sorted(glob.glob(f"{RAW}/amos22/amos22/labelsTr/*.nii.gz")):
    case_id = parse_amos_id(f)
    if case_id <= 0 or case_id > 500:
        continue
    symlink(f, f"{OUT}/AMOS22/labels/{os.path.basename(f)}")

report("AMOS22", 200, f"{OUT}/AMOS22/images", f"{OUT}/AMOS22/labels")
fill_missing_npz(
    "AMOS22",
    img_dir    = f"{OUT}/AMOS22/images",
    gt_dir     = f"{OUT}/AMOS22/labels",
    npz_dir    = f"{NPZ_ROOT}/AMOS22",
    prefix     = "CT_AMOS22_",
    img_suffix = ".nii.gz",
    gt_suffix  = ".nii.gz",
    wl=40, ww=400,
    remove_ids = [],
    expected   = 200,
)


# ── 3. COVID-19 ──────────────────────────────────────────────────────────────
print("\n[3/5] COVID-19 (Train only)")
os.makedirs(f"{OUT}/COVID-19/images", exist_ok=True)
os.makedirs(f"{OUT}/COVID-19/labels", exist_ok=True)

# Fix #4: Validation은 라벨이 비공개이므로 명시적으로 Train만 사용
for f in sorted(glob.glob(f"{RAW}/COVID-19-20_v2/Train/*_ct.nii.gz")):
    symlink(f, f"{OUT}/COVID-19/images/{os.path.basename(f)}")
for f in sorted(glob.glob(f"{RAW}/COVID-19-20_v2/Train/*_seg.nii.gz")):
    symlink(f, f"{OUT}/COVID-19/labels/{os.path.basename(f)}")

report("COVID-19", 199, f"{OUT}/COVID-19/images", f"{OUT}/COVID-19/labels")
fill_missing_npz(
    "COVID-19-20",
    img_dir    = f"{OUT}/COVID-19/images",
    gt_dir     = f"{OUT}/COVID-19/labels",
    npz_dir    = f"{NPZ_ROOT}/COVID-19-20",
    prefix     = "CT_COVID-19-20_",
    img_suffix = "_ct.nii.gz",
    gt_suffix  = "_seg.nii.gz",
    wl=-500, ww=1500,
    remove_ids = [],
    expected   = 199,
)


# ── 4. KiTS23 ────────────────────────────────────────────────────────────────
print("\n[4/5] KiTS23")
os.makedirs(f"{OUT}/KiTS23/images", exist_ok=True)
os.makedirs(f"{OUT}/KiTS23/labels", exist_ok=True)

for case_dir in sorted(glob.glob(f"{RAW}/kits23/dataset/case_*")):
    case_name = os.path.basename(case_dir)
    img_src = f"{case_dir}/imaging.nii.gz"
    gt_src = f"{case_dir}/segmentation.nii.gz"
    if os.path.exists(img_src) and os.path.exists(gt_src):
        symlink(img_src, f"{OUT}/KiTS23/images/{case_name}_0000.nii.gz")
        symlink(gt_src, f"{OUT}/KiTS23/labels/{case_name}.nii.gz")

report("KiTS23", 489, f"{OUT}/KiTS23/images", f"{OUT}/KiTS23/labels")
fill_missing_npz(
    "KiTS23",
    img_dir    = f"{OUT}/KiTS23/images",
    gt_dir     = f"{OUT}/KiTS23/labels",
    npz_dir    = f"{NPZ_ROOT}/KiTS23",
    prefix     = "CT_KiTS23_",
    img_suffix = "_0000.nii.gz",
    gt_suffix  = ".nii.gz",
    wl=100, ww=400,
    remove_ids = [],
    expected   = 489,
)


# ── 5. TotalSegmentator ──────────────────────────────────────────────────────
print("\n[5/5] TotalSegmentator — merging per-organ masks")
os.makedirs(f"{OUT}/TotalSegmentator/images", exist_ok=True)
os.makedirs(f"{OUT}/TotalSegmentator/labels", exist_ok=True)

# Abdominal organ label mapping
LABEL_MAP = {
    "spleen":                       1,
    "kidney_right":                 2,
    "kidney_left":                  3,
    "gallbladder":                  4,
    "liver":                        5,
    "stomach":                      6,
    "aorta":                        7,
    "inferior_vena_cava":           8,
    "pancreas":                     9,
    "adrenal_gland_right":         10,
    "adrenal_gland_left":          11,
    "duodenum":                    12,
    "colon":                       13,
    "small_bowel":                 14,
    "urinary_bladder":             15,
    "portal_vein_and_splenic_vein": 16,
    "esophagus":                   17,
}

ts_base = f"{RAW}/Totalsegmentator_dataset_v201"

if not os.path.isdir(ts_base):
    print(f"  WARNING: TotalSegmentator dir missing: {ts_base}")
    cases = []
else:
    # Fix #5: __MACOSX 등 잡 폴더 제외, s\d{4} 패턴만 허용
    case_pattern = re.compile(r"^s\d{4}$")
    cases = sorted(
        d for d in os.listdir(ts_base)
        if os.path.isdir(f"{ts_base}/{d}") and case_pattern.fullmatch(d)
    )

for case in tqdm(cases, desc="  merging"):
    ct_src = f"{ts_base}/{case}/ct.nii.gz"
    seg_dir = f"{ts_base}/{case}/segmentations"
    if not os.path.exists(ct_src):
        continue

    symlink(ct_src, f"{OUT}/TotalSegmentator/images/{case}_0000.nii.gz")

    gt_dst = f"{OUT}/TotalSegmentator/labels/{case}.nii.gz"
    if os.path.exists(gt_dst):
        continue

    try:
        ref = nib.load(ct_src)
        shape = ref.shape[:3]
        merged = np.zeros(shape, dtype=np.uint8)

        for label_name, label_id in LABEL_MAP.items():
            seg_file = f"{seg_dir}/{label_name}.nii.gz"
            if os.path.exists(seg_file):
                seg_arr = nib.load(seg_file).get_fdata(dtype=np.float32)
                merged[seg_arr > 0] = label_id

        out_nii = nib.Nifti1Image(merged, ref.affine, ref.header)
        nib.save(out_nii, gt_dst)
    except Exception as e:
        print(f"  Warning: {case} skipped — {e}")

report("TotalSegmentator", 1228, f"{OUT}/TotalSegmentator/images", f"{OUT}/TotalSegmentator/labels")
fill_missing_npz(
    "TotalSegmentator",
    img_dir    = f"{OUT}/TotalSegmentator/images",
    gt_dir     = f"{OUT}/TotalSegmentator/labels",
    npz_dir    = f"{NPZ_ROOT}/TotalSegmentator",
    prefix     = "CT_TotalSegmentator_",
    img_suffix = "_0000.nii.gz",
    gt_suffix  = ".nii.gz",
    wl=40, ww=400,
    remove_ids = [],
    expected   = 1174,  # 1228 중 54개는 raw organ mask가 전부 0 → 유효 케이스 1174
)


# ── Final summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Done. Summary of expected counts:")
print("  AbdomenCT-1K    : 1000 (labels) / 1062 (images)")
print("  AMOS22          : 200 (Tr CT only)")
print("  COVID-19        : 199 (Train only)")
print("  KiTS23          : 489")
print("  TotalSegmentator: 1174 (1228 - 54 empty GT)")
print("=" * 60)