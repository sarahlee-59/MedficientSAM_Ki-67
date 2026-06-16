#!/bin/bash
# Runs pre_CT_MR.py for each dataset sequentially.
# Output: /mnt/Disk1/sylee/CT/npz_output/MedSAM_train/CT_<anatomy>/

set -e
PYTHON=python3
SCRIPT=$(dirname "$0")/pre_CT_MR.py
BASE=/mnt/Disk1/sylee/CT
OUT=$BASE/npz_output
WORKERS=8

echo "========================================"
echo "[1/5] AbdomenCT-1K  (remove label 12=duodenum)"
echo "========================================"
$PYTHON $SCRIPT \
  -modality CT -anatomy Abd_AbdomenCT1K \
  -img_name_suffix _0000.nii.gz \
  -gt_name_suffix .nii.gz \
  -img_path $BASE/AbdomenCT-1K/images \
  -gt_path  $BASE/AbdomenCT-1K/labels \
  -output_path $OUT \
  -num_workers $WORKERS \
  -window_level 40 -window_width 400 \
  -remove_label_ids 12

echo "========================================"
echo "[2/5] AMOS22  (keep all labels)"
echo "========================================"
$PYTHON $SCRIPT \
  -modality CT -anatomy Abd_AMOS22 \
  -img_name_suffix .nii.gz \
  -gt_name_suffix .nii.gz \
  -img_path $BASE/AMOS22/images \
  -gt_path  $BASE/AMOS22/labels \
  -output_path $OUT \
  -num_workers $WORKERS \
  -window_level 40 -window_width 400 \
  -remove_label_ids ""

echo "========================================"
echo "[3/5] COVID-19  (binary: 0=bg, 1=infection)"
echo "========================================"
$PYTHON $SCRIPT \
  -modality CT -anatomy Lung_COVID19 \
  -img_name_suffix _ct.nii.gz \
  -gt_name_suffix _seg.nii.gz \
  -img_path $BASE/COVID-19/images \
  -gt_path  $BASE/COVID-19/labels \
  -output_path $OUT \
  -num_workers $WORKERS \
  -window_level -500 -window_width 1500 \
  -remove_label_ids ""

echo "========================================"
echo "[4/5] KiTS23  (1=kidney, 2=tumor, 3=cyst)"
echo "========================================"
$PYTHON $SCRIPT \
  -modality CT -anatomy Abd_KiTS23 \
  -img_name_suffix _0000.nii.gz \
  -gt_name_suffix .nii.gz \
  -img_path $BASE/KiTS23/images \
  -gt_path  $BASE/KiTS23/labels \
  -output_path $OUT \
  -num_workers $WORKERS \
  -window_level 100 -window_width 400 \
  -remove_label_ids ""

echo "========================================"
echo "[5/5] TotalSegmentator  (17 abdominal labels)"
echo "========================================"
$PYTHON $SCRIPT \
  -modality CT -anatomy Abd_TotalSeg \
  -img_name_suffix _0000.nii.gz \
  -gt_name_suffix .nii.gz \
  -img_path $BASE/TotalSegmentator/images \
  -gt_path  $BASE/TotalSegmentator/labels \
  -output_path $OUT \
  -num_workers $WORKERS \
  -window_level 40 -window_width 400 \
  -remove_label_ids ""

echo ""
echo "All preprocessing done! Output: $OUT"
