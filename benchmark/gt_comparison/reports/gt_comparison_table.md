# bench1.png cell 1~4: OpenVINO vs Torch(best.ckpt) vs 실제 GT 비교

`cells/ki67_hybrid_bench1_positive.json`의 4개 cell 클릭 prompt로 만든 결과를, 실제 GT(`train_npz/Pathology_new/gts_npz_s128/2D_S26-02873,A3,FDC00099,37255_3810_1_78609_72394_802_807_r00_c00.npz`)와 비교한 결과.

`bench1.png`가 위 GT npz의 `imgs`와 픽셀 단위로 100% 동일한 이미지임을 확인했고, cell 1~4는 각각 GT 인스턴스 id 4, 71, 7, 70과 매칭된다.

| cell | GT id | OpenVINO IoU | OpenVINO Dice | Torch IoU | Torch Dice |
|---|---|---|---|---|---|
| 1 | 4 | 0.8571 | 0.9231 | 0.8266 | 0.9051 |
| 2 | 71 | 0.8375 | 0.9115 | 0.6846 | 0.8128 |
| 3 | 7 | 0.7802 | 0.8765 | 0.8642 | 0.9272 |
| 4 | 70 | 0.8148 | 0.8979 | 0.7812 | 0.8772 |
| **평균** | | **0.8224** | **0.9023** | **0.7892** | **0.8806** |

OpenVINO가 4개 중 3개(1, 2, 4)에서 더 높고, cell 3에서만 Torch가 더 높음. 평균적으로는 OpenVINO가 IoU +0.033, Dice +0.022 더 높음.

생성 스크립트: `overlay_cells_vs_gt.py`(OpenVINO), `overlay_torch_cells_vs_gt.py`(Torch) — 결과 PNG는 `overlays/positive/`에 저장.
