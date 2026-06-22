# bench1.png negative cell 1~4: OpenVINO vs Torch(best.ckpt) vs 실제 GT 비교

`cells/ki67_hybrid_bench1_negative.json`의 4개 cell(전부 ki67_label=negative) 클릭 prompt로 만든 결과를, 실제 GT(`train_npz/Pathology_new/gts_npz_s128/2D_S26-02873,A3,FDC00099,37255_3810_1_78609_72394_802_807_r00_c00.npz`)와 비교한 결과.

cell 1~4는 각각 GT 인스턴스 id 36, 27, 38, 56과 매칭된다 (polyline 마스크와 GT 인스턴스 IoU가 가장 큰 id로 매칭).

| cell | GT id | OpenVINO IoU | OpenVINO Dice | Torch IoU | Torch Dice |
|---|---|---|---|---|---|
| 1 | 36 | 0.8913 | 0.9425 | 0.8638 | 0.9269 |
| 2 | 27 | 0.6980 | 0.8221 | 0.6023 | 0.7518 |
| 3 | 38 | 0.8683 | 0.9295 | 0.7582 | 0.8625 |
| 4 | 56 | 0.7707 | 0.8705 | 0.2900 | 0.4496 |
| **평균** | | **0.8071** | **0.8911** | **0.6286** | **0.7477** |

OpenVINO가 4개 전부에서 Torch보다 높음. cell4는 Torch 추론 결과에서 GT와 떨어진 분리(stray) 조각이 3개 발생해 IoU/Dice가 크게 낮아짐(시각화는 GT와 가장 겹치는 주 영역만 표시, 점수는 전체 마스크 기준). 평균적으로는 OpenVINO가 IoU +0.179, Dice +0.143 더 높음.

생성 스크립트: `overlay_cells_vs_gt_negative.py`(OpenVINO), `overlay_torch_cells_vs_gt_negative.py`(Torch) — 결과 PNG는 `overlays/negative/`에 저장.
