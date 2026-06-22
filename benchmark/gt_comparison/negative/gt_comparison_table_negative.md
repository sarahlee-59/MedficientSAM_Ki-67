# bench1.png negative cell 1~4: OpenVINO vs Torch(best.ckpt) vs 실제 GT 비교

`ki67_hybrid_bench1_negative.json`의 4개 cell(전부 ki67_label=negative, prompt 3/4/5/6개) 클릭 prompt로 만든 결과를, 실제 GT(`train_npz/Pathology_new/gts_npz_s128/2D_S26-02873,A3,FDC00099,37255_3810_1_78609_72394_802_807_r00_c00.npz`)와 비교한 결과.

cell 1~4는 각각 GT 인스턴스 id 36, 56, 27, 55와 매칭된다 (polyline 마스크와 GT 인스턴스 겹침이 가장 큰 id로 매칭). cell4는 같은 GT 인스턴스(id 55)를 6개 prompt로 다시 클릭한 버전으로 교체됨.

| cell | n_prompts | GT id | OpenVINO IoU | OpenVINO Dice | Torch IoU | Torch Dice |
|---|---|---|---|---|---|---|
| 1 | 3 | 36 | 0.8913 | 0.9425 | 0.8638 | 0.9269 |
| 2 | 4 | 56 | 0.7707 | 0.8705 | 0.3987 | 0.5701 |
| 3 | 5 | 27 | 0.6980 | 0.8221 | 0.6023 | 0.7518 |
| 4 | 6 | 55 | 0.6319 | 0.7744 | 0.3464 | 0.5146 |
| **평균** | | | **0.7480** | **0.8524** | **0.5728** | **0.7159** |

OpenVINO가 4개 전부에서 Torch보다 높음. cell2, cell4는 Torch 추론 결과에서 GT와 떨어진 분리(stray) 조각이 발생해 IoU/Dice가 크게 낮아짐(시각화는 GT와 가장 겹치는 주 영역만 표시, 점수는 전체 마스크 기준).

생성 스크립트: `overlay_negative.py` (OpenVINO·Torch 결과 모두 생성) — 결과 PNG는 `final/`에 저장.
