# bench1/bench4 positive cell: prompt 개수(3/4/5/6)별 OpenVINO vs Torch(best.ckpt) vs 실제 GT 비교

`ki67_hybrid_bench1_positive.json`에는 prompt 개수가 다른 positive cell 4개(3/4/5/6개)가 들어있다. `cell_index`는 개별 cell 식별자가 아니라 prompt 개수 그 자체이며, 파일명도 `cellp{prompt개수}_*`로 저장된다. cell마다 `image`/`gt_npz` 필드로 출처 이미지가 다를 수 있다 — prompt 3/4/5개는 `bench1.png`, prompt 6개는 `bench4.png` 기반이다.

각 prompt 개수는 GT 인스턴스 id 5(3개, bench1), 70(4개, bench1), 69(5개, bench1), 81(6개, bench4)과 매칭된다 (polyline 마스크와 GT 인스턴스 겹침이 가장 큰 id로 매칭). bench1.png는 `train_npz/Pathology_new/gts_npz_s128/...r00_c00.npz`, bench4.png는 같은 슬라이드의 다른 타일인 `...r00_c03.npz`와 픽셀 단위로 동일함을 확인했다.

| n_prompts | image | GT id | OpenVINO IoU | OpenVINO Dice | Torch IoU | Torch Dice |
|---|---|---|---|---|---|---|
| 3 | bench1.png | 5 | 0.7115 | 0.8314 | 0.6485 | 0.7867 |
| 4 | bench1.png | 70 | 0.8561 | 0.9225 | 0.6289 | 0.7722 |
| 5 | bench1.png | 69 | 0.8443 | 0.9156 | 0.8089 | 0.8944 |
| 6 | bench4.png | 81 | 0.8288 | 0.9064 | 0.6771 | 0.8074 |
| **평균** | | | **0.8102** | **0.8940** | **0.6909** | **0.8152** |

prompt 4개에서 Torch가 stray(GT와 떨어진 분리 조각) 8개가 발생해 IoU/Dice가 크게 낮아짐(시각화는 GT와 가장 겹치는 주 영역만 표시, 점수도 해당 영역 기준). 4개 전부 OpenVINO가 Torch보다 높음.

생성 스크립트: `overlay_positive.py` (OpenVINO·Torch 결과 모두 생성, cell별 image/gt_npz 참조) — 결과 PNG는 `final/`에 저장.
