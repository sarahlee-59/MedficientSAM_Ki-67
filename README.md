# Ki-67 Nucleus Segmentation — Training Pipeline & Web Service

Ki-67 IHC 병리 이미지에서 핵(nucleus)을 클릭 한 번으로 세그멘테이션하는 서비스.  
EfficientViT-SAM L1 공식 pretrained → Ki-67 도메인 Fine-tuning → OpenVINO FP32 배포까지의 전체 파이프라인과 서버 추론 기반 웹 서비스를 포함합니다.

---

## 레포지토리 구조

```
.
├── DATASET.md                        # 데이터셋 구성 및 전처리 상세 설명
├── EXPERIMENT_LOG.md                 # 실험 로그 및 학습 결과 분석
│
├── Ki-67_service/                   # Ki-67 특화 파이프라인 + 웹 서비스
│   ├── src/                         # Ki-67 실험용 학습 코드
│   │   ├── train.py                 # Hydra 학습 진입점
│   │   ├── export_onnx.py           # ONNX + INT8 양자화 export
│   │   ├── export_torch.py          # PyTorch checkpoint export
│   │   ├── infer.py                 # 검증셋 추론
│   │   ├── models/                  # distill_module, finetune_module, efficientvit 등
│   │   ├── data/                    # MedSAMDataModule, MedSAMDistillDataset 등
│   │   ├── losses/                  # SAMLoss
│   │   ├── metrics/                 # generalized_dice
│   │   └── utils/                   # 로깅, 전처리 유틸
│   ├── configs/                     # Ki-67 실험 Hydra 설정
│   ├── deployment/                  # 추론 배포 패키지
│   │   ├── encoder.quantized.onnx   # INT8 인코더 ~44MB
│   │   ├── decoder.quantized.onnx   # INT8 디코더 ~9MB
│   │   ├── openvino/                # OpenVINO FP32 — 현재 운영 중 (e2e ~125ms)
│   │   │   ├── encoder.xml / .bin   # FP32 인코더 IR ~167MB
│   │   │   ├── decoder.xml / .bin   # FP32 디코더 IR ~19MB
│   │   │   ├── infer.py             # Ki67Segmenter 클래스 (openvino)
│   │   │   ├── server.py            # FastAPI 추론 서버
│   │   │   ├── example.py           # 단독 실행 예시
│   │   │   ├── requirements.txt
│   │   │   └── README.md
│   │   └── onnx/                    # ONNX INT8 — 참고용 (e2e ~637ms)
│   │       ├── infer.py             # Ki67Segmenter 클래스 (onnxruntime)
│   │       ├── server.py            # FastAPI 추론 서버
│   │       ├── example.py           # 단독 실행 예시
│   │       └── README.md
│   ├── frontend/                    # Next.js 웹 서비스 (포트 3000)
│   │   ├── app/
│   │   │   ├── api/encode/          # FastAPI 프록시 — encode 전용
│   │   │   ├── api/decode/          # FastAPI 프록시 — decode 전용
│   │   │   ├── api/infer/           # FastAPI 프록시 — encode+decode 통합
│   │   │   ├── realtime/            # 실시간 세그멘테이션 UI
│   │   │   │   ├── page.tsx
│   │   │   │   ├── types.ts
│   │   │   │   └── utils/segmentation.ts
│   │   │   └── benchmark/           # 추론 속도 벤치마크 페이지
│   │   └── public/samples/          # 샘플 이미지
│   └── benchmark/                   # 추론 속도 벤치마크
│       ├── benchmark_speed.py       # ONNX-INT8 vs OpenVINO-FP32 측정 스크립트
│       ├── benchmark_results.md     # 결과 요약
│       ├── images/                  # 벤치마크용 샘플 이미지
│       └── results/                 # 실측 JSON (ki67_hybrid_bench1~5)
│
└── train_npz/                       # 학습 데이터 (gitignore, DATASET.md 참고)
```

---

## 모델 아키텍처

```
이미지 입력 (임의 해상도)
    ↓ longest-side resize → 512×512
EfficientViT-L1 Image Encoder   ← SAM ViT-B에서 distill
    ↓ (1, 256, 64, 64) embedding
SAM ViT-B Prompt Encoder + Mask Decoder
    ↓ point prompts (N개 클릭)
(N, 512, 512) 마스크 로짓 → 이진 마스크
```

- **Student**: EfficientViT-L1 (47.65M params, 51.05G FLOPs at 512×512)
- **Teacher**: SAM ViT-B (MedSAM 파인튜닝 버전)
- **Decoder**: SAM ViT-B의 prompt encoder + mask decoder를 그대로 사용 (point-prompt only)
- **배포 크기**: ~53MB (INT8 동적 양자화, PyTorch 불필요)

### 성능 (Ki-67 holdout 818 tiles, n=100, seed=42)

| 프롬프트 | Dice |
|----------|------|
| k=1 click | ~0.30 |
| k=3 clicks | ~0.73 |
| k=5 clicks | ~0.76 |

---

## 학습 파이프라인

### 1단계: Knowledge Distillation (배포 미채택)

EfficientViT-L1 image encoder를 SAM ViT-B encoder의 출력(image embeddings)에 MSE 회귀로 학습.  
실험 완료(400K steps, loss ▼50.9%)했으나 공식 pretrained 기반 fine-tuning 대비 성능이 낮아 최종 배포에서 제외됨.  
실험 상세는 `EXPERIMENT_LOG.md` 참고.

### 2단계: Fine-tuning (Ki-67 핵 데이터) — 배포 채택

> Fine-tuning 설계 및 실행은 별도로 진행됐습니다.

**EfficientViT-SAM L1 공식 pretrained** 가중치를 기반으로 병리 핵 데이터로 파인튜닝. 두 가지 조합으로 실험:

| 실험 | 데이터 | 총 패치 |
|------|--------|---------|
| 조합 학습 | Ki-67 (16,376) + PanNuke (2,538) + MoNuSeg (148) | 19,062 |
| Ki-67 단독 | 부산 백병원 89슬라이드, 90/5/5 분할 | 16,376 |

GlaS 데이터셋은 cell 단위가 아닌 gland 단위 구성이므로 제외.

증강 파이프라인: HorizontalFlip, VerticalFlip, RandomRotate90, **HEDJitter** (IHC 염색 변이), GaussianBlur/MotionBlur/Defocus (Scanner 품질), ImageCompression, RandomBrightnessContrast

### 3단계: ONNX Export + INT8 양자화 + OpenVINO 변환

`src/export_onnx.py`로 FP32 ONNX export → ORT 최적화 → INT8 동적 양자화를 순서대로 수행.  
최종 배포 파일: `encoder.quantized.onnx` (~44MB) + `decoder.quantized.onnx` (~9MB)

이후 ONNX FP32 모델을 OpenVINO IR 포맷(`encoder.xml/.bin`, `decoder.xml/.bin`)으로 변환하여 `deployment/openvino/`에 반영.  
Intel oneDNN 커널 활용으로 FP32 ONNX 대비 약 1.65× 빠른 추론 속도 확인.

---

## 데이터셋

### CVPR 2024 MedSAM Laptop Challenge 데이터 (`train_npz/`)

대부분의 데이터는 [공식 Google Drive](https://drive.google.com/drive/folders/1khEIdkO0MC_gG5EkQ7COdDS1jge5_XQs)에서 다운로드.  
챌린지 참가 후 [챌린지 페이지](https://www.codabench.org/competitions/1847/)에서도 접근 가능.

> **주의 — CT·Dermoscopy·PanNuke·MoNuSeg는 공식 Google Drive에 미포함**  
> 논문이 11개 모달리티를 사용했다고 명시되어 있어 동일하게 구성하려 했으나, 공식 Drive에 이 데이터셋들이 빠져 있었습니다.  
> 챌린지 공식 홈페이지([Codabench](https://www.codabench.org/competitions/1847/))에 새롭게 추가된 [Google Sheet](https://docs.google.com/spreadsheets/d/1QxjFs41eU6JG5KNhP576fc8MotrJ58KCrqH83HG-__E/edit?gid=2057737934#gid=2057737934)에서 별도 확보했습니다.  
> - **CT**: Google Sheet 경유, 챌린지 주최 측이 변환한 NPZ를 그대로 사용  
> - **Dermoscopy**: Google Sheet에서 원본 이미지 다운로드 → stride 128, crop 256×256 tiling 후 NPZ 생성  
> - **PanNuke**: Google Sheet에서 원본 다운로드 → tiling 없이 256×256 resize → NPZ 생성  
> - **MoNuSeg 2018**: Google Sheet에서 원본 다운로드 → 1000×1000 → 2×2 crop → 256×256 resize → NPZ 생성  
> - **TotalSegmentator 선정 이유**: 전신 117개 부위를 단일 데이터셋으로 커버하여 CT 학습 범위를 극대화

```
train_npz/
├── CT/          (AbdomenCT-1K, AMOS22, COVID-19-20, KiTS23, TotalSegmentator) ← 스프레드시트 경유
├── Dermoscopy/  (ISIC-2017) ← 스프레드시트 경유
├── Endoscopy/   (CholecSeg8k, Kvasir-SEG, m2caiSeg)
├── Fundus/      (IDRiD, PAPILA)
├── Mammography/ (CDD-CESM)
├── Microscopy/  (NeurIPS22CellSeg)
├── MR/          (AMOS, BraTS, CervicalCancer, crossmoda, Heart, ISLES2022, Prostate, SpineMR, WMH)
├── OCT/         (Intraretinal-Cystoid-Fluid)
├── Pathology_new/
│   ├── gts_npz_s128/            # Ki-67 IHC 슬라이드 타일 (128px)
│   ├── Cancer(PanNuke)_gts_npz/ # PanNuke 핵 데이터셋
│   └── MoNuSeg2018_gts_npz/     # MoNuSeg 2018
├── PET/         (autoPET)
├── US/          (Breast-Ultrasound, hc18)
└── XRay/        (ChestXray, COVID-19-Radiography, COVID-QU-Ex, Pneumothorax)
```

NPZ 파일 포맷: `{"imgs": (H,W,3) uint8, "gts": (H,W) uint8, "boxes": (N,4) float32}`

### NPZ 전처리 코드 출처

`train_npz/` 의 NPZ 파일은 이 레포 밖에서 생성되었습니다. 자세한 내용은 [`DATASET.md`](DATASET.md) 참고.

| 데이터 | 전처리 주체 | 방법 |
|--------|------------|------|
| Endoscopy, Fundus, MR, OCT 등 9개 모달리티 | 챌린지 주최 측 배포 NPZ | [MedSAM 공식 레포](https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM) |
| CT 5종 | 챌린지 주최 측 (Google Sheet 경유) | 이미 NPZ 형태 제공 |
| Dermoscopy (ISIC-2017) | 내부 전처리 | Google Sheet 원본 → stride 128, 256×256 tiling → NPZ |
| PanNuke | 내부 전처리 (Google Sheet 경유) | Google Sheet 원본 → tiling 없이 256×256 resize → NPZ |
| MoNuSeg 2018 | 내부 전처리 (Google Sheet 경유) | Google Sheet 원본 → 1000×1000 → 2×2 crop → 256×256 resize → NPZ |
| Ki-67 슬라이드 (`gts_npz_s128/`) | 자체 병리 파이프라인 | 256×256, stride 128 — `/mnt/Disk1/DP_IHC/Ki67_pytorchlightning/` |

---

## 배포 패키지

> **ONNX 모델 파일은 GitHub Releases에서 다운로드하세요.**  
> `encoder.quantized.onnx` (~44MB) + `decoder.quantized.onnx` (~9MB)를 `Ki-67_service/deployment/`에 배치.

### 의존성 (PyTorch 불필요)

```bash
pip install onnxruntime numpy opencv-python
```

### 사용 예시

```python
import sys
sys.path.insert(0, "Ki-67_service")
from deployment.onnx.infer import Ki67Segmenter
import numpy as np

seg = Ki67Segmenter(
    encoder_path="Ki-67_service/deployment/encoder.quantized.onnx",
    decoder_path="Ki-67_service/deployment/decoder.quantized.onnx",
)

# image: (H, W, 3) uint8 RGB
# points: (N, K, 2) float32 — N개 세포 × K번 클릭, (x, y) 좌표
image = ...
points = np.array([[[120.0, 80.0], [130.0, 95.0], [115.0, 110.0]]])  # 1개 세포, 3번 클릭
masks = seg.predict(image, points)  # (N, H, W) uint8 binary
```

여러 이미지에서 같은 이미지 반복 추론 시 `encode()` → `decode()` 분리로 encoder 1회 실행:

```python
emb = seg.encode(image)
masks_a = seg.decode(emb, points_a, image.shape[:2])
masks_b = seg.decode(emb, points_b, image.shape[:2])
```

자세한 CLI 사용법은 [`Ki-67_service/deployment/onnx/README.md`](Ki-67_service/deployment/onnx/README.md),  
OpenVINO 서버 사용법은 [`Ki-67_service/deployment/openvino/README.md`](Ki-67_service/deployment/openvino/README.md) 참고.

---

## 웹 서비스

실행 방법 및 UI 사용 가이드는 [`Ki-67_service/README.md`](Ki-67_service/README.md) 참고.

### 아키텍처

Next.js 프록시를 통해 FastAPI 추론 서버(OpenVINO FP32)에 요청을 전달하는 서버 추론 구조.  
브라우저는 이미지와 클릭 좌표를 전송하고, 서버에서 마스크를 반환합니다.

```
사용자 브라우저
    ↓ 이미지 업로드 시: POST /api/encode (FormData)
    ↓ 클릭마다:        POST /api/decode (embedding + points)
    ↓ 단일 요청 시:    POST /api/infer  (FormData)
Next.js /realtime 페이지 (포트 3000, systemd)
FastAPI 추론 서버 (포트 8000, OpenVINO FP32)
    ↓ 이미지 해시 캐시 → encode → decode → (H, W) uint8 마스크
브라우저 — 마스크 → 윤곽선 추출 → 캔버스 렌더링
```

### 추론 성능 (Intel CPU, 256×256, median 5회)

| 백엔드 | encode | decode (4셀) | e2e |
|--------|-------:|-------------:|----:|
| ONNX INT8 | 545 ms | 92 ms | 637 ms |
| **OpenVINO FP32** | **75 ms** | **50 ms** | **125 ms** |

---

## References

- [EfficientViT](https://github.com/mit-han-lab/efficientvit)
- [MedSAM](https://github.com/bowang-lab/MedSAM)
- [MedficientSAM](https://github.com/hieplpvip/medficientsam) — 본 학습 프레임워크의 기반
- [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)
- [CVPR 2024 SAM on Laptop Challenge](https://www.codabench.org/competitions/1847/)
