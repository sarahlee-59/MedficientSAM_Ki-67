# Ki-67 Nucleus Segmentation — Training Pipeline & Web Service

Ki-67 IHC 병리 이미지에서 핵(nucleus)을 클릭 한 번으로 세그멘테이션하는 서비스.  
EfficientViT-SAM L1 공식 pretrained → Ki-67 도메인 Fine-tuning → OpenVINO FP32 배포까지의 전체 파이프라인과 웹 서비스를 포함합니다.

---

## 레포지토리 구조

```
.
├── DATASET.md                        # 데이터셋 구성 및 전처리 상세 설명
├── EXPERIMENT_LOG.md                 # 실험 로그 및 학습 결과 분석
│
├── src/                             # 학습 코드
│   ├── train.py                     # Hydra 학습 진입점
│   ├── export_onnx.py               # ONNX + INT8 양자화 export
│   ├── export_torch.py              # PyTorch checkpoint export
│   ├── infer.py                     # 검증셋 추론
│   ├── models/                      # distill_module, finetune_module, efficientvit 등
│   ├── data/                        # MedSAMDataModule, MedSAMDistillDataset 등
│   ├── losses/                      # SAMLoss
│   ├── metrics/                     # generalized_dice
│   └── utils/                       # 로깅, 전처리 유틸
├── configs/                         # Hydra 설정
├── deployment/                      # 추론 배포 패키지
│   ├── encoder.quantized.onnx       # INT8 인코더 ~44MB
│   ├── decoder.quantized.onnx       # INT8 디코더 ~9MB
│   ├── openvino/                    # OpenVINO FP32 — 현재 운영 중
│   │   ├── encoder.xml / .bin       # FP32 인코더 IR ~167MB
│   │   ├── decoder.xml / .bin       # FP32 디코더 IR ~19MB
│   │   ├── infer.py                 # Ki67Segmenter 클래스 (openvino)
│   │   ├── server.py                # FastAPI 추론 서버
│   │   ├── example.py               # 단독 실행 예시
│   │   └── requirements.txt
│   └── onnx/                        # ONNX INT8 — 참고용
│       ├── infer.py                 # Ki67Segmenter 클래스 (onnxruntime)
│       ├── server.py                # FastAPI 추론 서버
│       └── example.py               # 단독 실행 예시
├── frontend/                        # Next.js 웹 서비스 (포트 3000)
│   ├── app/
│   │   ├── api/encode/              # FastAPI 프록시 — encode 전용
│   │   ├── api/decode/              # FastAPI 프록시 — decode 전용
│   │   ├── api/infer/               # FastAPI 프록시 — encode+decode 통합
│   │   ├── realtime/                # 실시간 세그멘테이션 UI
│   │   └── benchmark/               # 추론 속도 벤치마크 페이지
│   └── public/samples/              # 샘플 이미지
├── benchmark/                       # 추론 속도 벤치마크
│   ├── benchmark_speed.py           # ONNX-INT8 vs OpenVINO-FP32 측정
│   ├── benchmark_results.md         # 결과 요약
│   └── results/                     # 실측 JSON (ki67_hybrid_bench1~5)
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
- **Decoder**: SAM ViT-B의 prompt encoder + mask decoder 그대로 사용 (point-prompt only)
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

EfficientViT-L1 image encoder를 SAM ViT-B encoder 출력에 MSE 회귀로 학습.  
400K steps, loss ▼50.9% 완료했으나 공식 pretrained 기반 fine-tuning 대비 성능이 낮아 최종 배포에서 제외.  
실험 상세는 [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) 참고.

### 2단계: Fine-tuning (Ki-67 핵 데이터) — 배포 채택

EfficientViT-SAM L1 공식 pretrained 가중치 기반으로 병리 핵 데이터 파인튜닝.  
데이터 구성 및 증강 상세는 [`DATASET.md`](DATASET.md) 참고.

### 3단계: ONNX Export + INT8 양자화 + OpenVINO 변환

`src/export_onnx.py` → FP32 ONNX → ORT 최적화 → INT8 동적 양자화 → OpenVINO IR 변환.  
Export 설정 상세는 [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) 참고.

---

## 데이터셋

### CVPR 2024 MedSAM Laptop Challenge 데이터 (`train_npz/`)

대부분의 데이터는 [공식 Google Drive](https://drive.google.com/drive/folders/1khEIdkO0MC_gG5EkQ7COdDS1jge5_XQs)에서 다운로드.  
**CT·Dermoscopy·PanNuke·MoNuSeg는 공식 Drive에 미포함** — 챌린지 페이지([Codabench](https://www.codabench.org/competitions/1847/))의 추가 [Google Sheet](https://docs.google.com/spreadsheets/d/1QxjFs41eU6JG5KNhP576fc8MotrJ58KCrqH83HG-__E/edit?gid=2057737934#gid=2057737934)에서 별도 확보. 상세 내용은 [`DATASET.md`](DATASET.md) 참고.

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
│   ├── gts_npz_s128/            # Ki-67 IHC 슬라이드 타일 (stride 128)
│   ├── Cancer(PanNuke)_gts_npz/ # PanNuke 핵 데이터셋
│   └── MoNuSeg2018_gts_npz/     # MoNuSeg 2018
├── PET/         (autoPET)
├── US/          (Breast-Ultrasound, hc18)
└── XRay/        (ChestXray, COVID-19-Radiography, COVID-QU-Ex, Pneumothorax)
```

NPZ 파일 포맷: `{"imgs": (H,W,3) uint8, "gts": (H,W) uint8, "boxes": (N,4) float32}`

---

## 배포 패키지

> ONNX 모델 파일은 GitHub Releases에서 다운로드 후 `deployment/`에 배치.

사용법 및 API 상세:
- ONNX INT8: [`deployment/onnx/README.md`](deployment/onnx/README.md)
- OpenVINO FP32: [`deployment/openvino/README.md`](deployment/openvino/README.md)

---

## 웹 서비스 실행

### 이미 실행 중인지 확인

```bash
ss -tlnp | grep -E '3000|8000'
```

두 포트가 모두 보이면 바로 접속 → **http://10.10.40.194:3000/realtime**

---

### 1. 추론 서버 (포트 8000)

```bash
cd /mnt/Disk1/sylee/deployment/openvino

# 최초 1회
pip install fastapi "uvicorn[standard]" python-multipart
pip install -r requirements.txt

# 실행
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
echo $! > server.pid

# 중지
kill $(cat server.pid)
```

### 2. 프론트엔드 (포트 3000)

```bash
cd /mnt/Disk1/sylee/frontend

# 최초 1회
npm install && npm run build

# 실행
nohup npm start > ../frontend.log 2>&1 &
echo $! > ../frontend.pid

# 중지
kill $(cat ../frontend.pid)
```

### 3. 접속

```
http://10.10.40.194:3000/realtime
```

> 두 서버 모두 실행 후 5~10초 뒤 접속하세요.

---

## 웹 서비스 아키텍처

```
사용자 브라우저
    ↓ 이미지 업로드:  POST /api/encode (FormData)
    ↓ 클릭마다:       POST /api/decode (embedding + points)
    ↓ 단일 요청:      POST /api/infer  (FormData)
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

### API 엔드포인트 (FastAPI, 포트 8000)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/encode` | 이미지 업로드 → session_id 반환 (인코더 실행) |
| POST | `/decode` | session_id + 좌표 → 마스크 반환 (디코더만 실행) |
| POST | `/infer` | 이미지 + 좌표 → 마스크 (encode+decode 통합 fallback) |

---

## 기술 정보

| 항목 | 내용 |
|------|------|
| 추론 엔진 | OpenVINO FP32 (Intel CPU) |
| 임베딩 캐시 | 같은 이미지에서 반복 클릭 시 인코더 재실행 없음 |
| 프론트엔드 | Next.js (포트 3000) |
| 추론 서버 | FastAPI + OpenVINO (포트 8000) |

---

## References

- [EfficientViT](https://github.com/mit-han-lab/efficientvit)
- [MedSAM](https://github.com/bowang-lab/MedSAM)
- [MedficientSAM](https://github.com/hieplpvip/medficientsam) — 본 학습 프레임워크의 기반
- [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)
- [CVPR 2024 SAM on Laptop Challenge](https://www.codabench.org/competitions/1847/)
