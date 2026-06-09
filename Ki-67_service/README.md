# Ki-67 Segmentation Service

Ki-67 세포 이미지를 클릭만으로 핵(nucleus)을 세그멘테이션하고 Ki-67 지수를 계산하는 서비스입니다.

- 추론 엔진: 서버 사이드 ONNX Runtime Python / OpenVINO (FastAPI)
- 프론트엔드: Next.js — 이미지 및 클릭 좌표를 서버로 전송 후 마스크 수신

---

## 목차

1. [빠른 시작](#빠른-시작)
2. [서비스 사용 방법](#서비스-사용-방법)
3. [로컬 개발 환경](#로컬-개발-환경)
4. [Python 추론 API](#python-추론-api)
5. [아키텍처](#아키텍처)
6. [프로젝트 구조](#프로젝트-구조)
7. [트러블슈팅](#트러블슈팅)

---

## 빠른 시작

### 1. 추론 서버 실행

```bash
cd Ki-67_service/deployment
pip install fastapi "uvicorn[standard]" python-multipart onnxruntime numpy opencv-python
uvicorn server:app --host 0.0.0.0 --port 8000
```

OpenVINO 백엔드를 사용하는 경우:

```bash
cd Ki-67_service/deployment_openvino
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 2. 프론트엔드 실행

```bash
cd Ki-67_service/frontend
npm install
npm run dev
```

브라우저에서 http://localhost:3000/realtime 으로 접속합니다.

### Docker (프론트엔드만)

```bash
cd Ki-67_service
docker compose up -d
```

> 추론 서버는 Docker Compose에 포함되어 있지 않습니다. 별도로 실행해야 합니다.

---

## 서비스 사용 방법

### 화면 구성

```
┌──────────────────────────────────┬─────────────────────┐
│                                  │  라벨 선택           │
│         캔버스 영역               │  + 양성 / - 음성     │
│    (이미지 + 세그멘테이션 결과)    │                     │
│                                  │  도형 선택           │
│                                  │  △ □ ⬠ ⬡           │
│                                  │                     │
│                                  │  세포 목록 (결과)    │
│                                  │                     │
│                                  │  통계 (Ki-67 지수)   │
└──────────────────────────────────┴─────────────────────┘
```

### 기본 사용 흐름

**1단계: 이미지 업로드**
캔버스의 업로드 박스를 클릭하거나 드래그 앤 드롭합니다. 지원 형식: JPG, PNG, BMP, TIFF.

**2단계: 라벨 선택** — `P` 양성(+) / `N` 음성(−)

**3단계: 도형 선택 및 조정** — △ □ ⬠ ⬡ 중 선택, 마우스 휠로 크기 조정해 세포 핵에 맞춥니다.

**4단계: 클릭으로 세포 확정** — 핵 위에서 좌클릭. 양성은 빨간 윤곽선, 음성은 파란 윤곽선으로 표시됩니다.

완료 후 우측 하단 통계 패널에서 **Ki-67 지수 (%)** 와 세포 수를 확인하고, **JSON 저장** 버튼으로 결과를 다운로드합니다.

<details>
<summary>상세 조작 (도형 회전·편집·줌·단축키)</summary>

**도형 조작**

| 동작 | 효과 |
|---|---|
| 마우스 이동 | 도형 위치 이동 |
| 마우스 휠 | 전체 크기 조정 |
| 좌클릭 드래그 | 도형 회전 |
| 우클릭 드래그 | 가로/세로 폭 독립 조정 (X=가로, Y=세로) |
| `R` | 회전 0° 리셋 |

**편집**

| 동작 | 방법 |
|---|---|
| 마지막 세포 취소 | `Z` 키 또는 Undo 버튼 |
| 세포 삭제 | `Esc` → 삭제 모드 → 세포 클릭 |
| 전체 삭제 | `Ctrl + Esc` |
| 세포 재추론 | 우측 목록에서 세포 클릭 → 새 위치에서 좌클릭 |
| 라벨 변경 | 우측 목록에서 `+` / `−` 버튼 |
| 목록 순서 변경 | 우측 목록에서 드래그 앤 드롭 |

**줌·팬**

| 동작 | 효과 |
|---|---|
| `Ctrl + 휠` | 줌 인/아웃 (최대 8배) |
| `Ctrl + 좌드래그` | 화면 이동 |

**단축키 요약**

| 키 | 기능 |
|---|---|
| `P` | 양성(+) 라벨 |
| `N` | 음성(−) 라벨 |
| `Z` | Undo |
| `R` | 회전 리셋 |
| `Esc` | 삭제 모드 토글 |
| `Ctrl + Esc` | 전체 삭제 |
| `Ctrl + 휠` | 줌 인/아웃 |
| `Ctrl + 드래그` | 화면 이동 |

</details>

---

## 로컬 개발 환경

Node.js 20 이상이 필요합니다.

### 추론 서버 (필수)

```bash
cd deployment
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

서버가 실행 중이지 않으면 프론트엔드에서 추론이 동작하지 않습니다.

### 프론트엔드

```bash
cd frontend
npm install
npm run dev
```

`frontend/.env.local`의 `BACKEND_URL`이 추론 서버 주소와 일치하는지 확인합니다 (기본값: `http://localhost:8000`).

### 프로덕션 빌드 테스트

```bash
cd frontend
npm run build && npm run start
```

---

## Python 추론 API

### ONNX Runtime (deployment/)

```bash
pip install onnxruntime numpy opencv-python
```

```python
from deployment.infer import Ki67Segmenter

seg = Ki67Segmenter(
    encoder_path="deployment/encoder.quantized.onnx",
    decoder_path="deployment/decoder.quantized.onnx",
)

# image: (H, W, 3) uint8 RGB
# points: (N, K, 2) float32 — N개 인스턴스 × K개 클릭 좌표 (x, y)
masks = seg.predict(image, points)  # (N, H, W) uint8 바이너리 마스크
```

### OpenVINO (deployment_openvino/)

```bash
pip install -r deployment_openvino/requirements.txt
```

```python
from deployment_openvino.infer import Ki67Segmenter

seg = Ki67Segmenter(
    encoder_path="deployment_openvino/encoder.xml",
    decoder_path="deployment_openvino/decoder.xml",
)
masks = seg.predict(image, points)
```

두 패키지의 `Ki67Segmenter` API는 동일합니다. 파일 경로만 바꾸면 백엔드를 전환할 수 있습니다.

### 인코딩 캐시 (같은 이미지에서 반복 추론)

```python
emb = seg.encode(image)
masks_a = seg.decode(emb, points_a, image.shape[:2])
masks_b = seg.decode(emb, points_b, image.shape[:2])
```

### 성능 비교 (Intel CPU, 807×802 타일, median 30회)

| 백엔드 | 인코더 | 디코더 (N=1, K=1) | e2e |
|---|---|---|---|
| INT8 ONNX (onnxruntime) | 274 ms | 10 ms | 284 ms |
| **FP32 OpenVINO** | **139 ms** | **9 ms** | **148 ms** |

서버 추론에서는 인코딩 결과를 이미지 해시 기준으로 캐시합니다. 같은 이미지에서 여러 번 클릭할 때는 인코더를 한 번만 실행합니다.

---

## 아키텍처

```
사용자 브라우저
    │  이미지 + 클릭 좌표 전송
    ↓
/realtime 페이지 (Next.js, 포트 3000)
    │  POST /api/infer  (Next.js 프록시)
    ↓
FastAPI 추론 서버 (포트 8000)
    │  Ki67Segmenter.encode() — 이미지 해시 캐시
    │  Ki67Segmenter.decode() — 마스크 생성
    ↓
ONNX Runtime / OpenVINO
    │  (N, H, W) uint8 바이너리 마스크
    ↓
브라우저 — 마스크 → 윤곽선 변환 후 캔버스에 렌더링
```

---

## 프로젝트 구조

```
Ki-67_service/
├── frontend/                        # Next.js 앱 (Node.js 20)
│   ├── app/
│   │   ├── api/
│   │   │   └── infer/route.ts       # FastAPI 프록시 엔드포인트
│   │   └── realtime/                # 메인 어노테이션 페이지 (/realtime)
│   │       ├── page.tsx             # UI + 서버 추론 호출
│   │       ├── types.ts
│   │       └── utils/segmentation.ts
│   ├── public/
│   │   └── samples/                 # 샘플 이미지
│   ├── Dockerfile
│   ├── .env.local                   # BACKEND_URL 설정
│   └── package.json
├── deployment/                      # ONNX Runtime 추론 패키지
│   ├── infer.py                     # Ki67Segmenter 클래스
│   ├── server.py                    # FastAPI 추론 서버
│   ├── example.py                   # CLI 데모
│   ├── encoder.quantized.onnx       # INT8 인코더 (~44 MB)
│   └── decoder.quantized.onnx       # INT8 디코더 (~9 MB)
├── deployment_openvino/             # OpenVINO 추론 패키지
│   ├── infer.py                     # Ki67Segmenter 클래스 (OV 백엔드)
│   ├── example.py                   # CLI 데모
│   ├── encoder.xml / encoder.bin    # FP32 인코더 IR (~167 MB)
│   ├── decoder.xml / decoder.bin    # FP32 디코더 IR (~19 MB)
│   └── requirements.txt
├── docker-compose.yml
└── nginx.conf
```

> `src/`, `configs/` 는 모델 학습용 코드이며 서비스 실행과 무관합니다.

---

## 트러블슈팅

### 추론 서버에 연결할 수 없음

프론트엔드에서 추론 요청이 실패하는 경우 추론 서버가 실행 중인지 확인합니다.

```bash
curl http://localhost:8000/docs
```

`frontend/.env.local`의 `BACKEND_URL`이 실제 서버 주소와 일치하는지 확인합니다.

### 세포 세그멘테이션이 비어있거나 노이즈가 많음

- 도형 크기를 세포 핵에 맞게 조정합니다 (마우스 휠).
- 클릭 위치가 핵 **내부**에 오도록 조준합니다.
- 학습 타일 크기: 128~256 px. 더 큰 이미지는 내부에서 512px longest-side로 리사이즈됩니다.

### 컨테이너 코드가 반영되지 않음

```bash
docker compose build frontend && docker compose up -d
```

### 포트 3000 또는 8000이 이미 사용 중

```bash
sudo lsof -i :3000
sudo lsof -i :8000
```
