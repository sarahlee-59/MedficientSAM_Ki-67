# Ki-67 Segmentation Service

Ki-67 세포 이미지를 클릭만으로 핵(nucleus)을 세그멘테이션하고 Ki-67 지수를 계산하는 서비스입니다. 모든 추론이 브라우저 안에서 실행되어 **이미지가 서버로 전송되지 않으며, GPU 서버도 필요 없습니다.**

- 배포 주소: https://ki-67.vercel.app
- 추론 엔진: 브라우저 ONNX (onnxruntime-web, WASM)

---

## 목차

1. [빠른 시작 — Docker](#빠른-시작--docker)
2. [서비스 사용 방법](#서비스-사용-방법)
3. [로컬 개발 환경](#로컬-개발-환경)
4. [Python 추론 API](#python-추론-api)
5. [아키텍처](#아키텍처)
6. [프로젝트 구조](#프로젝트-구조)
7. [트러블슈팅](#트러블슈팅)

---

## 빠른 시작 — Docker

```bash
git clone <repo-url> Ki-67_service
cd Ki-67_service
docker compose up -d
```

빌드는 처음 실행 시 약 3~5분 소요됩니다. 완료 후 브라우저에서 접속합니다.

- 서버에서 직접: http://localhost:3000/realtime
- 외부 장치에서: http://서버-IP:3000/realtime

> **외부 IP + HTTP에서 모델 로드가 실패하는 경우**: `SharedArrayBuffer` 제한 때문입니다. [트러블슈팅](#onnx-모델-로드-실패-브라우저-콘솔-에러)을 참고하세요.

### 컨테이너 관리

```bash
# 상태 확인
docker compose ps
docker compose logs frontend --tail=30

# 중지
docker compose down

# 코드 변경 후 재빌드
docker compose build frontend && docker compose up -d
```

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
업로드 후 ONNX 모델이 자동 로드됩니다 (첫 로드 시 ~53 MB 다운로드, 이후 브라우저 캐시).

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

```bash
cd frontend
npm install
```

### HTTPS 개발 서버

외부 IP HTTP에서는 WASM이 동작하지 않으므로 HTTPS가 필요합니다.

```bash
# mkcert 설치 (최초 1회)
sudo apt install mkcert    # Ubuntu/Debian
# brew install mkcert      # macOS

# 인증서 생성 (최초 1회)
mkdir -p certs
mkcert -cert-file certs/cert.pem -key-file certs/key.pem <서버-IP>

# 서버 시작
npm run dev:https
```

https://서버-IP:3000/realtime 으로 접속합니다.

> 로컬에서 `localhost`로만 접속하는 경우엔 `npm run dev` 로 HTTP 서버를 바로 쓸 수 있습니다.

### 프로덕션 빌드 테스트

```bash
npm run build && npm run start
```

---

## Python 추론 API

서버 없이 Python에서 직접 ONNX 모델을 사용하는 방법입니다.

### 설치

```bash
pip install onnxruntime numpy opencv-python Pillow
```

### 기본 사용법

```python
import numpy as np
from deployment.infer import Ki67Segmenter

seg = Ki67Segmenter(
    encoder_path="deployment/encoder.quantized.onnx",
    decoder_path="deployment/decoder.quantized.onnx",
)

# image: (H, W, 3) uint8 RGB
# points: (N, K, 2) float32 — N개 인스턴스 × K개 클릭 좌표 (x, y)
masks = seg.predict(image, points)  # (N, H, W) uint8 바이너리 마스크
```

### 인코딩 캐시 (같은 이미지에서 반복 추론)

```python
emb = seg.encode(image)
masks_a = seg.decode(emb, points_a, image.shape[:2])
masks_b = seg.decode(emb, points_b, image.shape[:2])
```

### CLI 예제

```bash
# 단일 세포, 클릭 1개
python deployment/example.py my_tile.png --points "120,80"

# 단일 세포, 클릭 3개
python deployment/example.py my_tile.png --points "100,80;130,95;115,110"

# 전경 + 배경 혼합
python deployment/example.py my_tile.png --points "120,80;130,95" --neg-points "50,50"
```

결과는 `<이미지이름>.overlay.png`로 저장됩니다.

### 성능 지표 (Ki-67 holdout 818 타일, n=100, seed=42)

| 클릭 수 | Dice 점수 |
|---|---|
| k=1 | ~0.30 |
| k=3 | ~0.73 |
| k=5 | ~0.76 |

> 클릭 위치는 핵(nucleus) **내부**에 있어야 합니다. 학습 타일 크기 128~256 px — 큰 이미지는 내부에서 512px longest-side로 리사이즈됩니다.

---

## 아키텍처

```
사용자 브라우저
    ↓  이미지 업로드 (로컬, 서버 전송 없음)
/realtime 페이지 (Next.js)
    ↓  GET /models/encoder.quantized.onnx
    ↓  GET /models/decoder.quantized.onnx
nginx (포트 3000, 리버스 프록시)
    ↓
Next.js 서버 (Docker, 포트 3000)
    ↓  public/models/*.onnx (빌드 시 포함)
브라우저 WASM 런타임 (onnxruntime-web)
```

모든 추론은 브라우저 내 WASM으로 실행됩니다. 서버는 ONNX 모델 파일을 정적으로 서빙하는 역할만 합니다.

---

## 프로젝트 구조

```
Ki-67_service/
├── frontend/                        # Next.js 앱 (Node.js 20)
│   ├── app/
│   │   └── realtime/                # 메인 어노테이션 페이지 (/realtime)
│   │       ├── page.tsx             # UI + 추론 로직
│   │       ├── types.ts
│   │       └── utils/segmentation.ts
│   ├── public/
│   │   ├── models/                  # ONNX 모델 파일 (빌드에 포함, INT8 quantized 사용)¹
│   │   └── samples/                 # 샘플 이미지
│   ├── Dockerfile
│   └── package.json
├── deployment/                      # Python 추론 패키지
│   ├── infer.py                     # Ki67Segmenter 클래스
│   └── example.py
├── docker-compose.yml
└── nginx.conf
```

¹ `models/` 에는 INT8(quantized), FP16, FP32 세 가지 정밀도의 모델이 포함됩니다. 서비스는 기본으로 INT8을 사용합니다.

> `src/`, `configs/` 는 모델 학습용 코드이며 서비스 실행과 무관합니다.

---

## 트러블슈팅

### ONNX 모델 로드 실패 (브라우저 콘솔 에러)

**원인**: HTTP 환경에서 `SharedArrayBuffer`가 차단됨.

**해결**: HTTPS 환경을 구성합니다.
- 로컬 개발: `npm run dev:https` ([로컬 개발 환경](#로컬-개발-환경) 참고)
- Docker: nginx SSL 설정

`localhost`에서 접속하면 HTTP에서도 WASM이 정상 동작합니다.

### 첫 로드가 느림

모델 파일 합계 약 53 MB를 처음 다운로드합니다. 이후 브라우저 캐시에 저장됩니다.

### 세포 세그멘테이션이 비어있거나 노이즈가 많음

- 도형 크기를 세포 핵에 맞게 조정합니다 (마우스 휠).
- 클릭 위치가 핵 **내부**에 오도록 조준합니다.
- 추론 중 회색 테두리가 표시되면 완료될 때까지 기다립니다 (WASM CPU 추론 100~500ms).

### 컨테이너 코드가 반영되지 않음

```bash
docker compose build frontend && docker compose up -d
```

### 포트 3000이 이미 사용 중

```bash
sudo lsof -i :3000
# docker-compose.yml의 nginx 포트를 변경 (예: 3001)
```
