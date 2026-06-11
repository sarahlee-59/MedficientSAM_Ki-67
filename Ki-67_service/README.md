# Ki-67 Segmentation Service

Ki-67 세포 이미지에서 핵(nucleus)을 클릭만으로 세그멘테이션하고 Ki-67 지수를 계산하는 서비스입니다.

**접속 주소:** http://10.10.40.194:3000/realtime

---

## 화면 구성

```
┌─────────────────────────────────────┬──────────────────────┐
│                                     │  도구 / 라벨 선택     │
│          캔버스 영역                 │  양성(+) / 음성(−)   │
│   (이미지 + 세그멘테이션 결과 표시)  │                      │
│                                     │  프롬프트 도형        │
│                                     │  △ □ ⬠ ⬡            │
│                                     │                      │
├─────────────────────────────────────┤  단축키 / 액션       │
│  세포 목록 · Ki-67 통계 · JSON 저장  │                      │
└─────────────────────────────────────┴──────────────────────┘
```

---

## 사용 흐름

### 1단계 — 이미지 업로드

캔버스 중앙의 업로드 영역을 **클릭**하거나 이미지 파일을 **드래그 앤 드롭**합니다.

- 지원 형식: JPG, PNG, BMP, TIFF
- 또는 우측 툴바 **이미지 → 불러오기** 버튼 사용
- 샘플 이미지(bench 1~5)로 바로 시작할 수도 있습니다

### 2단계 — 라벨 선택

우측 툴바에서 세포 종류를 먼저 선택합니다.

| 버튼 | 단축키 | 의미 |
|------|--------|------|
| **양성(+)** | `P` | Ki-67 양성 세포 (갈색 염색) |
| **음성(−)** | `N` | Ki-67 음성 세포 (파란 염색) |

### 3단계 — 도형 조정 후 클릭

우측에서 프롬프트 도형(△ □ ⬠ ⬡)을 선택하고, 마우스로 핵 크기에 맞게 조정한 뒤 클릭합니다.

| 조작 | 효과 |
|------|------|
| 마우스 이동 | 도형 위치 이동 |
| 마우스 휠 | 도형 전체 크기 조정 |
| 좌드래그 | 도형 회전 |
| 우드래그 | 가로/세로 폭 독립 조정 |
| `R` | 회전 0° 리셋 |
| 좌클릭 | **세포 확정** — 추론 시작 |

추론이 완료되면 **양성은 빨간 윤곽선**, **음성은 파란 윤곽선**으로 표시됩니다.

### 4단계 — 결과 확인 및 저장

하단 패널에서 Ki-67 지수(%)와 세포 수를 확인합니다.  
**JSON 저장** 버튼으로 좌표·라벨·통계를 포함한 결과 파일을 다운로드합니다.

---

## 편집

| 동작 | 방법 |
|------|------|
| 마지막 세포 취소 | `Z` 또는 **Undo** 버튼 |
| 취소 되돌리기 | `Y` 또는 **Redo** 버튼 |
| 세포 삭제 | `Esc` → 삭제 모드 → 세포 클릭 |
| 전체 삭제 | `Ctrl + Esc` |
| 세포 재추론 | 우측 목록에서 세포 클릭 → 새 위치에서 좌클릭 |
| 라벨 변경 | 우측 목록에서 `+` / `−` 버튼 |
| 목록 순서 변경 | 우측 목록에서 드래그 앤 드롭 |

---

## 줌 · 화면 이동

| 조작 | 효과 |
|------|------|
| `Ctrl + 휠` | 줌 인/아웃 (최대 8×) |
| `Ctrl + 좌드래그` | 화면 이동 (팬) |
| **줌 초기화** 버튼 | 원래 배율로 복귀 |

---

## 단축키 전체 요약

| 키 | 기능 |
|----|------|
| `P` | 양성(+) 라벨 선택 |
| `N` | 음성(−) 라벨 선택 |
| `Z` | Undo — 마지막 세포 취소 |
| `Y` | Redo — 취소 되돌리기 |
| `R` | 도형 회전 0° 리셋 |
| `Esc` | 삭제 모드 토글 |
| `Ctrl + Esc` | 전체 세포 삭제 |
| `Ctrl + 휠` | 줌 인/아웃 |
| `Ctrl + 드래그` | 화면 이동 |

---

## 기술 정보

| 항목 | 내용 |
|------|------|
| 추론 엔진 | OpenVINO FP32 (Intel CPU) |
| 평균 추론 속도 | 인코딩 75 ms + 디코딩 50 ms ≈ **e2e 125 ms** |
| 임베딩 캐시 | 같은 이미지에서 반복 클릭 시 인코더 재실행 없음 |
| 프론트엔드 | Next.js (포트 3000) |
| 추론 서버 | FastAPI (포트 8000) |

---

## 인프라 구성

### 현재 Docker 구성

```
외부 :3000
    │
  [nginx]          docker-compose — 단순 리버스 프록시
    │
  [frontend]       docker-compose — Next.js (expose만, 외부 직접 접근 불가)
    │
  [FastAPI]        Docker 외부 — uvicorn 직접 실행 (포트 8000)
```

`docker-compose.yml`은 `frontend`(Next.js)와 `nginx` 두 서비스를 관리한다.  
`frontend`는 `expose`만 선언되어 있어 Docker 네트워크 내부에서만 접근 가능하고, nginx가 외부 포트 3000을 받아 내부로 전달한다.

### nginx의 역할과 필요성

현재 `nginx.conf`는 HTTP 리버스 프록시 하나만 수행한다(SSL 미설정, 정적 파일 분리 없음).  
즉, **nginx가 없어도 동일하게 동작한다.** `docker-compose.yml`에서 `expose → ports`로 바꾸면 nginx 없이 frontend를 직접 외부에 노출할 수 있다.

nginx가 유효해지는 시점:
- HTTPS 전환 시 (SSL 인증서 + `listen 443 ssl` 설정)
- 여러 서비스를 단일 포트로 묶을 때 (경로 기반 라우팅)

### Docker 없이 실행하는 방법

Docker를 쓰지 않아도 된다. 아래 순서로 직접 실행 가능하다.

```bash
# 1. FastAPI 추론 서버
cd /mnt/Disk1/sylee/Ki-67_service/deployment/openvino
pip install fastapi "uvicorn[standard]" python-multipart openvino numpy opencv-python
uvicorn server:app --host 0.0.0.0 --port 8000 &

# 2. Next.js 프론트엔드
cd /mnt/Disk1/sylee/Ki-67_service/frontend
ONNX_DIR=/mnt/Disk1/sylee/Ki-67_service/deployment/onnx npm run build
ONNX_DIR=/mnt/Disk1/sylee/Ki-67_service/deployment/onnx npm start
```

### Docker Compose로 실행하는 방법

```bash
cd /mnt/Disk1/sylee/Ki-67_service

# 빌드 후 시작
docker compose up -d --build

# 로그 확인
docker compose logs -f

# 중지
docker compose down
```

> **사전 준비**: `deployment/onnx/`에 `encoder.quantized.onnx`, `decoder.quantized.onnx` 파일이 있어야 한다.
