# Ki-67 Service — Docker 운영 가이드

## 서비스 구성

```
[클라이언트]
     │
     ▼ :3000
  [nginx]  ← docker-compose (nginx:alpine)
     │
     ▼ :3000 (내부)
[frontend] ← docker-compose (Next.js, node:20-alpine)
     │
     ▼ :8000
[FastAPI]  ← Docker 밖, uvicorn 직접 실행
```

- **frontend + nginx**: `docker-compose.yml`로 관리
- **FastAPI 추론 서버**: Docker 없이 uvicorn으로 별도 실행 (포트 8000)

---

## 사전 준비

```
Ki-67_service/
├── deployment/
│   └── onnx/              ← ONNX 모델 파일 (frontend 컨테이너에 마운트됨)
│       ├── encoder.quantized.onnx
│       └── decoder.quantized.onnx
└── frontend/
    └── certs/             ← nginx SSL 인증서 (HTTP만 쓰면 빈 디렉토리라도 필요)
```

---

## Docker Compose (frontend + nginx)

### 시작

```bash
# 이미지 빌드 후 백그라운드 실행
docker compose up -d --build

# 이미 빌드된 이미지로 시작 (코드 변경 없을 때)
docker compose up -d
```

### 중지 / 재시작

```bash
docker compose down            # 컨테이너 중지 + 삭제
docker compose down -v         # 볼륨까지 삭제 (주의)
docker compose restart         # 컨테이너 재시작 (이미지 재빌드 없음)
```

### 특정 서비스만 재빌드

```bash
# frontend 코드 변경 후 frontend만 재빌드
docker compose up -d --build frontend
```

### 상태 확인 / 로그

```bash
docker compose ps              # 실행 중인 서비스 목록
docker compose logs -f         # 전체 로그 스트림
docker compose logs -f frontend  # frontend 로그만
docker compose logs -f nginx     # nginx 로그만
```

### 컨테이너 내부 접속

```bash
docker compose exec frontend sh
docker compose exec nginx sh
```

---

## FastAPI 추론 서버 (Docker 외부)

`deployment/openvino/` 디렉토리에서 직접 실행.

### 환경 설치

```bash
cd /mnt/Disk1/sylee/Ki-67_service/deployment/openvino
pip install fastapi "uvicorn[standard]" python-multipart openvino numpy opencv-python
# 또는 (requirements.txt는 openvino/numpy/opencv/Pillow만 있으므로 fastapi 등 별도 설치)
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]" python-multipart
```

### 실행

```bash
cd /mnt/Disk1/sylee/Ki-67_service/deployment/openvino
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 백그라운드 실행 (nohup)

```bash
cd /mnt/Disk1/sylee/Ki-67_service/deployment/openvino
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
echo $! > server.pid
```

### 중지

```bash
kill $(cat /mnt/Disk1/sylee/Ki-67_service/deployment/openvino/server.pid)
```

---

## 전체 서비스 기동 순서

```bash
# 1. FastAPI 추론 서버 먼저 실행
cd /mnt/Disk1/sylee/Ki-67_service/deployment/openvino
uvicorn server:app --host 0.0.0.0 --port 8000 &

# 2. Docker Compose로 frontend + nginx 실행
cd /mnt/Disk1/sylee/Ki-67_service
docker compose up -d --build
```

접속 주소: **http://10.10.40.194:3000/realtime**

---

## API 엔드포인트 (FastAPI, 포트 8000)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/encode` | 이미지 업로드 → session_id 반환 (인코더 실행) |
| POST | `/decode` | session_id + 좌표 → 마스크 반환 (디코더만 실행) |
| POST | `/infer` | 이미지 + 좌표 → 마스크 (encode+decode 통합 fallback) |
