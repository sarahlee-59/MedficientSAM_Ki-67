# Ki-67 Segmentation Service

Ki-67 세포 이미지를 업로드하고 클릭하면 자동으로 세그멘테이션해주는 서비스.

- 사이트: https://ki-67.vercel.app
- 추론: 브라우저 ONNX (onnxruntime-web, WASM) — 서버 GPU 불필요

---

## 아키텍처

```
사용자 브라우저
    ↓  이미지 업로드 (로컬)
/realtime 페이지 (Next.js)
    ↓  /api/onnx/encoder, /api/onnx/decoder
nginx (포트 3000)
    ↓
Next.js 서버 (Docker)
    ↓  ONNX_DIR=/models (볼륨 마운트)
/mnt/Disk1/sylee/deployment/*.onnx
```

모든 추론은 브라우저 WASM으로 실행됩니다. 서버는 ONNX 모델 파일을 정적으로 서빙하는 역할만 합니다.

---

## 모델 파일

```
/mnt/Disk1/sylee/deployment/
├── encoder.quantized.onnx   (~44 MB)
└── decoder.quantized.onnx   (~9 MB)
```

docker-compose에서 위 디렉터리를 컨테이너 `/models`로 읽기 전용 마운트합니다.
모델 상세 스펙은 `deployment/README.md` 참고.

---

## 서버 시작 방법

```bash
docker compose up -d
```

서버 재시작 후 약 30초 뒤 http://localhost:3000 에서 접근 가능합니다.

상태 확인:
```bash
docker compose ps
docker compose logs frontend --tail=30
```

---

## 구조

```
Ki-67_service/
├── frontend/                   # Next.js 앱
│   ├── app/
│   │   ├── realtime/           # 메인 어노테이션 페이지 (/realtime)
│   │   │   ├── page.tsx        # 전체 UI + ONNX 추론 로직
│   │   │   ├── types.ts
│   │   │   └── utils/
│   │   └── api/onnx/           # 모델 파일 서빙 API 라우트
│   │       ├── encoder/
│   │       └── decoder/
│   └── Dockerfile
├── deployment/                 # ONNX 모델 파일 + Python 추론 예제
├── docker-compose.yml          # frontend + nginx
├── nginx.conf
├── configs/                    # 학습용 Hydra 설정 (서비스 운영과 무관)
├── src/                        # 학습 코드 (서비스 운영과 무관)
└── DATASET.md                  # 학습 데이터셋 설명
```

---

## 트러블슈팅

### 모델 파일 없음 오류

`/api/onnx/encoder` 또는 `/api/onnx/decoder`가 500을 반환하면 모델 파일 경로 확인:

```bash
ls -lh /mnt/Disk1/sylee/deployment/*.onnx
```

파일이 없으면 `deployment/README.md`를 참고해 모델을 준비한 뒤 컨테이너 재시작:

```bash
docker compose restart frontend
```

### 컨테이너 코드 미반영

코드 변경 후에는 이미지 재빌드 필요:

```bash
docker compose build frontend && docker compose up -d
```

### 브라우저에서 ONNX 로드 실패

첫 접속 시 모델 파일(~53 MB)을 다운로드합니다. 느린 네트워크에서는 로딩에 시간이 걸릴 수 있습니다.
`http://localhost:3000`(localhost) 또는 HTTPS 환경에서만 onnxruntime-web WASM이 정상 동작합니다.
사설 IP HTTP(예: `http://10.x.x.x:3000`)에서는 일부 브라우저 API가 제한됩니다.
