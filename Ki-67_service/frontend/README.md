# Ki-67 Frontend

Ki-67 세그멘테이션 서비스의 Next.js 프론트엔드입니다.

## 실행 방법

### 1. FastAPI 추론 서버 먼저 실행 (필수)

```bash
cd ../deployment/openvino
pip install -r requirements.txt   # 최초 1회
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 2. 프론트엔드 실행

```bash
npm install          # 최초 1회
npm run build
npm start
```

개발 중에는 `npm run dev`로 hot reload 모드로 실행합니다.

## 환경 변수

`.env.local`:

| 변수 | 기본값 | 설명 |
|---|---|---|
| `BACKEND_URL` | `http://localhost:8000` | FastAPI 추론 서버 주소 |

## 추론 흐름

```
브라우저 클릭
    → 이미지 로드 시: POST /api/encode (FormData) → session_id 캐시
    → 클릭마다:      POST /api/decode (session_id + 좌표) → 마스크
    → 세션 없을 때:  POST /api/infer  (FormData, fallback)
Next.js API 라우트 (프록시)
    → FastAPI 추론 서버 (포트 8000, OpenVINO FP32)
브라우저 — 마스크 → 윤곽선 변환 후 캔버스 렌더링
```

## 주요 파일

| 경로 | 역할 |
|---|---|
| `app/realtime/page.tsx` | 메인 어노테이션 UI + 추론 호출 (단일 파일, 컴포넌트 통합) |
| `app/realtime/utils/segmentation.ts` | 마스크 → 윤곽선 변환 유틸 |
| `app/realtime/types.ts` | 공용 타입 정의 |
| `app/api/encode/route.ts` | FastAPI `/encode` 프록시 |
| `app/api/decode/route.ts` | FastAPI `/decode` 프록시 |
| `app/api/infer/route.ts` | FastAPI `/infer` 프록시 (fallback) |
| `app/benchmark/page.tsx` | 브라우저 ONNX 추론 속도 벤치마크 (onnxruntime-web) |
| `public/models/` | ONNX INT8 모델 파일 — benchmark 페이지 전용 |
