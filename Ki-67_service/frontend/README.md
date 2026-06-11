# Ki-67 Frontend

Ki-67 세그멘테이션 서비스의 Next.js 프론트엔드입니다.

## 실행 방법

### 1. 추론 서버 먼저 실행 (필수)

```bash
cd ../deployment/openvino
pip install -r requirements.txt   # 최초 1회
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 2. 프론트엔드 실행

```bash
npm install
npm run dev
```

http://localhost:3000/realtime 으로 접속합니다.

## 환경 변수

`.env.local`:

| 변수 | 기본값 | 설명 |
|---|---|---|
| `BACKEND_URL` | `http://localhost:8000` | FastAPI 추론 서버 주소 |

## 추론 흐름

```
브라우저 클릭
    → POST /api/infer (Next.js 프록시, app/api/infer/route.ts)
    → POST http://BACKEND_URL/infer (FastAPI)
    → (N, H, W) uint8 마스크 반환
    → 브라우저에서 윤곽선 변환 후 캔버스 렌더링
```

## 주요 파일

| 경로 | 역할 |
|---|---|
| `app/realtime/page.tsx` | 메인 어노테이션 UI + 추론 호출 |
| `app/api/infer/route.ts` | FastAPI 프록시 라우트 |
| `app/realtime/utils/segmentation.ts` | 마스크 → 윤곽선 변환 유틸 |
| `app/realtime/types.ts` | 공용 타입 정의 |

## 빌드

```bash
npm run build
npm run start
```
