# Ki-67 Frontend

Ki-67 세그멘테이션 서비스의 Next.js 프론트엔드입니다.

실행 방법은 [`Ki-67_service/README.md`](../README.md) 참고.

## 환경 변수

`.env.local`:

| 변수 | 기본값 | 설명 |
|---|---|---|
| `BACKEND_URL` | `http://localhost:8000` | FastAPI 추론 서버 주소 |

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

## 개발 모드

```bash
npm run dev   # hot reload
```
