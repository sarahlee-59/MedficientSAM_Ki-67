# Ki-67 Segmentation Service — 설치 및 사용 가이드

Ki-67 세포 이미지에서 핵(nucleus)을 클릭만으로 세그멘테이션하고 Ki-67 지수를 계산하는 서비스입니다.

---

## 1. 사전 준비물

| 항목 | 확인 명령 | 참고 |
|------|-----------|------|
| Python 3.10+ | `python3 --version` | OpenVINO 추론 서버용 |
| Node.js 20+ (LTS) | `node -v` | Next.js 프론트엔드 빌드/실행용 |
| npm | `npm -v` | |
| git | `git --version` | submodule 포함 clone 필요 |

Node.js가 없다면 nvm으로 설치하는 것을 권장합니다.

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install --lts
```

---

## 2. 저장소 클론 (submodule 포함)

```bash
git clone --recurse-submodules <repo-url> ki67-service
cd ki67-service
```

이미 submodule 없이 clone했다면:

```bash
git submodule update --init --recursive
```

---

## 3. 모델 가중치 준비

`*.bin`, `*.onnx`, `*.pt`, `*.pth`, `*.ckpt` 등 모델 바이너리는 용량 문제로 git에 포함되지 않습니다 (`.gitignore` 참고). GitHub Releases에서 받아 배치합니다.

```bash
# deployment/openvino/ 에 필요한 파일
deployment/openvino/encoder.xml   # 코드에 포함됨
deployment/openvino/encoder.bin   # ← Releases에서 다운로드 필요
deployment/openvino/decoder.xml   # 코드에 포함됨
deployment/openvino/decoder.bin   # ← Releases에서 다운로드 필요
```

다운로드한 `.bin` 파일을 `deployment/openvino/` 디렉터리에 그대로 복사합니다.

> ONNX INT8 버전을 쓸 경우 `deployment/onnx/README.md` 참고.

---

## 4. 추론 서버 (FastAPI + OpenVINO) 의존성 설치

```bash
cd deployment/openvino
pip install -r requirements.txt
cd ../..
```

> 시스템 python을 그대로 쓰는 구성입니다. 격리하고 싶다면 `python3 -m venv .venv-infer && source .venv-infer/bin/activate` 후 위 설치를 진행해도 됩니다.

uvicorn이 설치되어 있는지 확인:

```bash
which uvicorn || pip install uvicorn
```

### 직접 실행 테스트 (systemd 등록 전 검증용)

```bash
cd deployment/openvino
uvicorn server:app --host 0.0.0.0 --port 8000
```

`Ctrl+C`로 종료 후 다음 단계로.

---

## 5. 프론트엔드 (Next.js) 빌드

```bash
cd frontend
npm ci
npm run build
cd ..
```

빌드 결과는 `frontend/.next/standalone/`에 생성됩니다 (systemd 서비스가 이 경로를 실행합니다).

### 직접 실행 테스트

```bash
cd frontend
npm run dev
```

브라우저에서 `http://localhost:3000/realtime` 접속 후 정상 동작 확인.

---

## 6. systemd 서비스 등록 (상시 운영용)

이미 작성된 유닛 파일을 `/etc/systemd/system/`에 복사하되, **경로와 실행 유저를 환경에 맞게 수정**해야 합니다.

```bash
sudo cp ki67-inference.service /etc/systemd/system/
sudo cp ki67-frontend.service /etc/systemd/system/
```

각 유닛 파일에서 아래 항목을 본인 환경에 맞게 수정하세요.

**`ki67-inference.service`**

| 항목 | 의미 | 수정 예 |
|------|------|---------|
| `User=` | 실행 유저 | 본인 계정명 |
| `WorkingDirectory=` | `deployment/openvino`의 절대경로 | `<repo 절대경로>/deployment/openvino` |
| `ExecStart=` | uvicorn 실행 경로 | `which uvicorn` 결과로 교체 |

**`ki67-frontend.service`**

| 항목 | 의미 | 수정 예 |
|------|------|---------|
| `User=` | 실행 유저 | 본인 계정명 |
| `WorkingDirectory=` | `frontend/.next/standalone`의 절대경로 | `<repo 절대경로>/frontend/.next/standalone` |
| `ExecStart=` | node 실행 경로 | `which node` 결과로 교체 (nvm 사용 시 버전별 경로 주의) |

> systemd 유닛 파일은 구조상 절대경로만 지원합니다. 위 두 항목만 예외적으로 절대경로를 사용하세요.

수정 후 등록 및 활성화:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ki67-inference.service
sudo systemctl enable --now ki67-frontend.service
```

---

## 7. 실행 / 관리

### 이미 실행 중인지 확인

```bash
ss -tlnp | grep -E '3000|8000'
```

두 포트가 모두 보이면 바로 접속 → `http://<서버 IP>:3000/realtime`

### 추론 서버 (포트 8000)

```bash
sudo systemctl status ki67-inference
sudo systemctl start ki67-inference
sudo systemctl stop ki67-inference
sudo systemctl restart ki67-inference
journalctl -u ki67-inference -f
```

### 프론트엔드 (포트 3000)

```bash
sudo systemctl status ki67-frontend
sudo systemctl start ki67-frontend
sudo systemctl stop ki67-frontend
sudo systemctl restart ki67-frontend
journalctl -u ki67-frontend -f
```

> 부팅 시 자동 시작됩니다 (`enabled`).

### 접속

```
http://<서버 IP>:3000/realtime
```

> 두 서버 모두 실행 후 5~10초 뒤 접속하세요.

### 코드 변경 후 재배포

최초 설치가 끝난 뒤 코드가 바뀔 때마다 저장소 루트의 `deploy.sh`를 사용합니다.

```bash
./deploy.sh
```

> 스크립트 내부에 `REPO_DIR` 절대경로가 박혀 있으므로, 본인 환경의 clone 경로에 맞게 스크립트 상단을 수정해야 합니다.

---

## 8. 화면 구성

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

## 9. 사용 흐름

### 1단계 — 이미지 업로드

캔버스 중앙의 업로드 영역을 **클릭**하거나 이미지 파일을 **드래그 앤 드롭**합니다.

- 지원 형식: JPG, PNG, BMP, TIFF
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

## 10. 편집

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

## 11. 줌 · 화면 이동

| 조작 | 효과 |
|------|------|
| `Ctrl + 휠` | 줌 인/아웃 (최대 8×) |
| `Ctrl + 좌드래그` | 화면 이동 (팬) |
| **줌 초기화** 버튼 | 원래 배율로 복귀 |

---

## 12. 단축키 전체 요약

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

## 13. 기술 정보

| 항목 | 내용 |
|------|------|
| 추론 엔진 | OpenVINO FP32 (Intel CPU) |
| 추론 성능 | e2e ~123 ms (Intel CPU, 256×256) — 상세 벤치마크는 [`benchmark/speed/benchmark_results.md`](benchmark/speed/benchmark_results.md) 참고 |
| 임베딩 캐시 | 같은 이미지에서 반복 클릭 시 인코더 재실행 없음 |
| 프론트엔드 | Next.js (포트 3000) |
| 추론 서버 | FastAPI + OpenVINO (포트 8000) |

---

## 14. API 엔드포인트 (FastAPI, 포트 8000)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/encode` | 이미지 업로드 → session_id 반환 (인코더 실행) |
| POST | `/decode` | session_id + 좌표 → 마스크 반환 (디코더만 실행) |
| POST | `/infer` | 이미지 + 좌표 → 마스크 (encode+decode 통합 fallback) |

---

## 15. 문제 해결

| 증상 | 확인 사항 |
|------|-----------|
| 8000번 포트 안 뜸 | `journalctl -u ki67-inference -f`, `encoder.bin`/`decoder.bin` 누락 여부 |
| 3000번 포트 안 뜸 | `journalctl -u ki67-frontend -f`, `frontend/.next/standalone` 빌드 여부 |
| 프론트는 뜨는데 추론 실패 | 8000 서버가 같이 떠 있는지, `frontend/app/api/*` 프록시가 8000을 가리키는지 확인 |
| systemd 등록 후 실행 안 됨 | 유닛 파일의 `WorkingDirectory`/`ExecStart`가 절대경로로 올바르게 들어갔는지 확인 (systemd는 상대경로를 지원하지 않음) |
