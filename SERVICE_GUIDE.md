# Ki-67 Segmentation Service — 설치 및 사용 가이드

Ki-67 세포 이미지에서 핵(nucleus)을 클릭만으로 세그멘테이션하고 Ki-67 지수를 계산하는 서비스입니다.

이 문서는 우분투에 서비스를 **처음 설치할 때** 해야 할 일과, 설치 후 **실행/사용 방법**을 함께 다룹니다.

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

```bash
# deployment/openvino/ 에 필요한 파일
deployment/openvino/encoder.xml   # 코드에 포함됨
deployment/openvino/encoder.bin   # ← 별도로 받아야 함
deployment/openvino/decoder.xml   # 코드에 포함됨
deployment/openvino/decoder.bin   # ← 별도로 받아야 함
```

다운로드한 `.bin` 파일을 `deployment/openvino/` 디렉터리에 그대로 복사합니다.

> ONNX INT8 버전을 쓸 경우 `deployment/onnx/README.md` 참고.


---

## 4. 설치 — `deploy.sh --init`

clone한 경로를 자동으로 감지해 의존성 설치 → 빌드 → systemd 서비스 등록까지 한 번에 처리합니다. 절대경로를 직접 수정할 필요가 없습니다.

```bash
./deploy.sh --init
```

내부적으로 다음을 순서대로 수행합니다.

1. `git submodule update --init --recursive`
2. 모델 가중치(`encoder.bin`/`decoder.bin`) 존재 확인 — 없으면 안내 메시지를 출력하고 중단
3. `python3 -m venv .venv-infer` 생성 후 그 안에 `pip install -r deployment/openvino/requirements.txt uvicorn` (시스템 python을 건드리지 않는 독립 환경)
4. `frontend/`에서 `npm ci && npm run build`
5. `ki67-frontend.service.template` / `ki67-inference.service.template`의 `__REPO_DIR__`, `__USER__`, `__NODE_BIN__`, `__UVICORN_BIN__`(`.venv-infer/bin/uvicorn`)을 현재 환경 값으로 채워 `/etc/systemd/system/`에 설치
6. `systemctl enable --now`로 두 서비스 활성화

마지막에 두 서비스가 `active`로 출력되면 설치 완료입니다.

> `sudo` 권한이 필요합니다 (systemd 유닛 설치 단계에서 비밀번호를 요구할 수 있습니다). 5~6단계는 `/etc/systemd/system/`에 파일을 쓰고 서비스를 켜는 단계라, 이미 같은 이름의 서비스가 운영 중이라면 잠깐 재시작될 수 있다는 점을 알고 실행하세요.

### (참고) 단계별로 직접 실행하고 싶다면

```bash
# 추론 서버용 독립 Python 환경 (venv) 생성
python3 -m venv .venv-infer
.venv-infer/bin/pip install -r deployment/openvino/requirements.txt uvicorn

# 직접 실행 테스트 (Ctrl+C로 종료)
cd deployment/openvino && ../../.venv-infer/bin/uvicorn server:app --host 0.0.0.0 --port 8000

# 프론트엔드 빌드 + 직접 실행 테스트
cd frontend && npm ci && npm run build
npm run dev   # http://localhost:3000/realtime 에서 확인
```

> venv를 쓰면 이 프로젝트의 Python 패키지가 시스템 전체 python과 분리되어, 다른 프로젝트와 패키지 버전이 충돌할 일이 없습니다.

systemd 유닛을 손으로 등록하려면 `ki67-frontend.service.template`, `ki67-inference.service.template`의 `__REPO_DIR__`(이 저장소의 절대경로), `__USER__`(실행 계정), `__NODE_BIN__`(`which node` 결과), `__UVICORN_BIN__`(`.venv-infer/bin/uvicorn`의 절대경로)를 직접 치환해 `/etc/systemd/system/ki67-frontend.service`, `ki67-inference.service`로 저장한 뒤 `daemon-reload` 하면 됩니다. (systemd 유닛은 구조상 절대경로만 지원하므로 이 두 파일만 예외입니다.)

> ⚠️ 이 가이드는 등록 **방법**을 설명하는 문서입니다. 실제 `sudo systemctl enable --now`는 본인이 직접 실행하세요 — 이미 운영 중인 서비스가 있다면 재시작 타이밍을 본인이 통제하는 것이 안전합니다.

---

## 5. 실행 / 관리

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

최초 설치(`--init`)가 끝난 뒤 코드가 바뀔 때마다 저장소 루트에서 인자 없이 실행합니다.

```bash
./deploy.sh
```

`git pull` → 프론트엔드 재빌드 → 두 systemd 서비스 재시작까지 자동으로 처리합니다. 저장소 경로는 스크립트가 자기 위치를 기준으로 자동 감지하므로 별도 수정이 필요 없습니다.

---

## 6. 화면 구성

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

## 7. 사용 흐름

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

## 8. 편집

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

## 9. 줌 · 화면 이동

| 조작 | 효과 |
|------|------|
| `Ctrl + 휠` | 줌 인/아웃 (최대 8×) |
| `Ctrl + 좌드래그` | 화면 이동 (팬) |
| **줌 초기화** 버튼 | 원래 배율로 복귀 |

---

## 10. 단축키 전체 요약

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

## 11. 기술 정보

| 항목 | 내용 |
|------|------|
| 추론 엔진 | OpenVINO FP32 (Intel CPU) |
| 추론 성능 | e2e ~123 ms (Intel CPU, 256×256) — 상세 벤치마크는 [`benchmark/speed/benchmark_results.md`](benchmark/speed/benchmark_results.md) 참고 |
| 임베딩 캐시 | 같은 이미지에서 반복 클릭 시 인코더 재실행 없음 |
| 프론트엔드 | Next.js (포트 3000) |
| 추론 서버 | FastAPI + OpenVINO (포트 8000) |

---

## 12. API 엔드포인트 (FastAPI, 포트 8000)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/encode` | 이미지 업로드 → session_id 반환 (인코더 실행) |
| POST | `/decode` | session_id + 좌표 → 마스크 반환 (디코더만 실행) |
| POST | `/infer` | 이미지 + 좌표 → 마스크 (encode+decode 통합 fallback) |
