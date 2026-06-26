# Ki-67 Segmentation Service — 설치 및 사용 가이드

Ki-67 세포 이미지에서 핵(nucleus)을 클릭만으로 세그멘테이션하고 Ki-67 지수를 계산하는 서비스입니다.

---

## 1. 사전 준비물

| 항목 | 확인 명령 |
|------|-----------|
| Python 3.10+ | `python3 --version` |
| Node.js 20+ (LTS) | `node -v` |
| npm | `npm -v` |
| git | `git --version` |

**없다면:**

```bash
# Python 3 + venv
sudo apt update && sudo apt install -y python3 python3-venv

# git
sudo apt install -y git

# Node.js (nvm 권장)
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

```
deployment/openvino/
├── encoder.xml   # 저장소에 포함
├── encoder.bin   # ← 별도 제공 파일
├── decoder.xml   # 저장소에 포함
└── decoder.bin   # ← 별도 제공 파일
```

제공받은 `.bin` 파일을 `deployment/openvino/` 에 복사합니다.

> 릴리즈 가중치 원본은 NAS의 `Archived/weight/DP_SAM/Released/openvino` 에 보관되어 있습니다.

> ONNX INT8 버전을 쓸 경우 `deployment/onnx/README.md` 참고.

---

## 4. 설치 — `deploy.sh --init`

```bash
chmod +x deploy.sh
./deploy.sh --init
```

> `sudo` 권한이 필요합니다 (systemd 유닛 설치 시 비밀번호 요구).

내부 동작 순서:

1. `git submodule update --init --recursive`
2. `encoder.bin` / `decoder.bin` 존재 확인 — 없으면 안내 후 중단
3. `.venv-infer/` 생성 후 `pip install -r deployment/openvino/requirements.txt uvicorn`
4. `frontend/`에서 `npm ci && npm run build`
5. systemd 유닛(`ki67-frontend.service`, `ki67-inference.service`) 설치 및 `enable --now`

마지막에 두 서비스가 `active`로 출력되면 설치 완료입니다.

### (참고) 단계별 수동 실행

```bash
# 추론 서버 (포트 8000)
python3 -m venv .venv-infer
.venv-infer/bin/pip install -r deployment/openvino/requirements.txt uvicorn
cd deployment/openvino
../../.venv-infer/bin/uvicorn server:app --host 0.0.0.0 --port 8000

# 프론트엔드 (포트 3000, 별도 터미널)
cd frontend && npm ci && npm run build && npm start
```

systemd 유닛을 직접 등록하려면 `ki67-frontend.service.template`, `ki67-inference.service.template`의 `__REPO_DIR__`(저장소 절대경로), `__USER__`(실행 계정), `__NODE_BIN__`(`which node`), `__UVICORN_BIN__`(`.venv-infer/bin/uvicorn` 절대경로)를 치환해 `/etc/systemd/system/`에 저장 후 `systemctl daemon-reload`하세요.

---

## 5. 실행 / 관리

### 상태 확인

```bash
ss -tlnp | grep -E '3000|8000'
```

두 포트가 모두 보이면 → `http://<서버 IP>:3000/realtime`

> 포트가 막혀 있다면: `sudo ufw allow 3000 && sudo ufw allow 8000`

### 서비스 관리

```bash
# 추론 서버 (포트 8000)
sudo systemctl status  ki67-inference   # 현재 상태(active/failed 등) 확인
sudo systemctl start   ki67-inference   # 서비스 시작
sudo systemctl stop    ki67-inference   # 서비스 중지
sudo systemctl restart ki67-inference   # 재시작 (코드·설정 변경 후)
journalctl -u ki67-inference -f         # 실시간 로그 스트리밍

# 프론트엔드 (포트 3000)
sudo systemctl status  ki67-frontend    # 현재 상태(active/failed 등) 확인
sudo systemctl start   ki67-frontend    # 서비스 시작
sudo systemctl stop    ki67-frontend    # 서비스 중지
sudo systemctl restart ki67-frontend    # 재시작 (코드·설정 변경 후)
journalctl -u ki67-frontend -f          # 실시간 로그 스트리밍
```

> 부팅 시 자동 시작됩니다 (`enabled`).

### 코드 변경 후 재배포

```bash
./deploy.sh
```

`git pull` → 프론트엔드 재빌드 → 두 서비스 재시작을 자동 처리합니다.

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
