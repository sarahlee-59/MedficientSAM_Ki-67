#!/bin/bash
# Ki-67 Segmentation Service — 설치 및 배포 스크립트
#
# 최초 설치:             ./deploy.sh --init
# 코드 업데이트 후 재배포: ./deploy.sh
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$REPO_DIR/frontend"
INFER_DIR="$REPO_DIR/deployment/openvino"
MODE="${1:-}"

cd "$REPO_DIR"

if [ "$MODE" = "--init" ]; then
    echo "[init 1/6] git submodule 초기화"
    git submodule update --init --recursive

    echo "[init 2/6] 모델 가중치 확인"
    if [ ! -f "$INFER_DIR/encoder.bin" ] || [ ! -f "$INFER_DIR/decoder.bin" ]; then
        echo "  ! $INFER_DIR/encoder.bin, decoder.bin 이 없습니다."
        echo "    GitHub Releases에서 다운로드하여 $INFER_DIR/ 에 넣은 뒤 다시 실행하세요."
        exit 1
    fi

    echo "[init 3/6] 추론 서버용 venv 생성 + 의존성 설치"
    python3 -m venv "$REPO_DIR/.venv-infer"
    "$REPO_DIR/.venv-infer/bin/pip" install --upgrade pip
    "$REPO_DIR/.venv-infer/bin/pip" install -r "$INFER_DIR/requirements.txt" uvicorn

    echo "[init 4/6] 프론트엔드 의존성 설치 + 빌드"
    cd "$FRONTEND_DIR"
    npm ci
    npm run build
    cd "$REPO_DIR"

    echo "[init 5/6] systemd 서비스 유닛 생성 (이 저장소 경로 기준)"
    NODE_BIN="$(command -v node)"
    UVICORN_BIN="$REPO_DIR/.venv-infer/bin/uvicorn"

    sed -e "s#__REPO_DIR__#$REPO_DIR#g" \
        -e "s#__USER__#$(whoami)#g" \
        -e "s#__NODE_BIN__#$NODE_BIN#g" \
        ki67-frontend.service.template > /tmp/ki67-frontend.service
    sed -e "s#__REPO_DIR__#$REPO_DIR#g" \
        -e "s#__USER__#$(whoami)#g" \
        -e "s#__UVICORN_BIN__#$UVICORN_BIN#g" \
        ki67-inference.service.template > /tmp/ki67-inference.service

    sudo cp /tmp/ki67-frontend.service /etc/systemd/system/ki67-frontend.service
    sudo cp /tmp/ki67-inference.service /etc/systemd/system/ki67-inference.service
    sudo systemctl daemon-reload

    echo "[init 6/6] 서비스 활성화"
    sudo systemctl enable --now ki67-inference.service
    sudo systemctl enable --now ki67-frontend.service

    sleep 3
    sudo systemctl is-active ki67-inference.service
    sudo systemctl is-active ki67-frontend.service
    echo "설치 완료 → http://<서버 IP>:3000/realtime"
    exit 0
fi

echo "[1/4] git pull"
git pull origin main

echo "[2/4] 프론트엔드 빌드"
cd "$FRONTEND_DIR"
npm ci --prefer-offline
npm run build
cd "$REPO_DIR"

echo "[3/4] 서비스 재시작"
sudo systemctl restart ki67-frontend.service
sudo systemctl restart ki67-inference.service

echo "[4/4] 상태 확인"
sleep 3
sudo systemctl is-active ki67-frontend.service
sudo systemctl is-active ki67-inference.service

echo "배포 완료"
