#!/bin/bash
set -e

REPO_DIR="/mnt/Disk1/sylee"
FRONTEND_DIR="$REPO_DIR/frontend"

echo "[1/4] git pull"
cd "$REPO_DIR"
git pull origin main

echo "[2/4] 프론트엔드 빌드"
cd "$FRONTEND_DIR"
npm ci --prefer-offline
npm run build

echo "[3/4] 서비스 재시작"
sudo systemctl restart ki67-frontend.service
sudo systemctl restart ki67-inference.service

echo "[4/4] 상태 확인"
sleep 3
sudo systemctl is-active ki67-frontend.service
sudo systemctl is-active ki67-inference.service

echo "배포 완료"
