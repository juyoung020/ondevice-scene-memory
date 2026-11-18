#!/usr/bin/env bash
# PC(WSL)에서: 보드에 남긴 녹화(보정 녹화 ~/imucal/run_* 와 board_all.sh 가 찍은 ~/board_rec/*)와 측정 결과(~/board_all/*)를 가져온다.
#   bash scripts/board_fetch.sh [보드=kau@192.168.219.101] [받을 곳=data/board]
# 먼저 크기를 보여 주고 rsync 로 받는다(이어 받기 · 바뀐 것만). 받은 녹화는 `ondevice slam --rsdrec <폴더> --imu-csv <폴더>/imu0.csv` 로 재생
set -euo pipefail
BOARD=${1:-kau@192.168.219.101}
DST=${2:-$(cd "$(dirname "$0")/.." && pwd)/data/board}
mkdir -p "$DST"
echo "보드에 있는 것:"
ssh "$BOARD" 'du -sh ~/imucal/run_* ~/board_rec/* ~/board_all/* 2>/dev/null; echo; du -shc ~/imucal ~/board_rec ~/board_all 2>/dev/null | tail -1'
rsync -a --info=progress2 --partial "$BOARD:imucal" "$BOARD:board_rec" "$BOARD:board_all" "$DST/" || true
echo "받음: $DST"; du -sh "$DST"/* 2>/dev/null
