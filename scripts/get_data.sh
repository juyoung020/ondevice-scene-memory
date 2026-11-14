#!/usr/bin/env bash
# 문서에서 쓰는 TUM RGB-D 영상을 공식 주소에서 받아 data/tum/ 에 푼다.
#   bash scripts/get_data.sh            # fr1_desk(328 MB) + fr3 long_office_household(1.4 GB)
#   bash scripts/get_data.sh desk       # fr1_desk 만
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=https://cvg.cit.tum.de/rgbd/dataset
SEQS=(freiburg1/rgbd_dataset_freiburg1_desk freiburg3/rgbd_dataset_freiburg3_long_office_household)
[ "${1:-}" = desk ] && SEQS=(freiburg1/rgbd_dataset_freiburg1_desk)
mkdir -p data/tum
cd data/tum
for s in "${SEQS[@]}"; do
  n=$(basename "$s")
  if [ -d "$n" ]; then echo "있음: data/tum/$n"; continue; fi
  curl -fL --retry 3 -C - -o "$n.tgz" "$BASE/$s.tgz"
  tar -xzf "$n.tgz" && rm "$n.tgz"
  echo "풀었음: data/tum/$n"
done
