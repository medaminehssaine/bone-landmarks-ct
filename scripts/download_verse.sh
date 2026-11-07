#!/usr/bin/env bash
# Download VerSe'20 (restructured, subject-based) and the Genant fracture grades.
#
#   bash scripts/download_verse.sh          # full dataset (~38 GB of zips)
#   bash scripts/download_verse.sh dev      # derivatives of all splits + 3 training CTs
#
# Data: CC BY-SA 4.0 - https://github.com/anjany/verse
# Note: VerSe'20 cases that were already in VerSe'19 (ids < 500) are NOT in these zips;
# they live in the VerSe'19 subject-based release. This project uses the VerSe'20 zips only.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=https://s3.bonescreen.de/public/VerSe-complete
OUT=data/verse20
PY=${PYTHON:-python}

mkdir -p data/fracture "$OUT"
curl -fL -o data/fracture/Genant_Fractures_thoracolumbar_reference.csv \
  https://osf.io/download/694557530f0cd6a1041594bf/

if [[ "${1:-full}" == "dev" ]]; then
  for s in training validation test; do
    $PY scripts/fetch_verse.py --split "$s" --part derivatives --out "$OUT"
  done
  $PY scripts/fetch_verse.py --split training --part ct --n-random 3 --seed 42 --out "$OUT"
  exit 0
fi

mkdir -p data/zips
for s in training validation test; do
  wget -c -P data/zips "$BASE/dataset-verse20$s.zip"
  unzip -q -n "data/zips/dataset-verse20$s.zip" -d data/zips/tmp -x "__MACOSX/*" "*.DS_Store"
done
mv data/zips/tmp/dataset-01training   "$OUT/01_training"
mv data/zips/tmp/dataset-02validation "$OUT/02_validation"
mv data/zips/tmp/dataset-03test       "$OUT/03_test"
rmdir data/zips/tmp
echo "done -> $OUT"
