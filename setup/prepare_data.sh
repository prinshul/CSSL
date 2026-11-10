#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?usage: prepare_data.sh <data root, e.g. /data/med>}
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/downloads" && ROOT=$(cd "$ROOT" && pwd) && cd "$ROOT/downloads"

if [ ! -f CHAOS_Train_Sets.zip ]; then
  wget -c "https://zenodo.org/records/3431873/files/CHAOS_Train_Sets.zip?download=1" -O CHAOS_Train_Sets.zip.part
  mv CHAOS_Train_Sets.zip.part CHAOS_Train_Sets.zip
fi
echo "df21053002a1cc86df918a87da3b2c19  CHAOS_Train_Sets.zip" | md5sum -c -
[ -d chaos_raw ] || { mkdir chaos_raw && unzip -q CHAOS_Train_Sets.zip -d chaos_raw; }
[ -f "$ROOT/chaos/meta.json" ] || python "$REPO/tools/prep_chaos.py" --src chaos_raw --out "$ROOT/chaos"

[ -s monuseg_train.zip ] || gdown 1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA -O monuseg_train.zip
[ -s monuseg_test.zip ] || gdown 1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw -O monuseg_test.zip
[ -d monuseg_raw ] || { mkdir monuseg_raw && unzip -q monuseg_train.zip -d monuseg_raw && unzip -q monuseg_test.zip -d monuseg_raw; }
[ -d "$ROOT/monuseg/test/masks" ] || python "$REPO/tools/prep_monuseg.py" --src monuseg_raw --out "$ROOT/monuseg"

python "$REPO/tools/make_splits.py" --dataset chaos --root "$ROOT/chaos"
python "$REPO/tools/make_splits.py" --dataset monuseg --root "$ROOT/monuseg"
echo "data ready in $ROOT"
