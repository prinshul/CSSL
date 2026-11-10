#!/usr/bin/env bash
set -uo pipefail
REPO=$(cd "$(dirname "$0")" && pwd)
cd "$REPO" || exit 1
DATA=${DATA:?set DATA=<data root, e.g. /data/med>}
STAGES=${STAGES:-"runs"}
DATASETS=${DATASETS:-"monuseg chaos"}
PAR=${PAR:-4}
E=$REPO/exp
LOGS=$E/paper_logs
WLIST=$LOGS/wave_$(echo "${DATASETS:-monuseg chaos}" | tr ' ' '_').list
mkdir -p "$LOGS"
export DATA E LOGS WLIST OMP_NUM_THREADS=1 DEVICE=${DEVICE:-cuda} EXTRA_OPTS=${EXTRA_OPTS:-}
has() { [[ " $DATASETS " == *" $1 "* ]]; }

if [[ " $STAGES " == *" setup "* ]]; then
  bash setup/prepare_data.sh "$DATA" || exit 1
  if [ ! -f "$DATA/chaos_dafnet/meta.json" ]; then
    python tools/prep_chaos_dafnet.py --src "$DATA/downloads/chaos_raw" --out "$DATA/chaos_dafnet" || exit 1
  fi
  [ -f splits/chaos/dafnet2.json ] || python tools/make_splits.py --dataset chaos --protocol dafnet \
    --root "$DATA/chaos_dafnet" || exit 1
  [ -f splits/monuseg/seed2.json ] || python tools/make_splits.py --dataset monuseg --root "$DATA/monuseg" || exit 1
  echo "setup done: $DATA/monuseg, $DATA/chaos_dafnet, splits/"
fi

one() {
  local out=$1 cfg=$2 frac=$3 k=$4; shift 4
  local droot=$DATA/monuseg name
  [[ $cfg == chaos* ]] && droot=$DATA/chaos_dafnet
  name=$(echo "${out#$REPO/}" | tr '/' '_')
  if [ -f "$out/results.json" ]; then echo "$(date +%H:%M) done     ${out#$REPO/}  [$(progress)]"; return; fi
  echo "$(date +%H:%M) start    ${out#$REPO/}"
  python train.py --device "${DEVICE:-cuda}" --config "configs/$cfg.yaml" --split-seed "$k" --label-frac "$frac" \
    --save-path "$out" --opts data_root="$droot" ${EXTRA_OPTS:-} "$@" > "$LOGS/$name.log" 2>&1
  if [ -f "$out/results.json" ]; then
    rm -f "$out/latest.pth"
    echo "$(date +%H:%M) finished ${out#$REPO/}  $(python tools/print_results.py --one "$out")  [$(progress)]"
  else echo "$(date +%H:%M) FAILED   ${out#$REPO/} (log: $LOGS/$name.log)  [$(progress)]"; fi
}
export -f one
export REPO
progress() {
  local n=0 d
  while read -r d; do [ -f "$d/results.json" ] && n=$((n + 1)); done < "$WLIST"
  echo "$WNAME: $n/$(wc -l < "$WLIST") runs done"
}
export -f progress
wave() {
  export WNAME=$1; shift
  printf '%s\n' "$@" | grep -v '^$' | awk '{print $2}' > "$WLIST"
  echo "== $(date +%H:%M)  $WNAME: $(wc -l < "$WLIST") runs, $PAR at a time"
  printf '%s\n' "$@" | grep -v '^$' | xargs -P "$PAR" -I{} bash -c '{}'
}
avg_lam() {
  python - "$1/metrics.jsonl" <<'PY'
import json, math, sys
v = [r["lam"] for r in map(json.loads, open(sys.argv[1])) if math.isfinite(r.get("lam", float("nan")))]
print("%.4f" % (sum(v) / len(v)))
PY
}
iters() { python -c "import yaml; print(yaml.safe_load(open('configs/$1.yaml'))['total_iters'])"; }
NONE="cssl.cl.enable=false cssl.cr.enable=false cssl.cr.fallback=none"
NOCR="cssl.cr.enable=false cssl.cr.fallback=none"
ADD="cssl.cr.mode=add cssl.cr.add_weight=0.5"
M=monuseg_selfloop; C=chaos_dafnet

CM="$ADD cssl.cl.embed_from=image cssl.cl.embed_level=-3 cssl.cl.queue=4096 cssl.cl.boundary=2"
CC="$ADD"

if [[ " $STAGES " == *" runs "* ]]; then
  echo "MoNuSeg CSSL configuration: $CM"
  echo "CHAOS CSSL configuration:   $CC"
  W1=()
  W2=()
  abl() {
    local cfg=$1 frac=$2 key=$3 k=$4 ch=$5 T="$E/sabl_%s/${3}_s$4"
    W1+=("one $(printf "$T" no_cl) $cfg $frac $k $ch cssl.cl.enable=false"
         "one $(printf "$T" no_cr) $cfg $frac $k $ch $NOCR"
         "one $(printf "$T" no_hnm) $cfg $frac $k $ch cssl.cl.hard_ratio=0")
    [[ " $ch " == *" cssl.cl.queue="* ]] && W1+=("one $(printf "$T" no_mem) $cfg $frac $k $ch cssl.cl.queue=0")
    [[ " $ch " == *" cssl.cl.boundary="* ]] && W1+=("one $(printf "$T" no_bnd) $cfg $frac $k $ch cssl.cl.boundary=0")
    [[ " $ch " == *" cssl.cl.embed_from=image"* ]] && \
      W1+=("one $(printf "$T" no_v2) $cfg $frac $k $ch cssl.cl.embed_from=patch cssl.cl.embed_level=-1")
  }
  if has monuseg; then
    IT=$(iters $M)
    for k in 0 1 2; do
      W1+=("one $E/cssl_strongest/monuseg_50_s$k $M 50 $k $CM"
           "one $E/mabl_none/monuseg_50_s$k $M 50 $k $NONE")
    done
    W1+=("one $E/main_sup/monuseg_50_s0 $M 50 0 warmup_iters=$IT"
         "one $E/sup100/monuseg_100_s0 $M 100 0 warmup_iters=$IT")
    abl $M 50 monuseg_50 0 "$CM"
  fi
  if has chaos; then
    IT=$(iters $C)
    for k in 0 1 2; do
      W1+=("one $E/cssl_strongest/chaos_25_s$k $C 25 $k $CC"
           "one $E/cabl_none/chaos_25_s$k $C 25 $k $NONE"
           "one $E/main_sup/chaos_25_s$k $C 25 $k warmup_iters=$IT")
    done
    abl $C 25 chaos_25 1 "$CC"
  fi
  wave "wave 1/2" "${W1[@]}"

  if has monuseg; then
    W2+=("one $E/sabl_fixlam/monuseg_50_s0 $M 50 0 $CM cssl.cr.fixed_lambda=$(avg_lam "$E/cssl_strongest/monuseg_50_s0")")
  fi
  if has chaos; then
    W2+=("one $E/sabl_fixlam/chaos_25_s1 $C 25 1 $CC cssl.cr.fixed_lambda=$(avg_lam "$E/cssl_strongest/chaos_25_s1")")
  fi
  wave "wave 2/2" "${W2[@]}"
fi

if [[ " $STAGES " == *" runs "* ]]; then
  python tools/audit.py --data "$DATA" --exp "$E" > "$E/audit.txt" 2>&1; tail -2 "$E/audit.txt"
fi
python tools/print_results.py --exp "$E"
