#!/usr/bin/env bash
set -e
export KMP_DUPLICATE_LIB_OK=TRUE
PY=/home/hyw/miniforge3/envs/cellpose/bin/python
ROOT=/home/hyw/chendawei/cellpose-DANN-opencode
OUT=/home/hyw/chendawei/dann_finetune
CACHE_LOG=$OUT/cache_cellprob_2012.log
LOG=$OUT/run_p02_c.log

cd "$ROOT"
echo "[$(date)] waiting cellprob cache" >> "$LOG"
for i in $(seq 1 120); do
  if grep -q "cellprob cache done" "$CACHE_LOG" 2>/dev/null; then break; fi
  sleep 20
done
if ! grep -q "cellprob cache done" "$CACHE_LOG" 2>/dev/null; then
  echo "[$(date)] cache not done; abort" >> "$LOG"
  exit 1
fi

echo "[$(date)] cache done; B (P0-2) start" >> "$LOG"
"$PY" train_dann_2012.py --stage 3 --in-chans 4 --seed 0 --tag _p02_seed0 \
  --init "$OUT/best_s3_seed0.pth" \
  --epochs 120 --eval-every 5 --neg-per-pos 40 --weight-decay 1e-4 \
  --loss ce --patience 8 > "$OUT/dann_s3_p02_seed0.log" 2>&1
echo "[$(date)] B done; C (P0-1+P0-2) start" >> "$LOG"

"$PY" train_dann_2012.py --stage 3 --in-chans 4 --use-gem --use-cbam --use-od-gate \
  --seed 0 --tag _p01p02_seed0 \
  --init "$OUT/best_s3_p01_seed0.pth" \
  --epochs 120 --eval-every 5 --neg-per-pos 40 --weight-decay 1e-4 \
  --loss ce --patience 8 > "$OUT/dann_s3_p01p02_seed0.log" 2>&1
echo "[$(date)] ALL DONE" >> "$LOG"
