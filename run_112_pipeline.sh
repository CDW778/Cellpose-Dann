#!/usr/bin/env bash
set -e
export KMP_DUPLICATE_LIB_OK=TRUE
PY=/home/hyw/miniforge3/envs/cellpose/bin/python
ROOT=/home/hyw/chendawei/cellpose-DANN-opencode
OUT=/home/hyw/chendawei/dann_finetune
LOG=$OUT/run_112_pipeline.log

cd "$ROOT"
echo "[$(date)] START 112 pipeline" >> "$LOG"

echo "[$(date)] train 112 start" >> "$LOG"
"$PY" train_dann_2012.py --stage 3 --input-size 112 --seed 0 --tag _112_seed0 \
  --init "$OUT/best_s3_seed0.pth" \
  --epochs 120 --eval-every 5 --neg-per-pos 40 --weight-decay 1e-4 \
  --loss ce --patience 8 > "$OUT/dann_s3_112_seed0.log" 2>&1
echo "[$(date)] train 112 done" >> "$LOG"

echo "[$(date)] eval two DANN start" >> "$LOG"
"$PY" eval_2012_two_dann.py --w224 "$OUT/best_s3_seed0.pth" --w112 "$OUT/best_s3_112_seed0.pth" \
  > "$OUT/eval_112_two_dann.log" 2>&1
echo "[$(date)] ALL DONE" >> "$LOG"
