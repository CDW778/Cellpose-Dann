#!/usr/bin/env bash
set -euo pipefail
export KMP_DUPLICATE_LIB_OK=TRUE

PY=/home/hyw/miniforge3/envs/cellpose/bin/python
REPO=/home/hyw/chendawei/cellpose-DANN-opencode
CV=/home/hyw/chendawei/dann_finetune/mask4_s3_seed2
LOG="$CV/auto_test.log"

exec >>"$LOG" 2>&1
echo "[$(date '+%F %T')] auto test start"

exec 9>"$CV/.auto_test.lock"
flock -n 9 || { echo "another instance, exit"; exit 0; }

if [ -f "$CV/TEST_DONE" ]; then
  echo "TEST_DONE already exists, skip"
  exit 0
fi

# 等待训练完成，最多 8 小时
for _ in $(seq 1 480); do
  [ -f "$CV/DONE" ] && break
  sleep 60
done

if [ ! -f "$CV/DONE" ]; then
  echo "ERROR: DONE not found after 8h"
  exit 1
fi

EP=$(tr -d '[:space:]' < "$CV/best_epoch_s3.txt")
if [ -z "$EP" ]; then
  echo "ERROR: best_epoch_s3.txt empty"
  exit 1
fi

THR=$(grep -E "^\[ep ${EP}/" "$CV/s3.log" | tail -n 1 \
      | sed -n 's/.*@thr=\([0-9.]*\).*/\1/p')

if [ -z "$THR" ]; then
  THR=$(grep '\*saved\*' "$CV/s3.log" \
        | sed -n 's/.*F1=\([0-9.]*\).*@thr=\([0-9.]*\).*/\1 \2/p' \
        | sort -k1,1nr | head -n 1 | awk '{print $2}')
fi

if [ -z "$THR" ]; then
  echo "ERROR: cannot parse val threshold"
  exit 1
fi

echo "best_epoch=$EP val_threshold=$THR"

cd "$REPO"
"$PY" eval_2012_dann_p0.py \
  --split test \
  --weight "$CV/best_s3.pth" \
  --in-chans 4 \
  --mask-channel \
  --crop-center centroid \
  --fixed-thr "$THR" \
  > "$CV/eval_test_seed2.log" 2>&1

grep -q '^fixed' "$CV/eval_test_seed2.log" || {
  echo "ERROR: eval did not produce fixed line"
  exit 1
}

touch "$CV/TEST_DONE"
echo "[$(date '+%F %T')] test done"
tail -n 5 "$CV/eval_test_seed2.log"
