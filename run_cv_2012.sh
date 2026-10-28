#!/usr/bin/env bash
set -euo pipefail
export KMP_DUPLICATE_LIB_OK=TRUE
PY=/home/hyw/miniforge3/envs/cellpose/bin/python
REPO=/home/hyw/chendawei/cellpose-DANN-opencode
CV=/home/hyw/chendawei/dann_finetune/cv
COMMON=(--img-root "$CV/images_all" --seed 0 --weight-decay 1e-4 --loss ce --workers 4)
cd "$REPO"
mkdir -p "$CV"

if [ ! -f "$CV/fold0/candidates_train.npz" ]; then
  "$PY" build_cv_folds_2012.py
fi

for f in 0 1 2 3; do
  OUT="$CV/fold$f"
  mkdir -p "$OUT"
  echo "[$(date '+%F %T')] fold$f stage1" | tee -a "$CV/progress.log"
  "$PY" train_dann_2012.py --stage 1 --out-dir "$OUT" --epochs 40 --eval-every 2 \
    --neg-per-pos 20 --patience 8 "${COMMON[@]}" > "$OUT/s1.log" 2>&1
  echo "[$(date '+%F %T')] fold$f stage2" | tee -a "$CV/progress.log"
  "$PY" train_dann_2012.py --stage 2 --out-dir "$OUT" --epochs 40 --eval-every 5 \
    --neg-per-pos 20 --patience 8 "${COMMON[@]}" > "$OUT/s2.log" 2>&1
  echo "[$(date '+%F %T')] fold$f stage3" | tee -a "$CV/progress.log"
  "$PY" train_dann_2012.py --stage 3 --out-dir "$OUT" --epochs 80 --eval-every 5 \
    --neg-per-pos 40 --patience 8 "${COMMON[@]}" > "$OUT/s3.log" 2>&1
  echo "[$(date '+%F %T')] fold$f dump" | tee -a "$CV/progress.log"
  "$PY" dump_val_preds_2012.py --weight "$OUT/best_s3.pth" \
    --cand "$OUT/candidates_val.npz" --img-root "$CV/images_all" \
    --out "$OUT/val_preds.npz" > "$OUT/dump.log" 2>&1
done

echo "[$(date '+%F %T')] pooling OOF" | tee -a "$CV/progress.log"
"$PY" pool_cv_eval_2012.py --cv-dir "$CV" > "$CV/pool.log" 2>&1
THR=$(cat "$CV/cv_best_thr.txt")
EP=$(cat "$CV/cv_median_epoch.txt")
echo "[$(date '+%F %T')] global_thr=$THR median_epoch=$EP" | tee -a "$CV/progress.log"

FINAL="$CV/final"
mkdir -p "$FINAL"
echo "[$(date '+%F %T')] final stage1 (00-03, no-val)" | tee -a "$CV/progress.log"
"$PY" train_dann_2012.py --stage 1 --out-dir "$FINAL" --epochs 40 \
  --neg-per-pos 20 --no-val "${COMMON[@]}" > "$FINAL/s1.log" 2>&1
echo "[$(date '+%F %T')] final stage2 (00-03, no-val)" | tee -a "$CV/progress.log"
"$PY" train_dann_2012.py --stage 2 --out-dir "$FINAL" --epochs 40 \
  --neg-per-pos 20 --no-val --init "$FINAL/last_s1.pth" "${COMMON[@]}" > "$FINAL/s2.log" 2>&1
echo "[$(date '+%F %T')] final stage3 (00-03, no-val, epoch=$EP)" | tee -a "$CV/progress.log"
"$PY" train_dann_2012.py --stage 3 --out-dir "$FINAL" --epochs "$EP" \
  --neg-per-pos 40 --no-val --init "$FINAL/last_s2.pth" "${COMMON[@]}" > "$FINAL/s3.log" 2>&1

echo "[$(date '+%F %T')] test 04 eval (thr=$THR)" | tee -a "$CV/progress.log"
"$PY" eval_2012_dann_p0.py --split test --weight "$FINAL/last_s3.pth" \
  --in-chans 3 --fixed-thr "$THR" > "$CV/eval_test_final.log" 2>&1

echo "[$(date '+%F %T')] ALL DONE thr=$THR epoch=$EP" | tee -a "$CV/progress.log"
touch "$CV/DONE"
