#!/usr/bin/env bash
set -euo pipefail
export KMP_DUPLICATE_LIB_OK=TRUE

PY=/home/hyw/miniforge3/envs/cellpose/bin/python
REPO=/home/hyw/chendawei/cellpose-DANN-opencode
OOF=/home/hyw/chendawei/dann_finetune/oof_d
IMG=/home/hyw/chendawei/dann_finetune/cv/images_all
MASKS="$OOF/masks_all"
ARGS=(--img-root "$IMG" --mask-dir-train "$MASKS" --mask-dir-val "$MASKS"
      --in-chans 4 --mask-channel --crop-center centroid
      --seed 0 --weight-decay 1e-4 --loss ce --workers 4)

cd "$REPO"
echo "[$(date '+%F %T')] OOF-D start" | tee -a "$OOF/progress.log"

for f in 0 1 2 3; do
  OUT="$OOF/fold$f"
  echo "[$(date '+%F %T')] fold$f stage1" | tee -a "$OOF/progress.log"
  "$PY" train_dann_2012.py --stage 1 --out-dir "$OUT" --epochs 40 \
    --neg-per-pos 20 --patience 25 "${ARGS[@]}" > "$OUT/s1.log" 2>&1

  echo "[$(date '+%F %T')] fold$f stage2" | tee -a "$OOF/progress.log"
  "$PY" train_dann_2012.py --stage 2 --out-dir "$OUT" --epochs 20 \
    --neg-per-pos 20 --patience 25 "${ARGS[@]}" > "$OUT/s2.log" 2>&1

  echo "[$(date '+%F %T')] fold$f stage3" | tee -a "$OOF/progress.log"
  "$PY" train_dann_2012.py --stage 3 --out-dir "$OUT" --epochs 120 \
    --neg-per-pos 40 --patience 25 "${ARGS[@]}" > "$OUT/s3.log" 2>&1

  echo "[$(date '+%F %T')] fold$f dump val scores" | tee -a "$OOF/progress.log"
  "$PY" dump_scores_2012.py --weight "$OUT/best_s3.pth" \
    --cand "$OUT/candidates_val.npz" --img-root "$IMG" --mask-dir "$MASKS" \
    --in-chans 4 --mask-channel --crop-center centroid \
    --out "$OUT/val_preds.npz" --workers 4 > "$OUT/dump.log" 2>&1

  test -s "$OUT/val_preds.npz" || { echo "ERROR: empty val_preds fold$f"; exit 1; }
  touch "$OUT/DONE"
  echo "[$(date '+%F %T')] fold$f done" | tee -a "$OOF/progress.log"
done

echo "[$(date '+%F %T')] pooling OOF" | tee -a "$OOF/progress.log"
"$PY" pool_cv_eval_2012.py --cv-dir "$OOF" > "$OOF/pool.log" 2>&1
touch "$OOF/ALL_DONE"
echo "[$(date '+%F %T')] ALL DONE" | tee -a "$OOF/progress.log"
