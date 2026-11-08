#!/usr/bin/env bash
# Depth-matched SatDINO fine-tuning: start from the ImageNet SiT-L/2 checkpoint trained with
# DINOv2-L REPA at block DEPTH, and fine-tune on EuroSAT with single-layer SatDINO-B REPA at
# the same block (fresh projector). Same arguments as the journal's SatDINO-B run otherwise.
#
#   scripts/train_satdino_depth.sh DEPTH [extra train.py args]        DEPTH in 6 8 10 12 14 16
#
# Environment (defaults in brackets):
#   CKPT_ROOT    folder with linear-sit-l-2-dinov2-l-enc<DEPTH>/checkpoints/0400000.pt
#                [/work/Jobs/data/ADTL-Assignment-2/REPA/exps]
#   EUROSAT      folder with train/ and val/ splits [/work/Jobs/data/eurosat]
#   OUTPUT_DIR   [exps]
#   STEPS        training steps [60000]
#   SAVE_EVERY   checkpoint interval; only the newest checkpoint is kept, and a rerun of this
#                script resumes from it, so a terminated job loses at most SAVE_EVERY steps [7500]
#   MICRO_BATCH  per-step batch; accumulation keeps the effective batch at 256 [256]
set -euo pipefail

DEPTH=${1:?usage: $0 DEPTH [extra train.py args]}
shift
CKPT_ROOT=${CKPT_ROOT:-/work/Jobs/data/ADTL-Assignment-2/REPA/exps}
EUROSAT=${EUROSAT:-/work/Jobs/data/eurosat}
OUTPUT_DIR=${OUTPUT_DIR:-exps}
STEPS=${STEPS:-60000}
SAVE_EVERY=${SAVE_EVERY:-7500}
BATCH=256
MICRO_BATCH=${MICRO_BATCH:-$BATCH}
if (( STEPS % SAVE_EVERY )); then
  echo "STEPS must be a multiple of SAVE_EVERY so the final step is saved" >&2; exit 2
fi
if (( BATCH % MICRO_BATCH )); then
  echo "MICRO_BATCH must divide $BATCH" >&2; exit 2
fi
PRETRAINED=$CKPT_ROOT/linear-sit-l-2-dinov2-l-enc$DEPTH/checkpoints/0400000.pt
[ -f "$PRETRAINED" ] || { echo "missing $PRETRAINED" >&2; exit 2; }

cd "$(dirname "$0")/.."
EXP="eurosat-finetune-satdino-b-enc${DEPTH}-sit-l-2"
LATEST=$(ls "$OUTPUT_DIR/$EXP/checkpoints/" 2>/dev/null | grep -E '^[0-9]{7}\.pt$' | sort | tail -1 || true)
if [ -n "$LATEST" ]; then
  INIT=(--resume-step=$((10#${LATEST%.pt})))
  echo "resuming $EXP from $LATEST"
else
  INIT=(--pretrained-path="$PRETRAINED" --reset-projectors)
fi
exec accelerate launch --num_processes=1 --num_machines=1 --mixed_precision=fp16 --dynamo_backend=no \
  train.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-L/2" \
  --enc-type="satdino-vit-b" \
  --proj-coeff=0.5 \
  --encoder-depth="$DEPTH" \
  --output-dir="$OUTPUT_DIR" \
  --num-classes=10 \
  --resolution=256 \
  --exp-name="$EXP" \
  "${INIT[@]}" \
  --learning-rate=1e-5 \
  --data-dir="$EUROSAT/train" \
  --val-dir="$EUROSAT/val" \
  --max-train-steps="$STEPS" \
  --checkpointing-steps="$SAVE_EVERY" \
  --keep-last-checkpoints=1 \
  --batch-size=$MICRO_BATCH \
  --gradient-accumulation-steps=$((BATCH / MICRO_BATCH)) \
  "$@"
