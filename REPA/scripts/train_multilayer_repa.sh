#!/usr/bin/env bash
# Fine-tune SiT-L/2 on EuroSAT with one fixed multi-layer SatDINO-B REPA configuration.
# Same arguments as the journal's SatDINO-B run, plus --repa-config. See JOURNAL.md.
#
#   scripts/train_multilayer_repa.sh CONFIG [extra train.py args]
#   CONFIG: a preset A B C D E, or a single pair S<j>T<i> (student block j -> teacher block i), e.g. S8T9
#
# Environment (defaults in brackets):
#   PRETRAINED   ImageNet SiT-L/2 + DINOv2-L enc8 400K checkpoint
#                [/work/Jobs/data/ADTL-Assignment-2/REPA/exps/linear-sit-l-2-dinov2-l-enc8/checkpoints/0400000.pt]
#   EUROSAT      folder with train/ and val/ splits [/work/Jobs/data/eurosat]
#   OUTPUT_DIR   [exps]
#   STEPS        training steps [60000]
#   SAVE_EVERY   checkpoint interval; only the newest checkpoint is kept, and a rerun of this
#                script resumes from it [7500]
#   MICRO_BATCH  per-step batch; gradient accumulation keeps the effective batch at 256 [256]
#                (256 on a full B200; e.g. 32 on a 23 GB MIG slice)
set -euo pipefail

CONFIG=${1:?usage: $0 CONFIG [extra train.py args]  (CONFIG: A-E or S<j>T<i>, e.g. S8T9)}
shift
[[ "$CONFIG" =~ ^([A-E]|S[0-9]+T[0-9]+)$ ]] || { echo "unknown config $CONFIG" >&2; exit 2; }

PRETRAINED=${PRETRAINED:-/work/Jobs/data/ADTL-Assignment-2/REPA/exps/linear-sit-l-2-dinov2-l-enc8/checkpoints/0400000.pt}
EUROSAT=${EUROSAT:-/work/Jobs/data/eurosat}
OUTPUT_DIR=${OUTPUT_DIR:-exps}
STEPS=${STEPS:-60000}
SAVE_EVERY=${SAVE_EVERY:-7500}
if (( STEPS % SAVE_EVERY )); then
  echo "STEPS must be a multiple of SAVE_EVERY so the final step is saved" >&2; exit 2
fi
BATCH=256
MICRO_BATCH=${MICRO_BATCH:-$BATCH}
if (( BATCH % MICRO_BATCH )); then
  echo "MICRO_BATCH must divide $BATCH" >&2; exit 2
fi

cd "$(dirname "$0")/.."
EXP="eurosat-finetune-satdino-b-ml${CONFIG}-sit-l-2"
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
  --repa-config="$CONFIG" \
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
