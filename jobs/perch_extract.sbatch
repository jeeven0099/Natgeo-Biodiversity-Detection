#!/bin/bash
# Optional. The whole corpus runs in ~22 min on a laptop RTX via
#   python scripts/extract_perch.py --model-name perch_v2_onnx --device cuda
# so this exists for the case where that machine is unavailable, or for a much
# larger corpus. See README, "GPU setup".
#SBATCH --job-name=perch-extract
#SBATCH --output=logs/perch_extract_%j.out
#SBATCH --error=logs/perch_extract_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# TODO: set the partition for your cluster, e.g. #SBATCH --partition=gpu

set -euo pipefail
mkdir -p logs

# TODO: replace with this cluster's module names.
# module load cuda/12.2 cudnn
# source activate natgeo-perch

PROJECT="${PROJECT:-$PWD}"

# perch_hoplite pulls the weights from Kaggle on first use and caches them.
# If the compute nodes have no outbound network, do the download once on a
# login node with the same KAGGLEHUB_CACHE and it will be a cache hit here.
export KAGGLEHUB_CACHE="${KAGGLEHUB_CACHE:-$PROJECT/.cache/kagglehub}"

# ~33k windows at 5 s. Run SMOKE=1 first: it does 20 clips and, more usefully,
# prints the model's head names so the class-list assumptions get checked
# before committing to the full pass.
python "$PROJECT/scripts/extract_perch.py" \
  --root "$PROJECT" \
  --model-name "${MODEL_NAME:-perch_v2}" \
  ${MODEL_PATH:+--model-path "$MODEL_PATH"} \
  --outdir "$PROJECT/data/perch" \
  --batch "${BATCH:-64}" \
  ${SMOKE:+--limit 20}

# Scoring is CPU-only and needs no GPU; run it here for convenience.
python "$PROJECT/scripts/score_zeroshot.py" --root "$PROJECT"
