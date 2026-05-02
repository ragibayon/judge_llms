#!/bin/bash
#SBATCH --job-name=prometheus-eval
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

CONTAINER="/opt/apptainer/images/huggingface-transformers-pytorch-gpu.sif"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"

export HF_HOME="${HOME}/.cache/huggingface"
mkdir -p "${HF_HOME}"

echo "[$(date)] Starting Prometheus evaluation job"
echo "[$(date)] Project dir: ${PROJECT_DIR}"

apptainer exec \
    --nv \
    --bind "${HOME}:${HOME}" \
    --pwd "${PROJECT_DIR}" \
    "${CONTAINER}" \
    python src/prometheus-eval/prometheus-eval.py

echo "[$(date)] Job finished"
