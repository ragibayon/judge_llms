#!/bin/bash
#SBATCH --job-name=prometheus-vllm
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -euo pipefail

CONTAINER="/opt/apptainer/images/vllm-openai-latest.sif"
MODEL_ID="${MODEL_ID:-prometheus-eval/prometheus-7b-v2.0}"
PORT="${PORT:-8000}"

export HF_HOME="${HOME}/.cache/huggingface"
mkdir -p "${HF_HOME}"

echo "[$(date)] Starting vLLM server"
echo "[$(date)] Model: ${MODEL_ID}"
echo "[$(date)] Port: ${PORT}"
echo "[$(date)] Host: $(hostname)"

apptainer exec \
    --nv \
    --bind "${HOME}:${HOME}" \
    "${CONTAINER}" \
    python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_ID}" \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --dtype auto \
        --max-model-len 8192
