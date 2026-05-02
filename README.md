## Prometheus Evaluation Pipeline

This repo runs `prometheus-eval/prometheus-7b-v2.0` as a remote judge model and evaluates summarization samples, including the local `summeval` dataset.

The intended workflow is:

1. Host the model on Minsky with `vLLM`
2. Tunnel the remote API back to your laptop
3. Run the local Python scripts against that API
4. Save row-level evaluation outputs to `results/`

## Repo Layout

- `data/download_hf_dataset.py`
  Downloads the Hugging Face dataset currently hardcoded in the script.
- `data/dataset_loader.py`
  Loads supported datasets and normalizes rows for evaluation.
- `src/prometheus-eval/prometheus-eval.py`
  Prometheus evaluator. Supports local `transformers` inference or remote `vLLM` API inference.
- `scripts/minsky-vllm-server.sbatch`
  SLURM job script to host the model on Minsky.
- `scripts/test_vllm_client.py`
  Smoke test against the remote API.
- `scripts/evaluate_summeval_vllm.py`
  Runs sampled `summeval` evaluation and saves outputs.
- `scripts/rich_logger.py`
  Shared console and file logging utilities.

## Local Setup

Install project dependencies:

```bash
uv sync
```

Activate the virtual environment if you want:

```bash
source .venv/bin/activate
```

## Environment Configuration

Create a local `.env` file from `.env.example`.

Example:

```env
PROMETHEUS_API_BASE=http://127.0.0.1:8000
PROMETHEUS_API_KEY=EMPTY
MODEL_ID=prometheus-eval/prometheus-7b-v2.0
PORT=8000
SAMPLE_COUNT=3
OUTPUT_PATH=results/summeval_vllm_results.json
TEST_OUTPUT_PATH=results/test_vllm_client_results.json
```

Important variables:

- `PROMETHEUS_API_BASE`
  Local URL your scripts call. This should match your SSH tunnel port.
- `MODEL_ID`
  Model served by `vLLM` on Minsky.
- `PORT`
  Remote port used by the `vLLM` server on Minsky. Default is `8000`.
- `SAMPLE_COUNT`
  Number of dataset rows to evaluate in `scripts/evaluate_summeval_vllm.py`.
- `OUTPUT_PATH`
  JSON output path for dataset evaluation. A CSV with the same basename is also written.
- `TEST_OUTPUT_PATH`
  JSON output path for the smoke test. A CSV with the same basename is also written.
- `HF_TOKEN`
  Optional but recommended on Minsky for Hugging Face downloads.

## Dataset Usage

### Download the Dataset

The dataset download script is currently hardcoded to:

```python
DATASET_NAME = "KnutJaegersberg/summeval_pairs"
```

Run:

```bash
uv run python data/download_hf_dataset.py
```

This saves the dataset to:

```text
data/datasets/KnutJaegersberg__summeval_pairs
```

### Load the Dataset in Code

Use the dataset loader:

```python
from data.dataset_loader import DatasetLoader

loader = DatasetLoader("summeval", None, 42)
dataset = loader.load_dataset()
samples = loader.sample_posts(dataset, 5)
print(samples[0])
```

For `summeval`, the loader prefers the local downloaded dataset if it exists.

### Normalized Output Shape

For `summeval`, each processed sample includes:

- `id`
- `source`
- `reference`
- `summary`
- `human_scores`
- `baseline_scores`
- `system_id`

The saved evaluation output keeps:

- `id`
- `source`
- `reference`
- `summary`
- `human_scores`
- `system_id`
- `<dimension>_score`
- `<dimension>_feedback`

## Hosting vLLM on Minsky

### 1. Push or Copy the Repo to Minsky

SSH to Minsky and enter the repo directory:

```bash
ssh your-netid@minsky.cs.txstate.edu
cd ~/your-repo
```

### 2. Submit the SLURM Job

Run:

```bash
sbatch scripts/minsky-vllm-server.sbatch
```

This job:

- loads the `vLLM` Apptainer image
- starts `prometheus-eval/prometheus-7b-v2.0`
- serves an OpenAI-compatible API on port `8000`

### 3. Check Job Status

Find the job:

```bash
squeue -u $USER
```

Check accounting:

```bash
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed
```

Read logs:

```bash
tail -f prometheus-vllm_<jobid>.out
tail -f prometheus-vllm_<jobid>.err
```

The key healthy log line is:

```text
Starting vLLM server on http://0.0.0.0:8000
```

## SSH Tunneling

Your local scripts should call the tunnel endpoint, not the remote hostname directly.

### Standard Tunnel on Local Port 8000

```bash
ssh -L 8000:minsky.cs.txstate.edu:8000 your-netid@minsky.cs.txstate.edu
```

Then set:

```env
PROMETHEUS_API_BASE=http://127.0.0.1:8000
```

### If Port 8000 Is Already in Use

Check:

```bash
lsof -i tcp:8000
```

Kill a process if needed:

```bash
kill <pid>
```

Or use another local port, for example `8001`:

```bash
ssh -L 8001:minsky.cs.txstate.edu:8000 your-netid@minsky.cs.txstate.edu
```

Then update `.env`:

```env
PROMETHEUS_API_BASE=http://127.0.0.1:8001
```

### Verify the Tunnel

Run:

```bash
curl http://127.0.0.1:8000/v1/models
```

or if using another local port:

```bash
curl http://127.0.0.1:8001/v1/models
```

If the tunnel works, you should see the served model metadata for:

```text
prometheus-eval/prometheus-7b-v2.0
```

## Changing the API Link

There are three common ways to change the API URL.

### Option 1: Update `.env`

```env
PROMETHEUS_API_BASE=http://127.0.0.1:8001
```

### Option 2: Export in the Shell

```bash
export PROMETHEUS_API_BASE=http://127.0.0.1:8001
```

### Option 3: Pass a Different Local Tunnel Port

If you tunnel `8002` locally:

```bash
ssh -L 8002:minsky.cs.txstate.edu:8000 your-netid@minsky.cs.txstate.edu
```

then set:

```env
PROMETHEUS_API_BASE=http://127.0.0.1:8002
```

## Running the Smoke Test

Run:

```bash
uv run python scripts/test_vllm_client.py
```

This:

- sends one fixed example through all Prometheus evaluation dimensions
- prints scores and feedback with `rich`
- writes a run log to `logs/`
- saves JSON and CSV outputs under `results/`

Output files:

- `logs/test_vllm_client_<timestamp>.log`
- `results/test_vllm_client_results.json`
- `results/test_vllm_client_results.csv`

## Running Dataset Evaluation

Run:

```bash
uv run python scripts/evaluate_summeval_vllm.py
```

To evaluate only one row:

```bash
SAMPLE_COUNT=1 uv run python scripts/evaluate_summeval_vllm.py
```

This script:

- loads local `summeval`
- samples `SAMPLE_COUNT` rows
- evaluates each summary across all Prometheus dimensions
- prints row-level tables and feedback
- saves JSON and CSV outputs

Output files:

- `logs/evaluate_summeval_vllm_<timestamp>.log`
- `results/summeval_vllm_results.json`
- `results/summeval_vllm_results.csv`

## Output Schema

### Smoke Test Output

Each saved row includes:

- `source`
- `summary`
- `domain_task`
- `<dimension>_score`
- `<dimension>_feedback`

### Dataset Evaluation Output

Each saved row includes:

- `index`
- `id`
- `source`
- `reference`
- `summary`
- `human_scores`
- `system_id`
- `coherence_score`
- `coherence_feedback`
- `factual_consistency_score`
- `factual_consistency_feedback`
- `fairness_score`
- `fairness_feedback`
- `fluency_score`
- `fluency_feedback`
- `relevance_score`
- `relevance_feedback`

## Troubleshooting

### `Address already in use`

A local port is already bound.

Check:

```bash
lsof -i tcp:8000
```

Either kill the process or use another local port like `8001`.

### Script Hangs at `Using Prometheus API at ...`

Usually this means:

- the tunnel is not active
- the wrong local port is configured
- the local port is bound by something other than SSH

Verify:

```bash
curl http://127.0.0.1:8000/v1/models
```

or:

```bash
curl http://127.0.0.1:8001/v1/models
```

### Minsky Login Node Shows `No devices were found`

That is expected on the login shell. The GPU is attached to the SLURM job, not your interactive login session.

Use:

```bash
squeue -u $USER
tail -f prometheus-vllm_<jobid>.out
```

to check server status instead.

### Hugging Face Warning About Unauthenticated Requests

Set an `HF_TOKEN` before submitting the job:

```bash
export HF_TOKEN=your_token_here
sbatch scripts/minsky-vllm-server.sbatch
```

## Typical End-to-End Run

### On Minsky

```bash
cd ~/your-repo
sbatch scripts/minsky-vllm-server.sbatch
squeue -u $USER
tail -f prometheus-vllm_<jobid>.out
```

### On Your Laptop

```bash
ssh -L 8000:minsky.cs.txstate.edu:8000 your-netid@minsky.cs.txstate.edu
curl http://127.0.0.1:8000/v1/models
uv run python scripts/test_vllm_client.py
SAMPLE_COUNT=1 uv run python scripts/evaluate_summeval_vllm.py
```
