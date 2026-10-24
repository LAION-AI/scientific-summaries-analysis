#!/bin/bash
#SBATCH --job-name=alexandria
#SBATCH --account=kisski-l3s-llm4sci
#SBATCH --output=/mnt/vast-kisski/projects/kisski-l3s-llm4sci/gollam/logs/alexandria_%x_%j.out
#
# Run ONE Alexandria stage as a batch job. Normally submitted by `alexandria submit` (which sets partition, GPUs,
# CPUs, memory, time and dependencies from configs/site.yaml and the stage registry):
#
#   alexandria submit ws1 features --profile full
#   sbatch slurm/stage.sbatch ENV=main -- ws1 features --profile debug      # by hand
#
# Arguments: KEY=VALUE settings first (ENV=main|llm|train), then `--`, then the `alexandria` command line.
# (Grete sets SBATCH_EXPORT=none, so settings go on the command line, not in the environment.)
set -euo pipefail
ENVNAME=main
while [[ $# -gt 0 && "$1" != "--" ]]; do
  case "$1" in
    ENV=*) ENVNAME="${1#ENV=}" ;;
    *) echo "unknown setting $1 (expected ENV=... then -- )"; exit 2 ;;
  esac
  shift
done
[[ "${1:-}" == "--" ]] && shift

STORE=/mnt/vast-kisski/projects/kisski-l3s-llm4sci/gollam
case "${ENVNAME}" in
  main) PY="${STORE}/envs/Alex/bin/python" ;;           # conda env "Alex"
  llm)  PY="${STORE}/envs/Alex-vllm/bin/python" ;;      # conda env "Alex-vllm" (vLLM)
  train) PY="${STORE}/envs/Alex-train/bin/python" ;;    # conda env "Alex-train" (Alex-vllm + TRL, PEFT, lm-eval; WS4)
  *) echo "unknown ENV ${ENVNAME}"; exit 2 ;;
esac
export HF_HOME="${STORE}/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1   # compute nodes have no internet
export NLTK_DATA="${STORE}/nltk_data"                       # IFEval (lm-eval) needs punkt_tab
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0   # compute nodes have no nvcc: FlashInfer cannot JIT-compile its sampler

echo "== $(date -Is) job ${SLURM_JOB_ID:-} on $(hostname) env=${ENVNAME} cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "== alexandria $*"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
"${PY}" -m alexandria.cli "$@"
