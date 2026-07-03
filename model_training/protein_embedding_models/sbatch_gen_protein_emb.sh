#!/bin/bash
#SBATCH -J esm_protein_emb
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --mail-type=END,FAIL

# NOTE on --mem: ESM-2 3B (~11 GB) lives in GPU VRAM; host RAM only holds the
# gene table + output matrix (13,986 x 2560 float32 ~= 143 MB) + batch tensors,
# so 24G is ample. Keep it modest to stay under QOSMaxMemoryPerUser.

# ESM-2 3B protein embeddings for every protein-coding gene (Stage 2a).
#
# 3B weights (~11 GB) load in fp16; a big-VRAM GPU is preferred (A100 on
# mit_preemptable, or an L40S/H200). Length-bucketed batching keeps a few long
# fly proteins (~1.5k are >1022 aa, truncated) from OOMing.
#
# Uses GENE_REP_ENV's python by ABSOLUTE PATH (env-retrieval robustness). Submit
# from the repo root with a SPACE-FREE log dir (the "3 - Gene Rep" path has
# spaces, which trips SLURM env retrieval):
#
#   cd /orcd/data/lhtsai/001/mabdel03/Flybase
#   L=/orcd/scratch/orcd/012/mabdel03/gene_rep/logs
#   sbatch --export=ALL,REPO_ROOT="$(pwd)" -p mit_preemptable \
#     --output="$L/esm_emb_%j.out" --error="$L/esm_emb_%j.out" \
#     "3 - Gene Rep/model_training/protein_embedding_models/sbatch_gen_protein_emb.sh"

set -euo pipefail

if [[ -n "${REPO_ROOT:-}" && -f "${REPO_ROOT}/config/paths.sh" ]]; then
    :
else
    _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${_SCRIPT_DIR}/../../.." && pwd)"
fi
source "${REPO_ROOT}/config/paths.sh"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH
PYBIN="${GENE_REP_ENV}/bin/python"
SCRIPT="${FLY_GENE_REP_TRAINING}/protein_embedding_models/gen_protein_emb.py"

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-interactive}  Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Start:  $(date)"
echo "model:  ${GENE_REP_ESM_MODEL}"
echo "python: ${PYBIN}"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "(no nvidia-smi)"

"${PYBIN}" "${SCRIPT}" \
    --model "${GENE_REP_ESM_MODEL}" \
    --max-residues 1022 \
    --max-tokens 16384 \
    --max-batch 32 \
    "$@"

echo "[esm] done: $(date)"
