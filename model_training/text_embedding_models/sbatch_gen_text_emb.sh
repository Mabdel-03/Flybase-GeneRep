#!/bin/bash
#SBATCH -J biobert_text_emb
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --mail-type=END,FAIL

# BioBERT embeddings of FlyBase gene descriptions (Stage 2b). Small model; a GPU
# makes it a few minutes but CPU also works. Uses GENE_REP_ENV's python (has a
# torch>=2.6-safe safetensors BioBERT mirror).
#
#   cd /orcd/data/lhtsai/001/mabdel03/Flybase
#   L=/orcd/scratch/orcd/012/mabdel03/gene_rep/logs
#   sbatch --export=ALL,REPO_ROOT="$(pwd)" -p mit_normal_gpu \
#     --output="$L/text_emb_%j.out" --error="$L/text_emb_%j.out" \
#     "3 - Gene Rep/model_training/text_embedding_models/sbatch_gen_text_emb.sh"

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
SCRIPT="${FLY_GENE_REP_TRAINING}/text_embedding_models/gen_gene_text_emb.py"

echo "Job ${SLURM_JOB_ID:-interactive} on ${SLURMD_NODENAME:-$(hostname)} @ $(date)"
echo "model: ${GENE_REP_TEXT_MODEL}"

"${PYBIN}" "${SCRIPT}" --model "${GENE_REP_TEXT_MODEL}" --pool cls "$@"

echo "[text] done: $(date)"
