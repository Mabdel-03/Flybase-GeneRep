#!/bin/bash
#SBATCH -J gene_rep_env
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --no-requeue
#SBATCH --mail-type=END,FAIL

# Create the dedicated GENE_REP_ENV conda env for the protein<->text CLIP stage.
#
# Runs on a CPU partition (no --gres=gpu): env creation is pure conda/pip work.
# Putting it on a contended GPU partition caused repeated preemption in the Cel
# Rep stage -> with --requeue it latched into user_env_retrieval_failed holds,
# and with --no-requeue preemption silently killed it. A CPU partition has no
# such contention. The CUDA-visibility check below is a non-fatal WARNING (no GPU
# on a CPU node); the GPU ESM job is the real first CUDA user.
#
# This single env covers: the ESM-2 embedding pass, all FlyBase data fetch/prep,
# and eval-label fetching. BioBERT text embedding can alternatively run in
# cel_rep. torch is pinned to a CUDA-matched wheel (cu121, covers L40S sm_89 /
# H200 sm_90) since the plain `pip install torch` can resolve to CPU-only.
#
# Idempotent: if the env already imports torch + esm + transformers, exit early.
#
# Submit from the repo root:
#   cd /orcd/data/lhtsai/001/mabdel03/Flybase
#   sbatch --export=ALL,REPO_ROOT="$(pwd)" -p mit_normal \
#     --output="<space-free-log-dir>/gene_rep_env_%j.out" \
#     "3 - Gene Rep/model_training/protein_embedding_models/sbatch_make_env.sh"

set -euo pipefail

if [[ -n "${REPO_ROOT:-}" && -f "${REPO_ROOT}/config/paths.sh" ]]; then
    :
else
    _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${_SCRIPT_DIR}/../../.." && pwd)"
fi
source "${REPO_ROOT}/config/paths.sh"

export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node:   ${SLURMD_NODENAME:-$(hostname)}"
echo "Start:  $(date)"
echo "GENE_REP_ENV: ${GENE_REP_ENV}"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"

set +u
source "${CONDA_INIT_SCRIPT}"
set -u

MAMBA="$(dirname "${CONDA_INIT_SCRIPT}")/../../bin/mamba"
if [[ ! -x "${MAMBA}" ]]; then MAMBA="mamba"; fi
echo "Using mamba: ${MAMBA}"

# Call the env's OWN python/pip by ABSOLUTE PATH — never `conda run -p ... pip`.
# On a node where another env (e.g. consortium) is base-active, `conda run` can
# leak PYTHONPATH / user-site so pip resolves "already satisfied" against the
# WRONG env and installs nothing into gene_rep. Absolute-path pip + a scrubbed
# PYTHONPATH/PYTHONNOUSERSITE avoids that entirely.
PYBIN="${GENE_REP_ENV}/bin/python"
PIPBIN="${GENE_REP_ENV}/bin/pip"
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# ── Idempotent early-exit: working torch + esm + transformers already present ──
if [[ -x "${PYBIN}" ]]; then
    if "${PYBIN}" -c "import torch, esm, transformers" 2>/dev/null; then
        echo "[env] gene_rep already has torch+esm+transformers — skipping create."
        "${PYBIN}" -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
        exit 0
    fi
    echo "[env] gene_rep exists but is incomplete — will (re)install packages into it."
fi

# ── Create env + install packages ─────────────────────────────────────────────
if [[ ! -x "${PYBIN}" ]]; then
    echo "[env] creating ${GENE_REP_ENV} (python=3.11)..."
    "${MAMBA}" create -y -p "${GENE_REP_ENV}" python=3.11
fi

echo "[env] installing CUDA-matched torch (cu121)..."
"${PIPBIN}" install --no-input --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121

echo "[env] installing the rest (esm + transformers + prep/eval stack)..."
"${PIPBIN}" install --no-input --no-cache-dir \
    fair-esm transformers scikit-learn scipy tqdm matplotlib pandas \
    pyarrow biopython requests numpy

# ── Verify ────────────────────────────────────────────────────────────────────
echo "[env] verifying imports + CUDA visibility..."
"${PYBIN}" - <<'PY'
import torch, esm, transformers, sklearn, numpy, pandas, matplotlib
import Bio, requests, pyarrow
print("torch       :", torch.__version__)
print("cuda avail  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu         :", torch.cuda.get_device_name())
print("fair-esm     : import OK (esm has ESM-2 loaders:", hasattr(esm.pretrained, "esm2_t36_3B_UR50D"), ")")
print("transformers:", transformers.__version__)
print("sklearn     :", sklearn.__version__)
print("biopython   :", Bio.__version__)
# This build runs on a CPU node, so cuda.is_available() is expected False here.
# Verify the wheel is a CUDA build so GPU jobs actually use the GPU.
print("torch.version.cuda:", torch.version.cuda)
if torch.version.cuda is None:
    print("WARNING: torch wheel has no CUDA runtime — GPU jobs would fall back to CPU!")
else:
    print("CUDA-enabled torch wheel OK (cuda runtime:", torch.version.cuda, ")")
print("ALL GOOD")
PY

echo "[env] done: $(date)"
