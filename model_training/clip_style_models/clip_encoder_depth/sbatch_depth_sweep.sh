#!/bin/bash
#SBATCH -J gene_clip_sweep
#SBATCH --array=0-8%6
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --mail-type=END,FAIL

# GeneCLIP depth x train-fraction sweep — Stage 3 dispatcher.
#
# 3 depths x 3 train-fracs = 9 cells, flattened into one SLURM array, all at the
# fixed test@42 / train@seed protocol. Each array task is an independent,
# resumable, single-GPU cell that trains the ESM<->BioBERT CLIP head on the
# shared clip_inputs_dmel.npz cache via gene_train_clip.py (which imports the Cel
# Rep Adapter/clip_loss/evaluate verbatim). The head is tiny, so cells are cheap.
#
# Idempotent resume: a cell is skipped if its metrics_<tag>.json exists (written
# LAST by gene_train_clip.py).
#
# Grid axis order (aggregate_gene_sweep.py mirrors this):
#   DI = IDX / 3   -> DEPTHS[DI]      (depth)
#   FI = IDX % 3   -> FRACS[FI]       (train fraction)
#
# CRITICAL: submit with SPACE-FREE --output/--error log paths and invoke python by
# absolute path ("${GENE_REP_ENV}/bin/python") — SLURM user-env retrieval breaks
# on the repo's spaced "3 - Gene Rep" paths. submit.sh does this for you.
#
# DRY_RUN=1 prints the (depth, frac, tag) mapping for every index and exits.

set -euo pipefail

if [[ -n "${REPO_ROOT:-}" && -f "${REPO_ROOT}/config/paths.sh" ]]; then
    :
else
    _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${_SCRIPT_DIR}/../../../.." && pwd)"
fi
source "${REPO_ROOT}/config/paths.sh"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

PY="${GENE_REP_ENV}/bin/python"
TRAIN="${FLY_GENE_REP_TRAINING}/clip_style_models"
CACHE="${GENE_REP_CLIP_CACHE:-${FLY_GENE_REP_DATA}/derived/clip_inputs_dmel.npz}"
RUNS="${GENE_SWEEP_OUTDIR:-${FLY_GENE_REP_EVALUATIONS}/clip_style_models/clip_encoder_depth/outputs/runs}"
mkdir -p "${RUNS}"

# ── grid (single source of truth) ──────────────────────────────────────────────
DEPTHS=(2 6 12)
FRACS=(0.50 0.80 0.90)
NF=${#FRACS[@]}
N=$(( ${#DEPTHS[@]} * NF ))

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "grid: ${#DEPTHS[@]} depths x ${NF} fracs = ${N} cells | cache=${CACHE}"
    for (( IDX=0; IDX<N; IDX++ )); do
        DI=$(( IDX / NF )); FI=$(( IDX % NF ))
        D=${DEPTHS[$DI]}; FR=${FRACS[$FI]}
        FTAG=$(printf '%02d' "$(python3 -c "print(int(round(${FR}*100)))" 2>/dev/null || echo 00)")
        printf "  idx=%2d  depth=%2d  frac=%s  tag=frac%s_d%02d_s42\n" "${IDX}" "${D}" "${FR}" "${FTAG}" "${D}"
    done
    exit 0
fi

IDX="${SLURM_ARRAY_TASK_ID:?run as a SLURM job array (sbatch --array=...) or set DRY_RUN=1}"
if (( IDX < 0 || IDX >= N )); then
    echo "ERROR: array index ${IDX} out of range (N=${N})" >&2; exit 1
fi

DI=$(( IDX / NF )); FI=$(( IDX % NF ))
DEPTH=${DEPTHS[$DI]}; FRAC=${FRACS[$FI]}
FTAG=$("${PY}" -c "print(f'{int(round(${FRAC}*100)):02d}')")
TAG="frac${FTAG}_d$(printf '%02d' "${DEPTH}")_s42"
echo "[cell] idx=${IDX}/${N} depth=${DEPTH} frac=${FRAC} tag=${TAG} host=$(hostname)"

METRICS="${RUNS}/metrics_${TAG}.json"
if [[ -f "${METRICS}" ]]; then
    echo "[cell] DONE ${TAG} — metrics JSON already present, skipping."
    exit 0
fi

if [[ ! -f "${CACHE}" ]]; then
    echo "ERROR: cache not found: ${CACHE} — run make_cache.py first" >&2; exit 1
fi

# --save-checkpoint keeps a .pt per cell (the GeneCLIP head is small — a few tens
# of MB even at depth 12) so the retrieval eval + inference can load the best one.
"${PY}" "${TRAIN}/gene_train_clip.py" \
    --cache "${CACHE}" \
    --depth "${DEPTH}" \
    --train-frac "${FRAC}" \
    --seed 42 \
    --run-tag "${TAG}" \
    --out-dir "${RUNS}" \
    --save-checkpoint \
    --device cuda

echo "[cell] done ${TAG}: $(date)"
