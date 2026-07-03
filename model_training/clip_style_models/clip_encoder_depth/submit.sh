#!/bin/bash
# submit.sh — launch the GeneCLIP depth x train-frac sweep.
#
# Submits sbatch_depth_sweep.sh as a SLURM array with SPACE-FREE log paths (the
# "3 - Gene Rep" path has spaces, which trips SLURM user-env retrieval and kills
# jobs ~2 min in), passing REPO_ROOT so the sourced config/paths.sh resolves.
#
#   cd /orcd/data/lhtsai/001/mabdel03/Flybase
#   bash "3 - Gene Rep/model_training/clip_style_models/clip_encoder_depth/submit.sh"
#
# Env overrides:
#   PARTITION   (default mit_normal_gpu)
#   LOGDIR      (default /orcd/scratch/orcd/012/mabdel03/gene_rep/logs)
#   DRY_RUN=1   print the index->(depth,frac,tag) table locally and exit.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "${REPO_ROOT}/config/paths.sh"

SWEEP="${FLY_GENE_REP_TRAINING}/clip_style_models/clip_encoder_depth/sbatch_depth_sweep.sh"
PARTITION="${PARTITION:-mit_normal_gpu}"
LOGDIR="${LOGDIR:-/orcd/scratch/orcd/012/mabdel03/gene_rep/logs}"

case "${LOGDIR}" in
    *" "*) echo "ERROR: LOGDIR '${LOGDIR}' contains a space — SLURM env retrieval will fail." >&2; exit 1 ;;
esac
mkdir -p "${LOGDIR}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    DRY_RUN=1 REPO_ROOT="${REPO_ROOT}" bash "${SWEEP}"
    exit 0
fi

echo "[submit] partition=${PARTITION} logdir=${LOGDIR}"
echo "[submit] cache=${FLY_GENE_REP_DATA}/derived/clip_inputs_dmel.npz"
sbatch --export=ALL,REPO_ROOT="${REPO_ROOT}" -p "${PARTITION}" \
    --output="${LOGDIR}/gene_clip_sweep_%A_%a.out" \
    --error="${LOGDIR}/gene_clip_sweep_%A_%a.out" \
    "${SWEEP}"
