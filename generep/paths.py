"""Central path and dataset helpers for the Gene Rep workspace.

Ported from ``2 - Cel Rep/Drosophila/celrep/paths.py``. Scripts should import this module
instead of relying on the current working directory or bare filenames. Every
path is overridable via the ``FLY_GENE_REP_*`` environment variables defined in
the repo-root ``config/paths.sh`` (Gene Rep block).

Layout (mirrors Cel Rep, adapted to gene/protein):

    3 - Gene Rep/
      generep/                          # this package
      data/
        raw/                            # FlyBase downloads + gene_table.parquet
        derived/protein_embeddings/     # ESM per-gene .npy + gene_ids.npy
        derived/text_embeddings/        # BioBERT per-gene .npy + gene_ids.npy
        derived/clip_inputs_*.npz       # aligned protein||text||labels cache
        model_weights/clip_style_models/
      model_training/{protein,text}_embedding_models/, clip_style_models/
      data_prep/
      evaluations/{protein_embedding_models,clip_style_models}/
      model_inference/clip_style_models/
"""

from __future__ import annotations

import os
from pathlib import Path


GENE_REP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = GENE_REP_ROOT.parent

DATA_DIR = Path(os.environ.get("FLY_GENE_REP_DATA", GENE_REP_ROOT / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
DERIVED_DATA_DIR = DATA_DIR / "derived"
PROTEIN_EMBEDDINGS_DIR = DERIVED_DATA_DIR / "protein_embeddings"
TEXT_EMBEDDINGS_DIR = DERIVED_DATA_DIR / "text_embeddings"
GENESETS_DIR = DERIVED_DATA_DIR / "genesets"
MODEL_WEIGHTS_DIR = Path(
    os.environ.get("FLY_GENE_REP_MODEL_WEIGHTS", DATA_DIR / "model_weights")
)
CLIP_MODEL_WEIGHTS_DIR = MODEL_WEIGHTS_DIR / "clip_style_models"

TRAINING_DIR = Path(os.environ.get("FLY_GENE_REP_TRAINING", GENE_REP_ROOT / "model_training"))
INFERENCE_DIR = Path(os.environ.get("FLY_GENE_REP_INFERENCE", GENE_REP_ROOT / "model_inference"))
EVALUATIONS_DIR = Path(os.environ.get("FLY_GENE_REP_EVALUATIONS", GENE_REP_ROOT / "evaluations"))
LOGS_DIR = Path(os.environ.get("FLY_GENE_REP_LOGS", EVALUATIONS_DIR / "logs" / "slurm"))

PROTEIN_EVAL_DIR = EVALUATIONS_DIR / "protein_embedding_models"
CLIP_EVAL_DIR = EVALUATIONS_DIR / "clip_style_models"

# The 2 - Cel Rep tree, whose Adapter / clip_loss / evaluate / safe_stratified_subset
# we import (never copy). Overridable if that stage is relocated.
CEL_REP_DIR = Path(os.environ.get("FLY_CEL_REP_DIR", REPO_ROOT / "2 - Cel Rep"))


def find_cel_rep_trainer() -> Path:
    """Locate the directory holding c2s_train_vae_clip.py inside 2 - Cel Rep.

    The Cel Rep submodule has been reorganized before (e.g. a per-species
    Drosophila/ subdir), so we DISCOVER the trainer rather than assume a fixed
    subpath. Honors GENE_REP_CEL_TRAINER_DIR if set. Raises if not found.
    """
    override = os.environ.get("GENE_REP_CEL_TRAINER_DIR")
    if override:
        p = Path(override)
        if (p / "c2s_train_vae_clip.py").exists():
            return p
        raise SystemExit(f"GENE_REP_CEL_TRAINER_DIR={override} has no c2s_train_vae_clip.py")
    # common locations first (fast path), then a bounded glob.
    for rel in ("model_training/clip_style_models",
                "Drosophila/model_training/clip_style_models"):
        p = CEL_REP_DIR / rel
        if (p / "c2s_train_vae_clip.py").exists():
            return p
    hits = [q for q in CEL_REP_DIR.rglob("c2s_train_vae_clip.py")
            if ".git" not in q.parts]
    if hits:
        return hits[0].parent
    raise SystemExit(
        f"Could not find c2s_train_vae_clip.py under {CEL_REP_DIR}. Is '2 - Cel Rep' "
        f"checked out? Override with GENE_REP_CEL_TRAINER_DIR=<dir>.")


def import_cel_rep_core():
    """Import (Adapter, clip_loss, evaluate, safe_stratified_subset) from the Cel
    Rep trainer, VERBATIM, without dragging in the single-cell stack.

    c2s_train_vae_clip.py has module-level `import scanpy` and `from celrep import
    paths` that its h5ad/VAE code paths need but the contrastive functions we use
    do NOT. To keep GENE_REP_ENV lean (no scanpy), we (a) put the discovered
    celrep package dir on sys.path and (b) install a minimal `scanpy` stub if it
    is absent. The four returned callables are pure torch/numpy/sklearn and never
    touch scanpy, so the stub is only ever a placeholder to satisfy the import.
    """
    import importlib
    import sys
    import types

    trainer_dir = find_cel_rep_trainer()
    # celrep package lives at the Cel Rep (sub)tree root; find it upward.
    cel_pkg_parent = None
    for parent in trainer_dir.parents:
        if (parent / "celrep").is_dir():
            cel_pkg_parent = parent
            break
    if cel_pkg_parent and str(cel_pkg_parent) not in sys.path:
        sys.path.insert(0, str(cel_pkg_parent))

    if "scanpy" not in sys.modules:
        try:
            importlib.import_module("scanpy")
        except ImportError:
            stub = types.ModuleType("scanpy")
            stub.__doc__ = "stub injected by generep.paths.import_cel_rep_core (scanpy unused here)"
            sys.modules["scanpy"] = stub

    if str(trainer_dir) not in sys.path:
        sys.path.insert(0, str(trainer_dir))
    mod = importlib.import_module("c2s_train_vae_clip")
    return mod.Adapter, mod.clip_loss, mod.evaluate, mod.safe_stratified_subset

# Default model ids (overridable via env, mirroring config/paths.sh).
ESM_MODEL = os.environ.get("GENE_REP_ESM_MODEL", "facebook/esm2_t36_3B_UR50D")
# BioBERT v1.1 (PubMed). The monologg mirror ships safetensors + a proper
# config.json/tokenizer_config.json, so it loads clean under transformers 5.x +
# torch<2.6 (the original dmis-lab repo ships only pytorch_model.bin, which the
# new torch.load security gate blocks). Same BioBERT weights, 768-d output.
TEXT_MODEL = os.environ.get("GENE_REP_TEXT_MODEL", "monologg/biobert_v1.1_pubmed")


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    """Create and return ``path`` as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _model_tag(model_id: str) -> str:
    """Filesystem-safe short tag for a HF model id (last path component)."""
    return model_id.rsplit("/", 1)[-1]


def gene_config(
    species: str = "dmel",
    *,
    esm_model: str | None = None,
    text_model: str | None = None,
    raw_dir: os.PathLike[str] | str | None = None,
    protein_dir: os.PathLike[str] | str | None = None,
    text_dir: os.PathLike[str] | str | None = None,
    cache_dir: os.PathLike[str] | str | None = None,
    checkpoint_dir: os.PathLike[str] | str | None = None,
) -> dict[str, object]:
    """Resolve canonical Gene Rep artifact paths for a species / model pair.

    ``species`` is the short tag used in filenames (default ``dmel``). The ESM and
    text embedding filenames carry the model tag so multiple encoders can coexist
    on disk without clobbering each other.
    """
    esm_model = esm_model or ESM_MODEL
    text_model = text_model or TEXT_MODEL
    esm_tag = _model_tag(esm_model)
    text_tag = _model_tag(text_model)

    raw = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    prot = Path(protein_dir) if protein_dir is not None else PROTEIN_EMBEDDINGS_DIR
    text = Path(text_dir) if text_dir is not None else TEXT_EMBEDDINGS_DIR
    cache = Path(cache_dir) if cache_dir is not None else DERIVED_DATA_DIR
    ckpt = Path(checkpoint_dir) if checkpoint_dir is not None else CLIP_MODEL_WEIGHTS_DIR

    return {
        "species": species,
        "esm_model": esm_model,
        "text_model": text_model,
        # Stage 1 — data prep.
        "protein_fasta": raw / f"{species}-all-translation.fasta.gz",
        "gene_summaries": raw / f"{species}_automated_gene_summaries.tsv.gz",
        "gene_table": raw / f"gene_table_{species}.parquet",
        # Stage 2 — embeddings. Each carries a gene-id sidecar with row order.
        "protein_emb_path": prot / f"esm_emb_{species}_{esm_tag}.npy",
        "protein_gene_ids": prot / f"gene_ids_{species}_{esm_tag}.npy",
        "text_emb_path": text / f"text_emb_{species}_{text_tag}.npy",
        "text_gene_ids": text / f"gene_ids_{species}_{text_tag}.npy",
        # Stage 2 — aligned training cache (Cel-Rep-compatible key names inside).
        "clip_cache": cache / f"clip_inputs_{species}.npz",
        # Stage 4 — eval labels (FBgn-keyed JSON).
        "go_bp_json": GENESETS_DIR / "go_bp.json",
        "gene_groups_json": GENESETS_DIR / "flybase_groups.json",
        # Stage 3 — checkpoint.
        "checkpoint_out": ckpt / f"gene_clip_{species}.pt",
    }
