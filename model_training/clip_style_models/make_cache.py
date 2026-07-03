"""make_cache.py — assemble the aligned protein<->text CLIP cache.

Stage 2c of the Gene Rep pipeline. Joins the ESM protein embeddings and the
BioBERT text embeddings BY FBgn (never positionally), attaches per-gene function
labels, and writes a single .npz that the trainer + evals consume:

    data/derived/clip_inputs_dmel.npz

Cel-Rep-COMPATIBLE key names (so the imported Adapter/clip_loss/evaluate and the
existing eval code work unchanged):
    vae_emb       (N x esm_dim)   -- the ESM protein matrix  (the "left" modality)
    text_emb      (N x 768)       -- the BioBERT matrix       (the "right" modality)
    labels        (N,)  str       -- fine gene-function label (most-specific GO-BP)
    labels_broad  (N,)  str       -- broad label (FlyBase gene group)
    barcodes      (N,)  str       -- FBgn ids (the join key / gene order)
Plus Gene-Rep extras: gene_ids (== barcodes), esm_dim, text_dim, esm_model,
text_model.

Labels: each gene's `labels` is its most-specific (smallest-membership) GO-BP
term; `labels_broad` is its FlyBase gene group (largest one if several). Genes
with no term get 'unlabeled'. Labels are used ONLY for the stratified train/test
split and the label-transfer evals — the CLIP objective itself is label-free
(protein<->text InfoNCE).

    python make_cache.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _parent in Path(__file__).resolve().parents:
    if (_parent / "generep").is_dir():
        sys.path.insert(0, str(_parent))
        break

from generep import paths as gpaths  # noqa: E402


def _load_emb(emb_path: Path, ids_path: Path) -> dict[str, np.ndarray]:
    """FBgn -> embedding row, from a .npy matrix + gene_ids sidecar."""
    emb = np.load(emb_path)
    ids = np.load(ids_path).astype(str)
    assert len(emb) == len(ids), f"{emb_path}: {len(emb)} rows vs {len(ids)} ids"
    return {g: emb[i] for i, g in enumerate(ids)}


def _fine_labels(fbgns: list[str], go_json: Path) -> dict[str, str]:
    """Assign each gene its MOST SPECIFIC GO-BP term (smallest membership)."""
    if not go_json.exists():
        return {}
    go = json.loads(go_json.read_text())
    # term -> size; and gene -> list of (size, term)
    gene_terms: dict[str, list[tuple[int, str]]] = {}
    for term, rec in go.items():
        genes = rec.get("genes", [])
        size = len(genes)
        for g in genes:
            gene_terms.setdefault(g, []).append((size, term))
    out = {}
    for g in fbgns:
        cands = gene_terms.get(g)
        if cands:
            out[g] = min(cands)[1]        # smallest-membership term
    return out


def _broad_labels(fbgns: list[str], gg_json: Path) -> dict[str, str]:
    """Assign each gene its FlyBase gene group (largest, if several)."""
    if not gg_json.exists():
        return {}
    gg = json.loads(gg_json.read_text())
    gene_groups: dict[str, list[tuple[int, str]]] = {}
    for gid, rec in gg.items():
        name = rec.get("name") or gid
        genes = rec.get("genes", [])
        for g in genes:
            gene_groups.setdefault(g, []).append((len(genes), name))
    out = {}
    for g in fbgns:
        cands = gene_groups.get(g)
        if cands:
            out[g] = max(cands)[1]        # largest group the gene belongs to
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--esm-model", default=None)
    ap.add_argument("--text-model", default=None)
    ap.add_argument("--esm-suffix", default="", help="e.g. _smoke for a smoke cache")
    ap.add_argument("--text-suffix", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = gpaths.gene_config("dmel", esm_model=args.esm_model, text_model=args.text_model)
    esm_tag = gpaths._model_tag(cfg["esm_model"])
    text_tag = gpaths._model_tag(cfg["text_model"])

    pdir = gpaths.PROTEIN_EMBEDDINGS_DIR
    tdir = gpaths.TEXT_EMBEDDINGS_DIR
    esm_emb_p = pdir / f"esm_emb_dmel_{esm_tag}{args.esm_suffix}.npy"
    esm_ids_p = pdir / f"gene_ids_dmel_{esm_tag}{args.esm_suffix}.npy"
    txt_emb_p = tdir / f"text_emb_dmel_{text_tag}{args.text_suffix}.npy"
    txt_ids_p = tdir / f"gene_ids_dmel_{text_tag}{args.text_suffix}.npy"
    for p in (esm_emb_p, esm_ids_p, txt_emb_p, txt_ids_p):
        if not p.exists():
            raise SystemExit(f"missing embedding artifact {p} — run the Stage-2 generators first")

    print("[cache] loading ESM + text embeddings...")
    esm = _load_emb(esm_emb_p, esm_ids_p)
    txt = _load_emb(txt_emb_p, txt_ids_p)

    # Intersect by FBgn (never positional); stable sorted order = the gene order.
    shared = sorted(set(esm) & set(txt))
    n_esm, n_txt, n = len(esm), len(txt), len(shared)
    print(f"[cache] ESM={n_esm} text={n_txt} shared(FBgn)={n}")
    if n < min(n_esm, n_txt):
        print(f"[cache] NOTE: dropped {min(n_esm, n_txt) - n} genes present in only one modality")

    esm_mat = np.stack([esm[g] for g in shared]).astype(np.float32)
    txt_mat = np.stack([txt[g] for g in shared]).astype(np.float32)

    # labels (for split + label-transfer evals only)
    fine = _fine_labels(shared, cfg["go_bp_json"])
    broad = _broad_labels(shared, cfg["gene_groups_json"])
    labels = np.array([fine.get(g, "unlabeled") for g in shared], dtype=object).astype(str)
    labels_broad = np.array([broad.get(g, "unlabeled") for g in shared], dtype=object).astype(str)
    n_fine = int((labels != "unlabeled").sum())
    n_broad = int((labels_broad != "unlabeled").sum())
    print(f"[cache] fine GO-BP labelled: {n_fine}/{n} ({pd.Series(labels).nunique()} classes) | "
          f"broad gene-group labelled: {n_broad}/{n} ({pd.Series(labels_broad).nunique()} classes)")

    gene_ids = np.array(shared, dtype=object).astype(str)
    out = Path(args.out) if args.out else cfg["clip_cache"]
    gpaths.ensure_dir(out.parent)
    np.savez(
        out,
        vae_emb=esm_mat,            # ESM protein matrix (Cel-Rep "left" slot)
        text_emb=txt_mat,          # BioBERT matrix (Cel-Rep "right" slot)
        labels=labels,
        labels_broad=labels_broad,
        barcodes=gene_ids,         # FBgn join key (Cel-Rep "barcodes" slot)
        gene_ids=gene_ids,
        esm_dim=np.int64(esm_mat.shape[1]),
        text_dim=np.int64(txt_mat.shape[1]),
        esm_model=np.str_(cfg["esm_model"]),
        text_model=np.str_(cfg["text_model"]),
    )
    print(f"[cache] wrote {out}")
    print(f"  vae_emb(ESM) {esm_mat.shape} | text_emb(BioBERT) {txt_mat.shape} | {n} genes")


if __name__ == "__main__":
    main()
