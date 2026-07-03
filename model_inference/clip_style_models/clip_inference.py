"""clip_inference.py — query a trained GeneCLIP in the shared protein<->text space.

Stage 5. Loads a gene_train_clip.py checkpoint + the embedding cache and answers
two query directions:

  text->gene   : encode a free-text description with BioBERT -> text adapter, then
                 return the top-k nearest GENES (by their projected protein
                 embedding). "Which fly genes look like this description?"
  gene->text   : take a gene's protein embedding -> protein adapter, then return
                 the top-k nearest genes' descriptions. "What is this protein like?"

The protein side of the index is precomputed once from the cache (all genes, not
just the test split). Gene symbols/descriptions come from gene_table_dmel.parquet.

    # which genes match a functional description?
    python clip_inference.py --ckpt <gene_clip.pt> --text "serine protease involved in immune response" --topk 10
    # nearest neighbours of a gene (by FBgn or symbol)
    python clip_inference.py --ckpt <gene_clip.pt> --gene Nep3 --topk 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

for _parent in Path(__file__).resolve().parents:
    if (_parent / "generep").is_dir():
        sys.path.insert(0, str(_parent))
        break
from generep import paths as gpaths  # noqa: E402

_TRAIN = gpaths.TRAINING_DIR / "clip_style_models"
sys.path.insert(0, str(_TRAIN))
from gene_train_clip import GeneCLIP  # noqa: E402


def load(ckpt_path, cache_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = GeneCLIP(int(ck["esm_dim"]), int(ck["text_dim"]),
                     projection_dim=int(ck.get("projection_dim", 2048)),
                     depth=int(ck["depth"])).to(device).eval()
    model.load_state_dict(ck["model_state_dict"])
    z = np.load(cache_path, allow_pickle=True)
    prot = np.asarray(z["vae_emb"], dtype=np.float32)
    text = np.asarray(z["text_emb"], dtype=np.float32)
    gene_ids = (z["barcodes"] if "barcodes" in z else z["gene_ids"]).astype(str)
    return model, ck, prot, text, gene_ids


def encode_text_query(query, text_model, device):
    """Encode a free-text query with the SAME BioBERT + pooling used in training."""
    sys.path.insert(0, str(gpaths.TRAINING_DIR / "text_embedding_models"))
    from gen_gene_text_emb import encode
    return encode([query], text_model, "cls", device)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--gene-table", default=None)
    ap.add_argument("--text", default=None, help="free-text query -> nearest genes")
    ap.add_argument("--gene", default=None, help="FBgn or symbol -> nearest genes")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()
    if not args.text and not args.gene:
        raise SystemExit("provide --text or --gene")

    cfg = gpaths.gene_config("dmel")
    cache = Path(args.cache) if args.cache else cfg["clip_cache"]
    table = Path(args.gene_table) if args.gene_table else cfg["gene_table"]
    device = torch.device(args.device)

    model, ck, prot, text, gene_ids = load(args.ckpt, cache, device)
    df = pd.read_parquet(table).set_index("fbgn")
    sym = {g: (df.loc[g, "symbol"] if g in df.index else g) for g in gene_ids}
    desc = {g: (str(df.loc[g, "description"])[:140] if g in df.index else "") for g in gene_ids}

    # project all genes' proteins into the shared space (the searchable index)
    with torch.inference_mode():
        gene_feats = F.normalize(model.protein_adapter(torch.tensor(prot).to(device)), dim=-1)

    if args.text:
        text_model = ck.get("text_model") or gpaths.TEXT_MODEL
        q = encode_text_query(args.text, text_model, device)
        with torch.inference_mode():
            qf = F.normalize(model.text_adapter(torch.tensor(q[None]).to(device)), dim=-1)
        scores = (qf @ gene_feats.T).squeeze(0).cpu().numpy()
        order = np.argsort(-scores)[: args.topk]
        print(f"\nQuery text: {args.text!r}\nTop {args.topk} genes:")
        for r, i in enumerate(order, 1):
            g = gene_ids[i]
            print(f"  {r:2d}. {sym[g]:<14} ({g})  score={scores[i]:.3f}  {desc[g]}")

    if args.gene:
        # resolve FBgn or symbol -> index
        want = args.gene
        idx = None
        if want in set(gene_ids):
            idx = int(np.where(gene_ids == want)[0][0])
        else:
            for i, g in enumerate(gene_ids):
                if sym[g] == want:
                    idx = i; break
        if idx is None:
            raise SystemExit(f"gene {want!r} not found in cache")
        qf = gene_feats[idx:idx + 1]
        scores = (qf @ gene_feats.T).squeeze(0).cpu().numpy()
        scores[idx] = -np.inf                              # exclude self
        order = np.argsort(-scores)[: args.topk]
        g0 = gene_ids[idx]
        print(f"\nQuery gene: {sym[g0]} ({g0})\nTop {args.topk} nearest genes (shared space):")
        for r, i in enumerate(order, 1):
            g = gene_ids[i]
            print(f"  {r:2d}. {sym[g]:<14} ({g})  score={scores[i]:.3f}  {desc[g]}")


if __name__ == "__main__":
    main()
