"""gen_gene_text_emb.py — BioBERT embeddings of FlyBase gene descriptions.

Stage 2b of the Gene Rep pipeline. Encodes the `description` column of
gene_table_dmel.parquet (the FlyBase best/automated gene summary — the text IS
the description, no caption templating) with BioBERT and writes:

    data/derived/text_embeddings/text_emb_dmel_<model_tag>.npy   (N x 768, float32)
    data/derived/text_embeddings/gene_ids_dmel_<model_tag>.npy   (N,)  FBgn order

Default model dmis-lab/biobert-base-cased-v1.1 (768-d), matching the BioBERT
pattern already used in
`2 - Cel Rep/Drosophila/evaluations/clip_style_models/original_fca_reference/code/textembeddings.py`.

Pooling: CLS token (last_hidden_state[:, 0, :]) by default — the repo's BioBERT
convention. `--pool mean` switches to masked-mean (the mpnet gen_text_emb.py
convention). Whatever is chosen here MUST match the eval QueryEncoder. Output is
L2-normalized.

Runs in cel_rep OR gene_rep (both have transformers). GPU optional but faster.

    python gen_gene_text_emb.py
    python gen_gene_text_emb.py --limit 256 --out-suffix _smoke
"""
from __future__ import annotations

import argparse
import sys
import time
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


def _load_tokenizer(model_id):
    """AutoTokenizer, falling back to BertTokenizer for old repos (e.g.
    dmis-lab/biobert-*) that ship only vocab.txt with no tokenizer_config.json,
    which transformers 5.x's AutoTokenizer can't infer a class for."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except (ValueError, KeyError, OSError) as e:
        from transformers import BertTokenizer
        print(f"  [text] AutoTokenizer failed ({type(e).__name__}); "
              f"using BertTokenizer for {model_id}")
        return BertTokenizer.from_pretrained(model_id)


def _load_model(model_id):
    """AutoModel, falling back to BertModel for old repos whose config.json lacks
    a `model_type` key (e.g. dmis-lab/biobert-*)."""
    from transformers import AutoModel
    try:
        return AutoModel.from_pretrained(model_id)
    except (ValueError, KeyError, OSError) as e:
        from transformers import BertModel
        print(f"  [text] AutoModel failed ({type(e).__name__}); "
              f"using BertModel for {model_id}")
        return BertModel.from_pretrained(model_id)


def encode(descriptions, model_id, pool, device, batch_size=32, max_length=256):
    tok = _load_tokenizer(model_id)
    model = _load_model(model_id).to(device).eval()
    out = []
    n = len(descriptions)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = descriptions[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt").to(device)
            hs = model(**enc).last_hidden_state           # (B, T, 768)
            if pool == "cls":
                emb = hs[:, 0, :]                         # CLS token
            else:  # masked mean
                mask = enc["attention_mask"].unsqueeze(-1).expand(hs.size()).float()
                emb = (hs * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
            emb = F.normalize(emb, dim=-1)
            out.append(emb.cpu().numpy())
            done = min(i + batch_size, n)
            if done % 2048 < batch_size:
                print(f"  [text] {done}/{n} ({100*done/n:.1f}%) | "
                      f"{done/max(time.time()-t0,1e-9):.1f} genes/s")
    return np.vstack(out).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene-table", default=None)
    ap.add_argument("--model", default=None, help="default GENE_REP_TEXT_MODEL")
    ap.add_argument("--pool", choices=["cls", "mean"], default="cls")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    model_id = args.model or gpaths.TEXT_MODEL
    cfg = gpaths.gene_config("dmel", text_model=model_id)
    table = Path(args.gene_table) if args.gene_table else cfg["gene_table"]
    out_dir = Path(args.out_dir) if args.out_dir else gpaths.TEXT_EMBEDDINGS_DIR
    gpaths.ensure_dir(out_dir)

    df = pd.read_parquet(table)
    if args.limit:
        df = df.iloc[: args.limit].copy()
    gene_ids = df["fbgn"].to_numpy().astype(str)
    descriptions = df["description"].fillna("").astype(str).tolist()
    # a genuinely empty description -> encode the gene symbol as a minimal fallback
    symbols = df["symbol"].astype(str).tolist()
    descriptions = [d if d.strip() else symbols[i] for i, d in enumerate(descriptions)]
    print(f"[text] {len(descriptions)} genes | model={model_id} | pool={args.pool} | "
          f"device={args.device}")

    emb = encode(descriptions, model_id, args.pool, torch.device(args.device),
                 batch_size=args.batch_size, max_length=args.max_length)

    tag = gpaths._model_tag(model_id)
    emb_path = out_dir / f"text_emb_dmel_{tag}{args.out_suffix}.npy"
    ids_path = out_dir / f"gene_ids_dmel_{tag}{args.out_suffix}.npy"
    np.save(emb_path, emb)
    np.save(ids_path, gene_ids)
    print(f"[text] done: {emb.shape} -> {emb_path.name} (+ {ids_path.name})")
    print(f"[text] finite={np.isfinite(emb).all()} | mean_norm={np.linalg.norm(emb, axis=1).mean():.4f}")


if __name__ == "__main__":
    main()
