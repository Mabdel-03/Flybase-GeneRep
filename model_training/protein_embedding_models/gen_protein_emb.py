"""gen_protein_emb.py — ESM-2 protein embeddings, one vector per gene.

Stage 2a of the Gene Rep pipeline. Embeds the representative (longest-isoform)
protein of every protein-coding gene in gene_table_dmel.parquet with ESM-2 and
writes:

    data/derived/protein_embeddings/esm_emb_dmel_<model_tag>.npy   (N x D, float32)
    data/derived/protein_embeddings/gene_ids_dmel_<model_tag>.npy  (N,)  FBgn order

Default model is ESM-2 3B (facebook/esm2_t36_3B_UR50D -> D=2560); override with
--model / GENE_REP_ESM_MODEL. Per protein: tokenize -> forward (fp16 on GPU) ->
MEAN-POOL over real residue positions (excluding BOS/EOS/pad) -> optional
L2-normalize.

3B is heavy — throughput/memory choices:
  * fp16 weights + torch.inference_mode().
  * Sequences are sorted by length and packed into LENGTH-BUCKETED batches with a
    token budget (--max-tokens), so a few long proteins don't blow up memory while
    short ones still batch efficiently.
  * ESM-2 processes <= 1022 residues per forward; longer fly proteins (~1.5k of
    13,986 are >1022 aa) are TRUNCATED to --max-residues (deterministic).

Run on a GPU (big-VRAM A100/H200 for 3B). Uses fair-esm from GENE_REP_ENV.

    python gen_protein_emb.py                        # all genes, ESM-2 3B
    python gen_protein_emb.py --limit 256 --out-suffix _smoke   # smoke test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

for _parent in Path(__file__).resolve().parents:
    if (_parent / "generep").is_dir():
        sys.path.insert(0, str(_parent))
        break

from generep import paths as gpaths  # noqa: E402

# fair-esm model name (e.g. esm2_t36_3B_UR50D) -> the repr layer index (t<N>).
def _esm_loader_and_layer(model_id: str):
    import esm
    name = model_id.rsplit("/", 1)[-1]        # facebook/esm2_t36_3B_UR50D -> esm2_t36_3B_UR50D
    loader = getattr(esm.pretrained, name, None)
    if loader is None:
        raise SystemExit(
            f"fair-esm has no loader '{name}'. Available esm2_* loaders: "
            + ", ".join(n for n in dir(esm.pretrained) if n.startswith('esm2_')))
    # repr layer = the number after 't' in the name (esm2_t36_... -> 36)
    import re
    m = re.search(r"_t(\d+)_", name)
    layer = int(m.group(1)) if m else 33
    return loader, layer, name


def bucketize(lengths: list[int], order: np.ndarray, max_tokens: int, max_batch: int):
    """Yield lists of dataset indices (already length-sorted via `order`) whose
    padded token count stays within max_tokens; cap count at max_batch."""
    batch: list[int] = []
    cur_max = 0
    for pos in order:
        L = lengths[pos] + 2                      # +BOS +EOS
        new_max = max(cur_max, L)
        if batch and (new_max * (len(batch) + 1) > max_tokens or len(batch) >= max_batch):
            yield batch
            batch, cur_max = [], 0
            new_max = L
        batch.append(int(pos))
        cur_max = new_max
    if batch:
        yield batch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene-table", default=None, help="default: gene_config['gene_table']")
    ap.add_argument("--model", default=None, help="HF/fair-esm model id (default GENE_REP_ESM_MODEL)")
    ap.add_argument("--out-dir", default=None, help="default: PROTEIN_EMBEDDINGS_DIR")
    ap.add_argument("--out-suffix", default="", help="appended to output filenames (e.g. _smoke)")
    ap.add_argument("--max-residues", type=int, default=1022, help="truncate proteins to this length")
    ap.add_argument("--max-tokens", type=int, default=16384, help="token budget per batch (buckets)")
    ap.add_argument("--max-batch", type=int, default=64, help="hard cap on sequences per batch")
    ap.add_argument("--limit", type=int, default=None, help="only embed the first N genes (smoke)")
    ap.add_argument("--no-normalize", action="store_true", help="skip final L2-normalize")
    ap.add_argument("--fp32", action="store_true", help="use fp32 (default fp16 on GPU)")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    model_id = args.model or gpaths.ESM_MODEL
    cfg = gpaths.gene_config("dmel", esm_model=model_id)
    table = Path(args.gene_table) if args.gene_table else cfg["gene_table"]
    out_dir = Path(args.out_dir) if args.out_dir else gpaths.PROTEIN_EMBEDDINGS_DIR
    gpaths.ensure_dir(out_dir)

    df = pd.read_parquet(table)
    if args.limit:
        df = df.iloc[: args.limit].copy()
    gene_ids = df["fbgn"].to_numpy().astype(str)
    seqs = [s[: args.max_residues] for s in df["protein_seq"].tolist()]
    lengths = [len(s) for s in seqs]
    n = len(seqs)
    print(f"[esm] {n} genes | model={model_id} | device={args.device} | "
          f"max_res={args.max_residues} | truncated={sum(l==args.max_residues for l in lengths)}")

    device = torch.device(args.device)
    loader, layer, name = _esm_loader_and_layer(model_id)
    print(f"[esm] loading {name} (repr layer {layer})...")
    t0 = time.time()
    model, alphabet = loader()
    model = model.to(device).eval()
    use_half = (device.type == "cuda") and (not args.fp32)
    if use_half:
        model = model.half()
    batch_converter = alphabet.get_batch_converter()
    print(f"[esm] model ready ({time.time()-t0:.1f}s) | fp16={use_half}")

    # length-sorted order for efficient bucketing; write back in original order.
    order = np.argsort(np.asarray(lengths))
    emb_dim = None
    out = None
    done = 0
    t0 = time.time()
    with torch.inference_mode():
        for batch_idx in bucketize(lengths, order, args.max_tokens, args.max_batch):
            data = [(gene_ids[i], seqs[i]) for i in batch_idx]
            _, _, toks = batch_converter(data)
            toks = toks.to(device)
            res = model(toks, repr_layers=[layer], return_contacts=False)
            reps = res["representations"][layer]            # (B, L, D), fp16
            # real-residue mask: exclude padding, BOS (cls), EOS.
            pad_mask = toks != alphabet.padding_idx
            pad_mask[:, 0] = False                          # BOS
            for b, i in enumerate(batch_idx):
                L = lengths[i]                              # residues before BOS/EOS
                # positions 1..L are residues (0 is BOS, L+1 is EOS)
                vec = reps[b, 1:1 + L].float().mean(dim=0)  # mean over residues
                v = vec.cpu().numpy()
                if out is None:
                    emb_dim = v.shape[0]
                    out = np.zeros((n, emb_dim), dtype=np.float32)
                out[i] = v
            done += len(batch_idx)
            if done % 1024 < len(batch_idx):
                el = time.time() - t0
                print(f"  [esm] {done}/{n} ({100*done/n:.1f}%) | {done/max(el,1e-9):.1f} genes/s")

    assert out is not None, "no embeddings produced"
    if not args.no_normalize:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.clip(norms, 1e-9, None)

    tag = gpaths._model_tag(model_id)
    emb_path = out_dir / f"esm_emb_dmel_{tag}{args.out_suffix}.npy"
    ids_path = out_dir / f"gene_ids_dmel_{tag}{args.out_suffix}.npy"
    np.save(emb_path, out)
    np.save(ids_path, gene_ids)
    print(f"[esm] done: {out.shape} -> {emb_path.name} (+ {ids_path.name})")
    print(f"[esm] wall={time.time()-t0:.1f}s | finite={np.isfinite(out).all()} "
          f"| mean_norm={np.linalg.norm(out, axis=1).mean():.4f}")


if __name__ == "__main__":
    main()
