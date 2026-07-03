"""gene_train_clip.py — train the protein<->text CLIP head (GeneCLIP).

Stage 3 of the Gene Rep pipeline. A thin trainer that IMPORTS the proven
contrastive machinery from 2 - Cel Rep verbatim (Adapter, clip_loss, evaluate,
safe_stratified_subset) and applies it to the new modality pair:

    left  = ESM protein embedding   (cache key 'vae_emb', esm_dim, e.g. 2560)
    right = BioBERT text embedding  (cache key 'text_emb', 768)

GeneCLIP is structurally identical to CellCLIP: two depth-swept Adapters + a
learnable logit_scale. Same AdamW(1e-5, wd0.01) + cosine schedule, same
logit_scale clamp, same frozen-comparable-test-split protocol
(safe_stratified_subset, stratified on the gene-function label), same
best-by-test-loss checkpoint + atomic metrics JSON.

Reads the .npz cache written by make_cache.py. recall@5 both directions
(protein->text and text->protein) via the imported evaluate().

  # build the shared embedding cache once (make_cache.py), then a grid cell:
  python gene_train_clip.py --cache <clip_inputs_dmel.npz> --depth 6 \
      --train-frac 0.80 --seed 42 --run-tag frac80_d06_s42 --out-dir <ckpts> --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

for _parent in Path(__file__).resolve().parents:
    if (_parent / "generep").is_dir():
        sys.path.insert(0, str(_parent))
        break
from generep import paths as gpaths  # noqa: E402

# ── import the Cel Rep contrastive core verbatim (never copied) ─────────────────
# Location inside 2 - Cel Rep is DISCOVERED (the submodule has been reorganized
# before) and the single-cell stack is not required — see import_cel_rep_core.
Adapter, clip_loss, evaluate, safe_stratified_subset = gpaths.import_cel_rep_core()


class GeneCLIP(nn.Module):
    """Protein<->text CLIP head — structurally identical to CellCLIP.

    protein_adapter maps the ESM embedding (esm_dim) and text_adapter maps the
    BioBERT embedding (text_dim=768) into a shared projection_dim space; both are
    L2-normalized. logit_scale is the learnable CLIP temperature.
    """
    def __init__(self, esm_dim, text_dim, projection_dim=2048, depth=6):
        super().__init__()
        self.protein_adapter = Adapter(esm_dim, projection_dim, projection_dim, depth=depth)
        self.text_adapter = Adapter(text_dim, projection_dim, projection_dim, depth=depth)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, protein_emb, text_emb):
        return (F.normalize(self.protein_adapter(protein_emb), dim=-1),
                F.normalize(self.text_adapter(text_emb), dim=-1))


class ProteinTextDataset(Dataset):
    def __init__(self, protein_emb, text_emb):
        self.p = torch.as_tensor(protein_emb, dtype=torch.float32)
        self.t = torch.as_tensor(text_emb, dtype=torch.float32)

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.p[i], self.t[i]


def _atomic_write_json(obj, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def _make_split(n, strat_labels, test_holdout, train_frac, seed):
    """Frozen-comparable train/test split.

    If strat_labels is given AND stratification is feasible (each split's size
    >= its class count), use the imported safe_stratified_subset (same recipe as
    Cel Rep: test carved at random_state=42, train drawn from the pool at --seed).
    Otherwise fall back to a plain random split (still test@42 / train@seed), so
    the protocol degrades gracefully for granular gene labels. Returns
    (train_idx, test_idx, mode)."""
    from sklearn.model_selection import train_test_split

    def _stratifiable(labels, frac):
        # sklearn's train_test_split requires BOTH the train side AND the
        # complement (test) side to have >= n_classes samples. At frac -> 1.0 the
        # complement gets tiny (this is what broke the frac0.90 cells), so we check
        # both sides, not just `frac`.
        vc = pd.Series(labels).value_counts()
        kept = vc[vc >= 2]
        n_kept = int(kept.sum())
        n_classes = int(len(kept))
        if n_kept == 0 or n_classes < 2:
            return False
        n_a = int(round(n_kept * frac))
        n_b = n_kept - n_a
        return n_a >= n_classes and n_b >= n_classes

    if strat_labels is not None and _stratifiable(strat_labels, test_holdout):
        obs = pd.DataFrame({"label": strat_labels})
        pool_idx, test_idx, _ = safe_stratified_subset(
            obs, "label", test_size=test_holdout, random_state=42, min_class_count=2)
        pool_obs = obs.iloc[pool_idx]
        pool_frac = min(train_frac / (1.0 - test_holdout), 0.999)
        if _stratifiable(pool_obs["label"].to_numpy(), pool_frac):
            mcc = max(2, int(np.ceil(1.0 / pool_frac)))
            keep_rel, _ = safe_stratified_subset(
                pool_obs, "label", train_size=pool_frac, random_state=seed, min_class_count=mcc)
            return pool_idx[keep_rel], test_idx, "stratified"
        # pool stratification infeasible -> random draw from the (stratified) pool
        rng = np.random.RandomState(seed)
        keep = rng.permutation(len(pool_idx))[: max(1, int(round(len(pool_idx) * pool_frac)))]
        return pool_idx[keep], test_idx, "stratified_test_random_train"

    # fully random fallback (test@42, train@seed)
    idx = np.arange(n)
    pool_idx, test_idx = train_test_split(idx, test_size=test_holdout, random_state=42)
    pool_frac = min(train_frac / (1.0 - test_holdout), 0.999)
    rng = np.random.RandomState(seed)
    keep = rng.permutation(len(pool_idx))[: max(1, int(round(len(pool_idx) * pool_frac)))]
    return pool_idx[keep], test_idx, "random"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=None, help="clip_inputs_*.npz (default gene_config)")
    ap.add_argument("--depth", type=int, default=6, help="# Linear layers per Adapter")
    ap.add_argument("--train-frac", type=float, default=0.80)
    ap.add_argument("--test-holdout", type=float, default=0.10,
                    help="fixed comparable test fraction (random_state=42)")
    ap.add_argument("--stratify-label", choices=["broad", "fine", "none"], default="broad",
                    help="label to stratify the split on (broad=gene group, "
                         "fine=GO-BP, none=random). Falls back to random if infeasible.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--projection-dim", type=int, default=2048)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--save-checkpoint", action="store_true")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    cfg = gpaths.gene_config("dmel")
    cache = Path(args.cache) if args.cache else cfg["clip_cache"]
    out_dir = Path(args.out_dir) if args.out_dir else gpaths.CLIP_EVAL_DIR / "clip_encoder_depth" / "outputs"
    gpaths.ensure_dir(out_dir)
    run_tag = args.run_tag or f"frac{int(round(args.train_frac*100)):02d}_d{args.depth:02d}_s{args.seed}"
    t0 = time.time()

    if not cache.exists():
        raise SystemExit(f"cache not found: {cache} — run make_cache.py first")
    print(f"[load] {cache}")
    z = np.load(cache, allow_pickle=True)
    prot_emb = np.asarray(z["vae_emb"], dtype=np.float32)     # ESM
    text_emb = np.asarray(z["text_emb"], dtype=np.float32)    # BioBERT
    labels_fine = z["labels"].astype(str)                     # most-specific GO-BP
    labels_broad = (z["labels_broad"].astype(str) if "labels_broad" in z
                    else labels_fine)                         # FlyBase gene group
    gene_ids = z["barcodes"].astype(str) if "barcodes" in z else z["gene_ids"].astype(str)
    print(f"  {len(labels_fine)} genes | protein(ESM) {prot_emb.shape} | text(BioBERT) {text_emb.shape}")

    # ── frozen comparable test split (seed 42, independent of --seed) ──────────
    # Stratify on the chosen gene-function label when feasible; the InfoNCE
    # objective is label-free, so the split only needs to be COMPARABLE across
    # sweep cells. Gene-function labels can be far more granular than the ~250
    # cell types in Cel Rep (fine GO-BP has ~2k classes), so we stratify on the
    # COARSER broad (gene-group) label by default and fall back to a plain random
    # split if even that is infeasible (test_size < n_classes).
    strat = {"broad": labels_broad, "fine": labels_fine, "none": None}[args.stratify_label]
    train_idx, test_idx, split_mode = _make_split(
        len(labels_fine), strat, args.test_holdout, args.train_frac, args.seed)
    print(f"[split] mode={split_mode} | train {len(train_idx)} | test {len(test_idx)} | "
          f"fine classes(train) {pd.Series(labels_fine[train_idx]).nunique()}")

    tr_p, te_p = prot_emb[train_idx], prot_emb[test_idx]
    tr_t, te_t = text_emb[train_idx], text_emb[test_idx]

    # ── model + optimizer (identical protocol to CellCLIP) ─────────────────────
    model = GeneCLIP(prot_emb.shape[1], text_emb.shape[1],
                     projection_dim=args.projection_dim, depth=args.depth).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    bs = min(512, max(32, len(train_idx) // 4))
    tr_loader = DataLoader(ProteinTextDataset(tr_p, tr_t), batch_size=bs, shuffle=True,
                           drop_last=(len(train_idx) % bs == 1))
    te_loader = DataLoader(ProteinTextDataset(te_p, te_t), batch_size=512, shuffle=False)
    assert len(tr_loader) > 0, "empty train loader"

    ckpt_path = out_dir / f"gene_clip_dmel_{run_tag}.pt"
    best_loss, best_metrics, best_epoch = float("inf"), None, -1
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for p, t in tr_loader:
            p, t = p.to(device), t.to(device)
            opt.zero_grad()
            pf, tf = model(p, t)
            sim = (pf @ tf.T) * model.logit_scale.exp()
            loss = clip_loss(sim)
            loss.backward()
            opt.step()
            with torch.no_grad():
                model.logit_scale.clamp_(min=np.log(1 / 100), max=np.log(100.0))
            tot += loss.item()
        sched.step()

        # evaluate() returns cell_recall@5 / text_recall@5 (protein is the "cell" side here)
        metrics = evaluate(model, te_loader, device, recall_k=5)
        print(f"  ep {epoch+1:03d}/{args.epochs} | train {tot/len(tr_loader):.4f} | "
              f"test {metrics['loss']:.4f} | gap {metrics['gap']:.3f} | "
              f"protein->text R@5 {metrics['cell_recall@5']:.3f} | "
              f"text->protein R@5 {metrics['text_recall@5']:.3f}")

        if metrics["loss"] < best_loss:
            best_loss, best_metrics, best_epoch = metrics["loss"], metrics, epoch
            if args.save_checkpoint:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "vae_emb": prot_emb,           # protein matrix (fig_eval reads shape[1])
                    "train_idx": train_idx, "test_idx": test_idx,
                    "gene_ids": gene_ids,
                    "esm_dim": prot_emb.shape[1], "text_dim": text_emb.shape[1],
                    "esm_model": str(z["esm_model"]) if "esm_model" in z else None,
                    "text_model": str(z["text_model"]) if "text_model" in z else None,
                    "projection_dim": args.projection_dim,
                    "depth": args.depth, "train_frac": args.train_frac, "seed": args.seed,
                }, ckpt_path, pickle_protocol=4)
                print(f"    saved {ckpt_path.name} (loss {metrics['loss']:.4f})")

    rand_baseline = 5.0 / len(test_idx)
    rec = {
        "run_tag": run_tag, "depth": args.depth, "train_frac": args.train_frac,
        "seed": args.seed, "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "n_params": n_params, "best_epoch": int(best_epoch),
        "loss": float(best_metrics["loss"]), "gap": float(best_metrics["gap"]),
        "protein_to_text_recall@5": float(best_metrics["cell_recall@5"]),
        "text_to_protein_recall@5": float(best_metrics["text_recall@5"]),
        "random_recall@5_baseline": float(rand_baseline),
        "esm_dim": int(prot_emb.shape[1]), "text_dim": int(text_emb.shape[1]),
        "projection_dim": args.projection_dim,
        "device": str(device), "epochs": args.epochs, "wall_sec": round(time.time() - t0, 1),
        "checkpoint": (str(ckpt_path) if args.save_checkpoint else None),
    }
    _atomic_write_json(rec, out_dir / f"metrics_{run_tag}.json")
    print(f"[done] {run_tag}: protein->text R@5={rec['protein_to_text_recall@5']:.4f} "
          f"text->protein R@5={rec['text_to_protein_recall@5']:.4f} "
          f"(rand {rand_baseline:.4g}) wall={rec['wall_sec']}s")


if __name__ == "__main__":
    main()
