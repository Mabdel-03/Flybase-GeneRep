"""eval_clip_retrieval.py — retrieval + confusion evals for a trained GeneCLIP.

Stage 4 tier 2. Loads a gene_train_clip.py checkpoint and, on its frozen test
split, reports:

  * protein->text and text->protein recall@k over the WHOLE test set (not just
    within a 512-batch, unlike the in-loop recall) — the honest retrieval number.
  * A gene-function retrieval-confusion grid: project every test protein into the
    shared space, score against per-gene-group text CENTROIDS, argmax -> predicted
    group; row-normalized confusion vs the true gene group, plus micro/macro
    accuracy and macro-F1 with a chance baseline.

Rebuilds GeneCLIP from the checkpoint (esm_dim/text_dim/depth/projection_dim) and
reuses the Cel Rep evaluate() philosophy; confusion uses the same
row_norm_confusion / predict_argmax / metrics_block idioms as the Cel Rep
fig2_eval.py, reimplemented compactly (sklearn/numpy only, no scanpy).

    python eval_clip_retrieval.py --ckpt <gene_clip_*.pt> --cache <clip_inputs_dmel.npz>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

for _parent in Path(__file__).resolve().parents:
    if (_parent / "generep").is_dir():
        sys.path.insert(0, str(_parent))
        break
from generep import paths as gpaths  # noqa: E402

# Reuse the GeneCLIP definition + the imported Cel Rep Adapter.
_TRAIN = gpaths.TRAINING_DIR / "clip_style_models"
sys.path.insert(0, str(_TRAIN))
from gene_train_clip import GeneCLIP  # noqa: E402


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    esm_dim = int(ck["esm_dim"]); text_dim = int(ck["text_dim"])
    depth = int(ck["depth"]); pdim = int(ck.get("projection_dim", 2048))
    model = GeneCLIP(esm_dim, text_dim, projection_dim=pdim, depth=depth)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    return model, ck


def recall_at_k(pf, tf, ks=(1, 5, 10)):
    """Full-matrix recall@k both directions. pf,tf are L2-normed (N,D) tensors."""
    sim = pf @ tf.T                                    # (N, N)
    n = sim.size(0)
    diag = torch.arange(n, device=sim.device)
    out = {}
    for k in ks:
        kk = min(k, n)
        p2t = sim.topk(kk, dim=1).indices
        t2p = sim.topk(kk, dim=0).indices
        out[f"protein_to_text_recall@{k}"] = float(
            (p2t == diag.unsqueeze(1)).any(1).float().mean())
        out[f"text_to_protein_recall@{k}"] = float(
            (t2p == diag.unsqueeze(0)).any(0).float().mean())
    return out


def _labels_broad(gene_ids, gg_json):
    gg = json.loads(Path(gg_json).read_text()) if Path(gg_json).exists() else {}
    gm = {}
    for gid, rec in gg.items():
        name = rec.get("name") or gid
        for g in rec.get("genes", []):
            gm.setdefault(g, []).append((len(rec.get("genes", [])), name))
    return np.array([max(gm[g])[1] if g in gm else "unlabeled" for g in gene_ids])


def group_confusion(protein_feats, text_feats, labels, min_per_class=5):
    """Project proteins, score vs per-group text centroids, argmax -> predicted
    group; row-normalized confusion + micro/macro/F1 vs true group."""
    from sklearn.metrics import f1_score
    lab_mask = labels != "unlabeled"
    pf, tf, lab = protein_feats[lab_mask], text_feats[lab_mask], labels[lab_mask]
    import pandas as pd
    vc = pd.Series(lab).value_counts()
    classes = sorted(vc[vc >= min_per_class].index.tolist())
    if len(classes) < 2:
        return None
    keep = np.isin(lab, classes)
    pf, tf, lab = pf[keep], tf[keep], lab[keep]
    cls_idx = {c: i for i, c in enumerate(classes)}
    # text centroid per group (unit-normed)
    centroids = np.stack([tf[lab == c].mean(0) for c in classes])
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    scores = pf @ centroids.T                          # (M, C)
    pred = np.array(classes)[scores.argmax(1)]
    true_code = np.array([cls_idx[c] for c in lab])
    pred_code = np.array([cls_idx[c] for c in pred])
    C = len(classes)
    conf = np.zeros((C, C), dtype=np.float64)
    for t, p in zip(true_code, pred_code):
        conf[t, p] += 1
    row = conf / np.clip(conf.sum(1, keepdims=True), 1, None)
    micro = float((pred == lab).mean())
    macro = float(np.mean([row[i, i] for i in range(C)]))
    f1 = float(f1_score(true_code, pred_code, average="macro"))
    return {"n_classes": C, "n_eval": int(len(lab)),
            "micro_accuracy": micro, "macro_accuracy_diag": macro,
            "macro_f1": f1, "chance_baseline": 1.0 / C,
            "classes": classes, "row_norm_confusion": row.tolist()}


def render_confusion_png(conf_block, out_png, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [fig] matplotlib unavailable ({e}); skipping PNG")
        return
    row = np.asarray(conf_block["row_norm_confusion"])
    classes = conf_block["classes"]
    C = len(classes)
    fig, ax = plt.subplots(figsize=(min(0.28 * C + 3, 22), min(0.28 * C + 3, 22)))
    im = ax.imshow(row, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_title(f"{title}\nmicro={conf_block['micro_accuracy']:.3f} "
                 f"macroF1={conf_block['macro_f1']:.3f} chance={conf_block['chance_baseline']:.3g}")
    ax.set_xlabel("predicted gene group"); ax.set_ylabel("true gene group")
    if C <= 40:
        ax.set_xticks(range(C)); ax.set_xticklabels(classes, rotation=90, fontsize=5)
        ax.set_yticks(range(C)); ax.set_yticklabels(classes, fontsize=5)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  [fig] wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    cfg = gpaths.gene_config("dmel")
    cache = Path(args.cache) if args.cache else cfg["clip_cache"]
    device = torch.device(args.device)
    out_dir = Path(args.out_dir) if args.out_dir else gpaths.CLIP_EVAL_DIR / "outputs"
    gpaths.ensure_dir(out_dir)

    model, ck = load_model(args.ckpt, device)
    z = np.load(cache, allow_pickle=True)
    prot = np.asarray(z["vae_emb"], dtype=np.float32)
    text = np.asarray(z["text_emb"], dtype=np.float32)
    gene_ids = (z["barcodes"] if "barcodes" in z else z["gene_ids"]).astype(str)
    test_idx = ck["test_idx"]
    print(f"[eval] ckpt depth={ck['depth']} | test set {len(test_idx)} genes")

    t0 = time.time()
    with torch.inference_mode():
        pf = F.normalize(model.protein_adapter(torch.tensor(prot[test_idx]).to(device)), dim=-1)
        tf = F.normalize(model.text_adapter(torch.tensor(text[test_idx]).to(device)), dim=-1)
    rec = recall_at_k(pf, tf)
    print("  recall:", {k: round(v, 4) for k, v in rec.items()})

    labels = _labels_broad(gene_ids[test_idx], cfg["gene_groups_json"])
    conf = group_confusion(pf.cpu().numpy(), tf.cpu().numpy(), labels)
    if conf:
        print(f"  gene-group retrieval: micro={conf['micro_accuracy']:.3f} "
              f"macroF1={conf['macro_f1']:.3f} chance={conf['chance_baseline']:.3g} "
              f"({conf['n_classes']} groups)")
        render_confusion_png(conf, out_dir / f"retrieval_confusion_d{ck['depth']:02d}.png",
                             f"GeneCLIP depth {ck['depth']} — protein→group retrieval")

    result = {"ckpt": str(args.ckpt), "cache": str(cache), "depth": int(ck["depth"]),
              "train_frac": ck.get("train_frac"), "n_test": int(len(test_idx)),
              "recall": rec, "gene_group_confusion": conf, "wall_sec": round(time.time() - t0, 1)}
    out = out_dir / f"clip_retrieval_d{ck['depth']:02d}.json"
    out.write_text(json.dumps({k: v for k, v in result.items() if k != "gene_group_confusion"}
                              | {"gene_group_confusion": (
                                  {kk: vv for kk, vv in (conf or {}).items()
                                   if kk not in ("row_norm_confusion", "classes")})}, indent=2))
    print(f"[eval] wrote {out}  ({result['wall_sec']}s)")


if __name__ == "__main__":
    main()
