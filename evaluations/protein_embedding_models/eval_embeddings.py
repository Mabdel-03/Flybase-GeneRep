"""eval_embeddings.py — embedding-quality evals for the raw ESM + BioBERT spaces.

Stage 4 tier 1. Answers "does each raw embedding carry gene-function structure
BEFORE any CLIP training?" on FBgn-keyed gene-function labels (both GO-BP and
FlyBase gene groups, per the plan). For each embedding (ESM protein, BioBERT
text) and each label set it reports:

  * kNN label-transfer accuracy + macro-F1 (k=15) on a stratified split
  * cell-type-style silhouette (ASW) of the label partition
  * GO/gene-group program decodability: ridge CV-R2 of predicting each program's
    membership indicator from the embedding (mean over programs)
  * participation ratio (effective dimensionality) of the embedding

Reuses the metric recipes from the Cel Rep label-free eval (knn_label_transfer /
silhouette style from tf_02_benchmark.py, _cv_r2 / participation_ratio /
t13-style program decodability from tf_03_labelfree_eval.py) but reimplemented
compactly here so it has no scanpy/h5ad dependency (pure sklearn/numpy).

    python eval_embeddings.py                       # both embeddings, both labels
    python eval_embeddings.py --embeddings esm      # ESM only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "generep").is_dir():
        sys.path.insert(0, str(_parent))
        break
from generep import paths as gpaths  # noqa: E402


def load_cache(cache_path):
    z = np.load(cache_path, allow_pickle=True)
    return {
        "esm": np.asarray(z["vae_emb"], dtype=np.float32),
        "text": np.asarray(z["text_emb"], dtype=np.float32),
        "gene_ids": (z["barcodes"] if "barcodes" in z else z["gene_ids"]).astype(str),
    }


def stratified_test_idx(labels, test_frac=0.2, seed=42, min_class=2):
    from sklearn.model_selection import train_test_split
    import pandas as pd
    ser = pd.Series(labels)
    vc = ser.value_counts()
    keep = vc[vc >= min_class].index
    pool = np.where(ser.isin(keep).to_numpy())[0]
    if len(pool) == 0 or ser.iloc[pool].nunique() < 2:
        return None
    y = ser.iloc[pool].to_numpy()
    if int(round(len(pool) * test_frac)) < ser.iloc[pool].nunique():
        # too granular to stratify -> plain split
        tr, te = train_test_split(pool, test_size=test_frac, random_state=seed)
    else:
        tr, te = train_test_split(pool, test_size=test_frac, random_state=seed, stratify=y)
    return np.sort(tr), np.sort(te)


def knn_label_transfer(emb, labels, tr, te, k=15):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, f1_score
    k = min(k, max(1, len(tr) - 1))
    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1).fit(emb[tr], labels[tr])
    pred = clf.predict(emb[te])
    return {"knn_accuracy": float(accuracy_score(labels[te], pred)),
            "knn_macro_f1": float(f1_score(labels[te], pred, average="macro")),
            "k": k, "n_test": int(len(te))}


def silhouette(emb, labels, max_cells=8000, seed=42):
    from sklearn.metrics import silhouette_score
    import pandas as pd
    ser = pd.Series(labels)
    mask = ser.isin(ser.value_counts()[lambda s: s >= 2].index).to_numpy()
    e, l = emb[mask], labels[mask]
    if len(e) > max_cells:
        rng = np.random.RandomState(seed)
        sel = rng.permutation(len(e))[:max_cells]
        e, l = e[sel], l[sel]
    if len(np.unique(l)) < 2:
        return None
    return float(silhouette_score(e, l))


def _cv_r2(X, y, seed=42, folds=5):
    """5-fold CV R2 of a ridge probe predicting y from X (per-fold standardized)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    kf = KFold(n_splits=min(folds, max(2, len(y) // 10)), shuffle=True, random_state=seed)
    scores = []
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        scores.append(r2_score(y[te], m.predict(sc.transform(X[te]))))
    return float(np.mean(scores))


def program_decodability(emb, gene_ids, program_json, max_programs=150, seed=42):
    """Mean ridge CV-R2 of predicting each program's membership indicator from emb."""
    if not Path(program_json).exists():
        return None
    programs = json.loads(Path(program_json).read_text())
    idx = {g: i for i, g in enumerate(gene_ids)}
    r2s = []
    rng = np.random.RandomState(seed)
    keys = list(programs.keys())
    if len(keys) > max_programs:
        keys = [keys[i] for i in rng.permutation(len(keys))[:max_programs]]
    for pid in keys:
        members = [idx[g] for g in programs[pid].get("genes", []) if g in idx]
        if len(members) < 5:
            continue
        y = np.zeros(len(gene_ids), dtype=np.float32)
        y[members] = 1.0
        # balance: all positives + an equal random sample of negatives
        pos = np.array(members)
        neg_all = np.setdiff1d(np.arange(len(gene_ids)), pos)
        neg = rng.choice(neg_all, size=min(len(pos) * 3, len(neg_all)), replace=False)
        sub = np.concatenate([pos, neg])
        r2s.append(_cv_r2(emb[sub], y[sub], seed=seed))
    if not r2s:
        return None
    return {"mean_r2": float(np.mean(r2s)), "n_programs": len(r2s)}


def participation_ratio(emb):
    """Effective dimensionality = (sum eig)^2 / sum(eig^2) of the covariance."""
    X = emb - emb.mean(0, keepdims=True)
    cov = (X.T @ X) / max(1, len(X) - 1)
    ev = np.linalg.eigvalsh(cov)
    ev = ev[ev > 0]
    return float((ev.sum() ** 2) / (np.square(ev).sum() + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--embeddings", nargs="+", choices=["esm", "text"], default=["esm", "text"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = gpaths.gene_config("dmel")
    cache = Path(args.cache) if args.cache else cfg["clip_cache"]
    if not cache.exists():
        raise SystemExit(f"cache not found: {cache} — run make_cache.py first")
    data = load_cache(cache)
    gene_ids = data["gene_ids"]
    n = len(gene_ids)
    print(f"[eval] {n} genes | ESM {data['esm'].shape} | text {data['text'].shape}")

    # single-label vectors for kNN/silhouette: most-specific GO-BP and gene group
    def fine_labels_local():
        go = json.loads(Path(cfg["go_bp_json"]).read_text()) if Path(cfg["go_bp_json"]).exists() else {}
        gt = {}
        for term, rec in go.items():
            for g in rec.get("genes", []):
                gt.setdefault(g, []).append((len(rec.get("genes", [])), term))
        return np.array([min(gt[g])[1] if g in gt else "unlabeled" for g in gene_ids])

    def broad_labels_local():
        gg = json.loads(Path(cfg["gene_groups_json"]).read_text()) if Path(cfg["gene_groups_json"]).exists() else {}
        gm = {}
        for gid, rec in gg.items():
            name = rec.get("name") or gid
            for g in rec.get("genes", []):
                gm.setdefault(g, []).append((len(rec.get("genes", [])), name))
        return np.array([max(gm[g])[1] if g in gm else "unlabeled" for g in gene_ids])

    label_sets = {"go_bp_fine": fine_labels_local(), "gene_group_broad": broad_labels_local()}

    results = {"cache": str(cache), "n_genes": int(n), "embeddings": {}}
    t0 = time.time()
    for emb_name in args.embeddings:
        emb = data[emb_name]
        block = {"dim": int(emb.shape[1]),
                 "participation_ratio": participation_ratio(emb)}
        for lname, labels in label_sets.items():
            labelled = labels != "unlabeled"
            e, l = emb[labelled], labels[labelled]
            n_lab = int(labelled.sum())
            split = stratified_test_idx(l) if n_lab > 50 else None
            lb = {"n_labelled": n_lab, "n_classes": int(len(np.unique(l)))}
            if split is not None:
                tr, te = split
                lb["knn"] = knn_label_transfer(e, l, tr, te)
                lb["silhouette"] = silhouette(e, l)
            prog_json = cfg["go_bp_json"] if lname.startswith("go") else cfg["gene_groups_json"]
            lb["program_decodability"] = program_decodability(emb, gene_ids, prog_json, seed=args.seed)
            block[lname] = lb
            print(f"  [{emb_name}/{lname}] knn_acc="
                  f"{lb.get('knn', {}).get('knn_accuracy', float('nan')):.3f} "
                  f"sil={lb.get('silhouette')} "
                  f"prog_r2={ (lb['program_decodability'] or {}).get('mean_r2') }")
        results["embeddings"][emb_name] = block

    results["wall_sec"] = round(time.time() - t0, 1)
    out = Path(args.out) if args.out else gpaths.PROTEIN_EVAL_DIR / "outputs" / "embedding_quality.json"
    gpaths.ensure_dir(out.parent)
    out.write_text(json.dumps(results, indent=2))
    print(f"[eval] wrote {out}  ({results['wall_sec']}s)")


if __name__ == "__main__":
    main()
