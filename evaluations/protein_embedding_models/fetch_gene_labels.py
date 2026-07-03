"""fetch_gene_labels.py — FBgn-keyed gene-function labels for the Gene Rep evals.

Downloads two annotation-independent, FBgn-keyed label sources (matching the
plan's decision to use BOTH):

  go_bp.json          GO Biological Process, FBgn -> GO terms, from the GO
                      Consortium FlyBase GAF.
  flybase_groups.json FlyBase Gene Groups (functional/family grouping),
                      FBgg -> member FBgn, from the s3ftp mirror.

Written to data/derived/genesets/ (Gene-Rep-owned; does not touch the Cel Rep or
TF_HOME copies). Adapted from
`2 - Cel Rep/Drosophila/evaluations/cell_embedding_models/fetch_external_genesets.py`.

IMPORTANT: ftp.flybase.org:443 is blocked from compute nodes — this uses the
https s3ftp mirror + the GO Consortium https endpoint, stdlib urllib only. Runs
in GENE_REP_ENV on a CPU partition.

    python evaluations/protein_embedding_models/fetch_gene_labels.py
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "generep").is_dir():
        sys.path.insert(0, str(_parent))
        break

from generep import paths as gpaths  # noqa: E402

MIRROR_GENES = "https://s3ftp.flybase.org/releases/current/precomputed_files/genes"
GO_URL = "https://current.geneontology.org/annotations/fb.gaf.gz"
UA = {"User-Agent": "fly-generep/1.0"}


def _get(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _current_gene_group_file() -> str:
    listing = _get(f"{MIRROR_GENES}/", timeout=60).decode("utf-8", "replace")
    cands = sorted(set(re.findall(r"gene_group_data_fb_\d{4}_\d{2}\.tsv\.gz", listing)))
    if not cands:
        raise SystemExit(f"no gene_group_data file found at {MIRROR_GENES}/")
    return cands[-1]


def fetch_gene_groups(min_genes: int, max_genes: int) -> dict:
    fname = _current_gene_group_file()
    data = gzip.decompress(_get(f"{MIRROR_GENES}/{fname}"))
    groups = defaultdict(lambda: {"name": None, "genes": set()})
    for ln in data.decode("utf-8", "replace").splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split("\t")
        if len(f) < 7:
            continue
        gid, _gsym, gname, _pid, _psym, fbgn, _sym = f[:7]
        if gid.startswith("FBgg") and fbgn.startswith("FBgn"):
            groups[gid]["name"] = gname
            groups[gid]["genes"].add(fbgn)
    out = {gid: {"name": v["name"], "genes": sorted(v["genes"])}
           for gid, v in groups.items() if min_genes <= len(v["genes"]) <= max_genes}
    return out, fname


def fetch_go_bp(min_genes: int, max_genes: int) -> dict:
    """FlyBase GAF: col2=FBgn (DB_Object_ID), col5=GO id, col9=aspect (P=BP)."""
    data = gzip.decompress(_get(GO_URL))
    terms = defaultdict(set)
    for ln in data.decode("utf-8", "replace").splitlines():
        if ln.startswith("!") or not ln.strip():
            continue
        f = ln.split("\t")
        if len(f) < 15:
            continue
        db_obj_id, go_id, aspect = f[1], f[4], f[8]
        if aspect == "P" and db_obj_id.startswith("FBgn") and go_id.startswith("GO:"):
            terms[go_id].add(db_obj_id)
    return {gid: {"name": gid, "genes": sorted(v)}
            for gid, v in terms.items() if min_genes <= len(v) <= max_genes}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None, help="default: generep GENESETS_DIR")
    ap.add_argument("--min-genes", type=int, default=5)
    ap.add_argument("--max-genes", type=int, default=500)
    args = ap.parse_args()

    out = Path(args.out_dir) if args.out_dir else gpaths.GENESETS_DIR
    gpaths.ensure_dir(out)
    prov = {"min_genes": args.min_genes, "max_genes": args.max_genes}

    print("[labels] FlyBase gene groups...")
    gg, gg_file = fetch_gene_groups(args.min_genes, args.max_genes)
    (out / "flybase_groups.json").write_text(json.dumps(gg, indent=2))
    prov["flybase_groups"] = {"file": gg_file, "n_groups": len(gg)}
    print(f"  groups (size {args.min_genes}-{args.max_genes}): {len(gg)}")

    print("[labels] GO Biological Process (GO Consortium FB GAF)...")
    go = fetch_go_bp(args.min_genes, args.max_genes)
    (out / "go_bp.json").write_text(json.dumps(go, indent=2))
    prov["go_bp"] = {"url": GO_URL, "n_terms": len(go)}
    print(f"  GO-BP terms (size {args.min_genes}-{args.max_genes}): {len(go)}")

    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"[labels] done -> {out}")


if __name__ == "__main__":
    main()
