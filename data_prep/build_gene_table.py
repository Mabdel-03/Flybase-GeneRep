"""build_gene_table.py — one row per Drosophila protein-coding gene.

Stage 1b of the Gene Rep pipeline. Consumes the FlyBase downloads from
fetch_flybase.py and produces a single canonical table:

    data/raw/gene_table_dmel.parquet
    columns: [fbgn, symbol, protein_seq, seq_len, n_isoforms, description, desc_source]

Rules (matching the plan):
  * Restrict to genes whose FlyBase gene_type == 'protein_coding_gene'
    (from fbgn_fbtr_fbpp_expanded), so no ncRNA/tRNA/etc.
  * Map every protein isoform (FBpp) -> parent FBgn straight from the FASTA
    header (`parent=FBgn...,FBtr...`), and pick the LONGEST isoform per gene as
    the gene's representative protein (deterministic; ties broken by FBpp id).
  * Join the gene text description: prefer best_gene_summary (curated-preferred),
    fall back to automated_gene_summaries. Genes with neither keep an empty
    description (kept if they still have a protein, but flagged).
  * Stable ordering: sort by FBgn. This order is the join key for every
    downstream embedding (ESM + BioBERT both index into this table).

Pure stdlib parse + pandas/pyarrow write (runs in GENE_REP_ENV or any env with
pandas+pyarrow; no GPU).

    python data_prep/build_gene_table.py
"""
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import pandas as pd

for _parent in Path(__file__).resolve().parents:
    if (_parent / "generep").is_dir():
        sys.path.insert(0, str(_parent))
        break

from generep import paths as gpaths  # noqa: E402


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace")


def parse_protein_coding_fbgn(expanded_map: Path) -> dict[str, str]:
    """FBgn -> gene_symbol for genes whose gene_type == protein_coding_gene."""
    keep: dict[str, str] = {}
    with _open_text(expanded_map) as fh:
        for ln in fh:
            if ln.startswith("#") or not ln.strip():
                continue
            f = ln.rstrip("\n").split("\t")
            if len(f) < 4:
                continue
            _org, gene_type, fbgn, symbol = f[0], f[1], f[2], f[3]
            if gene_type == "protein_coding_gene" and fbgn.startswith("FBgn"):
                keep.setdefault(fbgn, symbol)
    return keep


def parse_fasta(fasta_gz: Path) -> dict[str, dict]:
    """Parse the translation FASTA -> per-FBgn best (longest) isoform.

    Returns {fbgn: {"fbpp": str, "symbol": str, "seq": str, "n_isoforms": int}}.
    Header form: '>FBpp0070000 ... name=Nep3-PA; parent=FBgn0031081,FBtr0070000; ...'
    """
    import re
    parent_re = re.compile(r"parent=(FBgn\d+)")
    name_re = re.compile(r"name=([^;]+)")

    best: dict[str, dict] = {}
    counts: dict[str, int] = {}

    cur_fbgn = cur_fbpp = cur_name = None
    cur_seq: list[str] = []

    def _flush():
        if cur_fbgn is None:
            return
        seq = "".join(cur_seq).strip("*").replace("\n", "")
        counts[cur_fbgn] = counts.get(cur_fbgn, 0) + 1
        prev = best.get(cur_fbgn)
        # longest wins; tie -> lexicographically smaller FBpp for determinism
        if (prev is None or len(seq) > len(prev["seq"])
                or (len(seq) == len(prev["seq"]) and cur_fbpp < prev["fbpp"])):
            # derive gene symbol from the isoform name (strip trailing -P<x>)
            sym = cur_name
            if sym and "-" in sym:
                sym = sym.rsplit("-", 1)[0]
            best[cur_fbgn] = {"fbpp": cur_fbpp, "symbol": sym, "seq": seq}

    with _open_text(fasta_gz) as fh:
        for ln in fh:
            if ln.startswith(">"):
                _flush()
                cur_seq = []
                cur_fbpp = ln[1:].split(None, 1)[0]
                pm = parent_re.search(ln)
                nm = name_re.search(ln)
                cur_fbgn = pm.group(1) if pm else None
                cur_name = nm.group(1).strip() if nm else None
            else:
                cur_seq.append(ln.strip())
        _flush()

    for fbgn, rec in best.items():
        rec["n_isoforms"] = counts.get(fbgn, 1)
    return best


def parse_summaries(best_summary: Path | None, automated: Path | None) -> dict[str, tuple[str, str]]:
    """FBgn -> (description, source). best_gene_summary preferred, automated fallback."""
    desc: dict[str, tuple[str, str]] = {}
    # automated first (lower priority), then overwrite with best (higher priority)
    if automated and automated.exists():
        with _open_text(automated) as fh:
            for ln in fh:
                if ln.startswith("#") or not ln.strip():
                    continue
                f = ln.rstrip("\n").split("\t")
                if len(f) >= 2 and f[0].startswith("FBgn") and f[1].strip():
                    desc[f[0]] = (f[1].strip(), "automated")
    if best_summary and best_summary.exists():
        with _open_text(best_summary) as fh:
            for ln in fh:
                if ln.startswith("#") or not ln.strip():
                    continue
                f = ln.rstrip("\n").split("\t")
                # #FBgn_ID  Gene_Symbol  Summary_Source  Summary
                if len(f) >= 4 and f[0].startswith("FBgn") and f[3].strip():
                    desc[f[0]] = (f[3].strip(), f[2].strip() or "best")
    return desc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=None, help="default: generep RAW_DATA_DIR")
    ap.add_argument("--out", default=None, help="default: gene_config()['gene_table']")
    args = ap.parse_args()

    raw = Path(args.raw_dir) if args.raw_dir else gpaths.RAW_DATA_DIR
    cfg = gpaths.gene_config("dmel")
    out = Path(args.out) if args.out else cfg["gene_table"]
    gpaths.ensure_dir(out.parent)

    fasta = raw / "dmel-all-translation.fasta.gz"
    expanded = raw / "dmel_fbgn_fbtr_fbpp_expanded.tsv.gz"
    best_sum = raw / "dmel_best_gene_summary.tsv.gz"
    auto_sum = raw / "dmel_automated_gene_summaries.tsv.gz"
    for p in (fasta, expanded):
        if not p.exists():
            raise SystemExit(f"missing input {p} — run fetch_flybase.py first")

    print("[1/4] parsing protein-coding gene set (fbgn_fbtr_fbpp_expanded)...")
    pc_fbgn = parse_protein_coding_fbgn(expanded)
    print(f"      {len(pc_fbgn)} protein_coding_gene FBgn")

    print("[2/4] parsing translation FASTA -> longest isoform per gene...")
    best_iso = parse_fasta(fasta)
    print(f"      {len(best_iso)} FBgn with >=1 protein isoform")

    print("[3/4] parsing gene descriptions (best -> automated)...")
    desc = parse_summaries(best_sum, auto_sum)
    print(f"      {len(desc)} FBgn with a description")

    print("[4/4] assembling one row per protein-coding gene...")
    rows = []
    for fbgn in sorted(pc_fbgn):
        iso = best_iso.get(fbgn)
        if iso is None or not iso["seq"]:
            continue  # protein-coding but no translated isoform in FASTA — drop
        d_text, d_src = desc.get(fbgn, ("", "none"))
        rows.append({
            "fbgn": fbgn,
            "symbol": iso.get("symbol") or pc_fbgn[fbgn],
            "protein_seq": iso["seq"],
            "seq_len": len(iso["seq"]),
            "n_isoforms": iso["n_isoforms"],
            "description": d_text,
            "desc_source": d_src,
        })

    df = pd.DataFrame(rows).sort_values("fbgn").reset_index(drop=True)
    df.to_parquet(out, index=False)

    n = len(df)
    n_desc = int((df["description"].str.len() > 0).sum())
    print(f"[done] wrote {out}")
    print(f"  genes: {n} | with description: {n_desc} ({100*n_desc/max(n,1):.1f}%)")
    print(f"  seq_len: min={df['seq_len'].min()} median={int(df['seq_len'].median())} "
          f"max={df['seq_len'].max()} | >1022aa: {int((df['seq_len']>1022).sum())}")
    print("  head:")
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(df[["fbgn", "symbol", "seq_len", "n_isoforms", "desc_source"]].head(6).to_string(index=False))


if __name__ == "__main__":
    main()
