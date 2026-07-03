"""fetch_flybase.py — download the FlyBase inputs for the Gene Rep protein<->text CLIP.

Stage 1a of the Gene Rep pipeline. Downloads, for the current *D. melanogaster*
FlyBase release:

  1. Protein FASTA  : dmel-all-translation-r<rel>.fasta.gz   (one seq per FBpp;
                      header carries parent=FBgn...,FBtr... and name=<symbol>-P<x>)
  2. Gene summaries : automated_gene_summaries_fb_<rel>.tsv.gz   (computed, per FBgn)
                      best_gene_summary_fb_<rel>.tsv.gz          (curated-preferred)

IMPORTANT (cluster): ftp.flybase.org:443 is BLOCKED from the compute nodes. This
script uses the https mirror https://s3ftp.flybase.org/... (verified reachable),
stdlib urllib only — matching the working pattern in
`2 - Cel Rep/Drosophila/evaluations/cell_embedding_models/fetch_external_genesets.py`.

Runs in GENE_REP_ENV on a CPU partition (network egress, no GPU). Idempotent:
files already present with a plausible size are skipped unless --force.

    python data_prep/fetch_flybase.py            # -> data/raw/
    python data_prep/fetch_flybase.py --force    # re-download
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "generep").is_dir():
        sys.path.insert(0, str(_parent))
        break

from generep import paths as gpaths  # noqa: E402

MIRROR = "https://s3ftp.flybase.org/releases/current"
UA = {"User-Agent": "fly-generep/1.0"}


def _read(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def resolve_release() -> tuple[str, str]:
    """Return (release_dir, version) for the current dmel release, e.g.
    ('dmel_r6.68', '6.68'), by listing the `current/` mirror directory."""
    listing = _read(f"{MIRROR}/", timeout=60).decode("utf-8", "replace")
    rels = sorted(set(re.findall(r"dmel_r[0-9.]+", listing)))
    if not rels:
        raise SystemExit(f"Could not find a dmel release dir in {MIRROR}/")
    rel = rels[-1]
    version = rel.replace("dmel_r", "")
    return rel, version


def list_genes_dir() -> list[str]:
    """Filenames present under precomputed_files/genes/ (release-stamped)."""
    listing = _read(f"{MIRROR}/precomputed_files/genes/", timeout=60).decode("utf-8", "replace")
    return sorted(set(re.findall(r"[A-Za-z0-9_]+_fb_\d{4}_\d{2}\.tsv\.gz", listing)))


def _pick(files: list[str], prefix: str) -> str | None:
    """The release-stamped file whose name starts with `prefix`."""
    cands = [f for f in files if f.startswith(prefix)]
    return sorted(cands)[-1] if cands else None


def _download(url: str, dest: Path, force: bool, min_bytes: int = 1024) -> None:
    if dest.exists() and dest.stat().st_size >= min_bytes and not force:
        print(f"  [skip] {dest.name} ({dest.stat().st_size:,} bytes present)")
        return
    print(f"  [get ] {url}")
    t0 = time.time()
    data = _read(url)
    tmp = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    print(f"  [ok  ] {dest.name} ({len(data):,} bytes, {time.time()-t0:.1f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None, help="default: generep RAW_DATA_DIR")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    out = Path(args.out_dir) if args.out_dir else gpaths.RAW_DATA_DIR
    gpaths.ensure_dir(out)

    rel, version = resolve_release()
    print(f"[flybase] current release: {rel} (r{version})")
    genes_files = list_genes_dir()
    auto_sum = _pick(genes_files, "automated_gene_summaries")
    best_sum = _pick(genes_files, "best_gene_summary")
    fbgn_map = _pick(genes_files, "fbgn_fbtr_fbpp_expanded")
    print(f"[flybase] genes/ picks: summaries={auto_sum} best={best_sum} map={fbgn_map}")

    fasta_url = f"{MIRROR}/{rel}/fasta/dmel-all-translation-r{version}.fasta.gz"
    prov = {"release": rel, "version": version, "mirror": MIRROR, "files": {}}

    print("[flybase] downloading protein FASTA...")
    _download(fasta_url, out / "dmel-all-translation.fasta.gz", args.force, min_bytes=1_000_000)
    prov["files"]["protein_fasta"] = fasta_url

    print("[flybase] downloading gene summaries...")
    if auto_sum:
        _download(f"{MIRROR}/precomputed_files/genes/{auto_sum}",
                  out / "dmel_automated_gene_summaries.tsv.gz", args.force)
        prov["files"]["automated_gene_summaries"] = auto_sum
    if best_sum:
        _download(f"{MIRROR}/precomputed_files/genes/{best_sum}",
                  out / "dmel_best_gene_summary.tsv.gz", args.force)
        prov["files"]["best_gene_summary"] = best_sum
    if fbgn_map:
        # FBgn<->FBtr<->FBpp expanded map; not strictly needed (FASTA header carries
        # parent FBgn) but handy for cross-checks and gene-type filtering.
        _download(f"{MIRROR}/precomputed_files/genes/{fbgn_map}",
                  out / "dmel_fbgn_fbtr_fbpp_expanded.tsv.gz", args.force)
        prov["files"]["fbgn_fbtr_fbpp_expanded"] = fbgn_map

    (out / "flybase_provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"[flybase] done -> {out}")
    print(f"[flybase] provenance: {out / 'flybase_provenance.json'}")


if __name__ == "__main__":
    main()
