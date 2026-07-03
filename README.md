# Flybase — Gene Rep

Gene representation learning for *Drosophila melanogaster*. This stage builds a
joint **protein–language** model: for every protein-coding gene it co-embeds the
**ESM-2 embedding of the gene's protein** with the **BioBERT embedding of the
gene's FlyBase text description**, aligned by a **CLIP-style projection head
trained with symmetric InfoNCE**.

The contrastive core (projection MLP + InfoNCE loss + recall@k eval) is the same
machinery validated in `2 - Cel Rep` (CellCLIP) — it is **imported, not copied**
(`generep.paths.import_cel_rep_core`). Where CellCLIP aligns *cell ↔ caption*,
GeneCLIP aligns *protein ↔ gene-description*, one row per gene.

## Directory map

| Conceptual bucket | Directory |
| --- | --- |
| Shared helpers | `generep/` |
| Data prep | `data_prep/` |
| Embedding + CLIP training | `model_training/` |
| Inference | `model_inference/` |
| Evaluations | `evaluations/` |
| Data & caches (gitignored) | `data/` |

```text
3 - Gene Rep/
  generep/paths.py                     path + Cel-Rep-import helpers
  data_prep/
    fetch_flybase.py                   download FASTA + gene summaries (s3ftp mirror)
    build_gene_table.py                one row per gene, longest-isoform protein
  model_training/
    protein_embedding_models/gen_protein_emb.py   ESM-2 protein embeddings
    text_embedding_models/gen_gene_text_emb.py     BioBERT text embeddings
    clip_style_models/
      make_cache.py                    align protein||text||labels -> .npz
      gene_train_clip.py               GeneCLIP trainer (imports Cel Rep core)
      clip_encoder_depth/              depth x train-frac sbatch sweep
  evaluations/
    protein_embedding_models/
      fetch_gene_labels.py             GO-BP + FlyBase gene groups (FBgn-keyed)
      eval_embeddings.py               kNN / silhouette / decodability on raw embeddings
    clip_style_models/eval_clip_retrieval.py   recall@k + retrieval-confusion grid
  model_inference/clip_style_models/clip_inference.py   text->gene / gene->gene query
  data/                                (gitignored) raw downloads + derived embeddings + caches
```

## Environment

Gene Rep owns its **own** conda env `GENE_REP_ENV` (`config/paths.sh`), not
shared with `cel_rep`/`tf`/`consortium`. It has torch (cu121), `fair-esm`,
`transformers`, `biopython`, `requests`, `scikit-learn`, `pandas`,
`sentencepiece`. Create it once:

```bash
cd /orcd/data/lhtsai/001/mabdel03/Flybase
L=/orcd/scratch/orcd/012/mabdel03/gene_rep/logs
sbatch --export=ALL,REPO_ROOT="$(pwd)" -p mit_normal \
  --output="$L/gene_rep_env_%j.out" --error="$L/gene_rep_env_%j.out" \
  "3 - Gene Rep/model_training/protein_embedding_models/sbatch_make_env.sh"
```

Encoders (overridable via env): `GENE_REP_ESM_MODEL=facebook/esm2_t36_3B_UR50D`
(2560-d) and `GENE_REP_TEXT_MODEL=monologg/biobert_v1.1_pubmed` (768-d, a
safetensors BioBERT v1.1 mirror that loads under transformers 5.x + torch<2.6).

Always resolve paths through the repo config:

```bash
source config/paths.sh && check_paths
```

## Pipeline

Run each stage's python via `"${GENE_REP_ENV}/bin/python"` (absolute path avoids
SLURM env-retrieval issues). GPU jobs are submitted with **space-free log dirs**.

```bash
PY="${GENE_REP_ENV}/bin/python"

# 1. Data prep (CPU; s3ftp mirror — ftp.flybase.org is blocked on compute nodes)
$PY "3 - Gene Rep/data_prep/fetch_flybase.py"
$PY "3 - Gene Rep/data_prep/build_gene_table.py"        # -> gene_table_dmel.parquet (~13,986 genes)

# 2. Embeddings
#   ESM-2 3B (GPU, big VRAM):
sbatch --export=ALL,REPO_ROOT="$(pwd)" -p mit_normal_gpu \
  --output="$L/esm_emb_%j.out" --error="$L/esm_emb_%j.out" \
  "3 - Gene Rep/model_training/protein_embedding_models/sbatch_gen_protein_emb.sh"
#   BioBERT (GPU or CPU):
sbatch --export=ALL,REPO_ROOT="$(pwd)" -p mit_normal_gpu \
  --output="$L/text_emb_%j.out" --error="$L/text_emb_%j.out" \
  "3 - Gene Rep/model_training/text_embedding_models/sbatch_gen_text_emb.sh"

# 3. Labels + aligned cache
$PY "3 - Gene Rep/evaluations/protein_embedding_models/fetch_gene_labels.py"
$PY "3 - Gene Rep/model_training/clip_style_models/make_cache.py"   # -> clip_inputs_dmel.npz

# 4. Train the GeneCLIP sweep (depth x train-frac)
bash "3 - Gene Rep/model_training/clip_style_models/clip_encoder_depth/submit.sh"
#   (DRY_RUN=1 bash .../submit.sh  prints the index->(depth,frac,tag) table)

# 5. Evals
$PY "3 - Gene Rep/evaluations/protein_embedding_models/eval_embeddings.py"   # raw-embedding quality
$PY "3 - Gene Rep/evaluations/clip_style_models/eval_clip_retrieval.py" --ckpt <gene_clip_*.pt>

# 6. Inference
$PY "3 - Gene Rep/model_inference/clip_style_models/clip_inference.py" \
   --ckpt <gene_clip_*.pt> --text "serine protease in immune response" --topk 10
```

## Artifact locations

- Raw FlyBase downloads + `gene_table_dmel.parquet`: `data/raw/`
- ESM protein embeddings: `data/derived/protein_embeddings/`
- BioBERT text embeddings: `data/derived/text_embeddings/`
- Gene-function labels (GO-BP, gene groups): `data/derived/genesets/`
- Aligned CLIP cache: `data/derived/clip_inputs_dmel.npz`
- Sweep runs + metrics + figures: `evaluations/clip_style_models/clip_encoder_depth/outputs/`
- SLURM logs: a space-free scratch dir (`/orcd/scratch/orcd/012/mabdel03/gene_rep/logs`)

Large artifacts (`*.npy *.pt *.h5 *.npz data/ outputs/`) are gitignored.

## Cache schema (interoperability contract)

`clip_inputs_dmel.npz` uses **Cel-Rep-compatible key names** so the imported
trainer/eval code works unchanged:

| key | meaning |
| --- | --- |
| `vae_emb` | ESM protein matrix (N × 2560) — the "left" modality |
| `text_emb` | BioBERT text matrix (N × 768) — the "right" modality |
| `labels` | fine gene-function label (most-specific GO-BP term) |
| `labels_broad` | broad label (FlyBase gene group) |
| `barcodes` / `gene_ids` | FBgn ids — the join key / gene order |

Labels are used only for the stratified split and label-transfer evals; the CLIP
objective itself is label-free (protein↔text InfoNCE).

## Provenance

- FlyBase release r6.68 (FB2026_02), `https://s3ftp.flybase.org` mirror.
- ESM-2: Lin et al., Science 2023 (`fair-esm`).
- BioBERT: Lee et al., Bioinformatics 2020.
- Contrastive recipe adapted from `2 - Cel Rep` (Cell2Sentence / CellWhisperer /
  CLIP), imported verbatim.
