```markdown
# GWES (Tree/Ecology-aware GWES pipeline)

## 1) Introduction

This repository implements a **run-directory based pipeline** for detecting **epistasis-like statistical dependencies** between pangenome loci/unitigs while correcting for **phylogeny and shared ecology** using a per-locus logistic random-effect (“eco-bias”) model.

The core idea is:

- Fit per-locus **tip-level marginal probabilities** \(p_i(t)\) that account for structured confounding.
- For each candidate pair \((i,j)\), compute a **structured-mixture null**:
  \[
  p_{11,\text{null}} = \frac{1}{T}\sum_t p_i(t)p_j(t)
  \]
  (and corresponding \(p_{00},p_{01},p_{10}\)), then compute residual scores such as \(\Delta_{11}\), residual log-OR, residual MI, and signed residual MI.

The pipeline is organized as **stages** (stage0 … stage8). Each stage reads/writes artifacts under a single `--run-dir`, making runs reproducible and easier to debug.

---

## 2) Purpose of this tool

This tool is designed for:

- **High-throughput screening** of candidate epistatic pairs (unitigs/loci) from bacterial pangenomes.
- **Reducing false positives** driven by:
  - shared ancestry (phylogeny),
  - shared ecology/environmental structure,
  - structured population effects that inflate LD-like signals.
- Producing outputs suitable for **PAN-GWES style distance–signal plots** and follow-up significance testing with a **parametric bootstrap** under the structured null.

Typical outputs include:
- per-pair structured null probabilities,
- residual statistics (`delta11`, `rlogOR`, `rMI`, `srMI`),
- bootstrapped p-values and BH q-values for top pairs,
- optional plots (`gwes_plotting.r`).

---

## 3) Repository layout (high level)

```

.
├── configs/                     # optional configs/presets
├── gwes/                        # python package
│   ├── **init**.py
│   ├── cli/                     # optional CLI wrappers / future unified CLI
│   ├── stages/                  # stage0 ... stage8 entrypoints
│   ├── model_ecobias.py         # per-locus eco-bias model (MAP/Laplace etc.)
│   ├── phylo.py                 # phylo basis/cov utilities
│   ├── prob_store.py            # P_hat loading (npz/memmap abstraction)
│   ├── pair_stats.py            # logOR/MI utilities and null computations
│   └── ...                      # bits/matrix/fasta helpers
├── scripts/
│   └── gwes_plotting.r          # PAN-GWES style plotting for stage7/8 outputs
├── README.md
└── LICENSE

```

### Run directory layout (`--run-dir`)
A run directory is created/filled by stages. The expected structure is:

```

RUN_DIR/
├── work/
│   ├── stage0/                  # prepared artifacts (aligned tree/fake fasta/etc.)
│   ├── stage1/                  # phylo basis/cov
│   ├── stage2/                  # pairs_obs.tsv
│   ├── stage3/                  # P_hat + locus_fit + global_sigma.tsv
│   ├── stage4/                  # pairs_resid.tsv
│   ├── stage5/                  # flagged loci lists
│   ├── stage6/                  # refit_patch.npz (+ optional P_refit.npz)
│   ├── stage7/                  # pairs_resid_patched.tsv
│   ├── stage8/                  # stage8_bootstrap.tsv
│   └── plots/                   # optional plotting outputs
└── meta/
├── stage0.json
├── stage1.json
└── ...

````

---

## 4) Installation

### 4.1 Python environment

Recommended: create a dedicated environment.

Example (conda):
```bash
conda create -n gwes_env python=3.11 -y
conda activate gwes_env
````

Install the repo in editable mode (run from repo root):

```bash
cd /data/kell7366/developing
python -m pip install -U pip
python -m pip install -e .
```

If you prefer not installing, you can run from repo root (`cd .../developing`) as long as Python can resolve `gwes`, but editable install is more robust.

### 4.2 R (optional; for plotting only)

If you want plots:

* Install R
* Install `data.table`

Inside R:

```r
install.packages("data.table")
```

Or via conda:

```bash
conda install -c conda-forge r-base r-data.table
```

---

## 5) How to run

### 5.1 Inputs

You need:

1. **Tree**: Newick tree file (e.g. IQ-TREE `*.treefile` is fine if it is Newick).
2. **FASTA**: sequences with headers matching tree tips.
3. **Candidate pair file**: list of unitig/locus pairs (e.g. SpydrPick-like pairs).
   Stage0 expects this file and will align indices/tip order across artifacts.

Optional:

* **In-block unitig files** for “true epistasis” highlighting in plots (see §5.4).

### 5.2 Full pipeline (Stages 0–8)

Run from the **repo root** so `python -m gwes...` resolves cleanly.

**Important bash note:** do **not** leave a trailing `\` after the last argument of a command (it will escape the newline and break parsing).

Example:

```bash
#!/usr/bin/env bash
set -euo pipefail

treefile="/data/kell7366/test/eco.fa.treefile"
fastafile="/data/kell7366/test/simulation_eco.fasta"
pairfile="/data/kell7366/test/simulation_eco.mi_filtered.ud_sgg_0_based"

RUN_DIR="/data/kell7366/test/run_sim_eco"

cd /data/kell7366/developing

# Stage0
python -m gwes.stages.stage0_prepare \
  --tree "$treefile" \
  --fasta "$fastafile" \
  --pairs "$pairfile" \
  --run-dir "$RUN_DIR"

# Stage1
python -m gwes.stages.stage1_phylo_cov \
  --run-dir "$RUN_DIR" \
  --tree "$treefile"

# Stage2
python -m gwes.stages.stage2_score_pairs \
  --run-dir "$RUN_DIR" \
  --threads 16

# Stage3
python -m gwes.stages.stage3_fit_global_sigma \
  --run-dir "$RUN_DIR" \
  --threads 16

# Stage4
python -m gwes.stages.stage4_pairs_null_and_delta \
  --run-dir "$RUN_DIR" \
  --threads 16

# Stage5
python -m gwes.stages.stage5_fit_bias_refit_sigma \
  --run-dir "$RUN_DIR"

# Stage6
python -m gwes.stages.stage6_refit_flagged_loci_sigma_grid \
  --run-dir "$RUN_DIR" \
  --fasta "$fastafile" \
  --threads 8 \
  --write-probs

# Stage7
python -m gwes.stages.stage7_patch_pairs_refined_bias \
  --run-dir "$RUN_DIR"

# Stage8
python -m gwes.stages.stage8_bootstrap_top_pairs \
  --run-dir "$RUN_DIR" \
  --minimal \
  --top 1000 --B 2000 --threads 8
```

After this, your key outputs are:

* `work/stage4/pairs_resid.tsv`
* `work/stage7/pairs_resid_patched.tsv`
* `work/stage8/stage8_bootstrap.tsv`

### 5.3 Notes on “minimal” outputs

* **Stage7 requires full Stage4 output** (it patches null-dependent columns and residual stats).
  Do not attempt to run Stage7 starting from a Stage4 “minimal-only” file.

* Stage8 `--minimal` is intended for reducing Stage8 output verbosity (it does not remove required computations).

### 5.4 Plotting (Stage7 + Stage8 overlay)

Create the plots directory first:

```bash
mkdir -p "$RUN_DIR/work/plots"
```

Then run the plotting script:

```bash
INBLOCK_PREFIX="/data/kell7366/test/simulation_eco.unitigs_in_blocks"

Rscript scripts/gwes_plotting.r \
  "$RUN_DIR/work/stage7/pairs_resid_patched.tsv" \
  "$RUN_DIR/work/plots/stage7_8_trueepi.png" \
  "srMI_e" \
  0 \
  0 \
  0 \
  "$RUN_DIR/work/stage8/stage8_bootstrap.tsv" \
  0.05 \
  0 \
  0 \
  1 \
  "" \
  "$INBLOCK_PREFIX"
```

#### In-block (“true epistasis”) highlighting

The last argument (`INBLOCK_PREFIX`) is optional. If provided, the script looks for files matching either:

* a directory containing `*_block_<idx>_*.txt`, or
* a prefix that expands to `PREFIX_block_*.txt`

Example files:

```
simulation_eco.unitigs_in_blocks_block_0_20000_20001.txt
simulation_eco.unitigs_in_blocks_block_1_60000_60001.txt
...
```

The script caches unitig→blockmask mappings in an RDS file to speed up reruns.

---

## 6) Common failure modes / fixes

### “could not open file .../work/plots/....png”

Create the directory:

```bash
mkdir -p "$RUN_DIR/work/plots"
```

### Tip-order mismatch errors

You must not mix artifacts from different runs. Always re-run the pipeline (or keep all inputs/outputs under the same `--run-dir`).

### MI warnings (`invalid value encountered in log`)

These can occur for extreme probabilities (0/1) and near-zero ratios in MI terms. If you want silence, update the MI implementation to guard against `0 * log(0/0)` patterns (or use `np.errstate`). This does not necessarily indicate a fatal error.

---

## 7) Reproducibility

* Stage metadata is written under `RUN_DIR/meta/stageX.json`.
* Use a single `--run-dir` per run to keep all artifacts consistent.
* For comparisons across datasets or parameter sweeps, use distinct run directories.

---

```

If you want, I can also generate a **short “Quickstart”** at the top of the README and a **parameter reference section** for each stage (only listing flags actually supported by your current scripts).
```
