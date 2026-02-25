# NEW-GWES: Tree/Ecology-aware GWES pipeline

## 1) Overview

This repository implements a **run-directory based pipeline** for detecting **epistasis-like statistical dependencies** between pangenome loci/unitigs while correcting for **phylogeny and shared ecology** using a per-locus logistic random-effect (“eco-bias”) model.

Core idea:

- Fit per-locus **tip-level marginal probabilities** $p_i(t)$ that account for structured confounding.
- For each candidate pair $(i,j)$, compute a **structured-mixture null** by averaging over tips:
  
  $$
  p_{11,\mathrm{null}} = \frac{1}{N}\sum_{t=1}^{N} p_i(t)\,p_j(t)
  $$
  
  and similarly for $p_{00,\mathrm{null}}, p_{01,\mathrm{null}}, p_{10,\mathrm{null}}$, then compute residual statistics such as $\Delta_{11}$, residual log-OR, residual MI, and signed residual MI.

The pipeline is organized as **stages** (stage0 … stage8). Each stage reads/writes artifacts under a single `--run-dir`, making runs reproducible and easier to debug.

---

## 2) What this tool is for

Designed for:

- **High-throughput screening** of candidate epistatic pairs (unitigs/loci) from bacterial pangenomes.
- Reducing false positives driven by:
  - shared ancestry (phylogeny),
  - shared ecology/environmental structure,
  - structured population effects that inflate LD-like signals.
- Producing outputs suitable for **PAN-GWES style distance–signal plots** and follow-up significance testing with a **parametric bootstrap** under the structured null.

Inputs of this program is:
- output of [PAN-GWES](https://github.com/Sudaraka88/PAN-GWES) tool
- tree file of isolates

Typical outputs:
- per-pair structured null probabilities,
- residual statistics (`delta11`, `rlogOR`, `rMI`, `srMI`),
- bootstrapped p-values and BH q-values for top pairs,
- optional plots (`scripts/gwes_plotting.r`).

---

## 3) Repository layout (high level)

```text
.
├── configs/                     # optional configs/presets
├── gwes/                        # python package
│   ├── __init__.py
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
````

### Run directory layout (`--run-dir`)

A run directory is created/filled by stages. Expected structure:

```text
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
```

---

## 4) Installation

### 4.1 Python environment

Recommended: use a dedicated environment.

Example (conda):

```bash
conda create -n gwes_env python=3.11 -y
conda activate gwes_env
```

### 4.2 Install from git (placeholder)

```bash
git clone https://github.com/jeju2486/NEW-GWES.git
cd gwes
pip install .
```


### 4.3 R (optional; for plotting only)

If you want plots, you need R + `data.table`.

In R:

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

Optional:

* **In-block unitig files** for “true epistasis” highlighting in plots (see §5.4).

### 5.2 Example run script variables

```bash
treefile="/dir/to/treefile"
fastafile="/dir/to/fasta_ouput_of_gfa1_parser.fasta"
pairfile="/dir/to/pangwes_output.ud_sgg_0_based"
RUN_DIR="/dir/to/output_dir/"
```

**NOTE** We are assuming the running direcotory is this program directory. If it is run under different working directory, user should adjust the directory information.

---

## 6) Pipeline stages (0–8)

Each stage is run as `python -m gwes.stages.<stage_module> --run-dir "$RUN_DIR" ...`.

### Stage 0 — Preparation and quality checks

Normalizes and copies/links raw inputs into the run directory.

Aligns tip order across inputs and writes canonical artifacts for downstream stages.

```bash
python -m gwes.stages.stage0_prepare \
  --tree "$treefile" \
  --fasta "$fastafile" \
  --pairs "$pairfile" \
  --run-dir "$RUN_DIR"
```

### Stage 1 — Phylogeny covariance / basis build

Builds phylogenetic covariance (and/or a low-rank basis) aligned to Stage0 tip order.

```bash
python -m gwes.stages.stage1_phylo_cov \
  --run-dir "$RUN_DIR" \
  --tree "$treefile"
```

### Stage 2 — Pair counting (observed contingency tables)

Computes observed per-pair contingency counts (`n00,n01,n10,n11`) for candidate pairs.

Output: `work/stage2/pairs_obs.tsv` (path may vary by implementation, but lives under `work/stage2/`).

```bash
python -m gwes.stages.stage2_score_pairs \
  --run-dir "$RUN_DIR" \
  --threads 16
```

### Stage 3 — Fit global eco-bias model (global sigma)

Fits a per-locus eco-bias model using a **single global random-effect scale** (sigma).

Writes:

* `P_hat` (per-tip per-locus probabilities, NPZ or memmap abstraction),
* per-locus fit summaries (`locus_fit.*`),
* `global_sigma.tsv`.

```bash
python -m gwes.stages.stage3_fit_global_sigma \
  --run-dir "$RUN_DIR" \
  --threads 16
```

### Stage 4 — Structured nulls + residual statistics (pairwise)

Uses Stage2 counts + Stage3 `P_hat` to compute structured-mixture null probabilities and residual stats per pair:

* `p**_null`, `delta11`,
* `logOR_obs/null/rlogOR`,
* `MI_obs/null/rMI`, and signed residual MI variants (e.g. `srMI_*`).

Output: `work/stage4/pairs_resid.tsv`.

```bash
python -m gwes.stages.stage4_pairs_null_and_delta \
  --run-dir "$RUN_DIR" \
  --threads 16
```

### Stage 5 — Flag loci requiring per-locus sigma refit

Identifies loci that are not well-modeled by the global sigma (e.g. numerical issues, separation, saturated fits, extreme z-scores).

Outputs flagged locus lists under `work/stage5/`.

```bash
python -m gwes.stages.stage5_fit_bias_refit_sigma \
  --run-dir "$RUN_DIR"
```

### Stage 6 — Refit flagged loci (sigma grid)

Performs per-locus sigma refit for flagged loci (grid search or similar), producing a patch artifact.

Outputs under `work/stage6/`, e.g. `refit_patch.npz` and optionally `P_refit`/probability outputs.

```bash
python -m gwes.stages.stage6_refit_flagged_loci_sigma_grid \
  --run-dir "$RUN_DIR" \
  --fasta "$fastafile" \
  --threads 8 \
  --write-probs
```

### Stage 7 — Patch pairwise results with refined bias model

Recomputes/patches Stage4 quantities for pairs involving refit loci using Stage6 outputs.

Output: `work/stage7/pairs_resid_patched.tsv`.

```bash
python -m gwes.stages.stage7_patch_pairs_refined_bias \
  --run-dir "$RUN_DIR"
```

### Stage 8 — Bootstrap top pairs under the structured null

Parametric bootstrap for top-ranked pairs under the structured-mixture null, producing empirical p-values and BH q-values.

Output: `work/stage8/stage8_bootstrap.tsv`.

```bash
python -m gwes.stages.stage8_bootstrap_top_pairs \
  --run-dir "$RUN_DIR" \
  --minimal \
  --top 1000 --B 2000 --threads 8
```

---

## 7) Notes on “minimal” outputs

* **Stage7 requires full Stage4 output** (it patches null-dependent columns and residual stats).
  Do not run Stage7 from a Stage4 “minimal-only” file if it drops required columns.

* Stage8 `--minimal` is intended to reduce Stage8 output verbosity (it does not remove required computations).

---

## 8) Plotting (Stage7 + Stage8 overlay)

Create the plots directory first:

```bash
mkdir -p "$RUN_DIR/work/plots"
```

Then run:

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

### Tip-order mismatch errors

Do not mix artifacts from different runs. Keep all inputs/outputs under the same `--run-dir`.

---

## 10) For any unkown error

Please use the Issues tab or report to seungwon.ko@biology.ox.ac.uk
