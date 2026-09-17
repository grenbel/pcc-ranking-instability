# pcc-ranking-instability

Code release for **"Ranking Instability Under Corruption in Point Cloud Completion: A
Matched-Control Audit on the Point Completion Network Benchmark"** (IEEE Access, 2026).

The paper evaluates four public point cloud completion baselines (PoinTr, AdaPoinTr,
SnowflakeNet, SeedFormer) on the PCN test split under four corruption operators at five
severities, with every model seeing bit-identical corrupted inputs (matched control), and
reports how often the per-cell ranking differs from the aggregate-clean ranking.

This repository contains the three components of the pipeline described in Section V-G of
the paper:

| Component | Where |
|---|---|
| Cache builder + corruption operators: seeded, hash-stamped corrupted inputs for a PCN-format test split, single or composed operators | `src/corruptions/`, `scripts/sweep_common.py` (`cache_clean_pcn_test`), `scripts/forward_sweep*.py` |
| Per-cell metric emitter: consumes the dense predictions of any completion model for those inputs and writes the per-sample rows | `scripts/metric_emitter.py`, `src/metrics/` |
| Analysis scripts: stratified tables, paired statistics, bootstrap intervals | `analysis/` (the five drivers of the supplementary archive's `scripts/` directory, byte-identical, plus `main_grid_analysis.py`, `gen_table_data_rerun.py` and `arrange_metrics.py`) |

The dense per-sample prediction archives of all four sweeps (304 `.npz` files) are
published as a Hugging Face dataset, see [Dense prediction archives](#dense-prediction-archives).
The per-sample metric arrays, the forward manifests and the analysis JSONs are in the
supplementary archive published with the article.

## Repository layout

```
src/corruptions/      five corruption operators (noise, outlier, density, crop, pose) + valid-point-only
                      variants, the SHA-1 per-sample seed (make_seed) and ComposeCorruptions
src/metrics/          CD-L1 / CD-L2 (PoinTr's ignore_zeros mask convention), F-score, descriptive
                      decomposition metrics
src/stratify/         matched-control group index, paired Wilcoxon, Kendall tau
src/seedformer_loader.py  import shim that isolates SeedFormer's `models` package from PoinTr's
scripts/sanity_clean_pcn.py     clean-PCN sanity check against the stamped checkpoint metrics (Table 2)
scripts/sweep_common.py         clean-cache builder + legacy single-stage sweep
scripts/forward_sweep.py        forward stage, zero-pad protocol (main grid)
scripts/forward_sweep_validpt.py     forward stage, valid-point-only protocol (Section V-C)
scripts/forward_sweep_upsample.py    forward stage, UpSamplePoints protocol + clean@s0 cell (Section V-D)
scripts/forward_sweep_composed.py    forward stage, composed-operator pilot (Section V-E)
scripts/metric_emitter.py       metric stage: per-cell per-sample rows (all protocols)
scripts/chamfer_fourway_check.py     numpy vs CUDA chamfer agreement check
scripts/cardinality_sensitivity.py   cardinality-controlled sensitivity check on the flip cells
analysis/             instability_summary.py, validpt_sensitivity.py, upsample_sensitivity.py,
                      mixed_pilot.py, gen_table_data.py (archive-relative, see below);
                      main_grid_analysis.py (main-grid analysis JSON), gen_table_data_rerun.py
                      (table sources from a rerun), arrange_metrics.py (emitter output -> archive layout)
smoke_tests/          CPU-only tests that need no data or checkpoints
environment.yml       conda environment
```

## Setup

1. Create the environment and install PyTorch for your CUDA driver (the paper used
   PyTorch 2.8.0+cu128 on an RTX 5090 for the forward stages; the metric and analysis
   stages run on CPU):

   ```bash
   conda env create -f environment.yml
   conda activate pcc-ranking-instability
   # then install torch/torchvision from https://pytorch.org
   ```

2. Clone the official PoinTr repository into `baselines/PoinTr` and build its CUDA
   extensions (Chamfer distance and the others in `install.sh`). SnowflakeNet is bundled in
   that repository (`models/SnowFlakeNet.py`, `cfgs/PCN_models/SnowFlakeNet.yaml`). Building
   needs a CUDA Toolkit whose `nvcc` matches the PyTorch CUDA build (the paper used CUDA 12.8)
   and a C++ compiler supported by that toolkit (gcc on Linux, MSVC on Windows):

   ```bash
   git clone https://github.com/yuxumin/PoinTr baselines/PoinTr
   cd baselines/PoinTr && bash install.sh && cd ../..
   ```

   PoinTr's models also need `pointnet2_ops`. The upstream `Pointnet2_PyTorch` build hard-codes
   `TORCH_CUDA_ARCH_LIST="3.7+PTX;5.0;..."` in both `pointnet2_ops_lib/setup.py` and the JIT
   fallback in `pointnet2_ops/pointnet2_utils.py`; CUDA 12 no longer accepts `compute_37`, so make
   both files honour the environment variable and build for your GPU's compute capability
   (`12.0` for the RTX 5090 used in the paper, `8.9` for an RTX 4090, `8.6` for an RTX 3090):

   ```bash
   git clone https://github.com/erikwijmans/Pointnet2_PyTorch /path/to/Pointnet2_PyTorch
   cd /path/to/Pointnet2_PyTorch/pointnet2_ops_lib
   sed -i 's/os.environ\["TORCH_CUDA_ARCH_LIST"\] = "[^"]*"/os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "12.0")/' setup.py pointnet2_ops/pointnet2_utils.py
   python -m pip install setuptools wheel
   # no build isolation: the upstream setup.py imports torch without declaring it as a build requirement
   TORCH_CUDA_ARCH_LIST="12.0" python -m pip install --no-build-isolation .
   cd /path/to/pcc-ranking-instability   # back to this repository's root before the next step
   ```

3. Clone the official SeedFormer repository and expose its `codes/` directory as
   `baselines/SeedFormer` (a symlink or a copy; `baselines/SeedFormer/model.py` must exist):

   ```bash
   git clone https://github.com/hrzhou2/seedformer /path/to/seedformer
   ln -s /path/to/seedformer/codes baselines/SeedFormer
   ```

4. Download the four pretrained PCN checkpoints and place them as listed (the digests
   are those of the files used for the paper):

   | Registry name | Source | Path in this repo | Size (bytes) | MD5 |
   |---|---|---|---|---|
   | `PoinTr` | PCN_new checkpoint linked from the PoinTr README | `ckpts/pretrained_pointr/PCN_models/ckpt-best.pth` | 510,418,823 | `e7937f0e46e244f433feec9a3707ca9f` |
   | `AdaPoinTr` | AdaPoinTr PCN checkpoint linked from the PoinTr README | `ckpts/pretrained_adapointr/PCN_models/ckpt-best.pth` | 389,745,620 | `1097b56248b983a376f7f16c8e6be947` |
   | `SnowFlakeNet` | `ckpt-best-pcn-cd_l1.pth` from the SnowflakeNet repository (PCN CD-L1 variant) | `ckpts/pretrained_snowflakenet/PCN_models/ckpt-best.pth` | 77,427,782 | `baadb96d94174b4b7336a23d9d9cd8e6` |
   | `SeedFormer` | SeedFormer-dim128 PCN checkpoint from the SeedFormer repository | `ckpts/pretrained_seedformer/PCN_models/ckpt-best.pth` | 13,444,675 | `bb898a125eba57b6bccf8fe26b762d1e` |

5. Download the PCN dataset as described in PoinTr's `DATASET.md`. `--pcn-data-root` must
   contain `PCN.json` and `test/partial/...`, `test/complete/...`.

## Smoke tests (no data, no checkpoints)

```bash
python smoke_tests/pipeline_smoke.py              # operators, metrics, matched-control protocol (numpy/scipy only)
python smoke_tests/smoke_validpt_corruptions.py   # valid-point-only operator variants
python smoke_tests/smoke_composed_local.py        # composed-operator forward stage (needs torch, CPU is fine)
```

## Running the pipeline

All scripts are run from the repository root. Every forward script prints the exact
`metric_emitter.py` command for its output when it finishes.

**Sanity check** (Table 2): each baseline's clean CD-L1 must land within 10% of the value
stamped in its checkpoint.

```bash
python scripts/sanity_clean_pcn.py --pcn-data-root /path/to/PCN --output-dir logs/sanity \
    --models PoinTr,AdaPoinTr,SnowFlakeNet,SeedFormer --no-wandb
```

**Forward stage** (GPU). The cache builder inside each forward script snapshots the clean PCN
test split once under `--seed 42` and stamps the SHA-256 `cache_hash` of that cache into the
manifest and into every `.npz` it writes (the published zero-pad archives predate the per-file
stamp and are covered by their manifests). When a forward script finds an existing `.npz` in
`--pred-dir` it reuses it only if the file's stamp equals the current cache hash; unstamped files
are re-forwarded. The zero-pad, valid-point-only and composed
sweeps must all report the prefix `80d3efce468eba6c`, the UpSamplePoints sweep the prefix
`8b9a8168`.

```bash
# zero-pad protocol, main 20-cell grid (default --models is PoinTr,AdaPoinTr)
python scripts/forward_sweep.py --pcn-data-root /path/to/PCN --pred-dir preds/zero_pad \
    --models PoinTr,AdaPoinTr,SnowFlakeNet,SeedFormer --device cuda:0
# valid-point-only protocol (Section V-C)
python scripts/forward_sweep_validpt.py --pcn-data-root /path/to/PCN --pred-dir preds/validpt --device cuda:0
# UpSamplePoints protocol + clean@s0 pass-through cell (Section V-D)
python scripts/forward_sweep_upsample.py --pcn-data-root /path/to/PCN --pred-dir preds/upsample --device cuda:0
# composed-operator pilot (Section V-E)
python scripts/forward_sweep_composed.py --pcn-data-root /path/to/PCN --pred-dir preds/composed --device cuda:0
```

**Metric stage** (CPU; `--workers N` parallelizes over cells with bit-identical output). The
emitter rebuilds the clean cache, verifies the manifest alignment and the cache hash, and writes
`<model>_<op>_s<sev>_per_sample.json` per cell.

```bash
python scripts/metric_emitter.py --pcn-data-root /path/to/PCN --pred-dir preds/zero_pad --output-dir logs/zero_pad \
    --models PoinTr,AdaPoinTr,SnowFlakeNet,SeedFormer --ops noise,outlier,density,crop --severities 1,2,3,4,5 --seed 42 --workers 4
python scripts/metric_emitter.py --pcn-data-root /path/to/PCN --pred-dir preds/validpt --output-dir logs/validpt \
    --models PoinTr,AdaPoinTr,SnowFlakeNet,SeedFormer --ops noise_validpt,outlier_validpt,density_validpt,crop_validpt \
    --severities 1,2,3,4,5 --seed 42 --manifest-name validpt_forward_manifest.json --allow-validpt-ops --workers 4
python scripts/metric_emitter.py --pcn-data-root /path/to/PCN --pred-dir preds/upsample --output-dir logs/upsample \
    --models PoinTr,AdaPoinTr,SnowFlakeNet,SeedFormer --ops noise,outlier,density,crop,clean --severities 1,2,3,4,5,0 --seed 42 \
    --loader-cfg baselines/PoinTr/cfgs/PCN_models/SnowFlakeNet.yaml --manifest-name upsample_forward_manifest.json \
    --expect-protocol upsample_points_v1 --allow-clean-cell --workers 4
python scripts/metric_emitter.py --pcn-data-root /path/to/PCN --pred-dir preds/composed --output-dir logs/composed \
    --models PoinTr,AdaPoinTr,SnowFlakeNet,SeedFormer --ops mixed_noise_outlier,mixed_crop_noise,mixed_density_outlier \
    --severities 1,2,3,4,5 --seed 42 --manifest-name composed_forward_manifest.json --expect-protocol mixed_zero_pad_v1 \
    --allow-mixed-ops --workers 4
```

**Analysis.** Five of the scripts in `analysis/` are the archive-relative drivers shipped in
the supplementary archive: they read `../metrics*/<Baseline>/<op>_s<sev>.json` and
`../manifests/*.json` relative to their own directory and write into `../analysis/`. To
recompute the paper's statistics, unpack the supplementary archive and run them from its root
(`python scripts/instability_summary.py`, etc.). `analysis/main_grid_analysis.py` is the
generator of `analysis/4baseline_analysis.json` (per-cell means and ranks, paired Wilcoxon
tests, Friedman tests, Kendall tau, zero-row audit of the main 20-cell grid), the file that
`gen_table_data.py` turns into the table sources; run on the shipped rows it reproduces every
statistic of the shipped JSON (only the date and the zero-row audit differ: the shipped file
reports zeros for the 40 cells whose rows predate the `n_pred_zero_rows` field, this generator
reports them as unavailable). To analyse your own rerun, work on a copy of the unpacked archive: copy the
emitter's per-sample files into it with `analysis/arrange_metrics.py` (it applies the archive
naming: `metrics/SnowflakeNet/...` directories, the `_validpt` suffix dropped from file names,
`model_name` respelled to the archive form; `--overwrite` replaces the shipped rows), then run
the drivers in this order:

```bash
python analysis/arrange_metrics.py --protocol zero_pad --src logs/zero_pad --archive /path/to/archive_copy --overwrite
python analysis/arrange_metrics.py --protocol validpt  --src logs/validpt  --archive /path/to/archive_copy --overwrite
python analysis/arrange_metrics.py --protocol upsample --src logs/upsample --archive /path/to/archive_copy --overwrite
python analysis/arrange_metrics.py --protocol composed --src logs/composed --archive /path/to/archive_copy --overwrite
python analysis/main_grid_analysis.py --archive /path/to/archive_copy      # -> analysis/4baseline_analysis.json
cd /path/to/archive_copy
python scripts/instability_summary.py      # strict-Bonferroni winners, bootstrap CIs, permutation null
python scripts/validpt_sensitivity.py      # Table 7
python scripts/upsample_sensitivity.py     # Table 8
python scripts/mixed_pilot.py              # Table 9
cd -
python analysis/gen_table_data_rerun.py --archive /path/to/archive_copy   # Tables 3-6 sources from the regenerated analysis JSON
```

`scripts/gen_table_data.py` in the archive does the same for the shipped JSON but assumes every test is
defined; `gen_table_data_rerun.py` produces byte-identical tables from the shipped JSON and, on a rerun
where a test is undefined (two baselines with identical predictions in a cell), excludes it from the
significance counts and reports it in the ranking-statistics table. Table 2 (clean-PCN sanity) comes
from the sanity run, not from this JSON.

**Auditing an additional model.** Add an entry to `DEFAULT_MODELS` in
`scripts/sanity_clean_pcn.py` (config, checkpoint, dense-output index, whether the partial must be
concatenated, strict loading) and, if the model is not built through PoinTr's registry, a
builder like `src/seedformer_loader.py`. The forward scripts then feed it the cached inputs and
write its predictions in the per-cell archive format that the metric emitter consumes.

## Dense prediction archives

The per-cell dense predictions written by the forward stage for the paper's four sweeps
(304 `.npz` files, 65.5 GB) are published as a Hugging Face dataset:

**https://huggingface.co/datasets/spacetime1008/pcc-ranking-instability-predictions**

| Folder | Cells | Protocol / manifest | `cache_hash` prefix |
|---|---|---|---|
| `zero_pad/` | 80 (4 baselines x 4 operators x 5 severities) | `forward_manifest_PoinTr_AdaPoinTr.json`, `forward_manifest_SnowFlakeNet.json`, `forward_manifest_SeedFormer.json` | `80d3efce468eba6c` |
| `validpt/` | 80 | `validpt_forward_manifest.json` (`valid_point_only_v1`) | `80d3efce468eba6c` |
| `upsample/` | 84 (incl. the four `clean__s0` cells) | `upsample_forward_manifest.json` (`upsample_points_v1`) | `8b9a8168366080ba` |
| `composed/` | 60 (3 operator pairs x 5 severities x 4 baselines) | `composed_forward_manifest.json` (`mixed_zero_pad_v1`) | `80d3efce468eba6c` |

Each `.npz` holds the `(1200, 16384, 3)` float32 predictions of one baseline for one cell
plus the sample identities and the provenance stamps described in `scripts/forward_sweep.py`
(the composed cells additionally carry the constituent seeds, the composition order, the
per-sample input row audits and the per-cell corrupted-input SHA-256). `CHECKSUMS.md5` lists
the MD5 of every file. Pointing `metric_emitter.py --pred-dir` at one of these folders (with
the matching `--manifest-name`) recomputes the per-sample metric rows from the released
predictions; the zero-pad folder holds three manifests because that sweep was run per model
group, so run the emitter once per manifest with the corresponding `--models`.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The dense
prediction archives and the supplementary data files are released under CC BY 4.0.

## Citation

```bibtex
@article{gong2026rankinginstability,
  author  = {Gong, Wenbo},
  title   = {Ranking Instability Under Corruption in Point Cloud Completion: A Matched-Control Audit on the Point Completion Network Benchmark},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Manuscript Access-2026-42538, accepted for publication}
}
```

## Acknowledgements

The audit builds on the public PoinTr, AdaPoinTr, SnowflakeNet and SeedFormer repositories
and their pretrained checkpoints.
