# VMDM: Variant-aware, Missing-tolerant Dynamic Modelling

**A computational framework for tuberculosis drug-resistance prediction from low-coverage metagenomic sequencing data**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3113/)

VMDM accompanies the manuscript **"Sample-specific dynamic modeling predicts
Mycobacterium tuberculosis drug resistance from low-coverage metagenomes"**. It
combines WHO catalogue-based resistance evidence with sample-specific dynamic
machine-learning models to predict tuberculosis drug resistance from sparse
sequencing data.

The public release includes the database and model inputs used for four first-line anti-TB drugs:

- Rifampicin
- Isoniazid
- Pyrazinamide
- Ethambutol

VMDM is research software. It is not intended for direct clinical diagnosis or
treatment decisions without independent validation in the intended setting.

## Quick Start

The commands below install VMDM, decompress the packaged training tables for
faster runtime, and run the bundled FASTQ smoke test.

```bash
git clone https://github.com/hefush/vmdm.git
cd vmdm

mamba env create -f requirements.yaml -p ./venv
conda activate ./venv
```

If Mamba is not available, use Conda:

```bash
conda env create -f requirements.yaml -p ./venv
conda activate ./venv
```

For smoke tests, demonstrations and larger analyses, we recommend one-time
decompression of the packaged training tables. This adds about 1.5 GB of disk
usage but avoids repeated gzip reads during model fitting.

```bash
gzip -dk MTBdb/*.data.xls.gz
```

Run the bundled two-sample FASTQ smoke test:

```bash
python vmdm.py example.list examples/fastq_smoke/output --jobs 1 --threads 2
```

The command should exit with code 0 and write reports under
`examples/fastq_smoke/output/<sample_name>/`. Expected semantic checks and
report snapshots are described in `examples/fastq_smoke/README.md`.

Run VMDM on your own FASTQ list:

```bash
python vmdm.py input.list output_dir --jobs 4 --threads 20
```

For targeted sequencing data, use a minimum coverage threshold such as:

```bash
python vmdm.py input.list output_dir --min_cov 10 --jobs 4 --threads 20
```

## Installation

Requirements:

- Linux or another POSIX-compatible environment
- Conda or Mamba package manager
- Python 3.11 from the supplied Conda environment
- BWA, samtools, bcftools, bedtools, SnpEff, LightGBM, XGBoost and the
  scientific Python stack installed through `requirements.yaml`

The supplied `requirements.yaml` pins the versions used for this release and
selects OpenBLAS-backed BLAS/LAPACK for portable CPU parallelism. The file lists
mirror channel URLs first for faster environment solving in regions where those
mirrors are reliable, followed by standard `bioconda` and `conda-forge`
channels as public fallbacks.

The default database is included under `MTBdb/`. No Git LFS step is required for the current release.

## Input

Create a tab-separated file listing sample names and FASTQ paths. A paired-end
sample should be listed on two lines with the same sample name:

```text
sample1	/path/to/sample1_R1.fastq.gz
sample1	/path/to/sample1_R2.fastq.gz
sample2	/path/to/sample2.fastq.gz
```

VMDM starts from the FASTQ paths supplied in the input list and does not perform
adapter trimming or read quality filtering internally. If a dataset requires
trimming, run `fastp` or an equivalent preprocessing tool upstream and list the
processed FASTQ files in the VMDM input file.

## Usage

```bash
# mNGS mode with default low-coverage settings
python vmdm.py input.list output_dir

# tNGS mode with minimum coverage depth 10
python vmdm.py input.list output_dir --min_cov 10

# Run 4 samples in parallel, each sample using 20 bwa/samtools/model threads
python vmdm.py input.list output_dir --jobs 4 --threads 20
```

Command-line parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--drugs` | Target drugs for analysis | `Rifampicin,Isoniazid,Pyrazinamide,Ethambutol` |
| `--method` | ML algorithm (`LightGBM` or `XGBoost`) | `LightGBM` |
| `--max_snps` | Maximum SNPs for feature selection | `1000` |
| `--min_ppv` | PPV thresholds matched to `--drugs` order | `0.95,0.97,0.85,0.70` |
| `--jobs` | Number of samples to run in parallel | `5` |
| `--threads` | Threads per sample for bwa/samtools and model steps | `20` |
| `--min_cov` | Minimum coverage depth (`0` for mNGS mode; `10` recommended for tNGS mode) | `0` |
| `--refdb` | Reference FASTA path | `MTBdb/reference/ref.fa` |
| `--known` | WHO known resistance variants file | `MTBdb/WHO_known.info.xls` |
| `--anno` | Drug annotation file | `MTBdb/drug.anno` |
| `--bed` | BED file for targeted regions; use `""` to disable | `MTBdb/features.bed` |
| `--traindb` | Training database directory | `MTBdb` |

`--min_ppv` values are matched to `--drugs` by order. With the defaults,
Rifampicin uses `0.95`, Isoniazid uses `0.97`, Pyrazinamide uses `0.85`, and
Ethambutol uses `0.70`.

To derive prevalence-adjusted `--min_ppv` values from the training database, use:

```bash
python ppv_prevalence_adjustment.py --traindb MTBdb --target-prevalence 0.10 --target-ppv 0.90
```

`--target-prevalence` and `--target-ppv` can also be supplied as comma-separated
`drug=value` pairs when the target prevalence or PPV differs by drug. The
output `required_min_ppv` column is the value to pass to `vmdm.py --min_ppv` in
`--drugs` order.

VMDM uses `--threads` to set `VMDM_MODEL_THREADS`, `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` for
drug-specific model prediction. This avoids OpenMP/BLAS thread oversubscription
on shared servers and makes smoke-test runtime more stable across environments.

## Output

Each sample is written to `output_dir/<sample_name>/`.

`<sample_name>.report.xls` combines catalogue-supported variants and model predictions:

```text
Name    Drug          Evidence
M86434  Ethambutol    embB_p.Met306Val(R:34.23%),model(0.86)
M86434  Isoniazid     katG_p.Ser315Thr(R:77.81%),model(1.00)
M86434  Rifampicin    model(0.98)
```

Evidence-field interpretation:

- Variant evidence is written as `variant_label(category:contribution%)`.
- `variant_label` reports the detected catalogue variant. For example,
  `katG_p.Ser315Thr` means the katG amino-acid substitution Ser315Thr was
  detected.
- `category` follows the WHO catalogue annotation. `R` means "Assoc with R", and `I` means "Assoc with R - Interim".
- `contribution%` is the drug-specific resistance-contribution proportion from
  the WHO catalogue-derived annotation table used by VMDM. It is not the read
  allele frequency and not the model probability.
- `model(p)` is added when the machine-learning model predicts resistance at
  the configured `--min_ppv` threshold. If the model prediction does not pass
  the threshold, `model(...)` is not added to the report row.

`<sample_name>.report_anno.xls` adds drug-resistance mechanism annotations for clinical review.

## Data Files

Keep the packaged database files under `MTBdb/` unless using a custom training database through `--traindb`.

- `MTBdb/*.features.xls`: uncompressed feature tables used to build sample-specific model matrices.
- `MTBdb/*.data.xls.gz`: gzip-compressed drug-specific training tables.
- `MTBdb/reference/ref.fa`: packaged H37Rv reference used for read alignment.
- `MTBdb/WHO_known.info.xls`: WHO catalogue-derived resistance variant information.
- `MTBdb/drug.anno`: drug-resistance mechanism annotations.

VMDM automatically reads either `*.data.xls` or `*.data.xls.gz` training tables
and prefers the uncompressed `*.data.xls` version when both are present. To add
additional drugs, provide matching `*.features.xls` and `*.data.xls` or
`*.data.xls.gz` files in the directory supplied through `--traindb`.

## Reproduce Paper Figures and Tables

The `vmdm-paper-analysis/` directory contains the source data and scripts used
to regenerate the manuscript figures, supplementary figures, supplementary
tables and selected analysis checks. This directory is separate from the VMDM
prediction pipeline: `vmdm.py` runs drug-resistance prediction from FASTQ files,
whereas `vmdm-paper-analysis/` rebuilds the reported paper outputs from the
packaged plotting and analysis inputs.

Create the paper-analysis environment:

```bash
cd vmdm-paper-analysis
conda env create -f environment.yml
conda activate vmdm-paper-analysis
```

List and run figure-generation tasks:

```bash
python run_all_figures.py --list
python run_all_figures.py
```

Run selected analysis checks:

```bash
(cd analysis_checks && python verify_table.py)
(cd analysis_checks && python verify_description.py)
(cd analysis_checks && python verify_performance.py)
(cd analysis_checks && python verify_calculation.py)
```

Generated figure files are written under the corresponding figure directories
and are ignored by git. Raw sequencing reads are not included; the directory
contains the derived source data needed to reproduce the plotted panels and
inspect the table-level analyses.

## Method Summary

VMDM implements three components described in the accompanying manuscript:

1. PRESS feature selection for identifying resistance-linked variants robust to missing data.
2. Bayesian genotype QC for depth- and allele-frequency-aware handling of sparse calls.
3. Sample-specific dynamic modelling with drug-tunable PPV thresholds.

The pipeline stages are:

1. Read alignment to the packaged H37Rv reference.
2. Variant calling and optional targeted-region filtering.
3. WHO catalogue-based resistance evidence extraction.
4. Drug-specific feature matrix construction from sparse variants.
5. Dynamic LightGBM or XGBoost prediction.
6. Evidence aggregation and annotated report generation.

## Performance Context

The accompanying manuscript evaluates VMDM on 1,190 independent isolates across
8,330 down-sampled datasets from 0.01x to 10x coverage, plus two external
clinical sequencing cohorts. The packaged database in this repository is
intended to reproduce the software workflow for the four first-line drugs
evaluated in that study.

## Citation

Please cite the accompanying manuscript when using VMDM:

```text
Sample-specific dynamic modeling predicts Mycobacterium tuberculosis drug resistance from low-coverage metagenomes.
```

Code availability: <https://github.com/hefush/vmdm>

The scripts and source data for reproducing the manuscript figures and
supplementary analyses are included under `vmdm-paper-analysis/`.

## Public Use Notes

- VMDM is research software and is not a clinical diagnostic device.
- Do not commit private FASTQ files, clinical metadata, or local run outputs.
- The `images/` directory contains nonessential workflow illustrations; the
  command-line workflow does not depend on these files.
- Generated paper-analysis outputs under
  `vmdm-paper-analysis/figures/**/figs_png/` and PDF figure exports are ignored
  by git.
- The current release bundles a compact `MTBdb/` directory and does not require Git LFS.

## License

MIT License. See [LICENSE](LICENSE).

## Contact

For questions about the method or software, please use [GitHub Issues](https://github.com/hefush/vmdm/issues).
