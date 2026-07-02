# VMDM: Variant-aware, Missing-tolerant Dynamic Modelling

**A computational framework for tuberculosis drug-resistance prediction from low-coverage metagenomic sequencing data**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3113/)

## Description

VMDM is the software implementation accompanying the manuscript **"Robust prediction of Mycobacterium tuberculosis drug resistance from low-coverage metagenomic sequencing using sample-specific dynamic modelling"**. It combines WHO catalogue-based resistance detection with sample-specific dynamic machine-learning models to support tuberculosis drug-resistance prediction from sparse sequencing data.

The current public release includes the packaged database and models used for the four first-line anti-TB drugs evaluated in the manuscript:

- Rifampicin
- Isoniazid
- Pyrazinamide
- Ethambutol

Additional drugs can be added by providing matching `*.features.xls` and `*.data.xls` or `*.data.xls.gz` files in the directory supplied through `--traindb`.

## Repository

Public URL: <https://github.com/hefush/vmdm>

## Installation

### Requirements

- Linux or another POSIX-compatible environment
- Conda package manager
- Python 3.11, installed through the supplied Conda environment

### Setup

```bash
git clone https://github.com/hefush/vmdm.git
cd vmdm

conda env create -f requirements.yaml -p ./venv
conda activate ./venv
```

The default database is included under `MTBdb/`. No Git LFS step is required for the current release.

## Usage

### Input format

Create a tab-separated file listing sample names and FASTQ paths. A paired-end sample should be listed on two lines with the same sample name:

```text
sample1	/path/to/sample1_R1.fastq.gz
sample1	/path/to/sample1_R2.fastq.gz
sample2	/path/to/sample2.fastq.gz
```

### Execution

```bash
# mNGS mode with default low-coverage settings
python vmdm.py input.list output_dir

# tNGS mode with minimum coverage depth 10
python vmdm.py input.list output_dir --min_cov 10

# Run 4 samples in parallel, each sample using 20 bwa/samtools threads
python vmdm.py input.list output_dir --jobs 4 --threads 20
```

### Command-line Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--drugs` | Target drugs for analysis | `Rifampicin,Isoniazid,Pyrazinamide,Ethambutol` |
| `--method` | ML algorithm (`LightGBM` or `XGBoost`) | `LightGBM` |
| `--max_snps` | Maximum SNPs for feature selection | `1000` |
| `--min_ppv` | PPV thresholds matched to `--drugs` order | `0.95,0.97,0.85,0.70` |
| `--jobs` | Number of samples to run in parallel | `5` |
| `--threads` | Threads per sample for bwa/samtools steps | `20` |
| `--min_cov` | Minimum coverage depth (`0` for mNGS mode; `10` recommended for tNGS mode) | `0` |
| `--refdb` | Reference FASTA path | `MTBdb/reference/ref.fa` |
| `--known` | WHO known resistance variants file | `MTBdb/WHO_known.info.xls` |
| `--anno` | Drug annotation file | `MTBdb/drug.anno` |
| `--bed` | BED file for targeted regions; use `""` to disable | `MTBdb/features.bed` |
| `--traindb` | Training database directory | `MTBdb` |

`--min_ppv` values are matched to `--drugs` by order. With the defaults, Rifampicin uses `0.95`, Isoniazid uses `0.97`, Pyrazinamide uses `0.85`, and Ethambutol uses `0.70`.

## Output

Each sample is written to `output_dir/<sample_name>/`.

### Primary Report

`<sample_name>.report.xls` combines catalogue-supported variants and model predictions:

```text
Name    Drug          Evidence
M86434  Ethambutol    embB_p.Met306Val(R:34.23%),model(0.86)
M86434  Isoniazid     katG_p.Ser315Thr(R:77.81%),model(1.00)
M86434  Rifampicin    model(0.98)
```

### Annotated Report

`<sample_name>.report_anno.xls` adds drug-resistance mechanism annotations for clinical review.

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

The accompanying manuscript evaluates VMDM on 1,190 independent isolates across 8,330 down-sampled datasets from 0.01x to 10x coverage, plus two external clinical sequencing cohorts. The packaged database in this repository is intended to reproduce the software workflow for the four first-line drugs evaluated in that study.

## Citation

Please cite the accompanying manuscript when using VMDM:

```text
Robust prediction of Mycobacterium tuberculosis drug resistance from low-coverage metagenomic sequencing using sample-specific dynamic modelling.
```

Code availability: <https://github.com/hefush/vmdm>

## Notes for Public Use

- Do not commit private FASTQ files, clinical metadata, or local run outputs.
- The `images/` directory is intentionally excluded from the public repository because the manuscript figures are maintained separately.
- The current release bundles a compact `MTBdb/` directory and does not require Git LFS.

## License

MIT License. See [LICENSE](LICENSE).

## Contact

For questions about the method or software, please use [GitHub Issues](https://github.com/hefush/vmdm/issues).
