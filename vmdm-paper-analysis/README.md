# VMDM Paper Analysis

This directory contains the source data and scripts used to generate the
figures, supplementary figures, supplementary tables and selected analysis
checks for the manuscript:

Sample-specific dynamic modeling predicts Mycobacterium tuberculosis drug
resistance from low-coverage metagenomes.

The VMDM prediction pipeline itself is maintained in the parent repository at:

https://github.com/hefush/vmdm

This directory is intentionally narrower than the prediction pipeline: it is the
paper-analysis companion for transparent figure and table reproduction.

## Repository Layout

- `figures/`: plotting scripts and source data for main and supplementary
  figures.
- `supplementary_tables/`: final supplementary table files used in the
  submission.
- `metadata/`: public clinical cohort manifests and accession/QC summaries.
- `analysis_checks/`: selected scripts and derived tables used to verify
  manuscript claims.
- `fonts/`: font files used by the plotting scripts for stable rendering.
- `run_all_figures.py`: convenience wrapper for rebuilding figure outputs.

## Environment

```bash
conda env create -f environment.yml
conda activate vmdm-paper-analysis
```

If conda is not available, a standard Python environment is sufficient:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The wrapper sets `MPLBACKEND`, `MPLCONFIGDIR`, `XDG_CACHE_HOME` and
`ARIAL_FONT_PATH` automatically.

## Rebuild Figures

List available tasks:

```bash
python run_all_figures.py --list
```

Run all figure scripts:

```bash
python run_all_figures.py
```

Run selected figures:

```bash
python run_all_figures.py --only figure3 figure4 supplementary_figure5
```

Generated files are written next to each script, usually under `figs_png/` and
as PDF files in the corresponding figure directory. These generated outputs are
ignored by git.

To keep the repository focused, figure directories contain the plotted source
data needed to regenerate each manuscript panel. Bulky upstream intermediates
that are not read by the plotting or validation scripts are not included.
For example, Figure 2C is distributed as the plotted feature-stability summary,
and Figure 2D is distributed as the plotted LD-network subset.

## Validation

Run the table and claim checks from this directory:

```bash
(cd analysis_checks && python verify_table.py)
(cd analysis_checks && python verify_description.py)
(cd analysis_checks && python verify_performance.py)
(cd analysis_checks && python verify_calculation.py)
(cd analysis_checks && python discordant_enrichment_analysis.py)
(cd analysis_checks && python build_supplementary_table5_optimized.py)
```

Update the source-file manifest after any intentional file change:

```bash
python scripts/update_manifest.py
```

## Figure Mapping

| Manuscript item | Directory | Script |
|---|---|---|
| Figure 1 | `figures/figure1` | Manually assembled in Adobe Illustrator |
| Figure 2 | `figures/figure2` | `plot_figure2.py` |
| Figure 3 | `figures/figure3` | `plot_figure3.py` |
| Figure 4 | `figures/figure4` | `plot_figure4.py` |
| Figure 5 | `figures/figure5_and_supplementary_figure4` | `plot_figure5_and_suppfig4.py` |
| Supplementary Figure 1 | `figures/supplementary_figure1` | `plot_supplementary_figure1.py` |
| Supplementary Figure 2 | `figures/supplementary_figure2` | `plot_supplementary_figure2.py` |
| Supplementary Figure 3 | `figures/supplementary_figure3` | `plot_supplementary_figure3.py` |
| Supplementary Figure 4 | `figures/figure5_and_supplementary_figure4` | `plot_figure5_and_suppfig4.py` |
| Supplementary Figure 5 | `figures/supplementary_figure5` | `plot_supplementary_figure5.py` |

## Data Scope

Raw sequencing reads are not included here. They are publicly available from the
repositories and accessions listed in the manuscript and supplementary tables.
This repository includes the derived source data required to regenerate the
reported figures and inspect the reported table-level analyses.

## Suggested Code Availability Text

The custom code for the VMDM analysis pipeline is available on GitHub at
https://github.com/hefush/vmdm. The specific scripts and source data used to
generate the figures and perform the analyses reported in this paper are
included in the `vmdm-paper-analysis/` directory of the same repository. Cite
the associated repository release or archived record when available.

## License

MIT License. See `LICENSE`.
