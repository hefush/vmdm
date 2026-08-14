# FASTQ Smoke Test

This directory contains two small down-sampled public-read examples for
reviewer testing of the full VMDM FASTQ entry point.

Files:

- `SAMEA1102585.subseq.fq.gz`
- `SAMEA2535210.subseq.fq.gz`
- `example_fastq.list`
- `expected_output/`

From the package root, after creating and activating the conda environment:

```bash
python vmdm.py examples/fastq_smoke/example_fastq.list examples/fastq_smoke/output --jobs 1 --threads 2
```

This command uses the default four first-line drugs: Rifampicin, Isoniazid,
Pyrazinamide and Ethambutol.

The expected primary reports from our verification run are provided as tabular
files under `examples/fastq_smoke/expected_output/` for direct comparison with
the reports generated under `examples/fastq_smoke/output/`.

Expected semantic output for the down-sampled FASTQ files:

- `SAMEA1102585.report.xls` should contain resistance evidence for the default
  first-line drug run, including `rpoB_p.Ser450Leu`, an ethambutol
  `embB_p.Met306Val` catalogue marker, and `model(...)` predictions for
  Rifampicin, Isoniazid and Ethambutol.
- `SAMEA2535210.report.xls` should be generated successfully. In the packaged
  down-sampled reads, the catalogue evidence recovered from this file includes
  `embB_p.Met306Ile`, `katG_p.Ser315Thr` and `pncA_p.Arg154Gly`; the
  default first-line drug run should also include `model(...)` predictions for
  Rifampicin, Isoniazid and Ethambutol.

The exact model probability values can vary slightly with library builds and
threading, but the reports should be generated successfully and contain the
expected catalogue/model markers above.

In our review-package environment on a shared Linux server, this default
four-drug smoke test took about 1-2 minutes with `--jobs 1 --threads 2`.
Faster local disks or higher-clock CPUs may complete it in under 1 minute.

The default command runs machine-learning prediction for the four first-line
drugs. Catalogue-supported variants for other drugs can still appear in the
report if detected in the FASTQ files.

In our verification run of the default four-drug command, the primary report
files were:

`examples/fastq_smoke/output/SAMEA1102585/SAMEA1102585.report.xls`

```text
Name	Drug	Evidence
SAMEA1102585	Ethambutol	embB_p.Met306Val(R:34.23%),model(0.66)
SAMEA1102585	Isoniazid	model(0.98)
SAMEA1102585	Kanamycin	eis_c.-10G>A(R:6.60%)
SAMEA1102585	Rifampicin	rpoB_p.Ser450Leu(R:64.40%),model(0.99)
SAMEA1102585	Streptomycin	rrs_n.878G>A(I:0.48%)
```

`examples/fastq_smoke/output/SAMEA2535210/SAMEA2535210.report.xls`

```text
Name	Drug	Evidence
SAMEA2535210	Ethambutol	embB_p.Met306Ile(R:20.60%),model(0.86)
SAMEA2535210	Isoniazid	katG_p.Ser315Thr(R:77.81%),model(1.00)
SAMEA2535210	Pyrazinamide	pncA_p.Arg154Gly(R:0.39%)
SAMEA2535210	Rifampicin	model(0.95)
```
