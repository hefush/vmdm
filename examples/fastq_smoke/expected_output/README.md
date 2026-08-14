# Expected FASTQ Smoke-Test Output

This directory contains the primary report snapshots from our verification run
of the bundled FASTQ smoke test:

```bash
python vmdm.py examples/fastq_smoke/example_fastq.list examples/fastq_smoke/output --jobs 1 --threads 2
```

After running the smoke test, compare the generated reports under
`examples/fastq_smoke/output/<sample_name>/` with the corresponding files in
this directory. Exact model probability values can vary slightly with library
builds and threading, so the semantic checks in `../README.md` are the main
pass criteria.
