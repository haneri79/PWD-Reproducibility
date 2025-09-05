
````markdown
This repository contains the reproducibility artifact for Phase Wave Determinism (PWD), including code, benchmark data, and evaluation scripts.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17059069.svg)](https://doi.org/10.5281/zenodo.17059069)

# PWD Reproducibility Artifact (v1.1)

This Zenodo bundle contains a **single-file judge** and **minimal data** to reproduce the benchmark, plus outputs, logs, manifest, and citation metadata.

## Quick start

```bash
python -m pip install -U numpy pandas scipy
python Judge_Extended.py --input bench.csv \
  --ts-input dyn.csv --ts-ref-method REF --ts-delta 0.03 \
  --metal-input bands.csv --metal-window-ev 0.3 --metal-delta-ev 0.05 \
  --sc-input sc.csv --sc-ref-method REF --sc-delta 0.03 \
  --fit-scaling --predict-scale 2 4 8 16 32 \
  --out verdict.json --markdown summary.md
````

## Safe-run (OS/Colab/Windows)

All runs are handled by `tools/runner.py`, which ensures:

* **Timeout enforcement** (processes are killed after the limit)
* **Peak RAM tracking** including child processes
* `shell=False` safe execution

### Windows (PowerShell)

```powershell
$env:BASELINE_SOLVER="C:\path\to\baseline_solver.exe"
$env:PWD_SOLVER="C:\path\to\pwd_solver.exe"
.\hero\HS1_softCoulomb_3D\run_base.ps1
.\hero\HS1_softCoulomb_3D\run_pwd.ps1
```

### Linux / Colab

```bash
bash hero/HS1_softCoulomb_3D/run_base.sh
bash hero/HS1_softCoulomb_3D/run_pwd.sh
python hero/HS1_softCoulomb_3D/collect.py
python tools/verify_pwd.py --input hero/HS1_softCoulomb_3D/bench_hero.csv --bootstrap 200
```

* If `--input` is omitted, the script **auto-switches to selftest** (no error exit).
* In notebooks/Colab, any `-f kernel.json` is **ignored automatically**.

## Files

* `Judge_Extended.py` — main script (CLI/Jupyter/Colab compatible)
* `bench.csv`, `dyn.csv`, `bands.csv`, `sc.csv` — minimal sample data
* `verdict.json`, `summary.md` — produced by the smoke test during packaging
* `run_stdout.txt`, `run_stderr.txt` — logs from the local smoke test
* `MANIFEST.json` — file list with size and sha256
* `CITATION.cff` — cite this artifact
* `LICENSE` — license
* `README.md` — this file

## Re-run tests

```bash
python Judge_Extended.py --runtests --selftest --out /tmp/x.json
```

All **T1–T14** tests should pass.

## 🚀 Hero Suite (HS1) — external “hundreds×” verification

This repository also includes a Hero Suite for large grids (e.g., 256³ / 512³), enabling end-to-end verification of **fairness, accuracy (TOST), and speedup (hundreds×)** in a single workflow.

### Requirements

```bash
python -m pip install -U numpy pandas scipy matplotlib
```

## Citation and Preprint

If you use this artifact, please cite as:

[Author](mailto:haneri79@hanmail.net), *Super Speed Quantum Simulation — PWD Reproducibility Artifact (v1.1)*, Zenodo (2025).
Preprint / Artifact DOI: [https://doi.org/10.5281/zenodo.17059069](https://doi.org/10.5281/zenodo.17059069)

---

Generated on: 2025-09-04 22:09:45.338071+00:00

```
