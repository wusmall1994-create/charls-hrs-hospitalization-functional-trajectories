# CHARLS-HRS hospitalization and functional trajectories

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22351432.svg)](https://doi.org/10.5281/zenodo.22351432)

This repository contains the statistical analysis code for a coordinated study of functional trajectories around first observed hospitalization in the China Health and Retirement Longitudinal Study (CHARLS) and the US Health and Retirement Study (HRS). It contains no participant-level data, manuscript files, tables, figures, or personal contact information.

## Analysis scope

The code reproduces the following analyses:

- harmonized five-item activities of daily living (ADL) event studies;
- harmonized four-item instrumental activities of daily living (IADL) event studies;
- cross-cohort effect comparisons and sensitivity analyses;
- effect modification, age dose-response, and social-context analyses;
- missing-item imputation and mortality or competing-outcome analyses;
- publication figure generation and numerical result validation.

The exposure is first observed hospitalization during follow-up. The code does not identify first-ever hospitalization.

## Data access

Raw data are not redistributed here. Researchers must obtain the Harmonized CHARLS Version D, the CHARLS 2020 Exit Module, and the RAND HRS Longitudinal File 2022 Version 1 from their official providers and comply with the applicable terms of use.

- CHARLS: <https://charls.pku.edu.cn/en/>
- HRS: <https://hrs.isr.umich.edu/data-products>

Three Stata files are required. Their local locations are supplied through environment variables:

| Variable | Required file |
| --- | --- |
| `CHARLS_DATA` | Harmonized CHARLS Version D `.dta` file |
| `CHARLS_EXIT_DATA` | CHARLS 2020 Exit Module `.dta` file |
| `HRS_DATA` | RAND HRS Longitudinal File 2022 Version 1 `.dta` file |

An optional `CHARLS_HRS_OUTPUT_DIR` variable controls the output directory. If omitted, the pipeline writes to `results/`, which is excluded from version control.

## Software environment

Python 3.12 is recommended. Create an isolated environment and install the pinned dependencies:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the analysis

Set the required paths. For example, in PowerShell:

```powershell
$env:CHARLS_DATA = "path/to/H_CHARLS_D_Data.dta"
$env:CHARLS_EXIT_DATA = "path/to/Exit_Module.dta"
$env:HRS_DATA = "path/to/randhrs1992_2022v1.dta"
python scripts/run_all.py
```

On macOS or Linux:

```bash
export CHARLS_DATA="path/to/H_CHARLS_D_Data.dta"
export CHARLS_EXIT_DATA="path/to/Exit_Module.dta"
export HRS_DATA="path/to/randhrs1992_2022v1.dta"
python scripts/run_all.py
```

The pipeline runs the primary ADL analysis, multidomain extensions, targeted analyses, figure scripts, and numerical validation in sequence.

## Script map

| Script | Purpose |
| --- | --- |
| `scripts/config.py` | Portable input and output path configuration |
| `scripts/hrs_core.py` | HRS harmonization and shared estimation functions |
| `scripts/primary_analysis.py` | Locked two-cohort ADL analysis |
| `scripts/multidomain_analysis.py` | IADL, imputation, and competing-outcome extensions |
| `scripts/targeted_extensions.py` | Age dose-response and social-context analyses |
| `scripts/plot_main_results.py` | Main and supplementary multidomain figures |
| `scripts/plot_extensions.py` | Age and contextual extension figures |
| `scripts/validate_results.py` | Checks principal estimates and expected outputs |

## Repository checks

The tests compile every Python file and reject participant data, manuscripts, generated outputs, local absolute paths, and known personal contact strings:

```bash
python -m unittest discover -s tests -v
```

## Reproducibility boundary

The public-use datasets are versioned independently by CHARLS, HRS, and the Gateway to Global Aging Data. Small numerical differences may occur if providers revise source files or if dependency versions differ from `requirements.txt`. The validation script uses narrow tolerances for the principal standardized estimates.

## Licence

The analysis code is released under the MIT License. The licence does not apply to CHARLS or HRS data, which remain subject to their providers' terms.

## Citation

Wu X, Liang H, Wei H, Liu L. Statistical analysis code for hospitalization and functional trajectories in CHARLS and HRS. Version 1.0.0. Zenodo. 2026. https://doi.org/10.5281/zenodo.22351432
