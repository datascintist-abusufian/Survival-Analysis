# Post-Myocardial-Infarction Survival Modelling from Echocardiographic Variables

Code, data and verified outputs for the manuscript:

> **Post-Myocardial-Infarction Survival Modelling from Echocardiographic Variables: Cox Regression, Fractional Polynomials, Random Survival Forests, and Synthetic-Data Augmentation**\
> Md Abu Sufian, Wahiba Hamzi, Mai Ali, Amira Ali, Boumediene Hamzi\
> Revised manuscript submitted to *Scientific Reports* (under review).

## Overview

This repository re-evaluates survival after myocardial infarction (MI) using the original UCI Echocardiogram cohort. The 132 original records are reconstructed from the 13-field UCI schema and analysed separately from any synthetic observations.

The analysis covers:

- Kaplan–Meier estimation and multivariable Cox proportional hazards regression
- Fractional-polynomial (FP) Cox modelling with nested repeated cross-validation
- Random survival forests (RSF)
- A smoothed-bootstrap synthetic-data fidelity and utility experiment, with all models evaluated on held-out **real** observations only
- Confirmation of discrimination and calibration metrics (C-index, Brier score, integrated Brier score) with `scikit-survival`

## Repository structure

```text
Scientific Report/
├── Scientific_Reports_Reanalysis_Master.py   # main entry point (runs the full pipeline)
├── Real_Data_Core_Reanalysis.py              # real-data core analysis (KM, Cox, FP, bootstrap optimism)
├── Nested_FP_Cox_Validation_v1.py            # nested FP-Cox repeated cross-validation
├── Synthetic_Augmentation_Experiment.py      # synthetic-data fidelity/utility experiment
├── Standard_Package_Confirmation_sksurv.py   # scikit-survival confirmation of metrics
├── requirements_reanalysis.txt               # Python dependencies
├── README_REPRODUCIBILITY.md                 # detailed reproduction instructions
├── outcome/                                  # source data and core-analysis outputs
│   └── uci_echocardiogram_reconstructed_132.csv
├── validation_outputs/                       # model-comparison and confirmation tables
├── locked_results_ledger.json                # locked results reported in the manuscript
├── blockdiagram/                             # workflow diagram
├── Scientific_Reports_Revised_LaTeX_Bundle/  # LaTeX source and figures
├── Scientific_Reports_Revised_Full_Manuscript.tex
└── Scientific_Reports_Revised_Full_Manuscript.pdf
```

## Quick start

Python 3.11 or later is recommended.

```bash
cd "Scientific Report"
pip install -r requirements_reanalysis.txt
python Scientific_Reports_Reanalysis_Master.py --outdir rerun_outputs --standard-confirmation
```

The script reads `outcome/uci_echocardiogram_reconstructed_132.csv` by default and writes all regenerated outputs to `rerun_outputs/`. Omit `--standard-confirmation` to skip the `scikit-survival` step. See [`Scientific Report/README_REPRODUCIBILITY.md`](Scientific%20Report/README_REPRODUCIBILITY.md) for details.

## Data

The source data are the UCI Machine Learning Repository **Echocardiogram** dataset (Salzberg, 1989; [doi:10.24432/C5QW24](https://doi.org/10.24432/C5QW24)), distributed under CC BY 4.0. `uci_echocardiogram_reconstructed_132.csv` is a correctly parsed 132-record version of that dataset. Survival is measured in months after MI; `event = 1` denotes death and `event = 0` denotes censoring.

**Note on earlier material.** An earlier version of this project used a 100,000-row synthetic extension of the dataset (Mendeley Data, [doi:10.17632/x5nmsprjc9.1](https://doi.org/10.17632/x5nmsprjc9.1)). That extension is **not** used in the revised analysis and should not be used to reproduce the manuscript results.

## Citation

If you use this code, please cite the archived release (DOI to be added after the Zenodo release) and the manuscript once published.

## License

The code in this repository is released under the [MIT License](LICENSE). The UCI Echocardiogram data remain subject to their original CC BY 4.0 licence.

## Contact

Md Abu Sufian, University of East London — m.sufian@uel.ac.uk
Corresponding author: Boumediene Hamzi — b.hamzi@imperial.ac.uk
