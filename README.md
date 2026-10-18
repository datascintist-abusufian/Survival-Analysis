# Post-Myocardial-Infarction Survival Modelling: A Reproducible Reassessment Framework

This repository contains the reproducibility materials accompanying the revised Scientific Reports manuscript **“Post-Myocardial-Infarction Survival Modelling: A Reproducible Reassessment Framework.”**

The revision reassesses post-myocardial-infarction survival modelling using the original UCI Echocardiogram cohort and a transparent internal-validation workflow. The repository is intended to reproduce the reported statistical analyses and reviewer-requested validation checks; it is **not** a clinically validated prediction tool.

## Data source

The primary analyses use the **UCI Echocardiogram** dataset (132 source records).

- UCI dataset page: https://archive.ics.uci.edu/dataset/38/echocardiogram
- Dataset DOI: https://doi.org/10.24432/C5QW24
- UCI dataset license: **Creative Commons Attribution 4.0 International (CC BY 4.0)**

The reconstructed analysis copy is included only to make the revised analysis reproducible. The dataset remains subject to the UCI source license and is **not relicensed under the software license of this repository**.

## Reproducibility package

The analysis package is located in:

```text
Scientific Report/
```

The main entry point is:

```text
Scientific Report/Scientific_Reports_Reanalysis_Master.py
```

Detailed execution instructions and output provenance are provided in:

```text
Scientific Report/README_REPRODUCIBILITY.md
```

## Main analyses

The release workflow covers:

- dataset reconstruction/audit and missing-data accounting;
- Kaplan–Meier survival estimation;
- Cox proportional-hazards modelling and bootstrap optimism correction;
- fractional-polynomial assessment with nested power selection;
- Random Survival Forest comparison;
- repeated 5-fold cross-validation with 5 repeats;
- concordance index, IPCW Brier scores at 12/24/36 months, and IBS over 6–36 months;
- cross-validated RSF permutation importance;
- synthetic-data fidelity and held-out-real augmentation experiments;
- synthetic bandwidth sensitivity analysis; and
- deterministic generation of the final reviewer-facing comparison tables.

Synthetic observations are generated only for methodological experiments. They are not treated as independent patients and are not external validation data.

## Quick start

From the `Scientific Report` directory, install the recorded environment and run the complete release workflow:

```bash
python -m pip install -r requirements_reanalysis.txt
python Scientific_Reports_Reanalysis_Master.py --full-release --outdir rerun_outputs
```

`scikit-survival` is required for the final standard-package confirmation and RSF permutation-importance audit.

## Release status

This repository should be tagged only after the final rerun has been checked against the locked manuscript results. A Zenodo DOI can then be created from the tagged release and inserted into the manuscript Code Availability statement.

## License

Original analysis code in this repository is released under the **MIT License**; see `LICENSE`.

The UCI Echocardiogram data are distributed under **CC BY 4.0** by the UCI Machine Learning Repository and remain under that license.
