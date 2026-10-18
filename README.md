# Post-Myocardial-Infarction Survival Modelling:
## A Reproducible Reassessment Framework

This repository contains the reproducibility materials accompanying the
Scientific Reports manuscript.

## Data

The analyses use the UCI Echocardiogram dataset (132 records).

UCI dataset DOI:
https://doi.org/10.24432/C5QW24

The original dataset is licensed under CC BY 4.0.
The reconstructed analysis copy is provided for reproducibility and remains
subject to the source dataset's licensing and attribution requirements.

## Reproducibility

The complete analysis workflow is located in:

`Scientific Report/`

Main entry point:

`Scientific_Reports_Reanalysis_Master.py`

See:

`Scientific Report/README_REPRODUCIBILITY.md`

for detailed execution instructions.

## Main analyses

The repository reproduces:

- Kaplan-Meier survival estimates
- Cox proportional-hazards modelling
- bootstrap optimism correction
- nested fractional-polynomial Cox validation
- Random Survival Forest evaluation
- censoring-aware Brier score and integrated Brier score
- cross-validated RSF permutation importance
- synthetic-data fidelity and augmentation experiments
- fold-paired model comparisons

## License

Analysis code is released under the MIT License.

The UCI Echocardiogram dataset is distributed separately under CC BY 4.0
and is not relicensed by this repository.
