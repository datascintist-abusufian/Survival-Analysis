Post-Myocardial-Infarction Survival Modelling

A Reproducible Reassessment Framework

This repository contains the code, reconstructed analysis data, validation outputs, and reproducibility materials accompanying the revised Scientific Reports manuscript on post-myocardial-infarction survival modelling using the UCI Echocardiogram dataset.

Data

The primary analyses use the UCI Echocardiogram dataset, comprising 132 original clinical records.

UCI dataset DOI:
https://doi.org/10.24432/C5QW24

The dataset contains follow-up information for patients after myocardial infarction.

For the reproducibility analyses:

* survival time is defined using the available post-myocardial-infarction follow-up time;
* event = 1 denotes death;
* event = 0 denotes right-censoring for patients recorded as still alive.

The original dataset is licensed under CC BY 4.0.

A reconstructed analysis copy is included solely to support reproducibility and remains subject to the original dataset’s licensing and attribution requirements.

Synthetic Data

Synthetic observations used in this project are derived computationally from the original UCI data.

They must not be interpreted as:

* additional clinical patients;
* an independent cohort;
* external validation data; or
* evidence of a larger real-world sample size.

Synthetic data are used only for methodological experiments involving data fidelity and training-data augmentation.

Where augmentation is evaluated within cross-validation, synthetic observations are generated from the corresponding training data and are not used to replace held-out real observations.

Reproducibility

The complete revised analysis workflow is located in:

Scientific Report/

The main execution script is:

Scientific Report/Scientific_Reports_Reanalysis_Master.py

Detailed execution instructions, software requirements, analysis order, endpoint definitions, and output descriptions are provided in:

Scientific Report/README_REPRODUCIBILITY.md

The master workflow coordinates the corrected analyses and generates the principal numerical and graphical outputs used in the revised manuscript.

Main Analyses

The repository reproduces the following analyses:

* Kaplan-Meier survival estimation;
* Cox proportional-hazards modelling;
* bootstrap optimism correction;
* fractional-polynomial Cox modelling;
* nested fractional-polynomial Cox validation;
* Random Survival Forest evaluation;
* repeated cross-validation;
* Harrell concordance index estimation;
* censoring-aware Brier score calculation;
* integrated Brier score estimation;
* cross-validated Random Survival Forest permutation importance;
* age and wall-motion-index fractional-polynomial effect analyses;
* synthetic-data fidelity assessment;
* synthetic training-data augmentation experiments;
* bandwidth-sensitivity analyses for synthetic generation;
* fold-paired model comparisons; and
* final comparative model-performance summaries.

Validation Framework

Model performance is assessed using held-out real observations.

The revised framework includes repeated cross-validation and reports survival-model discrimination and prediction-error measures, including:

* concordance index;
* time-dependent, censoring-aware Brier scores; and
* integrated Brier score.

Synthetic observations are not treated as independent validation subjects.

Study Analysis Workflow

Original UCI Echocardiogram dataset
n = 132 records
        │
        ▼
Data reconstruction and audit
• Correct dataset schema
• Missing-value assessment
• Survival-time definition
• Death event = 1
• Right-censoring/alive = 0
        │
        ├───────────────────────────────┐
        │                               │
        ▼                               ▼
Primary real-data analyses       Synthetic-data experiments
                                Derived from original/training data
Kaplan-Meier                    • Synthetic-data generation
Cox proportional hazards       • Fidelity assessment
Bootstrap optimism correction  • Bandwidth sensitivity
Fractional-polynomial Cox      • Synthetic-only experiments
Nested FP-Cox validation       • Training-data augmentation
Random Survival Forest
        │                               │
        └──────────────┬────────────────┘
                       ▼
              Repeated validation
             5-fold CV × 5 repeats
                25 held-out folds
                       │
                       ▼
               Model comparison
• Harrell concordance index
• IPCW/censoring-aware Brier score
• Integrated Brier score
• Fold-paired model comparisons
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      FP effect analyses    RSF permutation
       Age and WMI            importance
             │                   │
             └─────────┬─────────┘
                       ▼
              Final locked results
                       │
                       ▼
         Scientific Reports revision


Repository Structure

Key reproducibility materials include:

Scientific Report/
│
├── Scientific_Reports_Reanalysis_Master.py
├── README_REPRODUCIBILITY.md
├── Real_Data_Core_Reanalysis.py
├── nested_fp_cox_validation.py
├── synthetic_data_experiments.py
├── bandwidth_sensitivity.py
├── standard_package_comparison.py
├── final_summary_tables.py
└── rsf_permutation_importance.py


Additional generated tables, figures, validation outputs, and intermediate files are produced by the reproducibility workflow.

Outcome Definition

The primary endpoint is post-myocardial-infarction survival/mortality as represented in the UCI Echocardiogram dataset.

The dataset does not contain a directly observed heart-failure endpoint.

Accordingly, heart failure should be interpreted only as relevant clinical context and not as the primary observed outcome of these analyses.

Fractional-Polynomial Analysis

Fractional-polynomial modelling is used to examine possible nonlinear relationships between continuous predictors and survival outcome.

Particular attention is given to:

* age; and
* wall-motion index (WMI).

The revised workflow evaluates fractional-polynomial transformations and generates effect analyses comparing conventional linear specifications with selected nonlinear forms.

Random Survival Forest Analysis

Random Survival Forest modelling is evaluated using held-out real observations within the validation framework.

The workflow includes:

* model fitting;
* concordance-based performance assessment;
* censoring-aware prediction-error assessment; and
* cross-validated permutation importance.

Variable-importance results are interpreted as predictive associations within this dataset and should not be interpreted as causal effects.

Model Performance Evaluation

The revised reproducibility framework uses comparable held-out real-data validation wherever possible.

Primary evaluation measures include:

Concordance Index

Harrell’s concordance index is used to quantify survival-model discrimination.

Brier Score

Time-dependent Brier scores are calculated using censoring-aware methodology.

Integrated Brier Score

Integrated Brier score summarises prediction error across an evaluated follow-up interval.

Performance results are reported with explicit reference to the model, validation framework, evaluation horizon, and underlying dataset to avoid ambiguity between previously reported concordance estimates.

Synthetic-Data Experiments

Synthetic data are retained only as a secondary methodological component.

The synthetic analyses investigate:

* similarity between original and generated distributions;
* sensitivity to synthetic-generation settings;
* whether synthetic training-data augmentation materially changes model performance; and
* whether observed changes persist when evaluation is restricted to held-out real records.

Synthetic-data results do not constitute external validation.

The effective number of independent clinical observations remains the size of the original UCI cohort.

Reproducibility Principles

The revised analysis follows the following principles:

1. The 132 original UCI records are clearly separated from derived synthetic observations.
2. Synthetic observations are not described as independent patients.
3. Held-out real observations are used for primary performance evaluation.
4. Survival time, event status, and censoring are explicitly defined.
5. Model-specific concordance and Brier results are reported separately.
6. Nonlinear fractional-polynomial relationships are evaluated directly rather than inferred from descriptive distributions.
7. Prediction results are distinguished from causal interpretation.
8. Lack of external clinical validation is explicitly acknowledged.
9. All principal analyses are linked to executable scripts.
10. Generated outputs are intended to be traceable to the reproducibility workflow.

Interpretation and Limitations

This repository supports a reproducible reassessment of survival modelling in a small historical echocardiographic dataset.

Important limitations include:

* the small original sample size;
* missing values in the source data;
* limited clinical variables;
* the historical nature of the dataset;
* absence of an independent external validation cohort;
* absence of a directly observed heart-failure endpoint; and
* the fact that synthetic observations do not increase the number of independent patients.

Results should therefore be interpreted primarily as a methodological and reproducibility analysis rather than as evidence of immediately generalisable clinical performance.

Clinical Scope

The analyses concern survival following myocardial infarction.

The repository does not claim to directly predict incident heart failure from the UCI Echocardiogram dataset.

Where heart failure is discussed in the accompanying manuscript, it is considered in the broader clinical context of adverse cardiovascular outcomes following myocardial infarction.

Code Availability

The complete analysis code required to reproduce the revised results is provided in this repository.

The recommended entry point is:

Scientific Report/Scientific_Reports_Reanalysis_Master.py

Users should consult:

Scientific Report/README_REPRODUCIBILITY.md

before running the workflow.

Data Availability

The original Echocardiogram dataset is publicly available through the UCI Machine Learning Repository.

Dataset DOI:
https://doi.org/10.24432/C5QW24

The reconstructed copy included in this repository is supplied for reproducibility and remains subject to the source dataset’s CC BY 4.0 licence.

Citation

Users of the original Echocardiogram dataset should cite the UCI Machine Learning Repository dataset using:

https://doi.org/10.24432/C5QW24

The associated Scientific Reports manuscript should also be cited once final publication details are available.

License

Analysis code in this repository is released under the MIT License.

The UCI Echocardiogram dataset is distributed separately under CC BY 4.0 and is not relicensed by this repository.

Any reconstructed or redistributed copy of the dataset remains subject to the original dataset’s licensing and attribution requirements.

Disclaimer

This repository is provided for research and reproducibility purposes.

The models and outputs are not intended for direct clinical decision-making, diagnosis, treatment selection, or individual patient risk assessment without appropriate independent validation.



