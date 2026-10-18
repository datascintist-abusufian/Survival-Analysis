#!/usr/bin/env python3
"""Generate final reviewer-facing model-comparison tables.

This script is a deterministic post-processing step. It does not fit models.
It reads ``standard_package_fold_results.csv`` produced by
``Standard_Package_Confirmation_sksurv.py`` and writes:

* final_overall_model_summary.csv
* final_corrected_model_comparison.csv
* paired_fold_model_differences.csv
* cox_rsf_repeated5fold_5repeats.csv

The FP-Cox training-description column contains fold-specific selected powers;
those rows are intentionally collapsed into one FP-Cox model family before
aggregation. Paired differences are descriptive only; no inferential p-values
are produced because repeated-CV folds are not independent observations.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = ["c_index", "brier_12m", "brier_24m", "brier_36m", "ibs_6_36m"]


def canonical_model(row: pd.Series) -> str:
    model = str(row["model"]).strip()
    training = str(row["training"]).strip()

    if model == "Cox PH" and training == "Real only":
        return "Cox PH, real only"
    if model == "FP-Cox" and training.startswith("Real only"):
        return "FP-Cox, real only (nested FP)"
    if model == "RSF" and training == "Real only":
        return "RSF, real only"
    if model == "Cox PH" and training == "Synthetic only 5x":
        return "Cox PH, synthetic only 5x"
    if model == "Cox PH" and training == "Real + 1x synthetic":
        return "Cox PH, real + 1x synthetic"
    return f"{model} | {training}"


def validate_fold_table(df: pd.DataFrame) -> pd.DataFrame:
    required = {"fold", "model", "training", *METRICS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    out = df.copy()
    out["model_family"] = out.apply(canonical_model, axis=1)
    for c in ["fold", *METRICS]:
        out[c] = pd.to_numeric(out[c], errors="raise")

    # The final standard-package run should contain one row per model family
    # for each of 25 held-out folds.
    duplicates = out.duplicated(["fold", "model_family"], keep=False)
    if duplicates.any():
        bad = out.loc[duplicates, ["fold", "model_family"]]
        raise ValueError(
            "Duplicate fold/model-family rows detected:\n" + bad.to_string(index=False)
        )
    return out


def overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("model_family", sort=False)[METRICS].agg(["mean", "std", "count"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    return agg.reset_index().rename(columns={"model_family": "model"})


def corrected_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "model",
        "c_index_mean", "c_index_std",
        "brier_12m_mean", "brier_24m_mean", "brier_36m_mean",
        "ibs_6_36m_mean", "ibs_6_36m_std",
        "c_index_count",
    ]
    out = summary[cols].copy()
    out = out.rename(columns={"c_index_count": "n_folds"})
    order = [
        "Cox PH, real only",
        "FP-Cox, real only (nested FP)",
        "RSF, real only",
        "Cox PH, synthetic only 5x",
        "Cox PH, real + 1x synthetic",
    ]
    rank = {name: i for i, name in enumerate(order)}
    out["_order"] = out["model"].map(rank).fillna(len(order))
    out = out.sort_values(["_order", "model"]).drop(columns="_order").reset_index(drop=True)
    return out


def paired_differences(df: pd.DataFrame) -> pd.DataFrame:
    baseline_name = "Cox PH, real only"
    base = df[df.model_family == baseline_name].set_index("fold")
    if base.empty:
        raise ValueError("Real-only Cox baseline is missing.")

    rows: list[dict] = []
    models = [m for m in df.model_family.drop_duplicates() if m != baseline_name]

    for model in models:
        cur = df[df.model_family == model].set_index("fold")
        common = base.index.intersection(cur.index)
        if len(common) == 0:
            continue

        for metric in ["c_index", "ibs_6_36m"]:
            diff = (
                cur.loc[common, metric].astype(float)
                - base.loc[common, metric].astype(float)
            )
            # For C-index, >0 favours the comparison model.
            # For IBS, <0 favours the comparison model.
            if metric == "c_index":
                n_better = int((diff > 0).sum())
                n_worse = int((diff < 0).sum())
            else:
                n_better = int((diff < 0).sum())
                n_worse = int((diff > 0).sum())

            rows.append({
                "model": model,
                "reference": baseline_name,
                "metric": metric,
                "mean_difference_model_minus_reference": float(diff.mean()),
                "sd_difference": float(diff.std(ddof=1)),
                "median_difference": float(diff.median()),
                "n_folds": int(diff.shape[0]),
                "n_folds_favouring_model": n_better,
                "n_folds_favouring_reference": n_worse,
                "n_ties": int((diff == 0).sum()),
                "inference": "descriptive only; repeated-CV folds are not independent",
            })
    return pd.DataFrame(rows)


def cox_rsf_fold_export(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact fold-level Cox-vs-RSF table from FINAL package results.

    This intentionally replaces the earlier provisional audit table of the same
    filename, so the archived file is aligned with the final scikit-survival
    confirmation rather than with superseded custom calculations.
    """
    cox = df[df.model_family == "Cox PH, real only"].set_index("fold")
    rsf = df[df.model_family == "RSF, real only"].set_index("fold")
    common = cox.index.intersection(rsf.index).sort_values()
    if len(common) == 0:
        raise ValueError("Could not find paired real-only Cox and RSF folds.")

    out = pd.DataFrame({
        "fold": common.astype(int),
        "cox_c": cox.loc[common, "c_index"].to_numpy(float),
        "cox_b12": cox.loc[common, "brier_12m"].to_numpy(float),
        "cox_b24": cox.loc[common, "brier_24m"].to_numpy(float),
        "cox_b36": cox.loc[common, "brier_36m"].to_numpy(float),
        "cox_ibs": cox.loc[common, "ibs_6_36m"].to_numpy(float),
        "rsf_c": rsf.loc[common, "c_index"].to_numpy(float),
        "rsf_b12": rsf.loc[common, "brier_12m"].to_numpy(float),
        "rsf_b24": rsf.loc[common, "brier_24m"].to_numpy(float),
        "rsf_b36": rsf.loc[common, "brier_36m"].to_numpy(float),
        "rsf_ibs": rsf.loc[common, "ibs_6_36m"].to_numpy(float),
    })
    return out.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fold-results",
        type=Path,
        default=Path("validation_outputs/standard_package_fold_results.csv"),
        help="Fold-level CSV produced by Standard_Package_Confirmation_sksurv.py",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("validation_outputs"),
    )
    ap.add_argument(
        "--cox-rsf-outdir",
        type=Path,
        default=Path("outcome"),
        help="Directory for cox_rsf_repeated5fold_5repeats.csv",
    )
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.cox_rsf_outdir.mkdir(parents=True, exist_ok=True)

    folds = validate_fold_table(pd.read_csv(args.fold_results))
    summary = overall_summary(folds)
    corrected = corrected_comparison(summary)
    paired = paired_differences(folds)
    cox_rsf = cox_rsf_fold_export(folds)

    summary.to_csv(args.outdir / "final_overall_model_summary.csv", index=False)
    corrected.to_csv(args.outdir / "final_corrected_model_comparison.csv", index=False)
    paired.to_csv(args.outdir / "paired_fold_model_differences.csv", index=False)
    cox_rsf.to_csv(args.cox_rsf_outdir / "cox_rsf_repeated5fold_5repeats.csv", index=False)

    print("Wrote:")
    print(args.outdir / "final_overall_model_summary.csv")
    print(args.outdir / "final_corrected_model_comparison.csv")
    print(args.outdir / "paired_fold_model_differences.csv")
    print(args.cox_rsf_outdir / "cox_rsf_repeated5fold_5repeats.csv")
    print("\nFinal comparison:\n", corrected.to_string(index=False))
    print("\nPaired descriptive differences:\n", paired.to_string(index=False))


if __name__ == "__main__":
    main()
