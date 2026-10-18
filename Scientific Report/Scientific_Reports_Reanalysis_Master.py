#!/usr/bin/env python3
"""Orchestrator for the Scientific Reports reproducibility release.

Expected layout inside ``Scientific Report/``:

    Scientific_Reports_Reanalysis_Master.py
    Real_Data_Core_Reanalysis.py
    Nested_FP_Cox_Validation_v1.py
    Synthetic_Augmentation_Experiment.py
    Synthetic_Bandwidth_Sensitivity.py
    Standard_Package_Confirmation_sksurv.py
    Generate_Final_Summary_Tables.py
    RSF_Permutation_Importance.py
    outcome/uci_echocardiogram_reconstructed_132.csv

Core analyses do not require scikit-survival. The final standard-package
confirmation and RSF permutation-importance audit do.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n>>>", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def require(here: Path, names: list[str]) -> None:
    missing = [name for name in names if not (here / name).exists()]
    if missing:
        raise FileNotFoundError("Missing required script(s): " + ", ".join(missing))


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Default: outcome/uci_echocardiogram_reconstructed_132.csv",
    )
    ap.add_argument("--outdir", type=Path, default=Path("rerun_outputs"))
    ap.add_argument(
        "--standard-confirmation",
        action="store_true",
        help="Run the final scikit-survival model comparison and derived tables.",
    )
    ap.add_argument(
        "--bandwidth-sensitivity",
        action="store_true",
        help="Reproduce synthetic bandwidth sensitivity (0.05/0.10/0.20).",
    )
    ap.add_argument(
        "--rsf-permutation-importance",
        action="store_true",
        help="Run cross-validated permutation importance for the final RSF.",
    )
    ap.add_argument(
        "--full-release",
        action="store_true",
        help="Run all optional release analyses (equivalent to all three flags).",
    )
    args = ap.parse_args()

    data = (
        args.data.resolve()
        if args.data is not None
        else (here / "outcome" / "uci_echocardiogram_reconstructed_132.csv").resolve()
    )
    out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not data.exists():
        raise FileNotFoundError(f"Data file not found: {data}")

    standard = args.standard_confirmation or args.full_release
    bandwidth = args.bandwidth_sensitivity or args.full_release
    permimp = args.rsf_permutation_importance or args.full_release

    required = [
        "Real_Data_Core_Reanalysis.py",
        "Nested_FP_Cox_Validation_v1.py",
        "Synthetic_Augmentation_Experiment.py",
    ]
    if bandwidth:
        required.append("Synthetic_Bandwidth_Sensitivity.py")
    if standard:
        required += [
            "Standard_Package_Confirmation_sksurv.py",
            "Generate_Final_Summary_Tables.py",
        ]
    if permimp:
        required.append("RSF_Permutation_Importance.py")
    require(here, required)

    run([
        sys.executable, here / "Real_Data_Core_Reanalysis.py",
        "--data", data,
        "--outdir", out / "real_data_core",
    ])

    run([
        sys.executable, here / "Nested_FP_Cox_Validation_v1.py",
        "--data", data,
        "--outdir", out / "nested_fp",
    ])

    run([
        sys.executable, here / "Synthetic_Augmentation_Experiment.py",
        "--data", data,
        "--outdir", out / "synthetic_primary",
        "--bandwidth", "0.10",
        "--global-ratio", "5",
    ])

    if bandwidth:
        run([
            sys.executable, here / "Synthetic_Bandwidth_Sensitivity.py",
            "--data", data,
            "--output", out / "synthetic_bandwidth_sensitivity_summary.csv",
        ])

    if standard:
        validation = out / "validation_outputs"
        run([
            sys.executable, here / "Standard_Package_Confirmation_sksurv.py",
            "--data", data,
            "--outdir", validation,
        ])
        run([
            sys.executable, here / "Generate_Final_Summary_Tables.py",
            "--fold-results", validation / "standard_package_fold_results.csv",
            "--outdir", validation,
            "--cox-rsf-outdir", out / "outcome",
        ])

    if permimp:
        run([
            sys.executable, here / "RSF_Permutation_Importance.py",
            "--data", data,
            "--outdir", out / "outcome",
        ])

    print(f"\nAll requested analyses completed. Outputs: {out}")


if __name__ == "__main__":
    main()
