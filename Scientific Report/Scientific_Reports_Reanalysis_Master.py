#!/usr/bin/env python3
"""Orchestrator for the Scientific Reports major-revision reanalysis.

Runs the corrected real-data core analysis, nested FP-Cox validation,
synthetic-data fidelity/utility experiment, and optionally the final
scikit-survival confirmation.

Expected repository layout (inside ``Scientific Report/``):
    Scientific_Reports_Reanalysis_Master.py
    Real_Data_Core_Reanalysis.py
    Nested_FP_Cox_Validation_v1.py
    Synthetic_Augmentation_Experiment.py
    Standard_Package_Confirmation_sksurv.py
    outcome/uci_echocardiogram_reconstructed_132.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n>>>", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True)


def main():
    here = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=None,
        help=(
            "Path to uci_echocardiogram_reconstructed_132.csv. "
            "Default: outcome/uci_echocardiogram_reconstructed_132.csv "
            "relative to this script."
        ),
    )
    ap.add_argument("--outdir", type=Path, default=Path("rerun_outputs"))
    ap.add_argument("--standard-confirmation", action="store_true")
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

    required = [
        "Real_Data_Core_Reanalysis.py",
        "Nested_FP_Cox_Validation_v1.py",
        "Synthetic_Augmentation_Experiment.py",
    ]
    if args.standard_confirmation:
        required.append("Standard_Package_Confirmation_sksurv.py")

    missing = [name for name in required if not (here / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required script(s): " + ", ".join(missing)
        )

    run(
        [
            sys.executable,
            here / "Real_Data_Core_Reanalysis.py",
            "--data",
            data,
            "--outdir",
            out / "real_data_core",
        ]
    )

    run(
        [
            sys.executable,
            here / "Nested_FP_Cox_Validation_v1.py",
            "--data",
            data,
            "--outdir",
            out / "nested_fp",
        ]
    )

    run(
        [
            sys.executable,
            here / "Synthetic_Augmentation_Experiment.py",
            "--data",
            data,
            "--outdir",
            out / "synthetic_v1",
            "--bandwidth",
            "0.10",
            "--global-ratio",
            "5",
        ]
    )

    if args.standard_confirmation:
        run(
            [
                sys.executable,
                here / "Standard_Package_Confirmation_sksurv.py",
                "--data",
                data,
                "--outdir",
                out / "validation_outputs",
            ]
        )
    else:
        print(
            "\nStandard-package confirmation not run. "
            "Install scikit-survival and re-run with --standard-confirmation."
        )


if __name__ == "__main__":
    main()
