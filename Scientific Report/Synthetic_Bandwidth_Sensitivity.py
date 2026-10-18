#!/usr/bin/env python3
"""Reproduce the synthetic-data bandwidth sensitivity analysis.

This script reuses the generator and held-out-real evaluation functions in
``Synthetic_Augmentation_Experiment.py``. It evaluates bandwidths 0.05, 0.10,
and 0.20 and writes ``synthetic_bandwidth_sensitivity_summary.csv``.

The global 5x synthetic sample at each bandwidth is used only for fidelity
metrics. Predictive utility is assessed by repeated 5-fold CV (5 repeats) with
the generator fitted inside each real training fold.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

import Synthetic_Augmentation_Experiment as sae

DEFAULT_BANDWIDTHS = (0.05, 0.10, 0.20)


def prepare_real(data: Path) -> pd.DataFrame:
    df = pd.read_csv(data)
    real = df.dropna(subset=["survival", "event"] + sae.PREDICTORS).copy()
    real["event"] = real["event"].astype(int)
    real["pericardial_effusion"] = real["pericardial_effusion"].astype(int)
    return real.reset_index(drop=True)


def fidelity_metrics(real: pd.DataFrame, bandwidth: float) -> dict:
    synthetic = sae.smoothed_bootstrap(
        real,
        len(real) * 5,
        sae.SEED,
        bandwidth,
    )
    fidelity, _, _, meta = sae.fidelity_summary(real, synthetic)
    continuous = fidelity[fidelity["type"] == "continuous"]
    return {
        "max_abs_continuous_SMD": float(continuous["SMD"].abs().max()),
        "max_KS_statistic": float(continuous["KS_statistic"].max()),
        "correlation_rmse": float(meta["spearman_correlation_matrix_rmse"]),
        "exact_duplicate_fraction": float(meta["exact_duplicate_fraction_8dp"]),
    }


def run_one_bandwidth(
    real: pd.DataFrame,
    bandwidth: float,
    workdir: Path,
) -> pd.DataFrame:
    fidelity = fidelity_metrics(real, bandwidth)
    _, summary, _ = sae.evaluate_cv(real, workdir, bandwidth=bandwidth)

    rows = []
    # evaluate_cv returns the groupby summary in alphabetical strategy order.
    for _, r in summary.iterrows():
        rows.append({
            "bandwidth": float(bandwidth),
            "strategy": r["strategy"],
            "c_index_mean": float(r["c_index_mean"]),
            "c_index_std": float(r["c_index_std"]),
            "ibs_6_36m_mean": float(r["ibs_6_36m_mean"]),
            "ibs_6_36m_std": float(r["ibs_6_36m_std"]),
            **fidelity,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("outcome/uci_echocardiogram_reconstructed_132.csv"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("synthetic_bandwidth_sensitivity_summary.csv"),
    )
    ap.add_argument(
        "--bandwidths",
        type=float,
        nargs="+",
        default=list(DEFAULT_BANDWIDTHS),
    )
    ap.add_argument(
        "--keep-fold-outputs",
        action="store_true",
        help="Keep per-bandwidth CV outputs under bandwidth_sensitivity_runs/.",
    )
    args = ap.parse_args()

    real = prepare_real(args.data)
    all_rows = []

    if args.keep_fold_outputs:
        base = args.output.parent / "bandwidth_sensitivity_runs"
        base.mkdir(parents=True, exist_ok=True)
        for bw in args.bandwidths:
            workdir = base / f"bandwidth_{bw:.2f}"
            workdir.mkdir(parents=True, exist_ok=True)
            all_rows.append(run_one_bandwidth(real, bw, workdir))
    else:
        with tempfile.TemporaryDirectory(prefix="synth_bw_") as tmp:
            base = Path(tmp)
            for bw in args.bandwidths:
                workdir = base / f"bandwidth_{bw:.2f}"
                workdir.mkdir(parents=True, exist_ok=True)
                all_rows.append(run_one_bandwidth(real, bw, workdir))

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(["bandwidth", "strategy"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
