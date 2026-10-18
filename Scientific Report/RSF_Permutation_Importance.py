#!/usr/bin/env python3
"""Cross-validated permutation importance for the FINAL RSF specification.

The model specification and repeated-CV splits match the final manuscript:

    RandomSurvivalForest(
        n_estimators=500,
        min_samples_split=10,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42 + fold,
    )

For each held-out fold, each predictor is permuted once within the held-out
fold without refitting. Importance is defined as:

    delta_c_index = C(permuted test feature) - C(original test data)

Therefore negative values indicate a reduction in held-out discrimination when
the feature is permuted. The summary is written to
``rsf_cv_permutation_importance.csv`` and the fold-level audit trail to
``rsf_cv_permutation_importance_by_fold.csv``.

This script intentionally uses the final RSF configuration. If an older
archived importance CSV was generated under an earlier/provisional RSF
configuration, replace that CSV with the output from this script and update any
manuscript sentence that quotes its numerical range.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

SEED = 42
PREDICTORS = [
    "age",
    "pericardial_effusion",
    "fractional_shortening",
    "epss",
    "lvdd",
    "wall_motion_index",
]


def c_index(y, risk) -> float:
    return float(
        concordance_index_censored(
            y["event"],
            y["time"],
            np.asarray(risk, float),
        )[0]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("outcome/uci_echocardiogram_reconstructed_132.csv"),
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("outcome"),
    )
    ap.add_argument("--n-repeats", type=int, default=5)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    real = (
        df.dropna(subset=["survival", "event"] + PREDICTORS)
        .copy()
        .reset_index(drop=True)
    )
    real["event"] = real["event"].astype(int)
    real["pericardial_effusion"] = real["pericardial_effusion"].astype(int)

    X = real[PREDICTORS].astype(float)
    y = Surv.from_arrays(
        event=real["event"].astype(bool).to_numpy(),
        time=real["survival"].astype(float).to_numpy(),
    )

    rkf = RepeatedKFold(
        n_splits=5,
        n_repeats=args.n_repeats,
        random_state=SEED,
    )

    rows = []
    for fold, (tr, te) in enumerate(rkf.split(X)):
        Xtr = X.iloc[tr].copy()
        Xte = X.iloc[te].copy()
        ytr = y[tr]
        yte = y[te]

        model = RandomSurvivalForest(
            n_estimators=500,
            min_samples_split=10,
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=-1,
            random_state=SEED + fold,
        )
        model.fit(Xtr, ytr)
        baseline_c = c_index(yte, model.predict(Xte))

        for feature_i, feature in enumerate(PREDICTORS):
            Xperm = Xte.copy()
            rng = np.random.default_rng(SEED + 10000 * fold + feature_i)
            Xperm[feature] = rng.permutation(Xperm[feature].to_numpy())
            permuted_c = c_index(yte, model.predict(Xperm))
            rows.append(
                {
                    "fold": int(fold),
                    "feature": feature,
                    "baseline_c_index": baseline_c,
                    "permuted_c_index": permuted_c,
                    "delta_c_index": permuted_c - baseline_c,
                }
            )

    detail = pd.DataFrame(rows)
    detail.to_csv(
        args.outdir / "rsf_cv_permutation_importance_by_fold.csv",
        index=False,
    )

    summary = (
        detail.groupby("feature", sort=False)["delta_c_index"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv(
        args.outdir / "rsf_cv_permutation_importance.csv",
        index=False,
    )

    print(summary.to_string(index=False))
    print(
        "\nInterpretation: negative delta_c_index means that permutation "
        "reduced held-out C-index."
    )
    print(f"Saved: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
