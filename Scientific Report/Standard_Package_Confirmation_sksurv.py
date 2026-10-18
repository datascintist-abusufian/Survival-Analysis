#!/usr/bin/env python3
"""Final standard-package survival-model confirmation.

Uses scikit-survival for Cox PH, RSF, concordance, IPCW Brier score and IBS.
All models are evaluated on the same 25 held-out REAL folds from repeated
5-fold cross-validation (5 repeats, seed 42). Synthetic observations are
created only inside the real training fold.

Prediction-error evaluation uses administrative censoring immediately after
36 months so the requested times (12/24/36 and the 6--36 IBS grid) are strictly
inside the evaluation support. Concordance uses full observed follow-up.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from statsmodels.duration.hazard_regression import PHReg

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import (
    concordance_index_censored,
    brier_score,
    integrated_brier_score,
)

from Synthetic_Augmentation_Experiment import smoothed_bootstrap

SEED = 42
PREDICTORS = [
    "age",
    "pericardial_effusion",
    "fractional_shortening",
    "epss",
    "lvdd",
    "wall_motion_index",
]
FP_POWERS = [-2, -1, -0.5, 0, 0.5, 1, 2, 3]
HORIZONS = np.array([12.0, 24.0, 36.0])
IBS_GRID = np.linspace(6.0, 36.0, 16)
ADMIN_HORIZON = 36.0
ADMIN_EPS = 1e-6


def fp(x, p):
    x = np.asarray(x, float)
    return np.log(x) if p == 0 else x ** p


def fp_design(d, pa, pw):
    X = d[PREDICTORS].copy().astype(float)
    X["age"] = fp(d["age"], pa)
    X["wall_motion_index"] = fp(d["wall_motion_index"], pw)
    return X


def fit_fp_train(train):
    best = None
    for pa in FP_POWERS:
        for pw in FP_POWERS:
            X = fp_design(train, pa, pw)
            try:
                model = PHReg(
                    train.survival.astype(float),
                    X,
                    status=train.event.astype(int),
                    ties="breslow",
                )
                fit = model.fit(disp=0)
                aic = 2 * X.shape[1] - 2 * float(fit.llf)
                if best is None or aic < best["aic"]:
                    best = {"pa": pa, "pw": pw, "fit": fit, "Xtr": X, "aic": aic}
            except Exception:
                pass
    if best is None:
        raise RuntimeError("No FP-Cox candidate converged in this training fold.")
    return best


def breslow_baseline(train, X, beta):
    T = train.survival.to_numpy(float)
    E = train.event.to_numpy(int)
    rr = np.exp(X.to_numpy(float) @ beta)
    rows = []
    cum = 0.0
    for t in np.sort(np.unique(T[E == 1])):
        d = np.sum((T == t) & (E == 1))
        den = np.sum(rr[T >= t])
        if den > 0:
            cum += d / den
            rows.append((t, cum))
    return np.asarray(rows, float)


def H0_at(base, t):
    if base.size == 0:
        return 0.0
    k = np.searchsorted(base[:, 0], t, side="right") - 1
    return 0.0 if k < 0 else float(base[k, 1])


def fp_survival_matrix(train, best, test, times):
    Xte = fp_design(test, best["pa"], best["pw"])
    beta = np.asarray(best["fit"].params)
    base = breslow_baseline(train, best["Xtr"], beta)
    rr = np.exp(Xte.to_numpy(float) @ beta)
    return np.column_stack([np.exp(-H0_at(base, t) * rr) for t in times]), Xte


def surv_matrix(model, X, times):
    fns = model.predict_survival_function(X)
    return np.asarray([[fn(t) for t in times] for fn in fns])


def surv_full(d: pd.DataFrame):
    return Surv.from_arrays(
        d.event.astype(bool).to_numpy(),
        d.survival.astype(float).to_numpy(),
    )


def surv_admin36(d: pd.DataFrame):
    """Administratively censor follow-up after 36 months at 36+epsilon."""
    time = d.survival.astype(float).to_numpy().copy()
    event = d.event.astype(bool).to_numpy().copy()
    after = time > ADMIN_HORIZON
    time[after] = ADMIN_HORIZON + ADMIN_EPS
    event[after] = False
    return Surv.from_arrays(event=event, time=time)


def prediction_error_metrics(train_real, test_real, sp_h, sp_g):
    y_train_eval = surv_admin36(train_real)
    y_test_eval = surv_admin36(test_real)
    _, bs = brier_score(y_train_eval, y_test_eval, sp_h, HORIZONS)
    ibs = integrated_brier_score(y_train_eval, y_test_eval, sp_g, IBS_GRID)
    return bs, float(ibs)


def evaluate_sksurv(train_real, test_real, train_model, model_kind, fold):
    y_test_full = surv_full(test_real)

    Xtr = train_model[PREDICTORS].astype(float)
    Xte = test_real[PREDICTORS].astype(float)
    y_model = surv_full(train_model)

    if model_kind == "cox":
        model = CoxPHSurvivalAnalysis(alpha=0.0).fit(Xtr, y_model)
    elif model_kind == "rsf":
        model = RandomSurvivalForest(
            n_estimators=500,
            min_samples_split=10,
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=-1,
            random_state=SEED + fold,
        )
        model.fit(Xtr, y_model)
    else:
        raise ValueError(model_kind)

    risk = model.predict(Xte)
    c = concordance_index_censored(
        y_test_full["event"], y_test_full["time"], risk
    )[0]
    sp_h = surv_matrix(model, Xte, HORIZONS)
    sp_g = surv_matrix(model, Xte, IBS_GRID)
    bs, ibs = prediction_error_metrics(train_real, test_real, sp_h, sp_g)
    return float(c), bs, ibs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("outcome/uci_echocardiogram_reconstructed_132.csv"),
    )
    ap.add_argument("--outdir", type=Path, default=Path("validation_outputs"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    real = (
        df.dropna(subset=["survival", "event"] + PREDICTORS)
        .copy()
        .reset_index(drop=True)
    )
    real.event = real.event.astype(int)
    real.pericardial_effusion = real.pericardial_effusion.astype(int)

    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    rows = []

    for fold, (tr, te) in enumerate(rkf.split(real)):
        train = real.iloc[tr].reset_index(drop=True)
        test = real.iloc[te].reset_index(drop=True)

        c, bs, ibs = evaluate_sksurv(train, test, train, "cox", fold)
        rows.append({
            "fold": fold, "model": "Cox PH", "training": "Real only",
            "c_index": c, "brier_12m": bs[0], "brier_24m": bs[1],
            "brier_36m": bs[2], "ibs_6_36m": ibs,
        })

        c, bs, ibs = evaluate_sksurv(train, test, train, "rsf", fold)
        rows.append({
            "fold": fold, "model": "RSF", "training": "Real only",
            "c_index": c, "brier_12m": bs[0], "brier_24m": bs[1],
            "brier_36m": bs[2], "ibs_6_36m": ibs,
        })

        best = fit_fp_train(train)
        y_test_full = surv_full(test)
        sp_h, Xte = fp_survival_matrix(train, best, test, HORIZONS)
        sp_g, _ = fp_survival_matrix(train, best, test, IBS_GRID)
        risk = Xte.to_numpy(float) @ np.asarray(best["fit"].params)
        c = concordance_index_censored(
            y_test_full["event"], y_test_full["time"], risk
        )[0]
        bs, ibs = prediction_error_metrics(train, test, sp_h, sp_g)
        rows.append({
            "fold": fold,
            "model": "FP-Cox",
            "training": f"Real only; nested powers age={best['pa']}, WMI={best['pw']}",
            "c_index": c, "brier_12m": bs[0], "brier_24m": bs[1],
            "brier_36m": bs[2], "ibs_6_36m": ibs,
        })

        syn5 = smoothed_bootstrap(train, len(train) * 5, SEED + 1000 + fold, 0.10)
        c, bs, ibs = evaluate_sksurv(train, test, syn5, "cox", fold)
        rows.append({
            "fold": fold, "model": "Cox PH", "training": "Synthetic only 5x",
            "c_index": c, "brier_12m": bs[0], "brier_24m": bs[1],
            "brier_36m": bs[2], "ibs_6_36m": ibs,
        })

        syn1 = smoothed_bootstrap(train, len(train), SEED + fold * 100 + 1, 0.10)
        aug = pd.concat(
            [train[["survival", "event"] + PREDICTORS], syn1],
            ignore_index=True,
        )
        c, bs, ibs = evaluate_sksurv(train, test, aug, "cox", fold)
        rows.append({
            "fold": fold, "model": "Cox PH", "training": "Real + 1x synthetic",
            "c_index": c, "brier_12m": bs[0], "brier_24m": bs[1],
            "brier_36m": bs[2], "ibs_6_36m": ibs,
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.outdir / "standard_package_fold_results.csv", index=False)

    # This raw summary intentionally preserves the full training description.
    # The corrected reviewer-facing aggregation is created by
    # Generate_Final_Summary_Tables.py, which collapses fold-specific FP powers.
    summary = out.groupby(["model", "training"])[
        ["c_index", "brier_12m", "brier_24m", "brier_36m", "ibs_6_36m"]
    ].agg(["mean", "std", "count"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(args.outdir / "standard_package_model_comparison.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
