#!/usr/bin/env python3
"""
Synthetic-data augmentation experiment for the Scientific Reports revision.

Design principles
-----------------
* Uses only correctly parsed, observed UCI Echocardiogram variables.
* No invented EF, BP, smoking, cholesterol, diabetes, etc.
* Synthetic generator is fitted separately inside each training fold.
* All predictive utility is evaluated on held-out REAL observations.
* No synthetic records are described as patients or external validation.
* Smoothed multivariate bootstrap preserves local joint structure while adding
  covariance-aware noise to continuous variables; binary variables are inherited
  from donor rows.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, norm, ttest_rel
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.duration.hazard_regression import PHReg

SEED = 42
PREDICTORS = [
    "age",
    "pericardial_effusion",
    "fractional_shortening",
    "epss",
    "lvdd",
    "wall_motion_index",
]
CONTINUOUS_FOR_SYNTH = [
    "survival",
    "age",
    "fractional_shortening",
    "epss",
    "lvdd",
    "wall_motion_index",
]
BINARY_FOR_SYNTH = ["event", "pericardial_effusion"]


def harrell_c_index(time, event, risk):
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    risk = np.asarray(risk, float)
    concordant = 0.0
    comparable = 0.0
    for i in range(len(time)):
        for j in range(i + 1, len(time)):
            if time[i] == time[j]:
                continue
            if time[i] < time[j] and event[i] == 1:
                comparable += 1
                if risk[i] > risk[j]: concordant += 1
                elif risk[i] == risk[j]: concordant += 0.5
            elif time[j] < time[i] and event[j] == 1:
                comparable += 1
                if risk[j] > risk[i]: concordant += 1
                elif risk[i] == risk[j]: concordant += 0.5
    return concordant / comparable if comparable else np.nan


def fit_cox(train):
    X = train[PREDICTORS].astype(float)
    model = PHReg(
        endog=train["survival"].astype(float),
        exog=X,
        status=train["event"].astype(int),
        ties="breslow",
    )
    return model.fit(disp=0)


def breslow_baseline(train, beta):
    T = train["survival"].to_numpy(float)
    E = train["event"].to_numpy(int)
    X = train[PREDICTORS].to_numpy(float)
    exp_lp = np.exp(X @ beta)
    event_times = np.sort(np.unique(T[E == 1]))
    cum = 0.0
    rows = []
    for t in event_times:
        d = np.sum((T == t) & (E == 1))
        risk_sum = np.sum(exp_lp[T >= t])
        if risk_sum <= 0: continue
        cum += d / risk_sum
        rows.append((t, cum))
    return np.asarray(rows, float)


def H0_at(baseline, t):
    if baseline.size == 0: return 0.0
    idx = np.searchsorted(baseline[:, 0], t, side="right") - 1
    if idx < 0: return 0.0
    return float(baseline[idx, 1])


def predict_survival(train, fit, test, times):
    beta = np.asarray(fit.params)
    baseline = breslow_baseline(train, beta)
    lp = test[PREDICTORS].to_numpy(float) @ beta
    rr = np.exp(lp)
    out = np.empty((len(test), len(times)), float)
    for j, t in enumerate(times):
        out[:, j] = np.exp(-H0_at(baseline, t) * rr)
    return out


def km_censoring(train):
    T = train["survival"].to_numpy(float)
    # censoring event = 1 when original event == 0
    C = (1 - train["event"].to_numpy(int)).astype(int)
    unique_times = np.sort(np.unique(T[C == 1]))
    g = 1.0
    rows = []
    for t in unique_times:
        n_risk = np.sum(T >= t)
        d = np.sum((T == t) & (C == 1))
        if n_risk > 0:
            g *= (1.0 - d / n_risk)
        rows.append((t, g))
    return np.asarray(rows, float)


def G_at(km, t, left=False):
    if km.size == 0: return 1.0
    side = "left" if left else "right"
    idx = np.searchsorted(km[:, 0], t, side=side) - 1
    if idx < 0: return 1.0
    return max(float(km[idx, 1]), 1e-8)


def ipcw_brier(train, test, surv_prob, times):
    km_g = km_censoring(train)
    T = test["survival"].to_numpy(float)
    E = test["event"].to_numpy(int)
    bs = []
    for j, t in enumerate(times):
        s = surv_prob[:, j]
        terms = np.zeros(len(test), float)
        for i in range(len(test)):
            if T[i] <= t and E[i] == 1:
                w = 1.0 / G_at(km_g, T[i], left=True)
                terms[i] = w * (0.0 - s[i]) ** 2
            elif T[i] > t:
                w = 1.0 / G_at(km_g, t, left=False)
                terms[i] = w * (1.0 - s[i]) ** 2
            else:
                terms[i] = 0.0
        bs.append(float(np.mean(terms)))
    return np.asarray(bs)


def smoothed_bootstrap(train, n_synth, seed, bandwidth=0.10):
    rng = np.random.default_rng(seed)
    donors = train.sample(n=n_synth, replace=True, random_state=seed).reset_index(drop=True)
    synth = donors.copy()

    Z = train[CONTINUOUS_FOR_SYNTH].to_numpy(float)
    cov = np.cov(Z, rowvar=False)
    # regularize for small data / collinearity
    scale = np.diag(np.diag(cov))
    cov_reg = (1 - 0.05) * cov + 0.05 * scale
    noise_cov = (bandwidth ** 2) * cov_reg
    noise = rng.multivariate_normal(np.zeros(len(CONTINUOUS_FOR_SYNTH)), noise_cov, size=n_synth)

    synth_vals = donors[CONTINUOUS_FOR_SYNTH].to_numpy(float) + noise
    for k, col in enumerate(CONTINUOUS_FOR_SYNTH):
        lo = float(train[col].min())
        hi = float(train[col].max())
        synth_vals[:, k] = np.clip(synth_vals[:, k], lo, hi)
    # survival time must be positive
    synth_vals[:, 0] = np.maximum(synth_vals[:, 0], 0.01)
    synth[CONTINUOUS_FOR_SYNTH] = synth_vals

    # Binary fields inherited unchanged from donor record.
    for col in BINARY_FOR_SYNTH:
        synth[col] = donors[col].astype(int)

    return synth[["survival", "event"] + PREDICTORS].copy()


def fidelity_summary(real, synthetic):
    rows = []
    continuous = ["survival", "age", "fractional_shortening", "epss", "lvdd", "wall_motion_index"]
    for c in continuous:
        r = real[c].to_numpy(float)
        s = synthetic[c].to_numpy(float)
        pooled = math.sqrt((np.var(r, ddof=1) + np.var(s, ddof=1)) / 2)
        smd = (np.mean(s) - np.mean(r)) / pooled if pooled > 0 else np.nan
        ks = ks_2samp(r, s).statistic
        rows.append({
            "feature": c,
            "type": "continuous",
            "real_mean": np.mean(r),
            "synthetic_mean": np.mean(s),
            "real_sd": np.std(r, ddof=1),
            "synthetic_sd": np.std(s, ddof=1),
            "SMD": smd,
            "KS_statistic": ks,
            "real_proportion": np.nan,
            "synthetic_proportion": np.nan,
            "absolute_proportion_difference": np.nan,
        })
    for c in ["event", "pericardial_effusion"]:
        pr = float(real[c].mean())
        ps = float(synthetic[c].mean())
        rows.append({
            "feature": c,
            "type": "binary",
            "real_mean": np.nan,
            "synthetic_mean": np.nan,
            "real_sd": np.nan,
            "synthetic_sd": np.nan,
            "SMD": np.nan,
            "KS_statistic": np.nan,
            "real_proportion": pr,
            "synthetic_proportion": ps,
            "absolute_proportion_difference": abs(ps-pr),
        })
    out = pd.DataFrame(rows)

    # Correlation preservation among continuous variables + binary event/PE.
    cols = ["survival", "event"] + PREDICTORS
    corr_r = real[cols].corr(method="spearman")
    corr_s = synthetic[cols].corr(method="spearman")
    corr_rmse = float(np.sqrt(np.mean((corr_r.values - corr_s.values) ** 2)))

    # Simple memorization indicator using standardized continuous space.
    scal = StandardScaler().fit(real[continuous])
    R = scal.transform(real[continuous])
    S = scal.transform(synthetic[continuous])
    # chunked nearest-neighbor distance to avoid extra deps
    mins = []
    for row in S:
        d2 = np.sum((R - row) ** 2, axis=1)
        mins.append(float(np.sqrt(d2.min())))
    nn_median = float(np.median(mins))
    nn_p01 = float(np.quantile(mins, 0.01))

    exact = 0
    rset = set(map(tuple, np.round(real[["survival", "event"] + PREDICTORS].to_numpy(float), 8)))
    for row in np.round(synthetic[["survival", "event"] + PREDICTORS].to_numpy(float), 8):
        if tuple(row) in rset:
            exact += 1

    meta = {
        "spearman_correlation_matrix_rmse": corr_rmse,
        "median_nearest_real_distance_standardized_continuous": nn_median,
        "p01_nearest_real_distance_standardized_continuous": nn_p01,
        "exact_duplicate_count_8dp": int(exact),
        "exact_duplicate_fraction_8dp": float(exact / len(synthetic)),
    }
    return out, corr_r, corr_s, meta


def evaluate_cv(real, outdir, ratios=(1, 2, 5, 10), bandwidth=0.10):
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    times = np.asarray([12.0, 24.0, 36.0])
    ibs_grid = np.linspace(6.0, 36.0, 16)
    rows = []

    for fold, (tr, te) in enumerate(rkf.split(real)):
        train = real.iloc[tr].reset_index(drop=True)
        test = real.iloc[te].reset_index(drop=True)

        strategies = {"real_only": train}
        # synthetic-only is fixed at 5x to represent TSTR.
        syn5 = smoothed_bootstrap(train, len(train)*5, SEED + 1000 + fold, bandwidth)
        strategies["synthetic_only_5x"] = syn5
        for ratio in ratios:
            syn = smoothed_bootstrap(train, len(train)*ratio, SEED + fold*100 + ratio, bandwidth)
            aug = pd.concat([train[["survival", "event"] + PREDICTORS], syn], ignore_index=True)
            strategies[f"real_plus_synth_{ratio}x"] = aug

        for strategy, train_model in strategies.items():
            try:
                fit = fit_cox(train_model)
                beta = np.asarray(fit.params)
                risk = test[PREDICTORS].to_numpy(float) @ beta
                c = harrell_c_index(test["survival"], test["event"], risk)
                sp = predict_survival(train_model, fit, test, times)
                bs = ipcw_brier(train, test, sp, times)  # censoring weights learned from REAL train only
                sp_grid = predict_survival(train_model, fit, test, ibs_grid)
                bs_grid = ipcw_brier(train, test, sp_grid, ibs_grid)
                ibs = float(np.trapezoid(bs_grid, ibs_grid) / (ibs_grid[-1] - ibs_grid[0]))
                rows.append({
                    "fold": fold,
                    "strategy": strategy,
                    "training_n": len(train_model),
                    "real_training_n": len(train),
                    "real_test_n": len(test),
                    "c_index": c,
                    "brier_12m": bs[0],
                    "brier_24m": bs[1],
                    "brier_36m": bs[2],
                    "ibs_6_36m": ibs,
                })
            except Exception as e:
                rows.append({
                    "fold": fold,
                    "strategy": strategy,
                    "training_n": len(train_model),
                    "real_training_n": len(train),
                    "real_test_n": len(test),
                    "c_index": np.nan,
                    "brier_12m": np.nan,
                    "brier_24m": np.nan,
                    "brier_36m": np.nan,
                    "ibs_6_36m": np.nan,
                    "error": str(e),
                })

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "synthetic_augmentation_repeated5fold_5repeats.csv", index=False)

    summary = out.groupby("strategy")[["c_index", "brier_12m", "brier_24m", "brier_36m", "ibs_6_36m"]].agg(["mean", "std", "count"])
    # flatten columns
    summary.columns = [f"{a}_{b}" for a,b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(outdir / "synthetic_augmentation_summary.csv", index=False)

    # paired comparisons vs real-only
    base = out[out.strategy == "real_only"].set_index("fold")
    comp_rows = []
    for strategy in sorted(set(out.strategy) - {"real_only"}):
        cur = out[out.strategy == strategy].set_index("fold")
        common = base.index.intersection(cur.index)
        for metric in ["c_index", "ibs_6_36m"]:
            a = cur.loc[common, metric].astype(float)
            b = base.loc[common, metric].astype(float)
            ok = a.notna() & b.notna()
            diff = (a[ok] - b[ok]).to_numpy()
            n = len(diff)
            md = float(np.mean(diff)) if n else np.nan
            sd = float(np.std(diff, ddof=1)) if n > 1 else np.nan
            se = sd / math.sqrt(n) if n > 1 else np.nan
            ci = (md - 1.96*se, md + 1.96*se) if n > 1 else (np.nan, np.nan)
            p = float(ttest_rel(a[ok], b[ok]).pvalue) if n > 1 else np.nan
            comp_rows.append({
                "strategy": strategy,
                "metric": metric,
                "mean_difference_vs_real_only": md,
                "ci95_lower": ci[0],
                "ci95_upper": ci[1],
                "paired_p": p,
                "n_folds": n,
            })
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(outdir / "synthetic_augmentation_paired_comparisons.csv", index=False)
    return out, summary, comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--bandwidth", type=float, default=0.10)
    ap.add_argument("--global-ratio", type=int, default=5)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    real = df.dropna(subset=["survival", "event"] + PREDICTORS).copy()
    real["event"] = real["event"].astype(int)
    real["pericardial_effusion"] = real["pericardial_effusion"].astype(int)

    # Global synthetic set ONLY for fidelity/supplementary description.
    synthetic = smoothed_bootstrap(real, len(real)*args.global_ratio, SEED, args.bandwidth)
    synthetic.to_csv(args.outdir / f"synthetic_smoothed_bootstrap_{args.global_ratio}x_global_fidelity_only.csv", index=False)

    fidelity, corr_r, corr_s, meta = fidelity_summary(real, synthetic)
    fidelity.to_csv(args.outdir / "synthetic_fidelity_univariate.csv", index=False)
    corr_r.to_csv(args.outdir / "synthetic_fidelity_real_spearman.csv")
    corr_s.to_csv(args.outdir / "synthetic_fidelity_synthetic_spearman.csv")
    with open(args.outdir / "synthetic_fidelity_multivariate_summary.json", "w") as f:
        json.dump(meta, f, indent=2)

    cv, summary, comp = evaluate_cv(real, args.outdir, bandwidth=args.bandwidth)

    report = {
        "real_complete_case_n": int(len(real)),
        "real_events": int(real.event.sum()),
        "real_censored": int(len(real)-real.event.sum()),
        "generator": "smoothed multivariate bootstrap",
        "bandwidth": args.bandwidth,
        "global_fidelity_synthetic_n": int(len(synthetic)),
        "global_fidelity_ratio": args.global_ratio,
        "validation": "5-fold repeated 5 times; generator fitted on real training fold only; all utility evaluated on held-out real observations",
        "fidelity_meta": meta,
    }
    with open(args.outdir / "synthetic_experiment_design_and_summary.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print("\nFidelity:\n", fidelity.to_string(index=False))
    print("\nUtility summary:\n", summary.to_string(index=False))
    print("\nPaired comparisons:\n", comp.to_string(index=False))

if __name__ == "__main__":
    main()
