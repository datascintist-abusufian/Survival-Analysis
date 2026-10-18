#!/usr/bin/env python3
"""
Scientific Reports major-revision reproducibility pipeline
=========================================================

Project:
    Long-term Heart Risk Prediction by Survival Analysis in Echocardiography:
    Leveraging Machine Learning, Interpretability Techniques, and Advanced
    Statistical Modelling

Purpose:
    This is the canonical rerun script for the corrected real-data analysis.
    It reconstructs the UCI Echocardiogram dataset using the correct 13-field
    schema, defines the survival endpoint correctly, and reproduces the
    corrected Kaplan-Meier, Cox PH, fractional-polynomial and validation
    analyses used during the major-revision audit.

Important:
    - Real source cohort: 132 UCI Echocardiogram records.
    - 'survival' = months observed after myocardial infarction.
    - 'still_alive' = 0 dead, 1 still alive/censored.
    - Event used in survival models = 1 - still_alive.
    - Synthetic data are NOT generated in this version. They will be added
      only after the real-data pipeline is locked.
    - The RSF section is implemented with scikit-survival and is intentionally
      guarded because the current audit environment did not have scikit-survival.
      Final manuscript RSF/Brier values must be generated with the standard
      package block in this file before submission.

This script does not hard-code manuscript results. It recalculates them from data.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import RepeatedKFold
from statsmodels.duration.hazard_regression import PHReg

SEED = 42
RNG = np.random.default_rng(SEED)

UCI_COLUMNS = [
    "survival",
    "still_alive",
    "age",
    "pericardial_effusion",
    "fractional_shortening",
    "epss",
    "lvdd",
    "wall_motion_score",
    "wall_motion_index",
    "mult",
    "name",
    "group",
    "alive_at_one",
]

PRIMARY_PREDICTORS = [
    "age",
    "pericardial_effusion",
    "fractional_shortening",
    "epss",
    "lvdd",
    "wall_motion_index",
]

FP_POWERS = [-2, -1, -0.5, 0, 0.5, 1, 2, 3]


# ---------------------------------------------------------------------
# 1. DATA RECONSTRUCTION
# ---------------------------------------------------------------------

def read_uci_raw(raw_path: Path) -> pd.DataFrame:
    """
    Parse the original echocardiogram.data file using the correct 13 fields.
    '?' is treated as missing.
    """
    rows = []
    with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 13:
                raise ValueError(
                    f"Line {line_number} contains {len(parts)} fields, expected 13."
                )
            rows.append(parts)

    df = pd.DataFrame(rows, columns=UCI_COLUMNS)

    numeric_cols = [c for c in UCI_COLUMNS if c != "name"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c].replace("?", np.nan), errors="coerce")

    # Correct survival endpoint:
    # still_alive == 0 -> death/event
    # still_alive == 1 -> censored/still alive at last follow-up
    df["event"] = np.where(
        df["still_alive"].isin([0, 1]),
        1 - df["still_alive"],
        np.nan,
    )
    return df


def load_analysis_data(data_path: Path) -> pd.DataFrame:
    """
    Accept either the original raw UCI data file or the reconstructed CSV.
    """
    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path)
        if "event" not in df.columns:
            df["event"] = np.where(
                df["still_alive"].isin([0, 1]),
                1 - df["still_alive"],
                np.nan,
            )
        return df
    return read_uci_raw(data_path)


def save_data_audit(df: pd.DataFrame, outdir: Path) -> dict:
    valid = df.dropna(subset=["survival", "still_alive"]).copy()
    cc = df.dropna(subset=["survival", "event"] + PRIMARY_PREDICTORS).copy()

    summary = {
        "raw_records": int(len(df)),
        "valid_survival_status_records": int(len(valid)),
        "deaths": int(valid["event"].sum()),
        "censored": int(len(valid) - valid["event"].sum()),
        "complete_case_primary_model_n": int(len(cc)),
        "complete_case_primary_model_deaths": int(cc["event"].sum()),
        "complete_case_primary_model_censored": int(len(cc) - cc["event"].sum()),
        "missingness": {
            c: int(df[c].isna().sum())
            for c in [
                "survival",
                "still_alive",
                "age",
                "pericardial_effusion",
                "fractional_shortening",
                "epss",
                "lvdd",
                "wall_motion_score",
                "wall_motion_index",
                "mult",
                "alive_at_one",
            ]
        },
    }

    with open(outdir / "data_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    df.to_csv(outdir / "uci_echocardiogram_reconstructed_132.csv", index=False)
    return summary


# ---------------------------------------------------------------------
# 2. KAPLAN-MEIER
# ---------------------------------------------------------------------

def kaplan_meier_table(time: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)

    event_times = np.sort(np.unique(time[event == 1]))
    s = 1.0
    rows = []

    for t in event_times:
        n_risk = int(np.sum(time >= t))
        d = int(np.sum((time == t) & (event == 1)))
        if n_risk > 0:
            s *= (1.0 - d / n_risk)
        rows.append((t, n_risk, d, s))

    return pd.DataFrame(
        rows, columns=["time", "n_at_risk", "deaths", "survival_probability"]
    )


def km_survival_at(km: pd.DataFrame, t: float) -> float:
    eligible = km[km["time"] <= t]
    if eligible.empty:
        return 1.0
    return float(eligible.iloc[-1]["survival_probability"])


def run_km(df: pd.DataFrame, outdir: Path) -> dict:
    d = df.dropna(subset=["survival", "event"]).copy()
    km = kaplan_meier_table(d["survival"].values, d["event"].values)
    km.to_csv(outdir / "kaplan_meier_curve.csv", index=False)

    horizons = [12, 24, 36, 48]
    estimates = {f"{h}_months": km_survival_at(km, h) for h in horizons}

    median_time = np.nan
    crossing = km[km["survival_probability"] <= 0.5]
    if not crossing.empty:
        median_time = float(crossing.iloc[0]["time"])

    summary = {
        "n": int(len(d)),
        "deaths": int(d["event"].sum()),
        "censored": int(len(d) - d["event"].sum()),
        "survival_estimates": estimates,
        "median_survival_months": median_time,
        "max_observed_followup_months": float(d["survival"].max()),
    }
    with open(outdir / "kaplan_meier_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(6.5, 4.5))
    x = np.r_[0.0, km["time"].values]
    y = np.r_[1.0, km["survival_probability"].values]
    plt.step(x, y, where="post")
    plt.xlabel("Months after myocardial infarction")
    plt.ylabel("Kaplan-Meier survival probability")
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(outdir / "kaplan_meier_corrected.png", dpi=300)
    plt.close()

    return summary


# ---------------------------------------------------------------------
# 3. COX PROPORTIONAL HAZARDS
# ---------------------------------------------------------------------

def fit_cox(df: pd.DataFrame, predictors: list[str]):
    d = df.dropna(subset=["survival", "event"] + predictors).copy()
    x = d[predictors].astype(float)
    model = PHReg(
        endog=d["survival"].astype(float),
        exog=x,
        status=d["event"].astype(int),
        ties="breslow",
    )
    result = model.fit()
    return d, model, result


def cox_summary_table(result, predictors: list[str]) -> pd.DataFrame:
    beta = np.asarray(result.params)
    se = np.asarray(result.bse)
    z = beta / se
    p = np.asarray(result.pvalues)
    hr = np.exp(beta)
    lower = np.exp(beta - 1.96 * se)
    upper = np.exp(beta + 1.96 * se)

    return pd.DataFrame(
        {
            "predictor": predictors,
            "beta": beta,
            "se": se,
            "HR": hr,
            "lower_95CI": lower,
            "upper_95CI": upper,
            "z": z,
            "p": p,
        }
    )


def harrell_c_index(time, event, risk) -> float:
    """
    Harrell C for right-censored data.
    Higher risk is assumed to mean shorter survival.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)

    concordant = 0.0
    comparable = 0.0

    n = len(time)
    for i in range(n):
        for j in range(i + 1, n):
            if time[i] == time[j]:
                continue

            # i had an observed event before j
            if time[i] < time[j] and event[i] == 1:
                comparable += 1
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5

            # j had an observed event before i
            elif time[j] < time[i] and event[j] == 1:
                comparable += 1
                if risk[j] > risk[i]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5

    return float(concordant / comparable) if comparable else np.nan


def run_cox(df: pd.DataFrame, outdir: Path):
    d, model, result = fit_cox(df, PRIMARY_PREDICTORS)

    summary = cox_summary_table(result, PRIMARY_PREDICTORS)
    summary.to_csv(outdir / "cox_complete_case_summary.csv", index=False)

    linear_predictor = np.dot(d[PRIMARY_PREDICTORS].values, result.params)
    apparent_c = harrell_c_index(
        d["survival"].values, d["event"].values, linear_predictor
    )

    info = {
        "n": int(len(d)),
        "deaths": int(d["event"].sum()),
        "censored": int(len(d) - d["event"].sum()),
        "log_likelihood": float(result.llf),
        "apparent_harrell_c_index": apparent_c,
    }
    with open(outdir / "cox_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    return d, result, info


# ---------------------------------------------------------------------
# 4. BOOTSTRAP OPTIMISM FOR COX C-INDEX
# ---------------------------------------------------------------------

def bootstrap_optimism_c_index(
    d: pd.DataFrame,
    predictors: list[str],
    n_boot: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap optimism:
        optimism_b = C(model_b on bootstrap_b) - C(model_b on original)
    corrected C = apparent C - mean(optimism_b)
    """
    rng = np.random.default_rng(seed)

    _, _, original_fit = fit_cox(d, predictors)
    lp_original = d[predictors].values @ original_fit.params
    apparent_c = harrell_c_index(d["survival"], d["event"], lp_original)

    rows = []
    n = len(d)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = d.iloc[idx].reset_index(drop=True)

        # skip degenerate bootstrap samples
        if boot["event"].sum() < 2:
            continue

        try:
            _, _, fit_b = fit_cox(boot, predictors)
            risk_boot = boot[predictors].values @ fit_b.params
            risk_orig = d[predictors].values @ fit_b.params

            c_boot = harrell_c_index(
                boot["survival"], boot["event"], risk_boot
            )
            c_orig = harrell_c_index(
                d["survival"], d["event"], risk_orig
            )
            rows.append((b, c_boot, c_orig, c_boot - c_orig))
        except Exception:
            continue

    out = pd.DataFrame(
        rows, columns=["bootstrap", "c_bootstrap", "c_original", "optimism"]
    )
    out.attrs["apparent_c"] = apparent_c
    out.attrs["mean_optimism"] = float(out["optimism"].mean())
    out.attrs["optimism_corrected_c"] = float(
        apparent_c - out["optimism"].mean()
    )
    return out


# ---------------------------------------------------------------------
# 5. FRACTIONAL POLYNOMIAL ANALYSIS
# ---------------------------------------------------------------------

def fp_transform(x: pd.Series, power: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("FP transformation requires positive values.")
    if power == 0:
        return np.log(x)
    return np.power(x, power)


def fit_cox_matrix(d: pd.DataFrame, X: pd.DataFrame):
    model = PHReg(
        d["survival"].astype(float),
        X.astype(float),
        status=d["event"].astype(int),
        ties="breslow",
    )
    result = model.fit()
    aic = 2 * X.shape[1] - 2 * float(result.llf)
    return result, aic


def run_fractional_polynomials(df: pd.DataFrame, outdir: Path):
    d = df.dropna(subset=["survival", "event"] + PRIMARY_PREDICTORS).copy()

    baseX = d[PRIMARY_PREDICTORS].copy()
    _, linear_aic = fit_cox_matrix(d, baseX)

    rows = [
        {
            "model": "Linear age + linear WMI",
            "age_power": 1.0,
            "wmi_power": 1.0,
            "AIC": linear_aic,
        }
    ]

    # Best age FP1, WMI kept linear
    best_age = None
    for p in FP_POWERS:
        X = baseX.copy()
        X["age"] = fp_transform(d["age"], p)
        r, aic = fit_cox_matrix(d, X)
        if best_age is None or aic < best_age["AIC"]:
            best_age = {"power": p, "AIC": aic, "result": r}
    rows.append(
        {
            "model": "Best age FP1 only",
            "age_power": best_age["power"],
            "wmi_power": 1.0,
            "AIC": best_age["AIC"],
        }
    )

    # Best WMI FP1, age kept linear
    best_wmi = None
    for p in FP_POWERS:
        X = baseX.copy()
        X["wall_motion_index"] = fp_transform(d["wall_motion_index"], p)
        r, aic = fit_cox_matrix(d, X)
        if best_wmi is None or aic < best_wmi["AIC"]:
            best_wmi = {"power": p, "AIC": aic, "result": r}
    rows.append(
        {
            "model": "Best WMI FP1 only",
            "age_power": 1.0,
            "wmi_power": best_wmi["power"],
            "AIC": best_wmi["AIC"],
        }
    )

    # Joint grid
    best_joint = None
    for pa in FP_POWERS:
        for pw in FP_POWERS:
            X = baseX.copy()
            X["age"] = fp_transform(d["age"], pa)
            X["wall_motion_index"] = fp_transform(
                d["wall_motion_index"], pw
            )
            r, aic = fit_cox_matrix(d, X)
            if best_joint is None or aic < best_joint["AIC"]:
                best_joint = {
                    "age_power": pa,
                    "wmi_power": pw,
                    "AIC": aic,
                    "result": r,
                    "X": X,
                }

    rows.append(
        {
            "model": "Best joint FP1",
            "age_power": best_joint["age_power"],
            "wmi_power": best_joint["wmi_power"],
            "AIC": best_joint["AIC"],
        }
    )

    comparison = pd.DataFrame(rows)
    comparison["delta_AIC_vs_linear"] = comparison["AIC"] - linear_aic
    comparison.to_csv(
        outdir / "fractional_polynomial_model_comparison.csv", index=False
    )

    # Create reviewer-requested effect curves:
    # compare linear and selected FP forms while holding all other covariates
    # at their medians.
    med = d[PRIMARY_PREDICTORS].median()

    # Linear reference fit
    linear_res, _ = fit_cox_matrix(d, baseX)

    # Age curve: use best age-only FP1 model
    age_grid = np.linspace(d["age"].min(), d["age"].max(), 200)
    age_ref = float(d["age"].median())

    X_ref_lin = med.to_frame().T
    lp_ref_lin = (X_ref_lin.values @ linear_res.params).item()

    age_rows = []
    for a in age_grid:
        xlin = med.copy()
        xlin["age"] = a
        lp_lin = (xlin.values @ linear_res.params).item()

        xfp = med.copy()
        xfp["age"] = fp_transform(pd.Series([a]), best_age["power"])[0]
        xref_fp = med.copy()
        xref_fp["age"] = fp_transform(
            pd.Series([age_ref]), best_age["power"]
        )[0]
        lp_fp = (xfp.values @ best_age["result"].params).item()
        lp_fp_ref = (xref_fp.values @ best_age["result"].params).item()

        age_rows.append(
            {
                "age": a,
                "linear_HR_vs_median": math.exp(lp_lin - lp_ref_lin),
                "fp_HR_vs_median": math.exp(lp_fp - lp_fp_ref),
            }
        )

    age_curve = pd.DataFrame(age_rows)
    age_curve.to_csv(outdir / "age_linear_vs_fp_curve.csv", index=False)

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(age_curve["age"], age_curve["linear_HR_vs_median"], label="Linear")
    plt.plot(age_curve["age"], age_curve["fp_HR_vs_median"], label=f"FP1 p={best_age['power']}")
    plt.axhline(1.0, linewidth=1)
    plt.xlabel("Age at heart attack (years)")
    plt.ylabel("Relative hazard vs median age")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        outdir / "age_linear_vs_fractional_polynomial.png", dpi=300
    )
    plt.close()

    # WMI curve
    wmi_grid = np.linspace(
        d["wall_motion_index"].min(),
        d["wall_motion_index"].max(),
        200,
    )
    wmi_ref = float(d["wall_motion_index"].median())
    wmi_rows = []

    for w in wmi_grid:
        xlin = med.copy()
        xlin["wall_motion_index"] = w
        lp_lin = (xlin.values @ linear_res.params).item()

        xfp = med.copy()
        xfp["wall_motion_index"] = fp_transform(
            pd.Series([w]), best_wmi["power"]
        )[0]
        xref_fp = med.copy()
        xref_fp["wall_motion_index"] = fp_transform(
            pd.Series([wmi_ref]), best_wmi["power"]
        )[0]
        lp_fp = (xfp.values @ best_wmi["result"].params).item()
        lp_fp_ref = (xref_fp.values @ best_wmi["result"].params).item()

        wmi_rows.append(
            {
                "wall_motion_index": w,
                "linear_HR_vs_median": math.exp(lp_lin - lp_ref_lin),
                "fp_HR_vs_median": math.exp(lp_fp - lp_fp_ref),
            }
        )

    wmi_curve = pd.DataFrame(wmi_rows)
    wmi_curve.to_csv(outdir / "wmi_linear_vs_fp_curve.csv", index=False)

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(
        wmi_curve["wall_motion_index"],
        wmi_curve["linear_HR_vs_median"],
        label="Linear",
    )
    plt.plot(
        wmi_curve["wall_motion_index"],
        wmi_curve["fp_HR_vs_median"],
        label=f"FP1 p={best_wmi['power']}",
    )
    plt.axhline(1.0, linewidth=1)
    plt.xlabel("Wall-motion index")
    plt.ylabel("Relative hazard vs median WMI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        outdir / "wmi_linear_vs_fractional_polynomial.png", dpi=300
    )
    plt.close()

    return comparison


# ---------------------------------------------------------------------
# 6. STANDARD-PACKAGE RSF + BRIER / IBS CONFIRMATION
# ---------------------------------------------------------------------

def run_standard_rsf_confirmation(df: pd.DataFrame, outdir: Path):
    """
    Final reviewer-facing RSF block.

    Requires:
        scikit-survival

    This section intentionally uses the standard package requested for the
    final manuscript. It should be run before submission and its values should
    replace any provisional custom/audit RSF numbers.
    """
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
        from sksurv.metrics import (
            concordance_index_censored,
            brier_score,
            integrated_brier_score,
        )
    except ImportError as exc:
        print(
            "\n[RSF confirmation skipped] scikit-survival is not installed.\n"
            "Install with: pip install scikit-survival\n"
            "The Cox/KM/FP outputs above are unaffected.\n"
        )
        return None

    d = df.dropna(subset=["survival", "event"] + PRIMARY_PREDICTORS).copy()
    X = d[PRIMARY_PREDICTORS].astype(float)
    y = Surv.from_arrays(
        event=d["event"].astype(bool).values,
        time=d["survival"].astype(float).values,
    )

    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    horizons = np.array([12.0, 24.0, 36.0])

    rows = []

    for fold, (tr, te) in enumerate(rkf.split(X)):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]

        rsf = RandomSurvivalForest(
            n_estimators=500,
            min_samples_split=10,
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=-1,
            random_state=SEED + fold,
        )
        rsf.fit(Xtr, ytr)

        risk = rsf.predict(Xte)
        c = concordance_index_censored(
            yte["event"], yte["time"], risk
        )[0]

        surv_funcs = rsf.predict_survival_function(Xte)
        prob = np.asarray(
            [[fn(t) for t in horizons] for fn in surv_funcs]
        )

        # brier_score expects shape (n_samples, n_times)
        times, bs = brier_score(ytr, yte, prob, horizons)

        # IBS over the same available horizon window
        ibs = integrated_brier_score(ytr, yte, prob, horizons)

        rows.append(
            {
                "fold": fold,
                "rsf_c_index": c,
                "brier_12m": float(bs[0]),
                "brier_24m": float(bs[1]),
                "brier_36m": float(bs[2]),
                "ibs_12_36m": float(ibs),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "rsf_standard_package_confirmation.csv", index=False)

    summary = out.agg(["mean", "std"]).T
    summary.to_csv(outdir / "rsf_standard_package_confirmation_summary.csv")

    print("\nStandard-package RSF confirmation:")
    print(summary)
    return out


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("uci_echocardiogram_reconstructed_132.csv"),
        help=(
            "Path to original echocardiogram.data or the reconstructed CSV."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("scientific_reports_rerun"),
    )
    parser.add_argument(
        "--run-rsf-confirmation",
        action="store_true",
        help="Run scikit-survival RSF/Brier/IBS confirmation if installed.",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_analysis_data(args.data)

    print("\n1. DATA AUDIT")
    audit = save_data_audit(df, args.outdir)
    print(json.dumps(audit, indent=2))

    print("\n2. KAPLAN-MEIER")
    km = run_km(df, args.outdir)
    print(json.dumps(km, indent=2))

    print("\n3. COX PH")
    cox_data, cox_result, cox_info = run_cox(df, args.outdir)
    print(json.dumps(cox_info, indent=2))
    print(pd.read_csv(args.outdir / "cox_complete_case_summary.csv"))

    print("\n4. BOOTSTRAP OPTIMISM")
    boot = bootstrap_optimism_c_index(
        cox_data, PRIMARY_PREDICTORS, n_boot=500, seed=SEED
    )
    boot.to_csv(args.outdir / "cox_bootstrap_optimism.csv", index=False)
    boot_summary = {
        "apparent_c": boot.attrs["apparent_c"],
        "mean_optimism": boot.attrs["mean_optimism"],
        "optimism_corrected_c": boot.attrs["optimism_corrected_c"],
        "successful_bootstraps": int(len(boot)),
    }
    with open(
        args.outdir / "cox_bootstrap_optimism_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(boot_summary, f, indent=2)
    print(json.dumps(boot_summary, indent=2))

    print("\n5. FRACTIONAL POLYNOMIAL ANALYSIS")
    fp = run_fractional_polynomials(df, args.outdir)
    print(fp)

    if args.run_rsf_confirmation:
        print("\n6. STANDARD-PACKAGE RSF CONFIRMATION")
        run_standard_rsf_confirmation(df, args.outdir)

    print("\nDone.")
    print(f"Outputs saved in: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
