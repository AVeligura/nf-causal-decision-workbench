from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold

from domain import EffectResult, IdentificationResult, IdentificationStatus


def _analysis_frame(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    frame = data.loc[(data["S"] == 1) & (data["V"] != "partial")].copy()
    return frame.dropna(subset=[outcome, "T"])


def _fit_aipw(
    data: pd.DataFrame,
    outcome: str,
    adjustment_set: tuple[str, ...],
    folds: int,
    seed: int,
    estimand: str,
) -> tuple[float, float, tuple[float, float], dict[str, Any], np.ndarray, np.ndarray]:
    frame = _analysis_frame(data, outcome)
    x = frame.loc[:, adjustment_set].to_numpy(dtype=float)
    t = frame["T"].to_numpy(dtype=int)
    y = frame[outcome].to_numpy(dtype=float)
    if len(np.unique(t)) < 2:
        raise ValueError("В аналитической выборке отсутствует один из режимов T")
    k = min(folds, int(np.bincount(t).min()))
    if k < 2:
        raise ValueError("Недостаточно наблюдений для cross-fitting")
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    p = np.empty(len(frame), dtype=float)
    m0 = np.empty(len(frame), dtype=float)
    m1 = np.empty(len(frame), dtype=float)
    for train, test in splitter.split(x, t):
        propensity = HistGradientBoostingClassifier(
            max_iter=80,
            max_depth=3,
            learning_rate=0.06,
            l2_regularization=1.0,
            random_state=seed,
        )
        propensity.fit(x[train], t[train])
        p[test] = propensity.predict_proba(x[test])[:, 1]
        for arm, target in ((0, m0), (1, m1)):
            arm_train = train[t[train] == arm]
            model = HistGradientBoostingRegressor(
                max_iter=45,
                max_depth=3,
                learning_rate=0.07,
                l2_regularization=0.5,
                random_state=seed + arm,
            )
            model.fit(x[arm_train], y[arm_train])
            target[test] = model.predict(x[test])
    p_raw = p.copy()
    # Conservative weight truncation protects AIPW from extrapolated nuisance
    # probabilities in heavy-tailed financial ratios. Raw values remain in the
    # diagnostics so overlap warnings are never hidden.
    p = np.clip(p, 0.10, 0.90)
    treated_share = float(t.mean())
    att_influence = t * (y - m0) / max(treated_share, 1e-8) - (
        (1 - t) * p * (y - m0) / ((1 - p) * max(treated_share, 1e-8))
    )
    att_estimate = float(np.mean(att_influence))
    att_se = float(np.std(att_influence, ddof=1) / np.sqrt(len(att_influence)))
    if estimand == "ATT":
        influence = att_influence
    else:
        influence = m1 - m0 + t * (y - m1) / p - (1 - t) * (y - m0) / (1 - p)
    estimate = float(np.mean(influence))
    se = float(np.std(influence, ddof=1) / np.sqrt(len(influence)))
    z = float(norm.ppf(0.975))
    interval = (estimate - z * se, estimate + z * se)
    diagnostics: dict[str, Any] = {
        "n_analysis": len(frame),
        "crossfit_folds": int(k),
        "propensity_min": float(p_raw.min()),
        "propensity_max": float(p_raw.max()),
        "propensity_estimated_extreme_share": float(
            np.mean((p_raw < 0.05) | (p_raw > 0.95))
        ),
        "propensity_weight_clipped_share": float(
            np.mean((p_raw < 0.10) | (p_raw > 0.90))
        ),
        "propensity_weight_bounds": (0.10, 0.90),
        "propensity_true_min": (
            float(frame["propensity_true"].min()) if "propensity_true" in frame else None
        ),
        "propensity_true_max": (
            float(frame["propensity_true"].max()) if "propensity_true" in frame else None
        ),
        "nuisance_outcome": "HistGradientBoostingRegressor",
        "nuisance_treatment": "HistGradientBoostingClassifier",
        "att_estimate": att_estimate,
        "att_standard_error": att_se,
        "att_interval": (
            att_estimate - float(norm.ppf(0.975)) * att_se,
            att_estimate + float(norm.ppf(0.975)) * att_se,
        ),
    }
    return estimate, se, interval, diagnostics, influence, x


def _cate_forest(
    influence: np.ndarray,
    x: np.ndarray,
    frame: pd.DataFrame,
    trees: int,
    seed: int,
    control_x: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    model = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=max(8, len(x) // 100),
        max_features=0.8,
        max_depth=10,
        bootstrap=True,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(x, influence)
    cate = model.predict(x)
    liquidity = frame["X4"].to_numpy() >= frame["X4"].median()
    debt = frame["X2"].to_numpy() >= frame["X2"].median()
    profiles: dict[str, float] = {}
    for label, mask in {
        "высокая L / низкий D": liquidity & ~debt,
        "высокая L / высокий D": liquidity & debt,
        "низкая L / низкий D": ~liquidity & ~debt,
        "низкая L / высокий D": ~liquidity & debt,
    }.items():
        profiles[label] = float(np.mean(cate[mask])) if mask.any() else float("nan")
    diagnostics = {
        "cate_model": "DR-learner with ExtraTrees regression on pseudo-outcomes",
        "cate_trees": trees,
        "cate_mean": float(cate.mean()),
        "cate_std": float(cate.std()),
        "cate_predictions": cate.tolist(),
        "cate_row_ids": (
            frame["row_id"].astype(int).tolist()
            if "row_id" in frame
            else frame.index.astype(int).tolist()
        ),
    }
    if control_x is not None:
        diagnostics["fixed_control_size"] = len(control_x)
        diagnostics["fixed_control_predictions"] = model.predict(control_x).tolist()
    return profiles, diagnostics


def _sequential_g_formula(
    data: pd.DataFrame,
    outcome: str,
    adjustment_set: tuple[str, ...],
) -> tuple[float, float, tuple[float, float], dict[str, Any]]:
    frame = _analysis_frame(data, outcome)
    x = frame.loc[:, adjustment_set].to_numpy(dtype=float)
    t = frame["T"].to_numpy(dtype=float)
    liquidity = frame["L"].to_numpy(dtype=float)
    d = frame["D"].to_numpy(dtype=float)
    y = frame[outcome].to_numpy(dtype=float)
    l_model = LinearRegression().fit(np.column_stack([x, t]), liquidity)
    d_model = LinearRegression().fit(np.column_stack([x, t, liquidity]), d)
    y_model = LinearRegression().fit(np.column_stack([x, liquidity, d]), y)
    predictions = []
    for arm in (0.0, 1.0):
        arm_vec = np.full(len(frame), arm)
        l_arm = l_model.predict(np.column_stack([x, arm_vec]))
        d_arm = d_model.predict(np.column_stack([x, arm_vec, l_arm]))
        y_arm = y_model.predict(np.column_stack([x, l_arm, d_arm]))
        predictions.append(y_arm)
    differences = predictions[1] - predictions[0]
    estimate = float(differences.mean())
    # Conservative residual-based interval for the sequential plug-in estimator.
    residual_scale = float(np.std(y - y_model.predict(np.column_stack([x, liquidity, d])), ddof=1))
    se = float(np.sqrt(np.var(differences, ddof=1) / len(frame) + residual_scale**2 / len(frame)))
    z = float(norm.ppf(0.975))
    return (
        estimate,
        se,
        (estimate - z * se, estimate + z * se),
        {
            "n_analysis": len(frame),
            "estimator": "sequential_g_formula",
        },
    )


def _partial_bounds(
    data: pd.DataFrame, outcome: str
) -> tuple[tuple[float, float], tuple[tuple[float, float], tuple[float, float]], dict[str, Any]]:
    complete = data.loc[data["S"] == 1, outcome].dropna().to_numpy(dtype=float)
    if not len(complete):
        raise ValueError("Нет наблюдаемых исходов для частичной идентификации")
    lower_y, upper_y = np.quantile(complete, [0.01, 0.99])
    observed_share = float(np.mean(data["S"] == 1))
    frame = _analysis_frame(data, outcome)
    treated_mean = float(frame.loc[frame["T"] == 1, outcome].mean())
    control_mean = float(frame.loc[frame["T"] == 0, outcome].mean())
    observed_effect = treated_mean - control_mean
    missing_penalty = (1.0 - observed_share) * (upper_y - lower_y)
    bounds = (observed_effect - missing_penalty, observed_effect + missing_penalty)
    n = max(len(frame), 2)
    se = float(np.std(complete, ddof=1) / np.sqrt(n))
    z = float(norm.ppf(0.975))
    bound_intervals = (
        (bounds[0] - z * se, bounds[0] + z * se),
        (bounds[1] - z * se, bounds[1] + z * se),
    )
    return (
        bounds,
        bound_intervals,
        {
            "observed_share": observed_share,
            "outcome_range_assumption": (float(lower_y), float(upper_y)),
        },
    )


def estimate_effect(
    data: pd.DataFrame,
    identification_result: IdentificationResult,
    *,
    outcome: str,
    estimand: str = "ATE",
    folds: int = 5,
    cate_trees: int = 300,
    seed: int = 20260814,
    compute_cate: bool = True,
) -> EffectResult:
    status = identification_result.status
    if status == IdentificationStatus.NOT_IDENTIFIED:
        return EffectResult(
            graph_id=identification_result.graph_id,
            query_id=identification_result.query_id,
            outcome=outcome,
            estimand=estimand,
            status=status,
            functional=identification_result.functional,
            adjustment_set=identification_result.adjustment_set,
            warnings=identification_result.warnings,
        )
    if status == IdentificationStatus.STRUCTURAL_ZERO:
        return EffectResult(
            graph_id=identification_result.graph_id,
            query_id=identification_result.query_id,
            outcome=outcome,
            estimand=estimand,
            status=status,
            estimate=0.0,
            interval=None,
            standard_error=None,
            functional=identification_result.functional,
            diagnostics={"origin": "structural", "estimated_from_sample": False},
        )
    if status == IdentificationStatus.PARTIALLY_IDENTIFIED:
        bounds, bound_intervals, diagnostics = _partial_bounds(data, outcome)
        return EffectResult(
            graph_id=identification_result.graph_id,
            query_id=identification_result.query_id,
            outcome=outcome,
            estimand=estimand,
            status=status,
            estimate=None,
            identified_bounds=bounds,
            bound_intervals=bound_intervals,
            functional=identification_result.functional,
            adjustment_set=identification_result.adjustment_set,
            diagnostics=diagnostics,
            warnings=identification_result.warnings,
        )

    if identification_result.graph_id == "G4":
        estimate, se, interval, diagnostics = _sequential_g_formula(
            data, outcome, identification_result.adjustment_set
        )
        profiles: dict[str, float] = {}
    else:
        estimate, se, interval, diagnostics, influence, x = _fit_aipw(
            data,
            outcome,
            identification_result.adjustment_set,
            folds,
            seed,
            estimand,
        )
        profiles = {}
        if compute_cate and estimand in {"ATE", "CATE"}:
            frame = _analysis_frame(data, outcome)
            control_frame = data.attrs.get("cate_control_x")
            control_x = (
                control_frame.loc[:, identification_result.adjustment_set].to_numpy(dtype=float)
                if isinstance(control_frame, pd.DataFrame)
                else None
            )
            profiles, cate_diag = _cate_forest(
                influence,
                x,
                frame,
                cate_trees,
                seed,
                control_x=control_x,
            )
            diagnostics.update(cate_diag)
    warnings = list(identification_result.warnings)
    if diagnostics.get("propensity_estimated_extreme_share", 0.0) >= 0.05:
        warnings.append(
            "Слабое overlap: оценка опирается на области ограниченной эмпирической поддержки"
        )
    return EffectResult(
        graph_id=identification_result.graph_id,
        query_id=identification_result.query_id,
        outcome=outcome,
        estimand=estimand,
        status=status,
        estimate=estimate,
        interval=interval,
        standard_error=se,
        functional=identification_result.functional,
        adjustment_set=identification_result.adjustment_set,
        cate_profiles=profiles,
        diagnostics=diagnostics,
        warnings=tuple(warnings),
    )
