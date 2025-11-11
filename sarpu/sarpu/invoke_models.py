"""Utilities to run SAR-PU compatible models individually.

This module re-packages the bespoke experiment harness that the user provided
in order to make it easier to trigger SAR-PU experiments programmatically.
The helpers mirror the structure of the original script so existing
configuration dictionaries keep working, but they now focus solely on SAR-PU
pipelines while still producing the familiar figures and tables for each run.

The main entry points are :func:`run_sarpu_static` (single configuration) and
:func:`run_sarpu_windows` (multiple rolling/expanding passes).  Both return the
paths to the generated artefacts so callers can post-process the CSV outputs or
embed the saved figures directly in reports.
"""

from __future__ import annotations

import importlib
import logging
import math
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

log = logging.getLogger(__name__)


def _winsorize_train_apply(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    lo: float = 0.01,
    hi: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same winsorisation to both training and test matrices."""

    q_lo = np.quantile(X_train, lo, axis=0)
    q_hi = np.quantile(X_train, hi, axis=0)
    return np.clip(X_train, q_lo, q_hi), np.clip(X_test, q_lo, q_hi)


@dataclass(frozen=True)
class Config:
    """Runtime configuration for the SAR-PU style experiments."""

    data_csv: Path = Path("./data-thesis/_master-data-yearly.csv")
    results_dir: Path = Path("./results-thesis")
    sarpu_classifier: str = "logit"
    sarpu_penalty: str = "l2"
    sarpu_l1_ratio: Optional[float] = None
    sarpu_solver: str = "saga"
    rf_n_estimators: int = 500
    rf_max_depth: Optional[int] = None
    rf_max_features: Optional[Any] = "sqrt"
    rf_min_samples_leaf: int = 1
    rf_class_weight: Optional[Any] = "balanced"
    rf_n_jobs: int = -1
    rf_positive_weight: Optional[float] = None
    start_year: int = 2006
    end_year: int = 2023
    target_col: str = "is_breached"
    label_col: str = "is_labeled"
    id_candidates: Sequence[str] = ("permno", "ffyear")
    price_col: Optional[str] = None
    min_price: Optional[float] = None
    mktcap_col: Optional[str] = None
    min_mktcap: Optional[float] = None
    exchcd_col: Optional[str] = None
    exchcd_list: Optional[Sequence[int]] = None
    features_with_labels: Mapping[str, str] = field(default_factory=dict)
    winsor_clip: float = 0.005
    global_standardise: bool = False
    use_jiang_preproc_for_mine: bool = True
    winsor_lo: float = 0.01
    winsor_hi: float = 0.99
    poly_degree: int = 4
    poly_include_bias: bool = True
    window_type: str = "rolling"
    recency_weighting: str = "exp"
    half_life_years: float = 3.0
    min_row_weight: float = 0.25
    max_row_weight: float = 4.0
    random_state: int = 42
    carry_propensity: bool = True
    validation_fraction: float = 0.20
    validation_fraction_grid: Optional[Sequence[float]] = None
    random_search_iterations: Optional[int] = 100
    threshold_objective: str = "youden"
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_points: int = 19
    tnr_min: float = 0.5
    tpr_min: float = 0.80
    cost_fn: float = 5.0
    cost_fp: float = 1.0
    include_balanced_weight: bool = True
    class_weight_grid_log_scale: bool = False
    class_weight_minority_min: float = 1.0
    class_weight_minority_max: float = 50.0
    class_weight_minority_min_exp: float = -2.0
    class_weight_minority_max_exp: float = 2.0
    class_weight_grid_points: int = 20
    class_weight_override: Optional[Any] = None
    c_grid_points: int = 20
    c_grid_min_exp: float = -4.0
    c_grid_max_exp: float = 4.0
    c_grid_override: Optional[Sequence[float]] = None
    sarpu_cv_folds: int = 5
    sarpu_c_grid: Optional[Sequence[float]] = None
    sarpu_reuse_c: bool = True
    sarpu_reuse_neighbors: int = 1
    sarpu_max_its: int = 1500
    sarpu_convergence_window: int = 10
    sarpu_slope_eps: float = 5e-5
    sarpu_ll_eps: float = 5e-5
    sarpu_platt_calibration: bool = False
    aul_k: float = 0.10
    cv_model_selection: str = "ba"
    pp_rate_cap: float | None = None
    threshold_ema_alpha: float | None = None


def _load_dataset(cfg: Config, feature_cols_union: List[str]) -> tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(cfg.data_csv, low_memory=False)
    df = df.loc[df["ffyear"].between(cfg.start_year, cfg.end_year)].copy()

    if cfg.price_col and cfg.price_col in df.columns and cfg.min_price is not None:
        price = pd.to_numeric(df[cfg.price_col], errors="coerce").abs()
        before = len(df)
        df = df.loc[price > float(cfg.min_price)].copy()
        log.info(
            "Price filter: %s > %.2f kept %d/%d rows (%.1f%%)",
            cfg.price_col,
            float(cfg.min_price),
            len(df),
            before,
            100 * len(df) / before if before > 0 else 0,
        )

    if cfg.mktcap_col and cfg.mktcap_col in df.columns and cfg.min_mktcap is not None:
        cap = pd.to_numeric(df[cfg.mktcap_col], errors="coerce")
        before = len(df)
        df = df.loc[cap > float(cfg.min_mktcap)].copy()
        log.info(
            "MktCap filter: %s > %.0f kept %d/%d rows (%.1f%%)",
            cfg.mktcap_col,
            float(cfg.min_mktcap),
            len(df),
            before,
            100 * len(df) / before if before > 0 else 0,
        )

    if cfg.exchcd_col and cfg.exchcd_col in df.columns and cfg.exchcd_list:
        exch = pd.to_numeric(df[cfg.exchcd_col], errors="coerce")
        before = len(df)
        df = df.loc[exch.isin(list(cfg.exchcd_list))].copy()
        log.info(
            "Exchange filter: %s in %s kept %d/%d rows (%.1f%%)",
            cfg.exchcd_col,
            cfg.exchcd_list,
            len(df),
            before,
            100 * len(df) / before if before > 0 else 0,
        )

    tgt = cfg.target_col
    lbl = cfg.label_col
    df[tgt] = pd.to_numeric(df[tgt], errors="coerce").fillna(0).astype(int)
    if lbl not in df.columns:
        df[lbl] = (df[tgt] == 1).astype(int)
    else:
        df[lbl] = pd.to_numeric(df[lbl], errors="coerce").fillna(0).astype(int)

    for col in feature_cols_union:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    id_cols = [c for c in cfg.id_candidates if c in df.columns]
    if not id_cols:
        df["row_id"] = np.arange(len(df))
        id_cols = ["row_id"]

    return df, id_cols


def _build_yearly_blocks(
    df: pd.DataFrame,
    *,
    feature_cols: List[str],
    label_col: str,
    target_col: str,
    id_cols: List[str],
) -> Dict[int, dict]:
    blocks: Dict[int, dict] = {}
    for year, group in df.groupby("ffyear"):
        g2 = group.reset_index(drop=True)
        blocks[int(year)] = {
            "x": g2[feature_cols].to_numpy(dtype=float),
            "s": g2[label_col].to_numpy(dtype=int),
            "y": g2[target_col].to_numpy(dtype=int),
            "ids": g2[id_cols].copy(),
            "frame": g2,
        }
    return blocks


def _import_sarpu_with_reload():
    def _reload_and_return():
        import sarpu.PUmodels as _p
        import sarpu.pu_learning as _l

        importlib.invalidate_caches()
        importlib.reload(_p)
        importlib.reload(_l)

        from sarpu.PUmodels import LogisticRegressionPU, RandomForestPU
        from sarpu.pu_learning import pu_learn_sar_em

        src = Path(_p.__file__).resolve().parent
        print(f"sarpu loaded from: {src}")
        return LogisticRegressionPU, RandomForestPU, pu_learn_sar_em, True

    try:
        return _reload_and_return()
    except Exception:
        pass

    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd()

    candidates = [
        (here / ".." / "bekker-sarpu" / "sarpu").resolve(),
        (here / ".." / "bekker-sarpu").resolve(),
        (here / ".." / "bekker-sarpu" / "sarpu" / "sarpu").resolve(),
    ]

    for path in candidates:
        if path.exists():
            sp = str(path)
            if sp not in sys.path:
                sys.path.insert(0, sp)

    try:
        return _reload_and_return()
    except Exception as exc:
        print(f"sarpu import failed: {exc}")
        return None, None, None, False


def _safe_auc(y_true, scores) -> float:
    try:
        return float(roc_auc_score(y_true, np.asarray(scores)))
    except Exception:
        return float("nan")


def _safe_ap(y_true, scores) -> float:
    try:
        return float(average_precision_score(y_true, np.asarray(scores)))
    except Exception:
        return float("nan")


def _lift_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    n = y.size
    if n == 0 or y.max() == y.min():
        r = np.linspace(1 / n if n else 0, 1.0, num=n if n else 1)
        return r, np.ones_like(r)
    order = np.argsort(-s)
    y_sorted = y[order]
    cum_pos = np.cumsum(y_sorted)
    r = np.arange(1, n + 1, dtype=float) / n
    prevalence = y.mean()
    precision = cum_pos / np.arange(1, n + 1, dtype=float)
    lift = np.where(prevalence > 0, precision / prevalence, np.ones_like(precision))
    return r, lift


def _aul_at_k(y_true: np.ndarray, scores: np.ndarray, k: float = 0.10) -> float:
    y = np.asarray(y_true, dtype=int).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    n = y.size
    if n == 0 or y.max() == y.min() or k <= 0:
        return float("nan")
    k = float(np.clip(k, 1.0 / n, 1.0))
    r, lift = _lift_curve(y, s)
    m = int(np.ceil(k * n))
    area = float(np.trapz(lift[:m], r[:m]))
    return area / k


def _compute_metrics(y_true, scores, y_pred, *, aul_k: float = 0.10) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    scores = np.asarray(scores).ravel()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    fnr = 1.0 - tpr
    out = {
        "f_roc_auc": _safe_auc(y_true, scores),
        "f_pr_auc": _safe_ap(y_true, scores),
        "f_precision": precision_score(y_true, y_pred, zero_division=0),
        "f_recall": recall_score(y_true, y_pred, zero_division=0),
        "f_f1": f1_score(y_true, y_pred, zero_division=0),
        "f_accuracy": accuracy_score(y_true, y_pred),
        "f_balanced_accuracy": 0.5 * (tpr + tnr),
        "f_gmean": math.sqrt(max(tpr, 0) * max(tnr, 0)),
        "tpr": tpr,
        "tnr": tnr,
        "fnr": fnr,
        "pp_rate": float(np.mean(y_pred)),
        "f_mse": mean_squared_error(y_true, scores),
        "f_mae": mean_absolute_error(y_true, scores),
        "f_aul": _aul_at_k(y_true, scores, k=float(aul_k)),
    }
    beta = 2.0
    p, r = out["f_precision"], out["f_recall"]
    out["f_fbeta"] = (1 + beta**2) * p * r / (beta**2 * p + r) if (p + r) > 0 else 0.0
    return out


def _sig_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def _unpaired_sig_test(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    t = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(t.pvalue)


def _paired_sig_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, str]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 3:
        return float("nan"), "insufficient"
    diffs = a - b
    if np.allclose(diffs, 0.0):
        return 1.0, "all_zero"
    try:
        w = stats.wilcoxon(diffs, zero_method="pratt", alternative="two-sided", correction=False, mode="auto")
        if np.isfinite(w.pvalue):
            return float(w.pvalue), "wilcoxon"
    except Exception:
        pass
    t = stats.ttest_rel(a, b, nan_policy="omit")
    return float(t.pvalue), "ttest_rel"


def _quantile_higher(scores: np.ndarray, q: float) -> float:
    scores = np.asarray(scores, dtype=float).ravel()
    q = float(np.clip(q, 0.0, 1.0))
    try:
        return float(np.quantile(scores, q, method="higher"))
    except TypeError:
        return float(np.quantile(scores, q, interpolation="higher"))


def _select_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    objective: str = "youden",
    tnr_min: float = 0.80,
    tpr_min: float = 0.60,
    grid: Sequence[float] = np.linspace(0.05, 0.95, 19),
    aul_k: float = 0.10,
) -> Tuple[float, float]:
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    if scores.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5, float("nan")
    if objective == "aul":
        q = float(np.clip(1.0 - float(aul_k), 0.0, 1.0))
        thr = _quantile_higher(scores, q)
        val = _aul_at_k(y_true, scores, k=float(aul_k))
        return float(thr), float(val)

    best_thr, best_val = 0.5, -np.inf
    best_infeasible: Optional[Tuple[float, float, float]] = None
    for thr in grid:
        y_pred = (scores >= float(thr)).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        if objective == "youden":
            val = tpr + tnr - 1.0
        elif objective == "balanced_accuracy":
            val = 0.5 * (tpr + tnr)
        elif objective == "gmean":
            val = math.sqrt(max(tpr, 0) * max(tnr, 0))
        elif objective == "f1":
            val = f1_score(y_true, y_pred, zero_division=0)
        elif objective == "tpr_at_tnr":
            if tnr >= tnr_min:
                val = tpr
            else:
                cand = (tnr, tpr, float(thr))
                if (
                    best_infeasible is None
                    or cand[0] > best_infeasible[0]
                    or (cand[0] == best_infeasible[0] and cand[1] > best_infeasible[1])
                ):
                    best_infeasible = cand
                val = -np.inf
        elif objective == "tnr_at_tpr":
            if tpr >= tpr_min:
                val = tnr
            else:
                cand = (tpr, tnr, float(thr))
                if (
                    best_infeasible is None
                    or cand[0] > best_infeasible[0]
                    or (cand[0] == best_infeasible[0] and cand[1] > best_infeasible[1])
                ):
                    best_infeasible = cand
                val = -np.inf
        else:
            val = 0.5 * (tpr + tnr)
        if val > best_val:
            best_val, best_thr = float(val), float(thr)

    if not np.isfinite(best_val) and best_infeasible is not None:
        best_thr = float(best_infeasible[2])
        best_val = float("nan")

    return float(best_thr), float(best_val)



def _run_sarpu_rolling(blocks: Dict[int, dict], feature_list: List[str], cfg: Config) -> Dict[str, Any]:
    (
        LogisticRegressionPU,
        RandomForestPU,
        pu_learn_sar_em,
        HAVE_SARPU,
    ) = _import_sarpu_with_reload()
    print("sarpu package found" if HAVE_SARPU else "sarpu package not found, using logistic fallback")

    penalty = str(getattr(cfg, "sarpu_penalty", "l2")).lower()
    l1_ratio = getattr(cfg, "sarpu_l1_ratio", None)
    solver = str(getattr(cfg, "sarpu_solver", "saga"))

    if penalty == "elasticnet" and l1_ratio is None:
        l1_ratio = 0.5
    if penalty == "elasticnet" and solver not in ["saga"]:
        solver = "saga"

    if not blocks:
        return {"metrics_df": pd.DataFrame(), "folds": []}

    any_year = next(iter(blocks))
    n_features = int(blocks[any_year]["x"].shape[1])
    if n_features < 1:
        raise ValueError("No features provided")

    raw_grid = list(getattr(cfg, "sarpu_c_grid", []) or []) or list(np.logspace(-3, 3, 7))
    base_C_grid = [float(c) for c in raw_grid]
    base_kfolds = max(2, int(getattr(cfg, "sarpu_cv_folds", 5)))
    aul_k = float(getattr(cfg, "aul_k", 0.10))
    obj = str(getattr(cfg, "threshold_objective", "balanced_accuracy")).lower()
    cv_choice = str(getattr(cfg, "cv_model_selection", "") or "").lower()
    if cv_choice in {"", "policy", "same"}:
        cv_choice = obj
    if cv_choice in {"aul", "pu_aul", "rank"}:
        cv_choice = obj
    if cv_choice == "ba":
        cv_choice = "balanced_accuracy"
    rng = int(getattr(cfg, "random_state", 42))
    thr_min = float(getattr(cfg, "threshold_min", 0.05))
    thr_max = float(getattr(cfg, "threshold_max", 0.95))
    thr_pts = int(getattr(cfg, "threshold_points", 19))
    tnr_min = float(getattr(cfg, "tnr_min", 0.80))
    tpr_min = float(getattr(cfg, "tpr_min", 0.60))
    win_type = str(getattr(cfg, "window_type", "rolling")).lower()
    reuse_c = bool(getattr(cfg, "sarpu_reuse_c", True))
    neighbour_span = max(0, int(getattr(cfg, "sarpu_reuse_neighbors", 1)))
    max_its = int(getattr(cfg, "sarpu_max_its", 1500))
    conv_win = int(getattr(cfg, "sarpu_convergence_window", 10))
    slope_eps = float(getattr(cfg, "sarpu_slope_eps", 5e-5))
    ll_eps = float(getattr(cfg, "sarpu_ll_eps", 5e-5))
    max_its_cv = min(max_its, 300)
    eps_cv = max(ll_eps, 2e-4)
    use_platt = bool(getattr(cfg, "sarpu_platt_calibration", False))
    clf_choice = str(getattr(cfg, "sarpu_classifier", "logit")).lower()
    val_frac = float(getattr(cfg, "validation_fraction", 0.20))
    pp_rate_cap = getattr(cfg, "pp_rate_cap", None)
    thr_ema_alpha = getattr(cfg, "threshold_ema_alpha", None)
    prev_thr_for_ema: Optional[float] = None

    use_parallel_cv = bool(getattr(cfg, "sarpu_parallel_cv", True))
    parallel_n_jobs = int(getattr(cfg, "sarpu_parallel_n_jobs", -1))

    def _clf_model(C: float):
        if HAVE_SARPU and clf_choice == "rf":
            leaf = int(max(1, round(C)))
            return RandomForestPU(
                n_estimators=int(getattr(cfg, "rf_n_estimators", 500)),
                max_depth=getattr(cfg, "rf_max_depth", None),
                max_features=getattr(cfg, "rf_max_features", "sqrt"),
                min_samples_leaf=leaf,
                class_weight=getattr(cfg, "rf_class_weight", "balanced"),
                n_jobs=int(getattr(cfg, "rf_n_jobs", -1)),
                random_state=int(getattr(cfg, "random_state", 42)),
                positive_weight=getattr(cfg, "rf_positive_weight", None),
            )
        if HAVE_SARPU and clf_choice == "et":
            from sarpu.PUmodels import ExtraTreesPU

            return ExtraTreesPU(
                n_estimators=int(getattr(cfg, "rf_n_estimators", 500)),
                max_depth=getattr(cfg, "rf_max_depth", None),
                max_features=getattr(cfg, "rf_max_features", "sqrt"),
                min_samples_leaf=int(max(1, round(C))),
                class_weight=getattr(cfg, "rf_class_weight", "balanced"),
                n_jobs=int(getattr(cfg, "rf_n_jobs", -1)),
                random_state=int(getattr(cfg, "random_state", 42)),
                positive_weight=getattr(cfg, "rf_positive_weight", None),
            )
        if HAVE_SARPU:
            from sarpu.PUmodels import LogisticRegressionPU

            return LogisticRegressionPU(
                C=C,
                penalty=penalty,
                solver=solver,
                max_iter=5000,
                n_jobs=-1,
                warm_start=True,
                l1_ratio=l1_ratio,
            )
        return LogisticRegression(
            C=C,
            penalty=penalty if penalty != "elasticnet" else "l2",
            solver=solver if penalty == "elasticnet" else "liblinear",
            max_iter=5000,
            n_jobs=-1,
            class_weight="balanced",
            l1_ratio=l1_ratio if penalty == "elasticnet" else None,
        )

    def _prop_model(C: float):
        if HAVE_SARPU:
            from sarpu.PUmodels import LogisticRegressionPU

            return LogisticRegressionPU(
                C=C,
                penalty=penalty,
                solver=solver,
                max_iter=5000,
                n_jobs=-1,
                warm_start=True,
                l1_ratio=l1_ratio,
            )
        return LogisticRegression(
            C=C,
            penalty=penalty if penalty != "elasticnet" else "l2",
            solver=solver if penalty == "elasticnet" else "liblinear",
            max_iter=5000,
            n_jobs=-1,
            class_weight="balanced",
            l1_ratio=l1_ratio if penalty == "elasticnet" else None,
        )

    all_years = sorted(blocks.keys())
    years = [y for y in all_years if (y >= cfg.start_year) and (y < cfg.end_year) and ((y + 1) in blocks)]
    if not years:
        log.warning("[SAR-PU] No train→test pairs inside [%s, %s).", cfg.start_year, cfg.end_year)
        return {"metrics_df": pd.DataFrame(), "folds": []}
    metrics_rows: List[Dict[str, Any]] = []
    folds: List[Dict[str, Any]] = []
    prev_best_C: Optional[float] = None

    def _adaptive_grid(base_grid, prev_c):
        if not reuse_c or prev_c is None or not base_grid:
            return list(base_grid)
        sorted_grid = sorted(set(float(c) for c in base_grid if c > 0))
        if not sorted_grid:
            return [float(prev_c)]
        log_prev = math.log(prev_c)
        idx = min(range(len(sorted_grid)), key=lambda i: abs(math.log(sorted_grid[i]) - log_prev))
        chosen = {sorted_grid[idx], float(prev_c)}
        for offset in range(1, neighbour_span + 1):
            if idx - offset >= 0:
                chosen.add(sorted_grid[idx - offset])
            if idx + offset < len(sorted_grid):
                chosen.add(sorted_grid[idx + offset])
        return sorted(chosen)

    def _score_at_policy(scores_val: np.ndarray, y_val: np.ndarray, objective: str) -> float:
        thr_cv, _ = _select_threshold(
            scores_val,
            y_val,
            objective=objective,
            aul_k=aul_k,
            tnr_min=tnr_min,
            tpr_min=tpr_min,
            grid=np.linspace(thr_min, thr_max, thr_pts),
        )
        yhat = (scores_val >= thr_cv).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, yhat, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        return 0.5 * (tpr + tnr)

    def _cv_score_from(scores_val, y_val, s_val):
        if cv_choice in {"balanced_accuracy", "ba"}:
            return _score_at_policy(scores_val, y_val, "balanced_accuracy")
        elif cv_choice in {"tpr_at_tnr", "tnr_at_tpr", "f1"}:
            return _score_at_policy(scores_val, y_val, cv_choice)
        elif cv_choice == "logloss":
            if np.unique(y_val).size < 2:
                return float("nan")
            eps = 1e-6
            p = np.clip(np.asarray(scores_val).ravel(), eps, 1 - eps)
            return -float(log_loss(y_val, p))
        else:
            return _score_at_policy(scores_val, y_val, "balanced_accuracy")

    def _eval_single_fold(X_tr, S_tr, X_va, Y_va, Ytr_fold, C):
        if int(S_tr.sum()) == 0:
            return float("nan")
        if np.unique(Y_va).size < 2:
            return float("nan")

        try:
            if HAVE_SARPU:
                fitted_cv, _, _ = pu_learn_sar_em(
                    X_tr,
                    S_tr,
                    list(range(X_tr.shape[1])),
                    classification_model=_clf_model(C),
                    propensity_model=_prop_model(C),
                    max_its=max_its_cv,
                    slope_eps=eps_cv,
                    ll_eps=eps_cv,
                    convergence_window=conv_win,
                    refit_classifier=True,
                )
                scores_va = np.asarray(fitted_cv.predict_proba(X_va)).ravel()
                scores_tr_fold = (
                    np.asarray(fitted_cv.predict_proba(X_tr)).ravel() if use_platt else None
                )
            else:
                model = _prop_model(C)
                model.fit(X_tr, S_tr)
                scores_va = model.predict_proba(X_va)[:, 1]
                scores_tr_fold = model.predict_proba(X_tr)[:, 1] if use_platt else None

            if use_platt and scores_tr_fold is not None and np.unique(Ytr_fold).size >= 2:
                eps = 1e-3
                p_tr = np.clip(scores_tr_fold, eps, 1 - eps)
                z_tr = np.log(p_tr / (1.0 - p_tr)).reshape(-1, 1)
                platt = LogisticRegression(penalty="none", solver="lbfgs", max_iter=200)
                platt.fit(z_tr, np.asarray(Ytr_fold).ravel())

                p_va = np.clip(scores_va, eps, 1 - eps)
                z_va = np.log(p_va / (1.0 - p_va)).reshape(-1, 1)
                scores_va = platt.predict_proba(z_va)[:, 1]

            return _cv_score_from(scores_va, Y_va, S_tr)
        except Exception as exc:
            log.warning("Fold evaluation failed for C=%s: %s", C, str(exc))
            return float("nan")

    for year in years:
        train_years = [y for y in all_years if y <= year] if win_type == "expanding" else [year]
        if win_type == "rolling":
            Xtr = blocks[year]["x"]
            Str = blocks[year]["s"]
            Ytr = blocks[year]["y"]
        else:
            Xtr = np.vstack([blocks[y]["x"] for y in train_years])
            Str = np.concatenate([blocks[y]["s"] for y in train_years])
            Ytr = np.concatenate([blocks[y]["y"] for y in train_years])
        Xte = blocks[year + 1]["x"]
        Yte = blocks[year + 1]["y"]
        Xtr = Xtr.astype(np.float32, copy=False)
        Xte = Xte.astype(np.float32, copy=False)
        lo = float(getattr(cfg, "winsor_lo", 0.01))
        hi = float(getattr(cfg, "winsor_hi", 0.99))
        if 0.0 <= lo < hi <= 1.0:
            Xtr, Xte = _winsorize_train_apply(Xtr, Xte, lo=lo, hi=hi)
        scaler = StandardScaler(copy=False).fit(Xtr)
        XtrT = scaler.transform(Xtr).astype(np.float32, copy=False)
        XteT = scaler.transform(Xte).astype(np.float32, copy=False)
        pos_lab, neg_lab = int(Str.sum()), int((Str == 0).sum())
        feasible_splits = max(0, min(base_kfolds, pos_lab, neg_lab))
        candidate_grid = _adaptive_grid(base_C_grid, prev_best_C)
        if HAVE_SARPU and clf_choice in {"rf", "et"}:
            candidate_grid = sorted(set(int(max(1, round(c))) for c in candidate_grid))
        use_cv = feasible_splits >= 2 and len(candidate_grid) > 1
        best_C, best_cv = candidate_grid[len(candidate_grid) // 2], float("nan")

        if use_cv:
            skf = StratifiedKFold(n_splits=feasible_splits, shuffle=True, random_state=rng)
            cv_scores_per_C: List[Tuple[float, float]] = []

            for C in candidate_grid:
                if use_parallel_cv:
                    fold_vals = Parallel(n_jobs=parallel_n_jobs, verbose=0)(
                        delayed(_eval_single_fold)(
                            XtrT[tr_idx],
                            Str[tr_idx],
                            XtrT[va_idx],
                            Ytr[va_idx],
                            Ytr[tr_idx],
                            C,
                        )
                        for tr_idx, va_idx in skf.split(XtrT, Str)
                    )
                else:
                    fold_vals = []
                    for tr_idx, va_idx in skf.split(XtrT, Str):
                        val = _eval_single_fold(
                            XtrT[tr_idx],
                            Str[tr_idx],
                            XtrT[va_idx],
                            Ytr[va_idx],
                            Ytr[tr_idx],
                            C,
                        )
                        fold_vals.append(float(val))

                cv_scores_per_C.append((C, float(np.nanmean(fold_vals)) if fold_vals else float("nan")))

            if cv_scores_per_C and any(np.isfinite(s) for _, s in cv_scores_per_C):
                def _key(item):
                    Cval, score = item
                    finite = 1 if np.isfinite(score) else 0
                    sval = float(score) if finite else -float("inf")
                    tie = -abs(math.log10(Cval if Cval > 0 else 1e-12))
                    return (finite, sval, tie)

                best_C, best_cv = max(cv_scores_per_C, key=_key)

        if HAVE_SARPU:
            fitted, _, _ = pu_learn_sar_em(
                XtrT,
                Str,
                list(range(XtrT.shape[1])),
                classification_model=_clf_model(best_C),
                propensity_model=_prop_model(best_C),
                max_its=max_its,
                slope_eps=slope_eps,
                ll_eps=ll_eps,
                convergence_window=conv_win,
                refit_classifier=True,
            )
            scores_te = np.asarray(fitted.predict_proba(XteT)).ravel()
            scores_tr = np.asarray(fitted.predict_proba(XtrT)).ravel()
        else:
            base = _prop_model(best_C)
            base.fit(XtrT, Str)
            scores_te = base.predict_proba(XteT)[:, 1]
            scores_tr = base.predict_proba(XtrT)[:, 1]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=rng)
        (tr_idx, va_idx), = list(sss.split(np.zeros_like(Ytr), Ytr))
        scores_va, Y_va = scores_tr[va_idx], Ytr[va_idx]

        if use_platt and np.unique(Ytr[tr_idx]).size >= 2:
            eps = 1e-3
            p_tr = np.clip(scores_tr[tr_idx], eps, 1 - eps)
            z_tr = np.log(p_tr / (1.0 - p_tr)).reshape(-1, 1)
            platt = LogisticRegression(penalty="none", solver="lbfgs", max_iter=200)
            platt.fit(z_tr, np.asarray(Ytr[tr_idx]).ravel())
            p_va = np.clip(scores_va, eps, 1 - eps)
            z_va = np.log(p_va / (1.0 - p_va)).reshape(-1, 1)
            scores_va = platt.predict_proba(z_va)[:, 1]
            p_te = np.clip(scores_te, eps, 1 - eps)
            z_te = np.log(p_te / (1.0 - p_te)).reshape(-1, 1)
            scores_te = platt.predict_proba(z_te)[:, 1]

        thr, _ = _select_threshold(
            scores_va,
            Y_va,
            objective=obj,
            aul_k=aul_k,
            tnr_min=tnr_min,
            tpr_min=tpr_min,
            grid=np.linspace(thr_min, thr_max, thr_pts),
        )
        thr_final = float(thr)
        if (
            thr_ema_alpha is not None
            and np.isfinite(thr_ema_alpha)
            and 0.0 < float(thr_ema_alpha) < 1.0
        ):
            if prev_thr_for_ema is not None and np.isfinite(prev_thr_for_ema):
                thr_final = float(thr_ema_alpha) * float(thr) + (1.0 - float(thr_ema_alpha)) * float(prev_thr_for_ema)
            prev_thr_for_ema = float(thr_final)
        if pp_rate_cap is not None and np.isfinite(pp_rate_cap) and 0.0 < float(pp_rate_cap) < 1.0:
            yhat_tmp = (scores_te >= thr_final).astype(int)
            pp_tmp = float(np.mean(yhat_tmp))
            if pp_tmp > float(pp_rate_cap):
                q = float(np.clip(1.0 - float(pp_rate_cap), 0.0, 1.0))
                thr_cap = _quantile_higher(np.asarray(scores_te), q)
                thr_final = max(thr_final, thr_cap)
        yhat = (scores_te >= thr_final).astype(int)
        met = _compute_metrics(Yte, scores_te, yhat, aul_k=aul_k)
        met.update(
            {
                "train_year": year,
                "test_year": year + 1,
                "threshold": float(thr_final),
                "best_cv": float(best_cv),
                "best_C": float(best_C),
                "window_type": win_type,
                "train_years_n": len(train_years),
                "train_years_start": int(train_years[0]),
                "train_years_end": int(train_years[-1]),
                "classifier": clf_choice,
            }
        )
        metrics_rows.append(met)
        folds.append(
            {
                "train_year": year,
                "test_year": year + 1,
                "train_years": train_years.copy(),
                "scores": scores_te,
                "threshold": float(thr_final),
                "y_true": Yte,
                "ids": blocks[year + 1]["ids"].copy(),
            }
        )
        prev_best_C = float(best_C)
        log.info(
            "[SAR-PU%s/%s][%s %s–%s] %d→%d | CV=%.3f | C=%s | OOS_BA=%.4f | AUL=%.3f | thr=%.3f",
            "" if HAVE_SARPU else "-fallback",
            clf_choice,
            win_type,
            train_years[0],
            train_years[-1],
            year,
            year + 1,
            best_cv,
            ("%g" % best_C),
            met["f_balanced_accuracy"],
            met["f_aul"],
            thr_final,
        )
    return {"metrics_df": pd.DataFrame(metrics_rows).sort_values("test_year"), "folds": folds}


def export_pu_scores_from_folds(folds, *, out_csv: Path) -> Path:
    parts = []
    for fold in folds:
        ids = fold.get("ids")
        if ids is None or ids.empty:
            continue
        df = ids.copy()
        if "permno" not in df.columns or "ffyear" not in df.columns:
            raise ValueError("ids must include 'permno' and 'ffyear' columns")
        df["pu_score"] = np.asarray(fold["scores"], dtype=float).ravel()
        df["test_year"] = int(fold["test_year"])
        df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
        df["ffyear"] = pd.to_numeric(df["ffyear"], errors="coerce").astype("Int64")
        parts.append(df[["permno", "ffyear", "test_year", "pu_score"]])
    if not parts:
        raise ValueError("No folds with ids were found")
    panel = pd.concat(parts, axis=0, ignore_index=True)
    panel = panel.sort_values(["permno", "ffyear", "test_year"]).drop_duplicates(
        ["permno", "ffyear", "test_year"], keep="last"
    )
    panel = panel.sort_values(["permno", "ffyear", "test_year"]).drop_duplicates(
        ["permno", "ffyear"], keep="last"
    )[["permno", "ffyear", "pu_score"]]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_csv, index=False)
    print(f"Wrote PU scores to: {out_csv.resolve()}")
    return out_csv


def ensure_directory(path: Path | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _write_scores_csv(
    folds: List[Dict[str, Any]],
    out_dir: Path | str,
    *,
    score_col: str = "pu_score",
    id_keep: Sequence[str] = ("permno", "ffyear"),
    logger: Optional[logging.Logger] = None,
) -> Path:
    lg = logger or logging.getLogger(__name__)
    out_dir = Path(out_dir)
    ensure_directory(out_dir)
    rows: List[pd.DataFrame] = []
    for fold in folds:
        ids_df: Optional[pd.DataFrame] = fold.get("ids", None)
        if ids_df is None or ids_df.empty:
            n = len(fold.get("scores", []))
            tmp = pd.DataFrame({"ffyear": int(fold["test_year"])}, index=range(n))
        else:
            tmp = ids_df.copy()
            if "ffyear" not in tmp.columns:
                tmp["ffyear"] = int(fold["test_year"])
        tmp = tmp.assign(**{score_col: np.asarray(fold["scores"]).ravel().astype(float)})
        keep = [c for c in id_keep if c in tmp.columns]
        cols = keep + [score_col]
        rows.append(tmp[cols])
    df_all = (
        pd.concat(rows, axis=0, ignore_index=True)
        if rows
        else pd.DataFrame(columns=list(id_keep) + [score_col])
    )
    path = out_dir / "pu_scores.csv"
    df_all.to_csv(path, index=False)
    lg.info("Wrote SAR-PU scores to %s", path)
    return path


def _write_metrics_csv(
    df: pd.DataFrame,
    out_dir: Path | str,
    *,
    logger: Optional[logging.Logger] = None,
) -> Path:
    lg = logger or logging.getLogger(__name__)
    out_dir = Path(out_dir)
    ensure_directory(out_dir)
    path = out_dir / "metrics.csv"
    df.to_csv(path, index=False)
    lg.info("Wrote metrics to %s", path)
    return path


def _write_yearly_confusion(
    folds: List[Dict[str, Any]],
    out_dir: Path | str,
    *,
    logger: Optional[logging.Logger] = None,
) -> Path:
    lg = logger or logging.getLogger(__name__)
    out_dir = Path(out_dir)
    rows: List[Dict[str, Any]] = []
    for fold in folds:
        y_true = np.asarray(fold["y_true"]).ravel()
        if "y_pred" in fold and fold["y_pred"] is not None:
            y_pred = np.asarray(fold["y_pred"]).ravel()
        else:
            thr = float(fold.get("threshold", 0.5))
            scores = np.asarray(fold["scores"]).ravel()
            y_pred = (scores >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        rows.append(
            {
                "test_year": int(fold["test_year"]),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    df = pd.DataFrame(rows).sort_values("test_year")
    ensure_directory(out_dir)
    path = Path(out_dir) / "yearly_confusion.csv"
    df.to_csv(path, index=False)
    lg.info("Wrote yearly confusion to %s", path)
    return path


def save_figure3_over_time(
    my_df: pd.DataFrame,
    jiang_ridge_df: pd.DataFrame,
    jiang_logit_df: pd.DataFrame,
    out_path: Path | str,
    *,
    logger: Optional[logging.Logger] = None,
    single_model_label: Optional[str] = None,
    cols: int = 2,
) -> None:
    lg = logger or logging.getLogger(__name__)
    ensure_directory(Path(out_path).parent)
    lg.info("Saving Figure 3 to %s", out_path)

    metric_cols = [
        ("f_pr_auc", "PR AUC"),
        ("f_aul", "AUL@10%"),
        ("f_recall", "Recall / TPR"),
        ("fnr", "FNR"),
        ("f_precision", "Precision"),
        ("f_f1", "F1 score"),
        ("f_balanced_accuracy", "Balanced accuracy"),
        ("f_gmean", "G-mean"),
        ("f_roc_auc", "ROC AUC"),
    ]

    def _has(df, col):
        return (df is not None) and (not df.empty) and (col in df.columns)

    selected = [
        (c, t)
        for (c, t) in metric_cols
        if _has(my_df, c) or _has(jiang_ridge_df, c) or _has(jiang_logit_df, c)
    ]
    if not any(c in dict(selected) for c in ("f_recall", "tpr")):
        selected = [(c, t) for (c, t) in selected if c not in ("f_recall", "tpr")]
    if not selected:
        lg.warning("No metric columns found to plot.")
        return

    n = len(selected)
    rows = math.ceil(n / cols)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.2), sharex=True)
    axes = axes.ravel() if isinstance(axes, np.ndarray) else np.array([axes])

    def years(df):
        return [] if df is None or df.empty else list(df["test_year"])

    all_years = sorted(set(years(my_df)) | set(years(jiang_ridge_df)) | set(years(jiang_logit_df)))
    label_me = single_model_label or "SAR-PU"

    style_map = {
        label_me: dict(linestyle="-", marker="o"),
        "Jiang ridge": dict(linestyle="--", marker="s"),
        "Jiang logistic": dict(linestyle=":", marker="^"),
    }

    def draw(df, col, label, ax):
        if df is not None and not df.empty and col in df.columns:
            st = style_map.get(label, dict(linestyle="-", marker="o"))
            ax.plot(
                df["test_year"],
                df[col],
                linestyle=st["linestyle"],
                marker=st["marker"],
                linewidth=1.6,
                label=label,
            )

    for ax, (col, title) in zip(axes, selected):
        actual_col = (
            "tpr"
            if (col == "f_recall" and not _has(my_df, col) and not _has(jiang_ridge_df, col) and not _has(jiang_logit_df, col))
            else col
        )
        draw(my_df, actual_col, label_me, ax)
        draw(jiang_ridge_df, actual_col, "Jiang ridge", ax)
        draw(jiang_logit_df, actual_col, "Jiang logistic", ax)

        if all_years:
            ax.set_xlim(min(all_years), max(all_years))
            ax.set_xticks(all_years)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    for j in range(len(selected), len(axes)):
        axes[j].set_visible(False)

    handles, labels = [], []
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, lab
            break

    nslots = len(axes)
    nused = len(selected)
    if handles and (nused < nslots):
        leg_ax = axes[-1]
        leg_ax.axis("off")
        leg_ax.legend(handles, labels, loc="center", fontsize=10, frameon=True, ncol=1)
    elif handles:
        fig.legend(handles, labels, loc="lower right", fontsize=10, frameon=True, bbox_to_anchor=(0.99, 0.01))

    fig.suptitle("Out-of-sample Performance Over Time")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    lg.info("Saved Figure 3.")


def save_table_means_2008_2018(
    my_df: pd.DataFrame,
    jiang_ridge_df: pd.DataFrame,
    jiang_logit_df: pd.DataFrame,
    out_path: Path | str,
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    lg = logger or logging.getLogger(__name__)
    ensure_directory(Path(out_path).parent)
    lg.info("Writing core metric Table 3 to %s", out_path)

    yrs = list(range(2008, 2019))
    m = (
        my_df.loc[my_df["test_year"].isin(yrs)].copy().sort_values("test_year")
        if not my_df.empty
        else pd.DataFrame()
    )
    r = (
        jiang_ridge_df.loc[jiang_ridge_df["test_year"].isin(yrs)].copy().sort_values("test_year")
        if not jiang_ridge_df.empty
        else pd.DataFrame()
    )
    l = (
        jiang_logit_df.loc[jiang_logit_df["test_year"].isin(yrs)].copy().sort_values("test_year")
        if not jiang_logit_df.empty
        else pd.DataFrame()
    )

    metrics = [
        ("f_pr_auc", "PR AUC"),
        ("f_aul", "AUL@10%"),
        ("f_recall", "Recall / TPR"),
        ("fnr", "FNR = 1−TPR"),
        ("f_precision", "Precision"),
        ("pp_rate", "Positive rate / workload"),
        ("f_f1", "F1 score"),
        ("f_gmean", "G-mean"),
        ("f_balanced_accuracy", "Balanced accuracy"),
    ]

    lines = [
        "| Metric | SAR-PU | Jiang ridge | Jiang logistic |",
        "|:--|--:|--:|--:|",
    ]

    def fnum(x):
        return "NA" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.3f}"

    for key, pretty in metrics:
        mv = float(m[key].mean()) if key in m else float("nan")
        rv = float(r[key].mean()) if key in r else float("nan")
        lv = float(l[key].mean()) if key in l else float("nan")
        lines.append(f"| {pretty} | {fnum(mv)} | {fnum(rv)} | {fnum(lv)} |")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    lg.info("Wrote streamlined Table 3 (core metrics only).")


def save_rankcorr_heatmaps(
    mine_scores_by_year: Dict[int, np.ndarray],
    jiang_scores_by_year: Dict[int, np.ndarray],
    frames_by_year: Dict[int, pd.DataFrame],
    mine_predictors: List[str],
    jiang_predictors: List[str],
    pretty_map: Mapping[str, str],
    out_dir: Path | str,
    *,
    logger=None,
) -> None:
    lg = logger or logging.getLogger(__name__)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lg.info("Saving rank-correlation heatmaps (signed, grayscale) to %s", out_dir)

    def _rankcorr_by_year(scores_by_year: Dict[int, np.ndarray], predictors: List[str]) -> pd.DataFrame:
        from scipy.stats import spearmanr

        rows = []
        years = sorted(set(scores_by_year.keys()).intersection(frames_by_year.keys()))
        for ty in years:
            dfy = frames_by_year[ty]
            scores = np.asarray(scores_by_year[ty], dtype=float)
            if len(dfy) != len(scores):
                continue
            for pred in predictors:
                if pred not in dfy.columns:
                    continue
                x = pd.to_numeric(dfy[pred], errors="coerce")
                if x.nunique(dropna=True) < 2:
                    continue
                r = spearmanr(scores, x, nan_policy="omit").correlation
                rows.append({"predictor": pred, "test_year": int(ty), "spearman": float(r)})
        return pd.DataFrame(rows)

    def _pivot_rankcorr(df_long: pd.DataFrame) -> pd.DataFrame:
        if df_long.empty:
            return pd.DataFrame()
        pv = df_long.pivot(index="predictor", columns="test_year", values="spearman")
        pv.index = [pretty_map.get(p, p) for p in pv.index]
        pv["Mean"] = pv.mean(axis=1, skipna=True)
        return pv.sort_index()

    def _grayscale_signed_heatmap(data: pd.DataFrame, title: str, path: Path):
        sns.set_style("whitegrid")

        if data.empty:
            fig, ax = plt.subplots(figsize=(9, 12))
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return

        years = [c for c in data.columns if c != "Mean"]
        mat = data[years].astype(float)

        if "Mean" in data.columns:
            means = data["Mean"]
            row_labels = [f"{idx} ({means.loc[idx]:+.3f})" for idx in mat.index]
        else:
            row_labels = list(mat.index)

        n_rows = mat.shape[0]
        width, height = 9.0, max(10, 0.40 * n_rows + 4.0)
        fig, ax = plt.subplots(figsize=(width, height))

        sns.heatmap(
            mat,
            cmap="Greys",
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            annot=True,
            fmt="+.2f",
            annot_kws={"fontsize": 8},
            linewidths=0.5,
            linecolor="white",
            cbar=True,
            cbar_kws={"orientation": "horizontal", "pad": 0.04, "fraction": 0.06, "aspect": 40},
            ax=ax,
        )

        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_label_position("bottom")
        ax.set_ylabel("Predictor (mean in parentheses)")
        ax.set_xlabel("Test year")
        ax.set_title(title, pad=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
        ax.set_yticklabels(row_labels, rotation=0)

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    mine_df_long = (
        _rankcorr_by_year(mine_scores_by_year, mine_predictors) if mine_scores_by_year else pd.DataFrame()
    )
    jiang_df_long = (
        _rankcorr_by_year(jiang_scores_by_year, jiang_predictors) if jiang_scores_by_year else pd.DataFrame()
    )

    pv_mine = _pivot_rankcorr(mine_df_long)
    pv_jg = _pivot_rankcorr(jiang_df_long)

    overlap = (
        sorted(set(pv_mine.index).intersection(pv_jg.index))
        if (not pv_mine.empty and not pv_jg.empty)
        else []
    )
    pv_diff = (
        pv_mine.loc[overlap].drop(columns=["Mean"], errors="ignore")
        - pv_jg.loc[overlap].drop(columns=["Mean"], errors="ignore")
    ) if overlap else pd.DataFrame()

    if not pv_mine.empty:
        _grayscale_signed_heatmap(pv_mine, "Rank Correlations (SAR-PU)", Path(out_dir) / "figure5_1_model.png")
    if not pv_jg.empty:
        _grayscale_signed_heatmap(pv_jg, "Rank correlations (Jiang ridge)", Path(out_dir) / "figure5_2_jiang.png")
    if not pv_diff.empty:
        _grayscale_signed_heatmap(
            pv_diff,
            "Difference in rank correlations (model − Jiang)",
            Path(out_dir) / "figure5_3_diff.png",
        )

    lg.info("Saved rank-correlation heatmaps (signed, grayscale).")


def _vals_over_years(df: pd.DataFrame, metric: str, years) -> np.ndarray:
    if df.empty or metric not in df.columns:
        return np.array([])
    x = df.loc[df["test_year"].isin(years), metric].astype(float)
    return x[np.isfinite(x)].to_numpy()


def save_table_pairwise_with_diffs(
    sarpu_df: pd.DataFrame,
    other_df: pd.DataFrame,
    other_label: str,
    out_path,
    *,
    years=tuple(range(2008, 2019)),
    logger=None,
) -> None:
    lg = logger or logging.getLogger(__name__)
    ensure_directory(Path(out_path).parent)
    lg.info("Writing pairwise table (unpaired Welch) to %s", out_path)

    metrics = [
        ("f_pr_auc", "PR AUC"),
        ("f_recall", "Recall/TPR"),
        ("fnr", "FNR"),
        ("f_precision", "Precision"),
        ("f_f1", "F1 score"),
        ("f_balanced_accuracy", "Balanced accuracy"),
        ("f_gmean", "G-mean"),
        ("f_roc_auc", "ROC AUC"),
    ]

    lines = [
        f"| Metric | SAR-PU | {other_label} | Δ | p-value |  |",
        "|:--|--:|--:|--:|--:|:--:|",
    ]

    def f(x):
        return "{:.3f}".format(x) if np.isfinite(x) else "NA"

    for key, pretty in metrics:
        pu_vals = _vals_over_years(sarpu_df, key, years)
        ot_vals = _vals_over_years(other_df, key, years)
        pu_mean = float(np.nanmean(pu_vals)) if pu_vals.size else float("nan")
        ot_mean = float(np.nanmean(ot_vals)) if ot_vals.size else float("nan")
        delta = pu_mean - ot_mean if np.isfinite(pu_mean) and np.isfinite(ot_mean) else float("nan")
        pval = _unpaired_sig_test(pu_vals, ot_vals)
        lines.append(f"| {pretty} | {f(pu_mean)} | {f(ot_mean)} | {f(delta)} | {f(pval)} | {_sig_stars(pval)} |")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def run_sarpu_static(
    *,
    features_with_labels: Mapping[str, str],
    random_search_iterations: int = 36,
    aul_k: float = 0.10,
    threshold_min: float = 0.01,
    threshold_max: float = 0.99,
    threshold_points: int = 25,
    c_grid_min_exp: float = -5.0,
    c_grid_max_exp: float = 1.0,
    c_grid_points: int = 10,
    class_weight_override: Optional[Any] = "balanced",
    include_balanced_weight: bool = True,
    window_type: str = "rolling",
    start_year: int = 2007,
    end_year: int = 2023,
    random_state: int = 42,
    validation_fraction: float = 0.20,
    threshold_objective: str = "aul",
    carry_propensity: bool = True,
    results_dir: str | Path = "./results-thesis/sarpu_static",
    cfg_overrides: Optional[Dict[str, Any]] = None,
    windows: Sequence[Tuple[str, int]] = (("main_2007_2018", 2018), ("extended_2007_2023", 2023)),
    debug_fast: bool = False,
    fast_pairs: int = 1,
    fast_cv_folds: int = 2,
    fast_c_grid: Optional[Sequence[float]] = (1e-2, 1e-1, 1.0),
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, Path]]:
    lg = logger or logging.getLogger(__name__)
    lg.info(
        "[run_sarpu_static] starting. years=%d–%d, window_type=%s, objective=%s",
        start_year,
        end_year,
        window_type,
        threshold_objective,
    )

    cfg_kwargs = dict(
        features_with_labels=features_with_labels,
        random_search_iterations=random_search_iterations,
        aul_k=aul_k,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_points=threshold_points,
        c_grid_min_exp=c_grid_min_exp,
        c_grid_max_exp=c_grid_max_exp,
        c_grid_points=c_grid_points,
        class_weight_override=class_weight_override,
        include_balanced_weight=include_balanced_weight,
        window_type=window_type,
        start_year=start_year,
        end_year=end_year,
        random_state=random_state,
        validation_fraction=validation_fraction,
        threshold_objective=threshold_objective,
        carry_propensity=carry_propensity,
        results_dir=str(results_dir),
    )

    if cfg_overrides:
        cfg_kwargs.update(cfg_overrides)

    if debug_fast:
        tiny_end = max(start_year + fast_pairs, start_year + 1)
        cfg_kwargs["start_year"] = start_year
        cfg_kwargs["end_year"] = min(end_year, tiny_end)
        if "sarpu_cv_folds" not in cfg_kwargs:
            cfg_kwargs["sarpu_cv_folds"] = int(fast_cv_folds)
        if "sarpu_c_grid" not in cfg_kwargs and fast_c_grid:
            cfg_kwargs["sarpu_c_grid"] = list(fast_c_grid)
        lg.info(
            "[run_sarpu_static] DEBUG-FAST enabled: years=%d–%d, folds=%s, C_grid=%s",
            cfg_kwargs["start_year"],
            cfg_kwargs["end_year"],
            cfg_kwargs.get("sarpu_cv_folds", "default"),
            cfg_kwargs.get("sarpu_c_grid", "default"),
        )

    cfg = Config(**cfg_kwargs)  # type: ignore[arg-type]

    feature_union = sorted(set(features_with_labels.keys()))
    if not feature_union:
        lg.warning("run_sarpu_static: empty features_with_labels; nothing to run.")
    df, id_cols = _load_dataset(cfg, feature_union)
    lg.info(
        "[run_sarpu_static] dataset loaded: rows=%d, years=[%s..%s]",
        len(df),
        df["ffyear"].min() if "ffyear" in df else "?",
        df["ffyear"].max() if "ffyear" in df else "?",
    )

    blocks = _build_yearly_blocks(
        df,
        feature_cols=list(features_with_labels.keys()),
        label_col=cfg.label_col,
        target_col=cfg.target_col,
        id_cols=id_cols,
    )
    years = sorted(blocks.keys())
    pairs = sum(1 for y in years if (y >= cfg.start_year) and (y < cfg.end_year) and ((y + 1) in blocks))
    lg.info(
        "[run_sarpu_static] built %d yearly blocks; available train→test pairs in window: %d",
        len(blocks),
        pairs,
    )
    if pairs == 0:
        lg.error(
            "[run_sarpu_static] no valid year pairs in [%d, %d). Aborting run.",
            cfg.start_year,
            cfg.end_year,
        )
        outputs: Dict[str, Dict[str, Path]] = {}
        for tag, _cap in windows:
            out_dir = Path(cfg.results_dir) / tag
            ensure_directory(out_dir)
            _write_metrics_csv(pd.DataFrame(), out_dir, logger=lg)
        return outputs

    lg.info("[run_sarpu_static] invoking _run_sarpu_rolling ...")
    sarpu_result = _run_sarpu_rolling(blocks, list(features_with_labels.keys()), cfg)
    lg.info(
        "[run_sarpu_static] _run_sarpu_rolling finished. metrics_rows=%s, folds=%s",
        (0 if sarpu_result.get("metrics_df") is None else len(sarpu_result["metrics_df"])),
        (0 if sarpu_result.get("folds") is None else len(sarpu_result["folds"])),
    )

    scores_csv = export_pu_scores_from_folds(
        sarpu_result["folds"],
        out_csv=Path(cfg.results_dir) / "josh_riskscores.csv",
    )
    lg.info("[run_sarpu_static] wrote consolidated scores: %s", scores_csv)

    outputs: Dict[str, Dict[str, Path]] = {}
    min_test_year = cfg.start_year + 1

    for tag, end_year_cap in windows:
        out_dir = Path(cfg.results_dir) / tag
        ensure_directory(out_dir)
        lg.info(
            "[run_sarpu_static] writing window '%s' (test years %d..%d)",
            tag,
            min_test_year,
            end_year_cap,
        )

        my_df = sarpu_result.get("metrics_df", pd.DataFrame()).copy()
        if not my_df.empty and "test_year" in my_df.columns:
            my_df = my_df[(my_df["test_year"] >= min_test_year) & (my_df["test_year"] <= end_year_cap)]
        else:
            my_df = pd.DataFrame()

        _write_metrics_csv(my_df, out_dir, logger=lg)

        folds = sarpu_result.get("folds", [])
        if not folds:
            lg.warning("[run_sarpu_static] no folds to write confusion for window '%s'", tag)
        _write_yearly_confusion(folds, out_dir, logger=lg)

        window_folds = [
            fd
            for fd in folds
            if "test_year" in fd and (min_test_year <= int(fd["test_year"]) <= end_year_cap)
        ]
        scores_csv_path = _write_scores_csv(
            window_folds,
            out_dir,
            score_col="pu_score",
            id_keep=("permno", "ffyear"),
            logger=lg,
        )
        outputs.setdefault(tag, {})["scores_csv"] = scores_csv_path

        if not my_df.empty:
            empty = pd.DataFrame()
            save_figure3_over_time(
                my_df,
                empty,
                empty,
                out_dir / "figure3.png",
                logger=lg,
                single_model_label="SAR-PU",
                cols=2,
            )
            save_table_means_2008_2018(my_df, empty, empty, out_dir / "table3.txt", logger=lg)
        else:
            lg.info(
                "[run_sarpu_static] no rows in metrics for '%s'; skipping Figure 3 and Table 3.",
                tag,
            )

        frames_by_year = {y: df.loc[df["ffyear"] == y].reset_index(drop=True) for y in sorted(df["ffyear"].unique())}
        mine_scores_by_year = {
            int(fd["test_year"]): fd["scores"]
            for fd in folds
            if "test_year" in fd and (min_test_year <= int(fd["test_year"]) <= end_year_cap)
        }
        pretty_map = dict(features_with_labels)
        pretty_map.setdefault("r_and_d", "R&D")
        pretty_map.setdefault("tobins_q", "Tobin's Q")

        if mine_scores_by_year:
            save_rankcorr_heatmaps(
                mine_scores_by_year,
                {},
                frames_by_year,
                list(features_with_labels.keys()),
                [],
                pretty_map,
                out_dir,
                logger=lg,
            )
        else:
            lg.info(
                "[run_sarpu_static] no fold scores within '%s'; skipping rank-corr heatmaps.",
                tag,
            )

        outputs.setdefault(tag, {})["sarpu_static"] = out_dir

    lg.info("[run_sarpu_static] done.")
    return outputs


def run_sarpu_windows(
    *,
    features_with_labels: Mapping[str, str],
    window_types: Sequence[str] = ("rolling", "expanding"),
    results_dir: str | Path = "./results-thesis/sarpu_windows",
    cfg_overrides: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs: Any,
) -> Dict[str, Dict[str, Dict[str, Path]]]:
    """Run SAR-PU experiments for multiple window strategies.

    Parameters
    ----------
    features_with_labels:
        Mapping of feature column name to a pretty label.
    window_types:
        Iterable containing the window_type values to evaluate (e.g. ``("rolling",
        "expanding")``).
    results_dir:
        Base directory under which per-window outputs will be stored.  Each
        window strategy receives its own sub-folder to avoid clobbering results.
    cfg_overrides:
        Optional dictionary merged into the :class:`Config` used for each
        invocation.
    logger:
        Optional logger for status updates.
    **kwargs:
        Additional keyword arguments forwarded to :func:`run_sarpu_static`.
    """

    lg = logger or logging.getLogger(__name__)
    outputs: Dict[str, Dict[str, Dict[str, Path]]] = {}
    base_overrides = dict(cfg_overrides or {})
    base_results = Path(results_dir)

    for win_type in window_types:
        lg.info("[run_sarpu_windows] executing window_type=%s", win_type)
        subdir = base_results / win_type
        overrides = dict(base_overrides)
        outputs[win_type] = run_sarpu_static(
            features_with_labels=features_with_labels,
            window_type=win_type,
            results_dir=subdir,
            cfg_overrides=overrides,
            logger=lg,
            **kwargs,
        )

    return outputs


__all__ = [
    "Config",
    "run_sarpu_static",
    "run_sarpu_windows",
    "export_pu_scores_from_folds",
    "save_figure3_over_time",
    "save_figure3_over_time_multi",
    "save_table_means_2008_2018",
    "save_rankcorr_heatmaps",
    "save_table_pairwise_with_diffs",
]

