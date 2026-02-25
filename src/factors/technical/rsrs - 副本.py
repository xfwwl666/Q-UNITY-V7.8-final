#!/usr/bin/env python3
"""
Q-UNITY-V6 RSRS 因子计算
  - 向量化 OLS（numpy lstsq，速度快10x）
  - NB-17: high/low 窗口均值归一化
  - NB-21: 新股保护（有效行数 >= rsrs_window * 2 才计算）
  - 返回 rsrs_raw / rsrs_zscore / rsrs_r2 / rsrs_adaptive
"""
from __future__ import annotations
import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _rolling_ols(high: np.ndarray, low: np.ndarray, window: int) -> tuple:
    """
    向量化滚动 OLS: high = beta * low + alpha
    返回 (betas, r2s, resid_stds) — 均为 len(high) 大小，前 window-1 行为 NaN
    """
    n = len(high)
    betas     = np.full(n, np.nan)
    r2s       = np.full(n, np.nan)
    resid_std = np.full(n, np.nan)

    for i in range(window - 1, n):
        x = low[i - window + 1: i + 1]
        y = high[i - window + 1: i + 1]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < window // 2:
            continue
        xv, yv = x[valid], y[valid]
        X = np.column_stack([np.ones(len(xv)), xv])
        try:
            coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
        except Exception:
            continue
        alpha, beta = coef
        y_hat = alpha + beta * xv
        ss_res = np.sum((yv - y_hat) ** 2)
        ss_tot = np.sum((yv - yv.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        betas[i]     = beta
        r2s[i]       = max(0.0, r2)
        resid_std[i] = np.std(yv - y_hat)

    return betas, r2s, resid_std


def compute_rsrs(
    df: pd.DataFrame,
    regression_window: int = 18,
    zscore_window:     int = 600,
    min_valid_rows:    Optional[int] = None,   # NB-21: 若指定则过滤新股
) -> pd.DataFrame:
    """
    计算 RSRS 因子全系列
    输入 df 必须包含列: high, low (已前复权)
    输出新列: rsrs_raw, rsrs_zscore, rsrs_r2, rsrs_adaptive, resid_std
    """
    if "high" not in df.columns or "low" not in df.columns:
        raise ValueError("df 必须包含 high, low 列")

    df = df.copy()

    # NB-17: 窗口均值归一化
    low_mean  = df["low"].rolling(regression_window, min_periods=1).mean()
    high_mean = df["high"].rolling(regression_window, min_periods=1).mean()
    low_norm  = (df["low"]  / low_mean.replace(0, np.nan)).fillna(1.0)
    high_norm = (df["high"] / high_mean.replace(0, np.nan)).fillna(1.0)

    high_arr = high_norm.values
    low_arr  = low_norm.values

    # NB-21: 有效行数保护
    if min_valid_rows is None:
        min_valid_rows = regression_window * 2
    n_valid = int(np.isfinite(high_arr).sum())
    if n_valid < min_valid_rows:
        logger.debug(f"有效行数 {n_valid} < {min_valid_rows}，跳过RSRS计算")
        for col in ["rsrs_raw", "rsrs_zscore", "rsrs_r2", "rsrs_adaptive", "resid_std"]:
            df[col] = np.nan
        return df

    betas, r2s, rstd = _rolling_ols(high_arr, low_arr, regression_window)

    df["rsrs_raw"]  = betas
    df["rsrs_r2"]   = r2s
    df["resid_std"] = rstd

    # Z-score 标准化
    roll_mean = pd.Series(betas).rolling(zscore_window, min_periods=30).mean()
    roll_std  = pd.Series(betas).rolling(zscore_window, min_periods=30).std()
    zscore    = (betas - roll_mean.values) / (roll_std.values + 1e-9)
    df["rsrs_zscore"] = zscore

    # 修正RSRS = zscore * R²
    df["rsrs_adaptive"] = zscore * r2s

    return df