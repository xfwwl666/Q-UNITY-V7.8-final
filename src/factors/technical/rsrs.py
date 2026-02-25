#!/usr/bin/env python3
"""
Q-UNITY-V6 RSRS 因子计算
  - 向量化 OLS（numpy lstsq，速度快10x）
  - NB-17: high/low 窗口均值归一化
  - NB-21: 新股保护（有效行数 >= rsrs_window * 2 才计算）
  - 返回 rsrs_raw / rsrs_zscore / rsrs_r2 / rsrs_adaptive
  - V7.7: 完全向量化 _rolling_ols，速度提升 50~100 倍
"""
from __future__ import annotations
import logging
from typing import Optional
import numpy as np
import pandas as pd

# 尝试使用 sliding_window_view (NumPy ≥1.20)
try:
    from numpy.lib.stride_tricks import sliding_window_view
    _HAS_SLIDING_WINDOW = True
except ImportError:
    from numpy.lib.stride_tricks import as_strided
    _HAS_SLIDING_WINDOW = False

logger = logging.getLogger(__name__)


def _rolling_ols_vectorized(high: np.ndarray, low: np.ndarray, window: int) -> tuple:
    """
    完全向量化滚动 OLS，一次性计算所有窗口的 beta、R²、残差标准差。

    参数:
        high: 一维数组，高点序列
        low:  一维数组，低点序列
        window: 回归窗口大小

    返回:
        (betas, r2s, resid_stds) — 均为 len(high) 大小，前 window-1 行为 NaN
    """
    n = len(high)
    out_len = n - window + 1
    if out_len <= 0:
        return (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))

    # 确保数组是 C 连续的
    high = np.ascontiguousarray(high)
    low  = np.ascontiguousarray(low)

    # 构建滑窗矩阵 (out_len, window)
    if _HAS_SLIDING_WINDOW:
        X = sliding_window_view(low, window)   # (out_len, window)
        Y = sliding_window_view(high, window)  # (out_len, window)
    else:
        # 手动 as_strided（注意必须连续）
        def make_windows(arr):
            shape   = (out_len, window)
            strides = (arr.strides[0], arr.strides[0])
            return as_strided(arr, shape=shape, strides=strides)
        X = make_windows(low)
        Y = make_windows(high)

    # 添加常数项列（1），形成设计矩阵 X_design = [1, low]
    ones = np.ones((out_len, window))
    X_design = np.stack([ones, X], axis=2)  # (out_len, window, 2)

    # 批量求解最小二乘：beta = (X'X)^(-1) X'Y
    # X'X 形状 (out_len, 2, 2)
    XtX = np.einsum('ijk,ijl->ikl', X_design, X_design)
    # X'Y 形状 (out_len, 2)
    XtY = np.einsum('ijk,ij->ik', X_design, Y)

    # 将 XtY 扩展为 (out_len, 2, 1) 以匹配 solve 的输入
    XtY = XtY[..., np.newaxis]  # (out_len, 2, 1)

    # 添加小正则化项防止奇异矩阵
    eps = 1e-10 * np.eye(2)
    XtX_reg = XtX + eps[None, :, :]  # (out_len, 2, 2)

    try:
        # 批量求解
        betas_all = np.linalg.solve(XtX_reg, XtY)  # (out_len, 2, 1)
        betas_all = betas_all.squeeze(-1)          # (out_len, 2)
        alpha = betas_all[:, 0]
        beta  = betas_all[:, 1]
    except np.linalg.LinAlgError:
        # 如果仍然奇异，回退到逐窗口 lstsq（极少发生）
        alpha = np.full(out_len, np.nan)
        beta  = np.full(out_len, np.nan)
        for i in range(out_len):
            try:
                coeffs = np.linalg.lstsq(X_design[i], Y[i], rcond=None)[0]
                alpha[i] = coeffs[0]
                beta[i]  = coeffs[1]
            except:
                pass

    # 计算拟合值 y_hat = alpha + beta * X
    y_hat = alpha[:, None] * ones + beta[:, None] * X

    # 残差
    resid = Y - y_hat
    ss_res = np.einsum('ij,ij->i', resid, resid)

    # 总平方和（减去均值）
    y_mean = Y.mean(axis=1)
    ss_tot = np.einsum('ij,ij->i', Y - y_mean[:, None], Y - y_mean[:, None])

    # R² = 1 - SS_res / SS_tot
    r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, 0.0)
    r2 = np.maximum(r2, 0.0)  # 确保非负

    # 残差标准差（使用有偏估计，与原代码一致）
    resid_std_win = np.sqrt(ss_res / window)

    # 填充到完整长度（前 window-1 行为 NaN）
    betas_out     = np.full(n, np.nan)
    r2s_out       = np.full(n, np.nan)
    resid_std_out = np.full(n, np.nan)

    betas_out[window-1:]     = beta
    r2s_out[window-1:]       = r2
    resid_std_out[window-1:] = resid_std_win

    return betas_out, r2s_out, resid_std_out


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

    high_arr = high_norm.values.astype(np.float64)
    low_arr  = low_norm.values.astype(np.float64)

    # NB-21: 有效行数保护
    if min_valid_rows is None:
        min_valid_rows = regression_window * 2
    n_valid = int(np.isfinite(high_arr).sum())
    if n_valid < min_valid_rows:
        logger.debug(f"有效行数 {n_valid} < {min_valid_rows}，跳过RSRS计算")
        for col in ["rsrs_raw", "rsrs_zscore", "rsrs_r2", "rsrs_adaptive", "resid_std"]:
            df[col] = np.nan
        return df

    # 处理缺失值（前向填充后再计算？原代码假设无缺失）
    # 为安全起见，用前向填充填充 NaN，但 RSRS 要求数据连续
    # high_arr = pd.Series(high_arr).fillna(method='ffill').fillna(1.0).values
    # low_arr  = pd.Series(low_arr).fillna(method='ffill').fillna(1.0).values
    # 使用 ffill() 替代 fillna(method='ffill')
    high_arr = pd.Series(high_arr).ffill().fillna(1.0).values
    low_arr = pd.Series(low_arr).ffill().fillna(1.0).values

    betas, r2s, rstd = _rolling_ols_vectorized(high_arr, low_arr, regression_window)

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