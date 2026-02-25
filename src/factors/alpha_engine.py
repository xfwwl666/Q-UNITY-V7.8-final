#!/usr/bin/env python3
"""
Q-UNITY-V6 多因子 Alpha 引擎
集成:
  - RSRS: 阻力支撑位相对强度
  - 动量因子
  - 波动率因子
  - 质量因子（ROE/净利润TTM）
  - NB-09: get_latest_factor 增加 date_boundary_idx
  - NB-21 @@NB21-CLOSED-LOOP-PATCH-v2@@ 新股防御蒙猴补丁（文件末）
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .technical.rsrs import compute_rsrs

logger = logging.getLogger(__name__)


class AlphaEngine:
    """多因子 Alpha 引擎"""

    def __init__(
        self,
        rsrs_window:   int = 18,
        zscore_window: int = 600,
        mom_window:    int = 20,
        vol_window:    int = 20,
    ) -> None:
        self.rsrs_window   = rsrs_window
        self.zscore_window = zscore_window
        self.mom_window    = mom_window
        self.vol_window    = vol_window

    # ── 因子计算 ─────────────────────────────────────────────────────────

    def compute_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有 Alpha 因子
        输入: OHLCV DataFrame，index=datetime
        输出: 原 df + 因子列
        """
        df = df.copy()

        # RSRS 因子（含 NB-21 min_valid_rows 保护）
        if "high" in df.columns and "low" in df.columns:
            df = compute_rsrs(df, self.rsrs_window, self.zscore_window,
                              min_valid_rows=self.rsrs_window * 2)

        # 动量因子：N日累计收益
        if "close" in df.columns:
            close = df["close"]
            df["mom"] = close.pct_change(self.mom_window)

            # 波动率因子（低波动为优）
            df["volatility"] = close.pct_change().rolling(self.vol_window).std()
            df["vol_factor"] = -df["volatility"]   # 取负：低波动=高分

            # 换手率动量（需 volume）
            if "volume" in df.columns:
                df["turnover"] = df["volume"] / df["volume"].rolling(self.vol_window).mean()

        return df

    # ── NB-09: 获取截止某日的最新因子值 ──────────────────────────────────

    def get_latest_factor(
        self,
        df: pd.DataFrame,
        factor: str,
        date_boundary: Optional[pd.Timestamp] = None,
        date_boundary_idx: Optional[int] = None,   # NB-09: 可指定整数索引
    ) -> Optional[float]:
        """
        获取 df[factor] 在 date_boundary 之前的最新非 NaN 值
        NB-09: 支持 date_boundary_idx（整数行索引上界），防前视
        """
        if factor not in df.columns:
            return None

        series = df[factor].dropna()
        if series.empty:
            return None

        # 整数索引截断（优先）
        if date_boundary_idx is not None:
            series = series.iloc[:date_boundary_idx]
        elif date_boundary is not None:
            series = series.loc[series.index <= date_boundary]

        if series.empty:
            return None
        return float(series.iloc[-1])

    # ── 批量评分 ──────────────────────────────────────────────────────────

    def score_universe(
        self,
        factor_data: Dict[str, pd.DataFrame],
        eval_date: pd.Timestamp,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        对股票池打分（截面 Z-score + 加权合成）
        factor_data: {code: factor_df}
        weights: {factor_name: weight}  默认等权
        """
        if weights is None:
            weights = {"rsrs_adaptive": 0.4, "mom": 0.3, "vol_factor": 0.3}

        rows = {}
        for code, df in factor_data.items():
            row = {}
            for fn in weights:
                v = self.get_latest_factor(df, fn, eval_date)
                row[fn] = v if v is not None else np.nan
            rows[code] = row

        scores_df = pd.DataFrame(rows).T
        # 截面 Z-score
        for col in scores_df.columns:
            s = scores_df[col]
            std = s.std()
            if std > 1e-9:
                scores_df[col] = (s - s.mean()) / std
            else:
                scores_df[col] = 0.0

        # 加权合成
        total_w = sum(weights.values())
        composite = sum(
            scores_df.get(fn, pd.Series(0, index=scores_df.index)) * w
            for fn, w in weights.items()
        ) / (total_w if total_w > 0 else 1.0)

        return composite.sort_values(ascending=False)

    # ── 批量计算（类方法接口，供策略调用）────────────────────────────────

    @classmethod
    def compute_from_history(
        cls,
        history: pd.DataFrame,
        rsrs_window: int = 18,
        zscore_window: int = 600,
    ) -> pd.DataFrame:
        """
        从历史行情计算 RSRS 因子 DataFrame
        此方法会被 NB-21 Monkey-Patch 替换（见文件末）
        """
        engine = cls(rsrs_window=rsrs_window, zscore_window=zscore_window)
        return engine.compute_factors(history)


# ============================================================================
# @@NB21-CLOSED-LOOP-PATCH-v2@@
# NB-21 新股防御闭环修补
# 策略: valid_count >= rsrs_window * 2 才允许计算
#       否则全部因子列强制置 NaN，防止 RSRS 在上市首几天产生噪声信号
# ============================================================================

def _nb21_valid_mask(history: pd.DataFrame, rsrs_window: int) -> np.ndarray:
    """
    生成每行的有效布尔掩码:
    仅当到当前行为止已有 >= rsrs_window*2 行非 NaN 收盘价时才有效
    """
    if "close" not in history.columns:
        return np.zeros(len(history), dtype=bool)
    close_valid = history["close"].notna().values.astype(int)
    cumcount = np.cumsum(close_valid)
    return cumcount >= (rsrs_window * 2)


def _apply_nb21_mask_to_rsrs(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """将所有 rsrs_* 列在 mask=False 处强制置 NaN"""
    rsrs_cols = [c for c in df.columns if c.startswith("rsrs_") or c == "resid_std"]
    for col in rsrs_cols:
        df.loc[~mask, col] = np.nan
    return df


def _patched_compute_from_history(
    history: pd.DataFrame,
    rsrs_window: int = 18,
    zscore_window: int = 600,
) -> pd.DataFrame:
    """NB-21 闭环版: 先正常计算，再用有效掩码清洗新股噪声"""
    engine = AlphaEngine(rsrs_window=rsrs_window, zscore_window=zscore_window)
    df = engine.compute_factors(history)
    mask = _nb21_valid_mask(history, rsrs_window)
    df = _apply_nb21_mask_to_rsrs(df, mask)
    return df


# 执行 Monkey-Patch
AlphaEngine.compute_from_history = staticmethod(_patched_compute_from_history)
logger.debug("AlphaEngine.compute_from_history 已应用 NB-21 闭环补丁 v2")