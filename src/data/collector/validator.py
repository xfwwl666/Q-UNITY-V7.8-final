#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validator.py — 数据验证层 (patch_v9)
======================================
采集后、写盘前执行三层验证：
  Layer 1: 行数检查   — 防止空数据写入 Parquet
  Layer 2: 必需列检查 — 保证 schema 完整
  Layer 3: OHLC 逻辑  — 防止脏数据（high < low / close <= 0）

通过验证返回 (True, "ok")；
失败返回 (False, "reason_string")，调用方记录到 RunReport。
"""

import logging
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# 必需列集合
REQUIRED_COLS = {"date", "open", "high", "low", "close", "vol"}

# 最小有效行数
MIN_ROWS = 10


class DataValidator:
    """
    三层数据验证器。

    使用示例:
        ok, reason = DataValidator.validate(df, code="000001")
        if not ok:
            report.add_validation_failure(code, reason)
    """

    @staticmethod
    def validate(
        df: Optional[pd.DataFrame],
        code: str = "",
        min_rows: int = MIN_ROWS,
    ) -> Tuple[bool, str]:
        """
        对采集结果执行三层验证。

        Args:
            df:       待验证的 DataFrame（可为 None）
            code:     股票代码（用于日志）
            min_rows: 最小行数阈值

        Returns:
            (True, "ok") 通过
            (False, reason_str) 失败，reason 供 RunReport 记录
        """
        # Layer 1: 空值/行数
        if df is None:
            return False, "df_is_none"
        if df.empty:
            return False, "df_empty"
        if len(df) < min_rows:
            return False, f"too_few_rows:{len(df)}"

        # Layer 2: 必需列完整性
        missing_cols = REQUIRED_COLS - set(df.columns)
        if missing_cols:
            return False, f"missing_cols:{sorted(missing_cols)}"

        # Layer 3: OHLC 逻辑合理性
        try:
            numeric_df = df[["open", "high", "low", "close"]].apply(
                pd.to_numeric, errors="coerce"
            )

            # high >= low
            invalid_hl = (numeric_df["high"] < numeric_df["low"]).sum()
            if invalid_hl > 0:
                return False, f"high_lt_low:{invalid_hl}_rows"

            # close > 0（允许停牌日为 0，但不允许负价）
            negative_close = (numeric_df["close"] < 0).sum()
            if negative_close > 0:
                return False, f"negative_close:{negative_close}_rows"

            # open/high/low/close 不能全部为 NaN
            all_nan_pct = numeric_df.isna().all(axis=1).mean()
            if all_nan_pct > 0.5:  # 超过 50% 行全 NaN
                return False, f"too_many_nan_rows:{all_nan_pct:.0%}"

        except Exception as exc:
            return False, f"ohlc_check_error:{exc}"

        return True, "ok"

    @staticmethod
    def validate_merge_result(
        merged_df: Optional[pd.DataFrame],
        code: str = "",
    ) -> Tuple[bool, str]:
        """
        对增量合并后的最终 DataFrame 做二次验证（更宽松）。
        主要确保合并没有引入严重问题。
        """
        if merged_df is None or merged_df.empty:
            return False, "merged_empty"

        # 检查日期是否有序
        if "date" in merged_df.columns:
            dates = merged_df["date"].tolist()
            if dates != sorted(dates):
                return False, "date_not_sorted"

            # 检查重复日期
            dup_count = merged_df.duplicated(subset=["date"]).sum()
            if dup_count > 0:
                return False, f"duplicate_dates:{dup_count}"

        return True, "ok"