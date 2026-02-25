#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
incremental.py — 智能增量更新逻辑 (patch_v9)
==============================================
1. read_local_max_date()  — 读取本地 Parquet 的最大日期
2. compute_missing_range() — 计算缺失区间
3. merge_incremental()     — pd.concat + drop_duplicates(keep='last')
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

HISTORY_START_DATE = "2005-01-01"
EARLIEST_DATE      = "1990-12-19"


def read_local_max_date(parquet_path: Path) -> Optional[str]:
    """读取本地 Parquet 文件中 date 列的最大值（仅加载 date 列，节省内存）。"""
    if not parquet_path.exists():
        return None
    try:
        df = pd.read_parquet(parquet_path, columns=["date"])
        if df.empty:
            return None
        return str(df["date"].max())
    except Exception as exc:
        logger.warning("读取本地文件失败 (%s): %s，视为无本地数据", parquet_path, exc)
        return None


def compute_missing_range(
    local_max_date: Optional[str],
    as_of_date: Optional[str] = None,
    history_start: str = HISTORY_START_DATE,
) -> Tuple[str, str]:
    """
    计算需要请求的 [start_date, end_date] 区间。
    无本地数据 → 全量；有 → 从 local_max + 1 天开始。
    """
    today = as_of_date or date.today().strftime("%Y-%m-%d")
    if local_max_date is None:
        return history_start, today
    try:
        max_dt   = datetime.strptime(local_max_date, "%Y-%m-%d").date()
        next_day = (max_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        if next_day > today:
            return today, today   # 调用方检查 start == end
        return next_day, today
    except ValueError:
        logger.warning("日期格式错误: %s，将全量下载", local_max_date)
        return history_start, today


def is_up_to_date(local_max_date: Optional[str], as_of_date: Optional[str] = None) -> bool:
    """本地数据是否已是最新（允许 1 天宽容）。"""
    if local_max_date is None:
        return False
    today = as_of_date or date.today().strftime("%Y-%m-%d")
    try:
        max_dt   = datetime.strptime(local_max_date, "%Y-%m-%d").date()
        today_dt = datetime.strptime(today, "%Y-%m-%d").date()
        return (today_dt - max_dt).days <= 1
    except ValueError:
        return False


def merge_incremental(
    local_df: Optional[pd.DataFrame],
    new_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    合并本地历史与新增数据。
    drop_duplicates(keep='last'): 新数据优先（复权因子可能刷新历史）。
    """
    if local_df is None or local_df.empty:
        result = new_df.copy()
    elif new_df.empty:
        result = local_df.copy()
    else:
        result = pd.concat([local_df, new_df], ignore_index=True)

    dedup_cols = ["date"] if "code" not in result.columns else ["code", "date"]
    result = result.drop_duplicates(subset=dedup_cols, keep="last")
    return result.sort_values("date").reset_index(drop=True)


def load_local_df(parquet_path: Path) -> Optional[pd.DataFrame]:
    if not parquet_path.exists():
        return None
    try:
        df = pd.read_parquet(parquet_path)
        return None if df.empty else df
    except Exception as exc:
        logger.warning("加载失败 (%s): %s", parquet_path, exc)
        return None


def save_df(df: pd.DataFrame, parquet_path: Path, compression: str = "zstd") -> bool:
    try:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False, compression=compression)
        logger.debug("保存: %s (%d 行)", parquet_path.name, len(df))
        return True
    except Exception as exc:
        logger.error("保存失败 (%s): %s", parquet_path, exc)
        return False