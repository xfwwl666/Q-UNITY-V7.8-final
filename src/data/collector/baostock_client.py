#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baostock_client.py — BaoStock 数据采集客户端（三级兜底）(patch_v9)
================================================================
复权方式统一为 hfq（后复权），adjustflag="1"。
每次 fetch 独立 login/logout，确保会话隔离。
"""

import time
import random
import logging
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import baostock as bs
    _BAOSTOCK_AVAILABLE = True
except ImportError:
    bs = None  # type: ignore[assignment]
    _BAOSTOCK_AVAILABLE = False

_BS_FIELDS  = "date,open,high,low,close,volume,amount,adjustflag,turn,tradestatus,pctChg"
_MARKET_PFX = {0: "sz", 1: "sh"}
_BS_COL_MAP = {
    "volume":      "vol",
    "amount":      "amount",
    "turn":        "turnover",
    "pctChg":      "pct_change",
    "tradestatus": "trade_status",
    "adjustflag":  "adjust_flag",
}
_REQUIRED_COLS = {"date", "open", "high", "low", "close", "vol"}


def _bs_code(code: str, market: int) -> str:
    return f"{_MARKET_PFX.get(market, 'sz')}.{code}"


def _to_date_str(d: str) -> str:
    """YYYYMMDD 或 YYYY-MM-DD 统一转 YYYY-MM-DD"""
    d = d.replace("/", "-")
    if len(d) == 8 and "-" not in d:
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return d


def _standardize(df: pd.DataFrame, code: str) -> pd.DataFrame:
    df = df.rename(columns=_BS_COL_MAP)
    df["code"]   = code
    df["source"] = "baostock"
    df["adjust"] = "hfq"

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    for col in ("vol", "amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    for col in ("turnover", "pct_change"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    # 过滤停牌日
    if "trade_status" in df.columns:
        df = df[df["trade_status"].astype(str) != "0"].copy()

    keep = [c for c in (
        "code", "date", "open", "high", "low", "close",
        "vol", "amount", "pct_change", "turnover", "source", "adjust"
    ) if c in df.columns]
    return df[keep].sort_values("date").reset_index(drop=True)


def fetch_baostock(
    code: str,
    market: int,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    delay_min: float = 0.5,
    delay_max: float = 1.0,
) -> Optional[pd.DataFrame]:
    """
    通过 BaoStock 获取 A 股日线数据（后复权 adjustflag=1）。

    Args:
        code:        股票代码，如 "600000"
        market:      市场 0=深圳, 1=上海
        start_date:  起始日期 "YYYY-MM-DD" 或 "YYYYMMDD"
        end_date:    截止日期
        max_retries: 最大重试次数
        delay_min/max: 随机 sleep 区间

    Returns:
        标准化 DataFrame（含 turnover/pct_change）或 None
    """
    if not _BAOSTOCK_AVAILABLE:
        logger.warning("baostock 未安装，跳过")
        return None

    bs_symbol = _bs_code(code, market)
    s_date    = _to_date_str(start_date)
    e_date    = _to_date_str(end_date)

    for attempt in range(max_retries):
        lg = None
        try:
            time.sleep(random.uniform(delay_min, delay_max))
            lg = bs.login()
            if lg.error_code != "0":
                raise RuntimeError(f"login 失败: {lg.error_msg}")

            rs = bs.query_history_k_data_plus(
                code=bs_symbol,
                fields=_BS_FIELDS,
                start_date=s_date,
                end_date=e_date,
                frequency="d",
                adjustflag="1",  # 后复权（hfq），与 AKShare 统一
            )
            if rs.error_code != "0":
                raise RuntimeError(f"查询失败: {rs.error_msg}")

            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                raise ValueError(f"空数据: {bs_symbol}")

            df = pd.DataFrame(rows, columns=rs.fields)
            df = _standardize(df, code)
            missing = _REQUIRED_COLS - set(df.columns)
            if missing:
                raise ValueError(f"缺少必须列: {missing}")

            logger.debug("BaoStock 成功: %s, %d 行", code, len(df))
            return df

        except Exception as exc:
            wait = (2 ** attempt) * (1 + random.random() * 0.3)
            if attempt < max_retries - 1:
                logger.warning("BaoStock 第%d/%d次失败 (%s): %s，等待%.1fs",
                               attempt + 1, max_retries, code, exc, wait)
                time.sleep(wait)
            else:
                logger.error("BaoStock 全部失败 (%s): %s", code, exc)
        finally:
            if lg is not None:
                try:
                    bs.logout()
                except Exception:
                    pass
    return None