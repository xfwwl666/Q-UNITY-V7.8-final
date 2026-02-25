#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enrich_baostock.py — 用 BaoStock 全量补充扩展字段（独立脚本）
=============================================================

功能：
  扫描 data/parquet/ 下所有 .parquet 文件，
  用 BaoStock 补充 turnover（换手率）和 pct_change（涨跌幅）。
  支持断点续传：已有这两列的文件自动跳过。

用法：
  python enrich_baostock.py                  # 使用默认路径 ./data/parquet
  python enrich_baostock.py --parquet ./data/parquet --workers 4
  python enrich_baostock.py --force          # 强制重写所有文件（忽略已有字段）

优点 vs AKShare：
  - BaoStock 无限流问题，速度稳定
  - 字段：turnover（换手率）+ pct_change（涨跌幅）完全覆盖
  - 缺点：无 amplitude（振幅），振幅只有 AKShare 才有

注意：
  - 复权方式 adjustflag="1"（后复权，与项目统一）
  - BaoStock 每次 login/logout 独立会话，无并发冲突
  - 建议 workers=4~8，BaoStock 无限流无需保守
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ── 日志配置 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("enrich_baostock.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── BaoStock 字段 ─────────────────────────────────────────────────────────────
_BS_FIELDS  = "date,turn,pctChg,tradestatus"
_BS_COL_MAP = {
    "turn":        "turnover",
    "pctChg":      "pct_change",
    "tradestatus": "trade_status",
}

try:
    import baostock as bs
    _BS_OK = True
except ImportError:
    _BS_OK = False
    logger.error("baostock 未安装！请执行: pip install baostock")
    sys.exit(1)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _guess_market(code: str) -> int:
    """6/9 开头 → 上海(1)，其余 → 深圳(0)。"""
    code = code.strip().zfill(6)
    return 1 if code.startswith(("6", "9")) else 0


def _bs_symbol(code: str, market: int) -> str:
    prefix = "sh" if market == 1 else "sz"
    return f"{prefix}.{code}"


def _to_ymd(d: str) -> str:
    """YYYYMMDD 或 YYYY-MM-DD → YYYY-MM-DD"""
    d = d.replace("/", "-")
    if len(d) == 8 and "-" not in d:
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return d


def _read_local_max_date(parquet_path: Path) -> Optional[str]:
    try:
        df = pd.read_parquet(parquet_path, columns=["date"])
        return str(df["date"].max()) if not df.empty else None
    except Exception:
        return None


def _needs_enrich(parquet_path: Path, force: bool) -> bool:
    """检查文件是否需要补充（有 turnover 且有 pct_change 则跳过，除非 --force）。"""
    if force:
        return True
    try:
        df = pd.read_parquet(parquet_path)
        has_turnover   = "turnover"   in df.columns and df["turnover"].notna().any()
        has_pct_change = "pct_change" in df.columns and df["pct_change"].notna().any()
        return not (has_turnover and has_pct_change)
    except Exception:
        return True


# ── BaoStock 采集（单股）────────────────────────────────────────────────────

def fetch_ext_fields(
    code: str,
    market: int,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
) -> Optional[pd.DataFrame]:
    """
    用 BaoStock 获取指定股票的 turnover + pct_change。

    Returns:
        DataFrame(date, turnover, pct_change) 或 None
    """
    symbol  = _bs_symbol(code, market)
    s_date  = _to_ymd(start_date)
    e_date  = _to_ymd(end_date)

    for attempt in range(max_retries):
        lg = None
        try:
            time.sleep(random.uniform(0.1, 0.3))   # BaoStock 无限流，轻度随机延迟即可
            lg = bs.login()
            if lg.error_code != "0":
                raise RuntimeError(f"login 失败: {lg.error_msg}")

            rs = bs.query_history_k_data_plus(
                code=symbol,
                fields=_BS_FIELDS,
                start_date=s_date,
                end_date=e_date,
                frequency="d",
                adjustflag="1",   # 后复权，与项目统一
            )
            if rs.error_code != "0":
                raise RuntimeError(f"查询失败: {rs.error_msg}")

            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())

            if not rows:
                logger.debug("BaoStock 空数据: %s", code)
                return None

            df = pd.DataFrame(rows, columns=rs.fields)
            df = df.rename(columns=_BS_COL_MAP)

            # 过滤停牌日（trade_status == "0"）
            if "trade_status" in df.columns:
                df = df[df["trade_status"].astype(str) != "0"].copy()

            df["date"]       = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df["turnover"]   = pd.to_numeric(df.get("turnover"),   errors="coerce").astype("float32")
            df["pct_change"] = pd.to_numeric(df.get("pct_change"), errors="coerce").astype("float32")

            keep = [c for c in ("date", "turnover", "pct_change") if c in df.columns]
            return df[keep].sort_values("date").reset_index(drop=True)

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


# ── 单股处理（采集 + 合并 + 写盘）──────────────────────────────────────────

def process_one(
    parquet_path: Path,
    force: bool,
    history_start: str,
) -> Tuple[str, str]:
    """
    处理单只股票。

    Returns:
        (code, status)  status ∈ {"skipped", "ok", "fail"}
    """
    code   = parquet_path.stem
    market = _guess_market(code)

    # 跳过已有字段的文件
    if not _needs_enrich(parquet_path, force):
        return code, "skipped"

    # 确定补充区间：全量（无历史数据时从 history_start 开始）
    local_max = _read_local_max_date(parquet_path)
    end_date  = date.today().strftime("%Y-%m-%d")

    if local_max is None:
        start_date = history_start
    else:
        try:
            next_day   = (datetime.strptime(local_max, "%Y-%m-%d").date()
                          + timedelta(days=1))
            start_date = next_day.strftime("%Y-%m-%d")
        except ValueError:
            start_date = history_start

    # 起始日期超过今天说明本地数据已是最新，但扩展字段可能仍缺失
    # 此时回退到全量补充（从 history_start 开始）
    if start_date > end_date:
        start_date = history_start

    # BaoStock 采集扩展字段
    ext_df = fetch_ext_fields(code, market, start_date, end_date)
    if ext_df is None or ext_df.empty:
        return code, "fail"

    # 读取本地 Parquet
    try:
        local_df = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.error("读取失败 (%s): %s", code, e)
        return code, "fail"

    # 删除已有的扩展字段列（避免重复）
    drop_cols = [c for c in ("turnover", "pct_change") if c in local_df.columns]
    if drop_cols:
        local_df = local_df.drop(columns=drop_cols)

    # left join 合并
    merged = pd.merge(local_df, ext_df, on="date", how="left")
    merged = merged.sort_values("date").reset_index(drop=True)

    # 写回
    try:
        merged.to_parquet(parquet_path, index=False, compression="zstd")
        logger.debug("写盘成功: %s (%d 行)", code, len(merged))
        return code, "ok"
    except Exception as e:
        logger.error("写盘失败 (%s): %s", code, e)
        return code, "fail"


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="用 BaoStock 全量补充扩展字段（turnover/pct_change）"
    )
    parser.add_argument(
        "--parquet", default="./data/parquet",
        help="Parquet 目录路径（默认: ./data/parquet）"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="并发线程数（默认: 4，BaoStock 无限流可适当提高）"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重写所有文件（默认跳过已有 turnover+pct_change 的文件）"
    )
    parser.add_argument(
        "--history-start", default="2005-01-01",
        help="历史起始日期（默认: 2005-01-01）"
    )
    args = parser.parse_args()

    parquet_dir = Path(args.parquet)
    if not parquet_dir.exists():
        logger.error("Parquet 目录不存在: %s", parquet_dir)
        sys.exit(1)

    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        logger.error("Parquet 目录为空，请先执行 TDX 全量采集")
        sys.exit(1)

    logger.info("扫描到 %d 只股票，workers=%d，force=%s",
                len(files), args.workers, args.force)

    # 统计
    ok_count      = 0
    fail_count    = 0
    skipped_count = 0
    fail_list: List[str] = []

    try:
        from tqdm import tqdm as _tqdm
        pbar = _tqdm(total=len(files), unit="股", dynamic_ncols=True)
    except ImportError:
        pbar = None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(process_one, f, args.force, args.history_start): f.stem
            for f in files
        }
        for future in as_completed(future_map):
            code = future_map[future]
            try:
                _, status = future.result()
            except Exception as exc:
                status = "fail"
                logger.error("worker 异常 (%s): %s", code, exc)

            if status == "ok":
                ok_count += 1
            elif status == "skipped":
                skipped_count += 1
            else:
                fail_count += 1
                fail_list.append(code)

            if pbar:
                pbar.update(1)
                pbar.set_postfix_str(
                    f"ok={ok_count} skip={skipped_count} fail={fail_count}"
                )

    if pbar:
        pbar.close()

    # 输出汇总
    logger.info("=" * 60)
    logger.info("完成！成功=%d  跳过=%d  失败=%d  共=%d",
                ok_count, skipped_count, fail_count, len(files))
    if fail_list:
        fail_path = parquet_dir.parent / "reports" / "bs_enrich_failed.txt"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        fail_path.write_text("\n".join(fail_list), encoding="utf-8")
        logger.info("失败列表已保存至: %s", fail_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()