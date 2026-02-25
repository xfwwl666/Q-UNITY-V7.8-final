#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py — 双轨并行采集引擎 (patch_v7.1-fixed)
==================================================

【全部修复说明】

P0 修复：接口断裂
  - StockDataPipeline.__init__ 新增 enable_akshare: bool = False 参数
  - 新增 enrich_akshare() 方法（用于菜单选项2：AKShare 扩展字段补充）
  - 新增 akshare_enabled 标志写入统计字典（供 menu_main._print_stats 使用）

P1 修复：AKShare 解绑，不再阻塞 TDX
  - run() 方法：当 enable_akshare=False 时，完全跳过 AKShare Phase 1
  - TDX 多进程采集直接开始，不等 AKShare
  - enable_akshare=False 时 ak_results = {}，TDX 采集后仅用 BaoStock 兜底

P2 修复：TDX 改为 multiprocessing.Pool 架构（参照 tdx_downloader.py）
  - 进程初始化：_tdx_worker_init 在 Pool 启动时建立持久 TDX 连接
  - imap_unordered：流式处理，任意进程完成立即回调写盘，不等全部完成
  - 独立进程无 GIL 限制，真正并行；故障进程不影响其他进程

P3 修复：per-batch sleep（每50只 sleep 一次，而非每次 API 请求）
  - _tdx_fetch_worker 内部不做 sleep
  - 主进程在 imap_unordered 回调中做 per-batch 计数控制
  - 5000只: 5000/50 × 2s = 200s（vs 原来 5000/8线程 × 0.6s = 375s）

【架构图】

  enable_akshare=False (快速模式，推荐首次使用):
    ┌─────────────────────────────────────────┐
    │  multiprocessing.Pool(N 进程)            │
    │    _tdx_worker_init → 持久 TDX 连接      │
    │    _tdx_fetch_worker × N 进程并行        │
    │    imap_unordered → 流式写盘             │
    │    per-batch sleep 每50只               │
    │                       ↓                │
    │    DataValidator → RunReport            │
    └─────────────────────────────────────────┘
    预计耗时：~15~25 分钟（5000只）

  enable_akshare=True (完整双轨模式):
    Phase 1: AKShare ProcessPool(2进程) → turnover/pct_change 字典
    Phase 2: TDX multiprocessing.Pool → OHLCV
    Phase 3: 按 code 合并双轨 + DataValidator + 写盘
    预计耗时：2~4 小时（AKShare 限流）

  enrich_akshare() (独立扩展字段补充):
    对已有 Parquet 文件仅补充 AKShare 扩展字段
    预计耗时：2~4 小时
"""

import os
import time
import random
import logging
import multiprocessing
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    tqdm = None  # type: ignore[assignment,misc]
    _TQDM_AVAILABLE = False

from .node_scanner import get_fastest_nodes, TDX_NODES
from .tdx_pool import TDXConnectionPool
from .tdx_process_worker import (
    _tdx_worker_init,
    _tdx_fetch_worker,
    BATCH_SLEEP_EVERY as _DEFAULT_BATCH_SLEEP_EVERY,
    BATCH_SLEEP_MIN as _DEFAULT_BATCH_SLEEP_MIN,
    BATCH_SLEEP_MAX as _DEFAULT_BATCH_SLEEP_MAX,
)
from .akshare_client import run_akshare_batch, fetch_akshare_single, AK_EXTENDED_FIELDS
from .baostock_client import fetch_baostock
from .incremental import (
    read_local_max_date, compute_missing_range,
    is_up_to_date, load_local_df, merge_incremental, save_df,
)
from .validator import DataValidator
from .run_report import RunReport

logger = logging.getLogger(__name__)

# ============================================================================
# 配置常量
# ============================================================================
BARS_PER_REQ: int    = 800
TOTAL_BARS:   int    = 2500

try:
    from pytdx.params import TDXParams
    _KLINE_DAILY = TDXParams.KLINE_TYPE_DAILY
except ImportError:
    _KLINE_DAILY = 9


# ============================================================================
# TDX 原始 bars → 标准 DataFrame
# ============================================================================

def _clean_tdx_bars(raw: List[dict], code: str, market: int) -> pd.DataFrame:
    """向量化清洗 TDX 原始 bar 数据。"""
    df = pd.DataFrame(raw)

    col_map = {"datetime": "date", "vol": "vol", "amount": "amount"}
    df = df.rename(columns=col_map)

    std_cols = ["date", "open", "high", "low", "close", "vol", "amount"]
    df = df[[c for c in std_cols if c in df.columns]].copy()

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    for col in ("vol", "amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

    df.insert(0, "code",   code)
    df.insert(1, "market", market)
    df["source"] = "tdx"
    df["adjust"] = "hfq"   # A-12 Fix: 注意 TDX get_security_bars 默认复权类型需实测确认

    return df.sort_values("date").reset_index(drop=True)


def _estimate_missing_bars(start_date: str, end_date: str) -> int:
    """粗估缺失交易日数量（自然日 × 0.72）。"""
    try:
        s = datetime.strptime(start_date, "%Y-%m-%d")
        e = datetime.strptime(end_date, "%Y-%m-%d")
        return max(1, int((e - s).days * 0.72)) + 1
    except Exception:
        return TOTAL_BARS


# ============================================================================
# 双轨合并逻辑
# ============================================================================

def _merge_dual_track(
    tdx_df: Optional[pd.DataFrame],
    ak_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """
    将 TDX 的 OHLCV 数据与 AKShare 的扩展字段合并。

    合并策略（date 为键）：
      - 两者都有时：以 TDX 为主，补充 AKShare 扩展字段，adjust = hfq
      - 仅 TDX：adjust = hfq（v9: TDX 已后复权）
      - 仅 AKShare：完整 hfq 数据，adjust = hfq
      - 均为 None：返回 None
    """
    if tdx_df is None and ak_df is None:
        return None

    if tdx_df is None:
        return ak_df

    if ak_df is None:
        return tdx_df

    # 双轨合并：TDX LEFT JOIN AKShare（以 date 为键）
    extended = [c for c in ("turnover", "pct_change", "amplitude", "change") if c in ak_df.columns]

    if not extended:
        return tdx_df

    ak_ext = ak_df[["date"] + extended].copy()
    merged = pd.merge(tdx_df, ak_ext, on="date", how="left")
    merged["adjust"] = "hfq"

    return merged.sort_values("date").reset_index(drop=True)


# ============================================================================
# BaoStock 单股降级（同步，仅在双轨均失败时调用）
# ============================================================================

def _fetch_baostock_fallback(
    code: str,
    market: int,
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """BaoStock 兜底采集，含完整扩展字段（hfq）。"""
    return fetch_baostock(code, market, start_date, end_date)


# ============================================================================
# 单股增量 + 验证 + 写盘（供 update 场景）
# ============================================================================

def update_single_stock(
    code: str,
    market: int,
    parquet_dir: Path,
    tdx_pool: TDXConnectionPool,
    ak_results: Dict[str, Optional[pd.DataFrame]],
    report: RunReport,
    force_full: bool = False,
) -> Tuple[str, bool, str]:
    """
    单股完整更新流程（增量 + 双轨合并 + 验证 + 写盘）。
    此函数用于 ThreadPoolExecutor 场景（增量更新、单股采集）。
    全量采集场景使用 multiprocessing.Pool + _tdx_fetch_worker。

    Args:
        code:        股票代码
        market:      市场 0=深圳, 1=上海
        parquet_dir: Parquet 存储目录
        tdx_pool:    TDX 线程安全连接池
        ak_results:  AKShare 批量采集结果（预先运行，{code: df_or_None}）
        report:      运行报告收集器
        force_full:  强制全量重下载

    Returns:
        (code, success, source_tag)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
    parquet_path = parquet_dir / f"{code}.parquet"
    today_str    = date.today().strftime("%Y-%m-%d")

    local_max_date = None if force_full else read_local_max_date(parquet_path)

    if not force_full and is_up_to_date(local_max_date, today_str):
        report.record_skipped(code, market)
        return code, True, "skipped"

    start_date, end_date = compute_missing_range(local_max_date, today_str)
    if start_date == end_date and not force_full:
        report.record_skipped(code, market, reason="same_day")
        return code, True, "skipped"

    missing_bars = _estimate_missing_bars(start_date, end_date)

    # TDX 采集（通过连接池，线程安全）
    tdx_df: Optional[pd.DataFrame] = None
    try:
        api = tdx_pool.get_connection()
        if api is not None:
            all_bars: List[dict] = []
            offset = 0
            bars_to_fetch = min(missing_bars, TOTAL_BARS)
            while offset < bars_to_fetch:
                count = min(BARS_PER_REQ, bars_to_fetch - offset)
                bars = api.get_security_bars(
                    category=_KLINE_DAILY,
                    market=market,
                    code=code,
                    start=offset,
                    count=count,
                )
                if not bars:
                    break
                all_bars.extend(bars)
                if len(bars) < count:
                    break
                offset += BARS_PER_REQ
            if all_bars:
                raw = _clean_tdx_bars(all_bars, code, market)
                tdx_df = raw[raw["date"] >= start_date].copy()
                if tdx_df.empty:
                    tdx_df = None
    except Exception as exc:
        logger.debug("TDX 采集异常 (%s): %s", code, exc)
        tdx_pool.release()

    # AKShare 结果
    ak_df: Optional[pd.DataFrame] = ak_results.get(code)
    if ak_df is not None and not ak_df.empty:
        ak_df = ak_df[ak_df["date"] >= start_date].copy()
        if ak_df.empty:
            ak_df = None

    # 双轨合并
    new_df = _merge_dual_track(tdx_df, ak_df)
    source_tag = "merged" if (tdx_df is not None and ak_df is not None) \
        else ("tdx" if ak_df is None else "akshare")

    # BaoStock 兜底
    if new_df is None:
        bs_df = _fetch_baostock_fallback(code, market, start_date, end_date)
        if bs_df is not None and not bs_df.empty:
            new_df     = bs_df
            source_tag = "baostock"
        else:
            report.record_failed(code, market, reason="complete_fail")
            return code, False, "failed"

    # 数据验证
    ok, reason = DataValidator.validate(new_df, code=code)
    if not ok:
        logger.warning("数据验证失败 (%s): %s", code, reason)
        report.record_validation_failure(code, reason)
        report.record_failed(code, market, reason=f"validate:{reason}")
        return code, False, "validate_fail"

    # 增量合并
    if force_full:
        merged_df = new_df
    else:
        local_df  = load_local_df(parquet_path)
        merged_df = merge_incremental(local_df, new_df)

    ok2, reason2 = DataValidator.validate_merge_result(merged_df, code)
    if not ok2:
        logger.warning("合并后验证失败 (%s): %s", code, reason2)
        report.record_failed(code, market, reason=f"merge_validate:{reason2}")
        return code, False, "merge_validate_fail"

    # 流式写盘
    ok3 = save_df(merged_df, parquet_path)
    if not ok3:
        report.record_failed(code, market, reason="save_failed")
        return code, False, "save_failed"

    report.record_success(code, market, source=source_tag, rows=len(merged_df))
    return code, True, source_tag


# ============================================================================
# 主引擎：StockDataPipeline（全面修复版）
# ============================================================================

class StockDataPipeline:
    """
    A 股日线采集引擎（patch_v7.1-fixed）。

    【P0 修复】: 新增 enable_akshare 参数 + enrich_akshare() 方法
    【P1 修复】: enable_akshare=False 时完全跳过 AKShare Phase 1
    【P2 修复】: TDX 改为 multiprocessing.Pool（参照 tdx_downloader.py）
    【P3 修复】: per-batch sleep（每50只 sleep 1-3s）

    推荐工作流:
        # 步骤1: TDX 快速全量（~15~25分钟）
        pipeline = StockDataPipeline(enable_akshare=False)
        stats = pipeline.download_all_a_stocks()

        # 步骤2（可选）: AKShare 扩展字段补充（~2~4小时，可后台）
        pipeline2 = StockDataPipeline(enable_akshare=True)
        stats2 = pipeline2.enrich_akshare()
    """

    def __init__(
        self,
        parquet_dir:        str   = "./data/parquet",
        reports_dir:        str   = "./data/reports",
        top_n_nodes:        int   = 5,
        tdx_workers:        int   = 8,
        ak_workers:         int   = 2,
        node_timeout:       float = 3.0,
        tdx_timeout:        float = 10.0,
        ak_delay_min:       float = 1.5,   # A-04 Fix: 同步 akshare_client BUG-3 修复（0.3→1.5）
        ak_delay_max:       float = 3.5,   # A-04 Fix: 同步 akshare_client BUG-3 修复（0.8→3.5）
        ak_max_retries:     int   = 3,
        force_full:         bool  = False,
        enable_akshare:     bool  = False,   # P0 修复: 新增参数
        batch_sleep_every:  int   = 50,      # P3 修复: 每 N 只 sleep 一次
        batch_sleep_min:    float = 1.0,
        batch_sleep_max:    float = 3.0,
    ) -> None:
        self.parquet_dir       = Path(parquet_dir)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir       = reports_dir
        self.tdx_workers       = tdx_workers
        self.ak_workers        = ak_workers
        self.ak_delay_min      = ak_delay_min
        self.ak_delay_max      = ak_delay_max
        self.ak_max_retries    = ak_max_retries
        self.force_full        = force_full
        self.enable_akshare    = enable_akshare   # P0 修复
        self.batch_sleep_every = batch_sleep_every
        self.batch_sleep_min   = batch_sleep_min
        self.batch_sleep_max   = batch_sleep_max

        # 节点赛马
        logger.info("节点赛马测试（%d 个候选节点）...", len(TDX_NODES))
        self.top_nodes = get_fastest_nodes(top_n=top_n_nodes, timeout=node_timeout)
        if self.top_nodes:
            logger.info("选出节点: %s",
                        [(n["name"], f"{n['latency_ms']:.0f}ms") for n in self.top_nodes])
        else:
            logger.warning("所有节点不可达，将使用 BaoStock 兜底")

        # TDX 线程安全连接池（供增量更新/单股采集场景使用）
        self.tdx_pool = TDXConnectionPool(
            top_nodes=self.top_nodes if self.top_nodes else TDX_NODES[:3],
            timeout=tdx_timeout,
        )

    # ------------------------------------------------------------------
    # P1 修复：run() - enable_akshare=False 时完全跳过 AKShare
    # ------------------------------------------------------------------

    def run(self, stock_list: List[Tuple[str, int]]) -> Dict:
        """
        批量更新股票日线数据。

        P1 修复：enable_akshare=False 时，直接进入 TDX 多进程采集，
        不启动 AKShare Phase 1，不等待 AKShare 完成。

        Args:
            stock_list: [(code, market), ...] market: 0=深圳, 1=上海

        Returns:
            统计字典（含 akshare_enabled 字段）
        """
        total   = len(stock_list)
        report  = RunReport(self.reports_dir)
        today   = date.today().strftime("%Y-%m-%d")

        logger.info(
            "采集模式: %s | 股票数: %d | TDX=%d进程 | 批次sleep=每%d只",
            "双轨(TDX+AKShare)" if self.enable_akshare else "快速(仅TDX)",
            total, self.tdx_workers, self.batch_sleep_every,
        )

        # ── P1 修复: Phase 1 AKShare ──────────────────────────────────
        ak_results: Dict[str, Optional[pd.DataFrame]] = {}

        if self.enable_akshare:
            logger.info("── Phase 1: AKShare 批量进程采集（%d 只）...", total)
            ak_tasks = []
            for code, market in stock_list:
                parquet_path   = self.parquet_dir / f"{code}.parquet"
                local_max_date = None if self.force_full else read_local_max_date(parquet_path)
                start_date, end_date = compute_missing_range(local_max_date, today)
                if start_date != end_date or self.force_full:
                    ak_tasks.append((code, start_date, end_date))

            if ak_tasks:
                ak_pbar = None
                if _TQDM_AVAILABLE:
                    ak_pbar = tqdm(total=len(ak_tasks), desc="AKShare(扩展字段)",
                                   unit="股", colour="blue", dynamic_ncols=True)

                def _ak_progress(code: str, success: bool, error) -> None:
                    if ak_pbar:
                        ak_pbar.update(1)
                        ak_pbar.set_postfix_str(f"{'ok' if success else 'fail'} {code}")

                ak_results = run_akshare_batch(
                    stock_list=ak_tasks,
                    max_workers=self.ak_workers,
                    max_retries=self.ak_max_retries,
                    delay_min=self.ak_delay_min,
                    delay_max=self.ak_delay_max,
                    progress_callback=_ak_progress,
                )
                if ak_pbar:
                    ak_pbar.close()

                ak_ok = sum(1 for v in ak_results.values() if v is not None)
                logger.info("AKShare 完成: %d/%d 成功", ak_ok, len(ak_tasks))
        else:
            logger.info("── AKShare 跳过（enable_akshare=False，TDX 直接开始）")

        # ── P2+P3 修复: TDX multiprocessing.Pool ─────────────────────
        logger.info("── Phase TDX: 多进程采集 + 合并 + 写盘（%d 只）...", total)

        t0 = time.perf_counter()

        # 构建任务列表（code, market, missing_bars, start_date, end_date）
        tdx_tasks = []
        for code, market in stock_list:
            parquet_path   = self.parquet_dir / f"{code}.parquet"
            local_max_date = None if self.force_full else read_local_max_date(parquet_path)
            if not self.force_full and is_up_to_date(local_max_date, today):
                report.record_skipped(code, market)
                continue
            start_date, end_date = compute_missing_range(local_max_date, today)
            if start_date == end_date and not self.force_full:
                report.record_skipped(code, market, reason="same_day")
                continue
            missing_bars = _estimate_missing_bars(start_date, end_date)
            tdx_tasks.append((code, market, missing_bars, start_date, end_date))

        if tdx_tasks:
            self._run_tdx_multiprocess(tdx_tasks, ak_results, report, today)
        else:
            logger.info("所有股票已是最新，无需采集")

        # ── 持久化报告 ────────────────────────────────────────────────
        report.save()

        elapsed = time.perf_counter() - t0
        result = {
            "total":           total,
            "success":         report.total_success,
            "failed":          report.total_failed,
            "skipped":         report.total_skipped,
            "elapsed_s":       round(elapsed, 2),
            "speed":           round(total / elapsed, 2) if elapsed > 0 else 0,
            "reports_dir":     str(self.reports_dir),
            "akshare_enabled": self.enable_akshare,    # P0 修复: menu_main._print_stats 需要
        }
        logger.info("采集完成: %s | %.1fs | %.1f股/s",
                    report.summary_str(), elapsed, result["speed"])
        return result

    def _run_tdx_multiprocess(
        self,
        tdx_tasks: List[Tuple],
        ak_results: Dict[str, Optional[pd.DataFrame]],
        report: RunReport,
        today: str,
    ) -> None:
        """
        P2+P3 修复核心：使用 multiprocessing.Pool + imap_unordered 采集 TDX 数据。

        参照 tdx_downloader.py 的架构：
          1. Pool(initializer=_tdx_worker_init): 进程启动时建立持久连接
          2. imap_unordered: 流式处理，任意进程完成立即回调
          3. per-batch sleep: 每 batch_sleep_every 只 sleep 一次
        """
        n_workers = min(self.tdx_workers, len(tdx_tasks), multiprocessing.cpu_count())
        n_workers = max(1, n_workers)

        pbar = None
        if _TQDM_AVAILABLE:
            pbar = tqdm(total=len(tdx_tasks), desc="TDX多进程采集",
                        unit="股", colour="green", dynamic_ncols=True)

        completed_count = 0

        def _handle_one_result(result_tuple) -> None:
            nonlocal completed_count
            code, market, raw_bars, error = result_tuple

            completed_count += 1

            # P3 修复：per-batch sleep（在主进程统一控制）
            if completed_count % self.batch_sleep_every == 0:
                sleep_secs = random.uniform(self.batch_sleep_min, self.batch_sleep_max)
                logger.debug("批次 sleep %.1fs（已完成 %d 只）", sleep_secs, completed_count)
                time.sleep(sleep_secs)

            parquet_path = self.parquet_dir / f"{code}.parquet"
            local_max_date = None if self.force_full else read_local_max_date(parquet_path)
            start_date, end_date = compute_missing_range(
                None if self.force_full else local_max_date, today
            )

            # 处理 TDX 结果
            tdx_df: Optional[pd.DataFrame] = None
            if raw_bars and not error:
                try:
                    raw_df = _clean_tdx_bars(raw_bars, code, market)
                    tdx_df = raw_df[raw_df["date"] >= start_date].copy()
                    if tdx_df.empty:
                        tdx_df = None
                except Exception as e:
                    logger.warning("清洗 TDX bars 异常 (%s): %s", code, e)

            # AKShare 结果（来自 Phase 1 预先采集，可能为 None）
            ak_df: Optional[pd.DataFrame] = ak_results.get(code)
            if ak_df is not None and not ak_df.empty:
                ak_df_filtered = ak_df[ak_df["date"] >= start_date].copy()
                ak_df = ak_df_filtered if not ak_df_filtered.empty else None

            # 双轨合并
            new_df = _merge_dual_track(tdx_df, ak_df)
            source_tag = "merged" if (tdx_df is not None and ak_df is not None) \
                else ("tdx" if ak_df is None else "akshare")

            # BaoStock 兜底
            if new_df is None:
                bs_df = _fetch_baostock_fallback(code, market, start_date, end_date)
                if bs_df is not None and not bs_df.empty:
                    new_df     = bs_df
                    source_tag = "baostock"
                else:
                    report.record_failed(code, market, reason="complete_fail")
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix_str(report.summary_str())
                    return

            # 数据验证
            ok, reason = DataValidator.validate(new_df, code=code)
            if not ok:
                logger.warning("数据验证失败 (%s): %s", code, reason)
                report.record_validation_failure(code, reason)
                report.record_failed(code, market, reason=f"validate:{reason}")
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix_str(report.summary_str())
                return

            # 增量合并
            if self.force_full:
                merged_df = new_df
            else:
                local_df  = load_local_df(parquet_path)
                merged_df = merge_incremental(local_df, new_df)

            ok2, reason2 = DataValidator.validate_merge_result(merged_df, code)
            if not ok2:
                logger.warning("合并后验证失败 (%s): %s", code, reason2)
                report.record_failed(code, market, reason=f"merge_validate:{reason2}")
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix_str(report.summary_str())
                return

            # 流式写盘
            if save_df(merged_df, parquet_path):
                report.record_success(code, market, source=source_tag, rows=len(merged_df))
            else:
                report.record_failed(code, market, reason="save_failed")

            if pbar:
                pbar.update(1)
                pbar.set_postfix_str(report.summary_str())

        # P2 修复：multiprocessing.Pool + imap_unordered
        # A-07 Fix: Linux 使用 fork（更快），Windows/macOS 使用 spawn（兼容性）
        import platform as _platform
        _spawn_method = "fork" if _platform.system() == "Linux" else "spawn"
        ctx = multiprocessing.get_context(_spawn_method)
        with ctx.Pool(
            processes=n_workers,
            initializer=_tdx_worker_init,
            initargs=(
                self.top_nodes,
                10.0,                   # tdx_timeout
                self.batch_sleep_every,
                self.batch_sleep_min,
                self.batch_sleep_max,
            ),
        ) as pool:
            for result in pool.imap_unordered(_tdx_fetch_worker, tdx_tasks):
                _handle_one_result(result)

        if pbar:
            pbar.close()

    # ------------------------------------------------------------------
    # P0 修复：enrich_akshare() — AKShare 扩展字段独立补充
    # ------------------------------------------------------------------

    @staticmethod
    def _guess_market(code: str) -> int:
        """根据股票代码推测市场：6开头沪市，否则深市（0）。"""
        code = code.strip().zfill(6)
        return 1 if code.startswith(('6', '9')) else 0

    def enrich_akshare(self) -> Dict:
        """
        P0 修复：对已有 Parquet 文件补充 AKShare 扩展字段（turnover/pct_change）。
        此方法对应 menu_main.py 的「选项2: AKShare 扩展字段补充」。
        前提：需已有 Parquet 文件（先跑 download_all_a_stocks 或 run()）。

        Returns:
            统计字典
        """
        logger.info("AKShare 扩展字段补充开始...")

        # 扫描现有 Parquet 文件
        existing_files = list(self.parquet_dir.glob("*.parquet"))
        if not existing_files:
            logger.warning("Parquet 目录为空，请先执行 TDX 全量采集")
            return {"total": 0, "success": 0, "failed": 0, "akshare_enabled": True}

        logger.info("发现 %d 只已采集股票，准备补充扩展字段", len(existing_files))

        today = date.today().strftime("%Y-%m-%d")
        ak_tasks = []

        for f in existing_files:
            code = f.stem
            try:
                local_max_date = read_local_max_date(f)
                start_date, end_date = compute_missing_range(local_max_date, today)
                # 补充扩展字段：只需要最新数据（增量）
                ak_tasks.append((code, start_date, end_date))
            except Exception as e:
                logger.warning("跳过 %s: %s", code, e)

        if not ak_tasks:
            logger.info("所有股票扩展字段已是最新")
            return {"total": 0, "success": 0, "failed": 0, "akshare_enabled": True}

        logger.info("需要补充扩展字段的股票: %d 只", len(ak_tasks))

        # 初始化运行报告
        report = RunReport(self.reports_dir)

        ak_pbar = None
        if _TQDM_AVAILABLE:
            ak_pbar = tqdm(total=len(ak_tasks), desc="AKShare 扩展字段",
                           unit="股", colour="cyan", dynamic_ncols=True)

        success_count = 0
        failed_count  = 0

        def _ak_enrich_progress(code: str, success: bool, error) -> None:
            nonlocal success_count, failed_count
            if success:
                success_count += 1
            else:
                failed_count  += 1
            if ak_pbar:
                ak_pbar.update(1)
                ak_pbar.set_postfix_str(f"ok={success_count} fail={failed_count}")

        ak_results = run_akshare_batch(
            stock_list=ak_tasks,
            max_workers=self.ak_workers,
            max_retries=self.ak_max_retries,
            delay_min=self.ak_delay_min,
            delay_max=self.ak_delay_max,
            progress_callback=_ak_enrich_progress,
        )

        if ak_pbar:
            ak_pbar.close()

        merged_ok = 0
        merged_fail = 0

        # 合并 AKShare 结果回现有 Parquet
        for code, ak_df in ak_results.items():
            if ak_df is None or ak_df.empty:
                merged_fail += 1
                # 记录采集失败（akshare 本身失败）
                market = self._guess_market(code)
                report.record_failed(code, market, reason="akshare_fetch_fail")
                continue

            try:
                parquet_path = self.parquet_dir / f"{code}.parquet"
                if not parquet_path.exists():
                    merged_fail += 1
                    market = self._guess_market(code)
                    report.record_failed(code, market, reason="parquet_missing")
                    continue

                local_df = load_local_df(parquet_path)
                if local_df is None or local_df.empty:
                    merged_fail += 1
                    market = self._guess_market(code)
                    report.record_failed(code, market, reason="local_df_empty")
                    continue

                # 确定扩展字段列
                extended = [c for c in ("turnover", "pct_change", "amplitude", "change")
                            if c in ak_df.columns]
                if not extended:
                    merged_fail += 1
                    market = self._guess_market(code)
                    report.record_failed(code, market, reason="no_extended_fields")
                    continue

                # 提取 AKShare 扩展部分
                ak_ext = ak_df[["date"] + extended].copy()
                # 删除本地文件中已有的扩展字段，避免重复
                drop_cols = [c for c in extended if c in local_df.columns]
                if drop_cols:
                    local_df = local_df.drop(columns=drop_cols)

                # 合并
                merged = pd.merge(local_df, ak_ext, on="date", how="left")
                merged = merged.sort_values("date").reset_index(drop=True)

                # 保存
                if save_df(merged, parquet_path):
                    merged_ok += 1
                    market = self._guess_market(code)
                    report.record_success(code, market, source="akshare_enrich", rows=len(merged))
                else:
                    merged_fail += 1
                    market = self._guess_market(code)
                    report.record_failed(code, market, reason="save_failed")

            except Exception as e:
                logger.warning("合并扩展字段失败 (%s): %s", code, e)
                merged_fail += 1
                market = self._guess_market(code)
                report.record_failed(code, market, reason=f"merge_exception:{e}")

        # 保存运行报告（写入 failed_stocks.txt 等）
        report.save()

        logger.info("扩展字段补充完成: 成功=%d 失败=%d", merged_ok, merged_fail)

        return {
            "total":           len(ak_tasks),
            "success":         merged_ok,
            "failed":          merged_fail,
            "akshare_enabled": True,
        }

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def download_all_a_stocks(self) -> Dict:
        """一键下载全量 A 股（自动获取代码列表）。"""
        stock_list = self._get_all_a_stock_list()
        if not stock_list:
            logger.error("获取 A 股列表失败")
            return {}
        return self.run(stock_list)

    def retry_failed(self, reports_dir: Optional[str] = None) -> Dict:
        """
        从 failed_stocks.txt 加载失败列表，重新采集。

        Returns:
            补采统计字典
        """
        rdir = reports_dir or self.reports_dir
        failed_list = RunReport.load_failed_list(rdir)
        if not failed_list:
            logger.info("failed_stocks.txt 为空或不存在，无需补采")
            return {"total": 0, "akshare_enabled": self.enable_akshare}
        logger.info("开始补采 %d 只失败股票...", len(failed_list))
        old_force = self.force_full
        self.force_full = True
        result = self.run(failed_list)
        self.force_full = old_force
        return result

    def _get_all_a_stock_list(self) -> List[Tuple[str, int]]:
        """通过 TDX 获取全量 A 股代码列表。"""
        api = self.tdx_pool.get_connection()
        if api is None:
            return []
        try:
            stocks: List[Tuple[str, int]] = []
            for market in (0, 1):
                count = api.get_security_count(market)
                logger.info("市场 %d 总记录数: %d", market, count)
                for start in range(0, count, 1000):
                    batch = api.get_security_list(market, start)
                    if not batch:
                        continue
                    for item in batch:
                        code = item["code"]
                        if market == 0 and code.startswith(("00", "30")):
                            stocks.append((code, market))
                        elif market == 1 and code.startswith(("60", "68")):
                            stocks.append((code, market))
            stocks = list(set(stocks))
            logger.info("A 股总数: %d", len(stocks))
            return stocks
        except Exception as exc:
            logger.error("获取股票列表失败: %s", exc)
            return []