#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tdx_process_worker.py — TDX 多进程采集工作函数 (patch_v7.1-fixed)
==================================================================

【P2 修复：TDX 改为 multiprocessing.Pool 架构】

原 ThreadPoolExecutor 问题：
  - Python GIL 限制 CPU 密集操作（DataFrame 构造），8线程实际并发受限
  - 连接为懒初始化，首批请求时有建连延迟
  - per-request sleep 0.1-0.2s 导致每只股票额外浪费 0.6s

新 multiprocessing.Pool 架构（参照 tdx_downloader.py 模式）：
  - _tdx_worker_init(): 进程启动时立即建立持久 TDX 连接，采集阶段直接使用
  - _tdx_fetch_worker(): 每50只才 sleep 一次（per-batch），而非每次请求都 sleep
  - Pool.imap_unordered(): 流式处理结果，任意进程完成立即回调，不等全部完成
  - N 个独立进程 = N 个独立 Python 解释器 = 真正的并行，无 GIL 瓶颈

【P3 修复：per-batch sleep】
  原来：每次 API 请求后 sleep 0.1-0.2s = 4次/股 × 0.15s = 0.6s/股
        5000只/8线程 = 375s 纯 sleep
  修复：每50只才 sleep 1-3s = 5000只/50 × 2s = 200s（且分散在多个进程）

【注意】
  此模块的顶层函数（_tdx_worker_init, _tdx_fetch_worker）必须在模块级定义，
  才能被 multiprocessing.Pool 序列化（pickle）传入子进程。
  不能在 class 内部或函数内部定义。
"""

import os
import time
import random
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================================
# 全局配置（子进程通过 _tdx_worker_init 注入）
# ============================================================================
_top_nodes: List[Dict] = []
_tdx_timeout: float    = 10.0
_api = None   # 子进程内的持久连接（进程级全局变量）

BARS_PER_REQ: int = 800
TOTAL_BARS:   int = 2500

# 批次 sleep 控制（P3 修复）
BATCH_SLEEP_EVERY: int   = 50    # 每 N 只股票 sleep 一次
BATCH_SLEEP_MIN:   float = 1.0   # 批次 sleep 最小值（秒）
BATCH_SLEEP_MAX:   float = 3.0   # 批次 sleep 最大值（秒）

try:
    from pytdx.hq import TdxHq_API
    from pytdx.params import TDXParams
    _KLINE_DAILY = TDXParams.KLINE_TYPE_DAILY
    _PYTDX_AVAILABLE = True
except ImportError:
    TdxHq_API = None
    _KLINE_DAILY = 9
    _PYTDX_AVAILABLE = False


# ============================================================================
# P2: 进程初始化函数（Pool initializer）
# ============================================================================

def _tdx_worker_init(
    top_nodes: List[Dict],
    tdx_timeout: float = 10.0,
    batch_sleep_every: int = 50,
    batch_sleep_min: float = 1.0,
    batch_sleep_max: float = 3.0,
) -> None:
    """
    multiprocessing.Pool 进程初始化函数。

    在子进程启动时调用一次，建立持久 TDX 连接。
    采集阶段直接复用此连接，零建连延迟。

    注意：此函数是顶层函数，必须可 pickle，不能在 class 内部定义。
    """
    global _top_nodes, _tdx_timeout, _api
    global BATCH_SLEEP_EVERY, BATCH_SLEEP_MIN, BATCH_SLEEP_MAX

    _top_nodes = top_nodes
    _tdx_timeout = tdx_timeout
    BATCH_SLEEP_EVERY = batch_sleep_every
    BATCH_SLEEP_MIN = batch_sleep_min
    BATCH_SLEEP_MAX = batch_sleep_max

    pid = os.getpid()

    if not _PYTDX_AVAILABLE:
        logger.warning("[PID-%d] pytdx 未安装，子进程将无法采集 TDX 数据", pid)
        return

    # 尝试连接 top_nodes 中的节点，优先选择延迟最低的
    for node in top_nodes:
        try:
            api = TdxHq_API(heartbeat=True, auto_retry=True)
            api.connect(node["host"], node["port"], time_out=tdx_timeout)
            # 验证连接可用
            count = api.get_security_count(0)
            if count and count > 0:
                _api = api
                logger.info("[PID-%d] TDX 连接成功: %s (%s:%s)",
                            pid, node["name"], node["host"], node["port"])
                return
            else:
                api.disconnect()
        except Exception as exc:
            logger.debug("[PID-%d] 连接失败 %s: %s", pid, node["name"], exc)
            try:
                api.disconnect()
            except Exception:
                pass

    logger.error("[PID-%d] 所有节点连接失败，此进程将使用 BaoStock 兜底", pid)


def _tdx_reconnect() -> bool:
    """尝试重新连接 TDX，返回是否成功。"""
    global _api
    pid = os.getpid()

    if not _PYTDX_AVAILABLE or not _top_nodes:
        return False

    for node in _top_nodes:
        try:
            api = TdxHq_API(heartbeat=True, auto_retry=True)
            api.connect(node["host"], node["port"], time_out=_tdx_timeout)
            count = api.get_security_count(0)
            if count and count > 0:
                _api = api
                logger.info("[PID-%d] TDX 重连成功: %s", pid, node["name"])
                return True
            api.disconnect()
        except Exception as exc:
            logger.debug("[PID-%d] 重连失败 %s: %s", pid, node["name"], exc)
    return False


# ============================================================================
# P2+P3: 进程工作函数（Pool worker）
# ============================================================================

def _tdx_fetch_worker(
    task: Tuple[str, int, int, str, str],
) -> Tuple[str, int, Optional[List[dict]], Optional[str]]:
    """
    multiprocessing.Pool 工作函数，在子进程中执行。

    【P3 修复】：batch sleep 控制在调用方（Pipeline.run）完成，
    本函数内部不做 per-request sleep，只做纯粹的数据拉取。
    这样可以精确控制 sleep 频率，避免 per-request 的浪费。

    Args:
        task: (code, market, missing_bars, start_date, end_date)

    Returns:
        (code, market, raw_bars_list_or_None, error_msg_or_None)
        注意：返回 list[dict] 而非 DataFrame，因为 DataFrame pickle 有额外开销。
    """
    global _api

    code, market, missing_bars, start_date, end_date = task
    pid = os.getpid()

    if not _PYTDX_AVAILABLE:
        return (code, market, None, "pytdx_not_available")

    if _api is None:
        # 尝试重连
        if not _tdx_reconnect():
            return (code, market, None, "no_connection")

    bars_to_fetch = min(missing_bars, TOTAL_BARS)
    all_bars: List[dict] = []
    offset = 0

    try:
        while offset < bars_to_fetch:
            count = min(BARS_PER_REQ, bars_to_fetch - offset)
            try:
                bars = _api.get_security_bars(
                    category=_KLINE_DAILY,
                    market=market,
                    code=code,
                    start=offset,
                    count=count,
                )
            except Exception as api_exc:
                # 连接可能断开，尝试重连一次
                logger.debug("[PID-%d] API 请求异常 (%s), 尝试重连: %s",
                             pid, code, api_exc)
                try:
                    _api.disconnect()
                except Exception:
                    pass
                _api = None
                if not _tdx_reconnect():
                    return (code, market, None, f"reconnect_failed:{api_exc}")
                # 重连后重试本次请求
                try:
                    bars = _api.get_security_bars(
                        category=_KLINE_DAILY,
                        market=market,
                        code=code,
                        start=offset,
                        count=count,
                    )
                except Exception as retry_exc:
                    return (code, market, None, f"retry_failed:{retry_exc}")

            if not bars:
                break
            all_bars.extend(bars)
            if len(bars) < count:   # 到达最早数据
                break
            offset += BARS_PER_REQ

    except Exception as exc:
        logger.warning("[PID-%d] TDX 分片请求异常 (%s): %s", pid, code, exc)
        try:
            _api.disconnect()
        except Exception:
            pass
        _api = None
        return (code, market, None, f"fetch_exception:{exc}")

    if not all_bars:
        return (code, market, None, "empty_bars")

    return (code, market, all_bars, None)