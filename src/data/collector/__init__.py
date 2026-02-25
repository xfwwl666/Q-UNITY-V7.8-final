#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/data/collector — 双轨并行采集包 (patch_v7.1-fixed)
=======================================================
主要接口:
  StockDataPipeline   — 主采集引擎（P0/P1/P2/P3 全修复）
  TDXConnectionPool   — 线程安全 TDX 连接池（供增量更新使用）
  get_fastest_nodes   — 24节点赛马
  run_akshare_batch   — AKShare 进程池批量采集
  fetch_baostock      — BaoStock 单股采集
  DataValidator       — 数据验证（三层）
  RunReport           — 运行报告持久化
"""

from .node_scanner import TDX_NODES, get_fastest_nodes, race_nodes
from .tdx_pool import TDXConnectionPool, get_global_pool, reset_global_pool
from .tdx_process_worker import _tdx_worker_init, _tdx_fetch_worker
from .akshare_client import (
    run_akshare_batch,
    fetch_akshare_single,
    _akshare_process_worker,
    AK_EXTENDED_FIELDS,
)
from .baostock_client import fetch_baostock
from .incremental import (
    read_local_max_date, compute_missing_range,
    is_up_to_date, merge_incremental, load_local_df, save_df,
)
from .validator import DataValidator, REQUIRED_COLS, MIN_ROWS
from .run_report import RunReport
from .pipeline import StockDataPipeline, update_single_stock

__all__ = [
    "TDX_NODES", "get_fastest_nodes", "race_nodes",
    "TDXConnectionPool", "get_global_pool", "reset_global_pool",
    "_tdx_worker_init", "_tdx_fetch_worker",
    "run_akshare_batch", "fetch_akshare_single", "AK_EXTENDED_FIELDS",
    "fetch_baostock",
    "read_local_max_date", "compute_missing_range",
    "is_up_to_date", "merge_incremental", "load_local_df", "save_df",
    "DataValidator", "REQUIRED_COLS", "MIN_ROWS",
    "RunReport",
    "StockDataPipeline", "update_single_stock",
]