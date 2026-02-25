#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
node_scanner.py — TDX 24 节点赛马筛选模块
==========================================
并发 TCP 探针，按延迟升序排序。
"""

import socket
import time
import logging
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# ============================================================================
# 24 个实测优选节点 (Port 均为 7709)
# ============================================================================
TDX_NODES: List[Dict] = [
    {"name": "node-01", "host": "116.205.183.150", "port": 7709},
    {"name": "node-02", "host": "116.205.163.254", "port": 7709},
    {"name": "node-03", "host": "110.41.2.72",     "port": 7709},
    {"name": "node-04", "host": "110.41.147.114",  "port": 7709},
    {"name": "node-05", "host": "111.230.186.52",  "port": 7709},
    {"name": "node-06", "host": "124.71.9.153",    "port": 7709},
    {"name": "node-07", "host": "116.205.171.132", "port": 7709},
    {"name": "node-08", "host": "124.71.187.122",  "port": 7709},
    {"name": "node-09", "host": "123.60.84.66",    "port": 7709},
    {"name": "node-10", "host": "123.60.70.228",   "port": 7709},
    {"name": "node-11", "host": "122.51.232.182",  "port": 7709},
    {"name": "node-12", "host": "115.238.56.198",  "port": 7709},
    {"name": "node-13", "host": "122.51.120.217",  "port": 7709},
    {"name": "node-14", "host": "124.70.133.119",  "port": 7709},
    {"name": "node-15", "host": "123.60.73.44",    "port": 7709},
    {"name": "node-16", "host": "115.238.90.165",  "port": 7709},
    {"name": "node-17", "host": "218.75.126.9",    "port": 7709},
    {"name": "node-18", "host": "121.36.225.169",  "port": 7709},
    {"name": "node-19", "host": "118.25.98.114",   "port": 7709},
    {"name": "node-20", "host": "119.97.185.59",   "port": 7709},
    {"name": "node-21", "host": "124.71.187.72",   "port": 7709},
    {"name": "node-22", "host": "124.70.199.56",   "port": 7709},
    {"name": "node-23", "host": "111.229.247.189", "port": 7709},
    {"name": "node-24", "host": "180.153.18.170",  "port": 7709},
]


def _probe_sync(node: Dict, timeout: float) -> Dict:
    """TCP 握手探针，测量实际连接延迟。"""
    start = time.perf_counter()
    try:
        with socket.create_connection((node["host"], node["port"]), timeout=timeout):
            latency_ms = (time.perf_counter() - start) * 1000
            return {**node, "latency_ms": round(latency_ms, 2), "status": "ok"}
    except (socket.timeout, OSError) as exc:
        return {**node, "latency_ms": -1.0, "status": f"fail:{type(exc).__name__}"}


def _sort_results(results: List[Dict]) -> List[Dict]:
    return sorted(results, key=lambda x: (x["latency_ms"] < 0, x["latency_ms"]))


def race_nodes(
    nodes: Optional[List[Dict]] = None,
    timeout: float = 3.0,
    workers: int = 32,
) -> List[Dict]:
    """并发赛马：同时向所有节点发出 TCP 探针。"""
    nodes = nodes or TDX_NODES
    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(nodes))) as pool:
        future_map = {pool.submit(_probe_sync, node, timeout): node for node in nodes}
        for future in as_completed(future_map):
            try:
                results.append(future.result())
            except Exception as exc:
                node = future_map[future]
                results.append({**node, "latency_ms": -1.0, "status": f"fail:{exc}"})
    sorted_results = _sort_results(results)
    ok_count = sum(1 for r in sorted_results if r["status"] == "ok")
    logger.info("节点赛马: %d/%d 可达，最优 %s (%.1f ms)",
                ok_count, len(nodes),
                sorted_results[0]["name"] if ok_count else "None",
                sorted_results[0]["latency_ms"] if ok_count else -1)
    return sorted_results


def get_fastest_nodes(top_n: int = 5, timeout: float = 3.0) -> List[Dict]:
    """返回延迟最低的 top_n 个可用节点。"""
    all_results = race_nodes(timeout=timeout)
    ok_nodes = [r for r in all_results if r["status"] == "ok"]
    selected = ok_nodes[:top_n]
    if not selected:
        logger.warning("所有节点不可达！返回默认节点（未验证）")
        return [dict(n, latency_ms=-1.0, status="unknown") for n in TDX_NODES[:top_n]]
    return selected