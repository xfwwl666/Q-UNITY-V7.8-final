#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tdx_pool.py — TDX 线程安全连接池 (threading.local)
====================================================
pytdx 的 TdxHq_API 不是线程安全的。
本模块通过 threading.local() 保证每线程独立 socket 连接。

注意：本模块用于 ThreadPoolExecutor 场景（如增量更新、单股采集）。
全量采集场景已改为 multiprocessing.Pool，使用 tdx_process_worker 模块。
"""

import random
import threading
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from pytdx.hq import TdxHq_API
    from pytdx.params import TDXParams
    _PYTDX_AVAILABLE = True
except ImportError:
    TdxHq_API = None   # type: ignore[assignment,misc]
    TDXParams  = None  # type: ignore[assignment]
    _PYTDX_AVAILABLE = False


class TDXConnectionPool:
    """
    线程安全的 TDX 连接池。
    每个线程维护独立的 TdxHq_API 实例（threading.local），
    避免多线程共享同一 socket 连接导致的数据错乱。
    """

    def __init__(
        self,
        top_nodes: List[Dict],
        timeout: float = 10.0,
    ) -> None:
        if not top_nodes:
            raise ValueError("top_nodes 不能为空")
        self._nodes   = top_nodes
        self._timeout = timeout
        self._local   = threading.local()

    def _create_connection(self) -> Optional["TdxHq_API"]:
        if not _PYTDX_AVAILABLE:
            logger.error("pytdx 未安装，无法创建 TDX 连接")
            return None

        for node in self._nodes:
            try:
                api = TdxHq_API(heartbeat=True, auto_retry=True)
                api.connect(node["host"], node["port"], time_out=self._timeout)
                logger.info("[TID-%s] TDX 已连接: %s (%s:%s)",
                            threading.get_ident(), node["name"],
                            node["host"], node["port"])
                return api
            except Exception as exc:
                logger.debug("[TID-%s] 连接失败 %s: %s",
                             threading.get_ident(), node["name"], exc)
                try:
                    api.disconnect()
                except Exception:
                    pass
        logger.error("[TID-%s] 所有节点连接失败", threading.get_ident())
        return None

    def get_connection(self) -> Optional["TdxHq_API"]:
        """获取/复用当前线程的 TdxHq_API 连接（含心跳检测）。"""
        api: Optional[TdxHq_API] = getattr(self._local, "api", None)
        if api is not None:
            try:
                api.get_security_count(0)   # 最轻量心跳
            except Exception:
                logger.info("[TID-%s] 连接失效，重建", threading.get_ident())
                self._safe_disconnect(api)
                api = None
        if api is None:
            api = self._create_connection()
            self._local.api = api
        return api

    def release(self) -> None:
        api = getattr(self._local, "api", None)
        if api is not None:
            self._safe_disconnect(api)
            self._local.api = None

    def close_all(self) -> None:
        self.release()

    @staticmethod
    def _safe_disconnect(api: "TdxHq_API") -> None:
        try:
            api.disconnect()
        except Exception:
            pass

    def __enter__(self) -> Optional["TdxHq_API"]:
        return self.get_connection()

    def __exit__(self, *_) -> None:
        pass


_global_pool: Optional[TDXConnectionPool] = None
_pool_lock = threading.Lock()


def get_global_pool(top_nodes: Optional[List[Dict]] = None, timeout: float = 10.0) -> TDXConnectionPool:
    global _global_pool
    if _global_pool is None:
        with _pool_lock:
            if _global_pool is None:
                if top_nodes is None:
                    raise ValueError("首次调用必须提供 top_nodes")
                _global_pool = TDXConnectionPool(top_nodes, timeout=timeout)
    return _global_pool


def reset_global_pool() -> None:
    global _global_pool
    with _pool_lock:
        if _global_pool is not None:
            _global_pool.close_all()
        _global_pool = None