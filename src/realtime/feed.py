# -*- coding: utf-8 -*-
"""
src/realtime/feed.py -- TDX real-time price feed (V7.5)

V7.5 enhancements:
  - Dynamic TDX node discovery via node_scanner.get_fastest_nodes()
  - Support for tdx_node_list in config (manual node override)
  - Configurable: use_node_scanner, tdx_top_n, tdx_node_list
  - Logging: connection success/failure and node selection (INFO)
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Setup realtime file logger (logs/realtime.log)
_rt_file_handler_added = False


def _ensure_realtime_logger():
    global _rt_file_handler_added
    if _rt_file_handler_added:
        return
    rt_logger = logging.getLogger("realtime")
    if not any(isinstance(h, logging.FileHandler) for h in rt_logger.handlers):
        try:
            Path("logs").mkdir(exist_ok=True)
            fh = logging.FileHandler("logs/realtime.log", encoding="utf-8")
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            rt_logger.addHandler(fh)
            rt_logger.setLevel(logging.INFO)
        except Exception:
            pass
    _rt_file_handler_added = True


def _code_to_tdx(code: str):
    c = code.strip().zfill(6)
    if c.startswith(("60", "68", "900")):
        return 1, c   # Shanghai
    return 0, c       # Shenzhen / default


# Default hardcoded fallback nodes
_HARDCODED_NODES = [
    ("119.147.212.81", 7709),
    ("119.147.212.83", 7709),
    ("218.108.98.244", 7709),
    ("221.194.181.176", 7709),
    ("10.0.3.5", 7709),
]


class RealtimeFeed:
    """
    TDX-based real-time price polling feed. (V7.5)

    Config path: realtime.feed.*
      enabled:          bool   (default True)
      interval_seconds: int    (default 3)
      source:           str    (default "tdx")
      batch_size:       int    (default 80)
      use_node_scanner: bool   (default True)
      tdx_top_n:        int    (default 5)
      tdx_node_list:    list   (default []) -- manual [{host, port}, ...]
    """

    def __init__(self, config=None, codes=None):
        _ensure_realtime_logger()
        self.config = config or {}
        feed_cfg = self.config.get("realtime", {}).get("feed", {})
        self.interval = int(feed_cfg.get("interval_seconds", 3))
        self.batch_size = int(feed_cfg.get("batch_size", 80))
        self.source = feed_cfg.get("source", "tdx")

        # V7.5: node discovery config
        self.use_node_scanner = bool(feed_cfg.get("use_node_scanner", True))
        self.tdx_top_n = int(feed_cfg.get("tdx_top_n", 5))
        raw_node_list = feed_cfg.get("tdx_node_list", [])
        if raw_node_list and isinstance(raw_node_list, list):
            self._nodes = [(n["host"], int(n["port"])) for n in raw_node_list
                           if "host" in n and "port" in n]
        else:
            self._nodes = []

        self._codes = list(codes or [])
        self._lock = threading.Lock()
        self._prices = {}
        self._quotes = {}
        self._running = False
        self._thread = None
        self._api = None
        self._last_error = None

    def set_codes(self, codes):
        with self._lock:
            self._codes = list(codes)

    def add_codes(self, codes):
        with self._lock:
            existing = set(self._codes)
            for c in codes:
                if c not in existing:
                    self._codes.append(c)
                    existing.add(c)

    def start(self):
        if self._running:
            return
        self._running = True
        self._connect()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="realtime-feed")
        self._thread.start()
        logger.info("RealtimeFeed started (interval=%ds, source=%s)",
                    self.interval, self.source)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self._disconnect()
        logger.info("RealtimeFeed stopped")

    def is_running(self):
        return self._running and self._thread is not None and self._thread.is_alive()

    def get_price(self, code):
        with self._lock:
            return self._prices.get(code)

    def get_all_prices(self):
        with self._lock:
            return dict(self._prices)

    def get_quote(self, code):
        with self._lock:
            return self._quotes.get(code)

    def get_last_error(self):
        return self._last_error

    def _resolve_nodes(self):
        """
        V7.5: Resolve node list in priority order:
          1. tdx_node_list (manual config)
          2. node_scanner dynamic discovery
          3. hardcoded fallback
        """
        if self._nodes:
            logger.info("RealtimeFeed using manual tdx_node_list (%d nodes)", len(self._nodes))
            return self._nodes

        if self.use_node_scanner:
            try:
                from src.data.collector.node_scanner import get_fastest_nodes
                scanned = get_fastest_nodes(top_n=self.tdx_top_n, timeout=3.0)
                if scanned:
                    nodes = [(n["host"], n["port"]) for n in scanned
                             if n.get("status") == "ok"]
                    if nodes:
                        logger.info(
                            "RealtimeFeed node_scanner found %d nodes (top: %s:%d)",
                            len(nodes), nodes[0][0], nodes[0][1])
                        return nodes
                logger.warning("RealtimeFeed node_scanner returned no OK nodes, using fallback")
            except Exception as e:
                logger.warning("RealtimeFeed node_scanner failed: %s, using fallback", e)

        logger.info("RealtimeFeed using hardcoded node list (%d nodes)", len(_HARDCODED_NODES))
        return list(_HARDCODED_NODES)

    def _connect(self):
        if self.source != "tdx":
            return False
        try:
            from pytdx.hq import TdxHq_API
            self._api = TdxHq_API(raise_exception=False)
            nodes = self._resolve_nodes()
            for host, port in nodes:
                try:
                    t0 = time.time()
                    result = self._api.connect(host, port)
                    latency_ms = (time.time() - t0) * 1000
                    if result:
                        logger.info(
                            "RealtimeFeed connected to %s:%d (latency=%.1fms)",
                            host, port, latency_ms)
                        return True
                    else:
                        logger.debug("RealtimeFeed connect failed for %s:%d", host, port)
                except Exception as ex:
                    logger.debug("RealtimeFeed connect error %s:%d: %s", host, port, ex)
            logger.warning("RealtimeFeed could not connect to any TDX node")
            self._api = None
            return False
        except ImportError:
            logger.warning("pytdx not installed; RealtimeFeed running in simulation mode")
            self._api = None
            return False

    def _disconnect(self):
        if self._api is not None:
            try:
                self._api.disconnect()
            except Exception:
                pass
            self._api = None

    def _poll_loop(self):
        fail_count = 0
        while self._running:
            try:
                with self._lock:
                    codes_snapshot = list(self._codes)
                if codes_snapshot:
                    self._fetch_quotes(codes_snapshot)
                    fail_count = 0
            except Exception as e:
                fail_count += 1
                self._last_error = str(e)
                logger.warning("RealtimeFeed poll error (#%d): %s", fail_count, e)
                if fail_count >= 3:
                    self._disconnect()
                    self._connect()
                    fail_count = 0
            time.sleep(self.interval)

    def _fetch_quotes(self, codes):
        if self._api is None:
            return
        for batch_start in range(0, len(codes), self.batch_size):
            batch = codes[batch_start: batch_start + self.batch_size]
            params = [_code_to_tdx(c) for c in batch]
            try:
                data = self._api.get_security_quotes(params)
                if not data:
                    continue
                with self._lock:
                    for i, item in enumerate(data):
                        if i >= len(batch):
                            break
                        code = batch[i]
                        if not isinstance(item, dict):
                            continue
                        price = item.get("price", 0.0)
                        if price and price > 0:
                            self._prices[code] = float(price)
                            self._quotes[code] = {
                                "price":     float(price),
                                "open":      float(item.get("open", 0)),
                                "high":      float(item.get("high", 0)),
                                "low":       float(item.get("low", 0)),
                                "last_close": float(item.get("last_close", 0)),
                                "vol":       int(item.get("vol", 0)),
                                "amount":    float(item.get("amount", 0)),
                                "ask1":      float(item.get("ask1", 0)),
                                "bid1":      float(item.get("bid1", 0)),
                                "ask1_vol":  int(item.get("ask1_vol", 0)),
                                "bid1_vol":  int(item.get("bid1_vol", 0)),
                                "ts":        float(item.get("active2", 0)),
                            }
            except Exception as e:
                logger.debug("Batch fetch error: %s", e)

    def fetch_once(self, codes):
        """Connect, fetch, disconnect in one call. Returns {code: price}."""
        connected = self._connect()
        if not connected:
            return {}
        try:
            self._fetch_quotes(codes)
            return self.get_all_prices()
        finally:
            self._disconnect()