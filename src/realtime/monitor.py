# -*- coding: utf-8 -*-
"""
src/realtime/monitor.py -- Real-time monitor engine (V7.4)

V7.4 enhancements:
  - Integrates RealtimeFeed (TDX live price polling)
  - Multi-strategy scanning with configurable merge rules (any/majority/weighted)
  - Real-time risk control check via trader.update_prices
  - Loads active_strategies and strategy_params from config
  - Falls back to MA cross strategy when active_strategies is empty
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _setup_realtime_log_handler():
    """V7.5: Setup shared realtime.log file handler"""
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


class MonitorEngine:
    """
    Real-time monitor engine (V7.4)

    Workflow per scan:
      1. get prices from RealtimeFeed (or fallback to last close)
      2. for each code in universe:
         a. load historical parquet data
         b. call each active strategy's generate_realtime_signal(code, df, price)
         c. merge signals per merge_rule
         d. if merged signal != hold: send alert (single or merged)
         e. if auto_execute: call trader.buy / trader.sell
      3. call trader.update_prices for risk control
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 alerter=None, trader=None):
        self.config = config or {}
        rt = self.config.get("realtime", {})

        self.scan_interval: int = rt.get("scan_interval_seconds", 300)
        self.universe_mode: str = rt.get("universe", rt.get("scan_universe", "all"))
        self.watchlist: List[str] = rt.get("watchlist", [])
        self.auto_execute: bool = rt.get("auto_execute", False)
        self.min_score: float = rt.get("trading", {}).get("min_signal_score", 0.5)
        self.data_dir: str = self.config.get("data", {}).get("parquet_dir", "data/parquet")

        # V7.4: multi-strategy config
        self.active_strategies: List[str] = rt.get("active_strategies", [])
        self.signal_merge_rule: str = rt.get("signal_merge_rule", "any")
        self.strategy_params: Dict[str, Dict] = rt.get("strategy_params", {})

        self.alerter = alerter
        self.trader = trader

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._recent_signals: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._scan_count = 0

        # V7.5: LRU cache for parquet files
        self._parquet_cache = OrderedDict()   # code -> (df, timestamp)
        self._cache_max_size = 100
        self._cache_ttl = 300.0  # seconds
        self._cache_lock = threading.Lock()

        # Setup realtime.log
        _setup_realtime_log_handler()

        # V7.4: feed
        self._feed: Optional[Any] = None
        feed_cfg = rt.get("feed", {})
        if feed_cfg.get("enabled", True):
            try:
                from .feed import RealtimeFeed
                self._feed = RealtimeFeed(config=self.config)
                logger.info("RealtimeFeed initialized")
            except Exception as e:
                logger.warning("RealtimeFeed init failed (will use close price): %s", e)

        # V7.4: load strategy instances
        self._strategies: Dict[str, Any] = {}
        self._load_strategies()

    # ------------------------------------------------------------------
    def _load_strategies(self) -> None:
        """Load configured strategy instances"""
        try:
            from src.strategy.strategies import STRATEGY_REGISTRY
        except ImportError:
            try:
                from ..strategy.strategies import STRATEGY_REGISTRY
            except ImportError:
                logger.warning("Cannot import STRATEGY_REGISTRY")
                return

        targets = self.active_strategies if self.active_strategies else ["rsrs_momentum"]
        for name in targets:
            if name not in STRATEGY_REGISTRY:
                logger.warning("Unknown strategy: %s", name)
                continue
            params = self.strategy_params.get(name, {})
            try:
                self._strategies[name] = STRATEGY_REGISTRY[name](config=params)
                logger.info("Loaded strategy: %s with params %s", name, params)
            except Exception as e:
                logger.warning("Failed to load strategy %s: %s", name, e)

    def reload_strategies(self) -> None:
        """Reload strategies (call after changing active_strategies/strategy_params)"""
        self._strategies.clear()
        self._load_strategies()

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            logger.warning("MonitorEngine already running")
            return
        self._running = True
        if self._feed is not None:
            try:
                self._feed.start()
            except Exception as e:
                logger.warning("Feed start failed: %s", e)
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="monitor-engine")
        self._thread.start()
        logger.info("MonitorEngine started (interval=%ds, universe=%s, strategies=%s)",
                    self.scan_interval, self.universe_mode,
                    list(self._strategies.keys()))

    def stop(self) -> None:
        self._running = False
        if self._feed is not None:
            try:
                self._feed.stop()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("MonitorEngine stopped")

    def is_running(self) -> bool:
        return self._running and (self._thread is not None) and self._thread.is_alive()

    def get_recent_signals(self, n: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._recent_signals[-n:])

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self._running:
            try:
                self.scan_once()
            except Exception as e:
                logger.error("Monitor scan error: %s", e, exc_info=True)
            for _ in range(self.scan_interval * 10):
                if not self._running:
                    break
                time.sleep(0.1)

    def scan_once(self) -> List[Dict[str, Any]]:
        """Execute one full scan. Returns list of triggered signals."""
        self._scan_count += 1
        logger.info("Scan #%d (universe=%s, strategies=%s, rule=%s)",
                    self._scan_count, self.universe_mode,
                    list(self._strategies.keys()), self.signal_merge_rule)

        codes = self._load_universe()
        if not codes:
            logger.warning("Empty universe, skipping scan")
            return []

        signals = []
        price_map: Dict[str, float] = {}

        for code in codes:
            try:
                result, price = self._scan_code(code)
                if result:
                    signals.append(result)
                if price > 0:
                    price_map[code] = price
            except Exception as e:
                logger.debug("Scan %s error: %s", code, e)

        # V7.4: risk control update
        if self.trader and price_map:
            try:
                triggered = self.trader.update_prices(price_map)
                for t in triggered:
                    logger.info("Risk triggered: %s %s pnl=%.1f%%",
                                t["code"], t["event"], t.get("pnl_pct", 0) * 100)
                    if self.alerter:
                        self.alerter.send_position_alert(
                            t["code"], t["code"], t["event"],
                            t.get("pnl_pct", 0), t.get("price", 0)
                        )
                    if self.auto_execute and self.trader:
                        self.trader.sell(t["code"], t.get("price", 0))
            except Exception as e:
                logger.warning("Risk update error: %s", e)

        logger.info("Scan #%d done: %d signals from %d codes",
                    self._scan_count, len(signals), len(codes))

        with self._lock:
            self._recent_signals.extend(signals)
            if len(self._recent_signals) > 500:
                self._recent_signals = self._recent_signals[-500:]

        return signals

    def _load_universe(self) -> List[str]:
        if self.universe_mode == "watchlist":
            return list(self.watchlist)
        p = Path(self.data_dir)
        if not p.exists():
            return []
        return [f.stem for f in p.glob("*.parquet")]

    def _get_price(self, code: str, df) -> float:
        """Get realtime price from feed or fallback to last close"""
        if self._feed is not None:
            price = self._feed.get_price(code)
            if price and price > 0:
                return price
        # fallback: last close from df
        if df is not None and "close" in df.columns and len(df) > 0:
            return float(df["close"].iloc[-1])
        return 0.0

    def _scan_code(self, code: str) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Scan a single stock. Returns (signal_dict_or_None, current_price).
        """
        try:
            df = self._get_cached_parquet(code)
            if df is None:
                return None, 0.0
        except Exception as e:
            logger.debug("Failed to read %s: %s", code, e)
            return None, 0.0

        if len(df) < 30:
            return None, 0.0

        current_price = self._get_price(code, df)
        if current_price <= 0:
            return None, 0.0

        # If no strategies loaded, use simple MA fallback
        if not self._strategies:
            signal, score = self._fallback_strategy(df, current_price)
            if signal == "hold" or score < self.min_score:
                return None, current_price
            name = code
            result = {
                "code": code, "name": name, "signal": signal,
                "score": score, "price": current_price,
                "strategy": "ma_fallback",
                "strategies_triggered": ["ma_fallback"],
                "merge_rule": "fallback",
                "ts": time.time(),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reason": "MA fallback signal",
            }
            if self.alerter:
                self.alerter.send_signal_alert(
                    code, name, signal, score, "ma_fallback", current_price)
            if self.auto_execute and self.trader:
                if signal == "buy":
                    self.trader.buy(code, name, current_price)
                elif signal == "sell":
                    self.trader.sell(code, current_price)
            return result, current_price

        # Multi-strategy scan
        strategy_results: List[Tuple[str, str, float, str]] = []
        for strat_name, strat_obj in self._strategies.items():
            try:
                sig, score, reason = strat_obj.generate_realtime_signal(
                    code, df, current_price)
                if sig != "hold" and score >= self.min_score:
                    strategy_results.append((strat_name, sig, score, reason))
            except Exception as e:
                logger.debug("Strategy %s failed for %s: %s", strat_name, code, e)

        if not strategy_results:
            return None, current_price

        # Merge signals
        merged_signal, merged_score, triggered = self._merge_signals(strategy_results)
        if merged_signal == "hold" or not triggered:
            return None, current_price

        name = code
        strat_names = [t[0] for t in triggered]
        reasons = [t[3] for t in triggered]
        result = {
            "code": code,
            "name": name,
            "signal": merged_signal,
            "score": merged_score,
            "price": current_price,
            "strategy": "+".join(strat_names),
            "strategies_triggered": strat_names,
            "merge_rule": self.signal_merge_rule,
            "ts": time.time(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": " | ".join(reasons[:3]),
        }

        # Send alert
        if self.alerter:
            if len(triggered) == 1:
                self.alerter.send_signal_alert(
                    code, name, merged_signal, merged_score,
                    triggered[0][0], current_price, triggered[0][3])
            else:
                self.alerter.send_merged_signal_alert(
                    code, name, merged_signal, merged_score,
                    current_price, strat_names, self.signal_merge_rule)

        # Auto execute
        if self.auto_execute and self.trader:
            if merged_signal == "buy":
                self.trader.buy(code, name, current_price)
            elif merged_signal == "sell":
                self.trader.sell(code, current_price)

        return result, current_price

    def _merge_signals(
        self,
        results: List[Tuple[str, str, float, str]],
    ) -> Tuple[str, float, List]:
        """
        Merge multi-strategy signals.
        Returns (final_signal, final_score, triggered_list)
        triggered_list: list of (strat_name, signal, score, reason) that contributed
        """
        if not results:
            return "hold", 0.0, []

        rule = self.signal_merge_rule

        buy_results  = [(n, s, sc, r) for n, s, sc, r in results if s == "buy"]
        sell_results = [(n, s, sc, r) for n, s, sc, r in results if s == "sell"]

        if rule == "any":
            # Any strategy triggers -> alert
            if buy_results and not sell_results:
                best = max(buy_results, key=lambda x: x[2])
                return "buy", best[2], buy_results
            if sell_results and not buy_results:
                best = max(sell_results, key=lambda x: x[2])
                return "sell", best[2], sell_results
            if buy_results and sell_results:
                # conflicting: pick the higher score side
                buy_score  = max(x[2] for x in buy_results)
                sell_score = max(x[2] for x in sell_results)
                if buy_score >= sell_score:
                    return "buy", buy_score, buy_results
                return "sell", sell_score, sell_results
            return "hold", 0.0, []

        elif rule == "majority":
            # V7.5 fix: use len(results) as denominator (strategies that returned signals)
            # not len(self._strategies) (total enabled strategies, some may return hold)
            total = len(results)
            if not total:
                return "hold", 0.0, []
            buy_ratio  = len(buy_results)  / total
            sell_ratio = len(sell_results) / total
            if buy_ratio > 0.5:
                avg_score = sum(x[2] for x in buy_results) / len(buy_results)
                return "buy", avg_score, buy_results
            if sell_ratio > 0.5:
                avg_score = sum(x[2] for x in sell_results) / len(sell_results)
                return "sell", avg_score, sell_results
            return "hold", 0.0, []

        elif rule == "weighted":
            # Weighted average by individual scores
            buy_wt  = sum(x[2] for x in buy_results)
            sell_wt = sum(x[2] for x in sell_results)
            if buy_wt > sell_wt and buy_wt > self.min_score:
                norm = buy_wt / max(len(buy_results), 1)
                return "buy", min(norm, 1.0), buy_results
            if sell_wt > buy_wt and sell_wt > self.min_score:
                norm = sell_wt / max(len(sell_results), 1)
                return "sell", min(norm, 1.0), sell_results
            return "hold", 0.0, []

        # Default: same as "any"
        if buy_results:
            return "buy", max(x[2] for x in buy_results), buy_results
        if sell_results:
            return "sell", max(x[2] for x in sell_results), sell_results
        return "hold", 0.0, []

    def _get_cached_parquet(self, code: str):
        """V7.5: LRU cache for parquet files. Returns DataFrame or None."""
        import pandas as pd
        now = time.time()
        with self._cache_lock:
            if code in self._parquet_cache:
                df, ts = self._parquet_cache[code]
                if now - ts < self._cache_ttl:
                    # Cache hit: move to end (most recently used)
                    self._parquet_cache.move_to_end(code)
                    return df
                else:
                    # Expired: remove from cache
                    del self._parquet_cache[code]
            # Cache miss: load from disk
            path = Path(self.data_dir) / (code + ".parquet")
            if not path.exists():
                return None
            try:
                df = pd.read_parquet(path)
                self._parquet_cache[code] = (df, now)
                # Evict LRU if over max size
                while len(self._parquet_cache) > self._cache_max_size:
                    self._parquet_cache.popitem(last=False)
                return df
            except Exception as e:
                logger.debug("Parquet read error %s: %s", code, e)
                return None

    def _fallback_strategy(self, df, current_price: float):
        """Simple MA cross fallback when no strategies are loaded"""
        try:
            import numpy as np
            close = df["close"].values
            if len(close) < 60 or close[-20] <= 0:
                return "hold", 0.0
            ma5  = float(np.mean(close[-5:]))
            ma20 = float(np.mean(close[-20:]))
            ma60 = float(np.mean(close[-60:]))
            if current_price > ma5 > ma20 > ma60:
                score = min((current_price / ma60 - 1) * 10, 1.0)
                return "buy", score
            elif current_price < ma5 < ma20 < ma60:
                score = min((1 - current_price / ma60) * 10, 1.0)
                return "sell", score
            return "hold", 0.0
        except Exception:
            return "hold", 0.0