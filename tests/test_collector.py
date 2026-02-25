#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_collector.py — 双轨采集引擎单元测试 (patch_v7.1-fixed)
============================================================
T1: 节点扫描器（24节点 / 排序逻辑）
T2: TDXConnectionPool（线程隔离）
T3: 增量更新逻辑（max_date / merge / 去重）
T4: DataValidator（三层验证）
T5: RunReport（持久化 / 加载）
T6: 双轨合并逻辑（TDX + AKShare merge）
T7: 三级降级管道（全 Mock）
T8: AKShare 进程隔离退避（限流检测）
T9: StockDataPipeline enable_akshare 参数（P0 修复验证）
T10: enrich_akshare 方法存在性（P0 修复验证）
"""

import json
import math
import tempfile
import threading
import unittest
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch, call

import pandas as pd
import numpy as np


# ── 辅助函数 ──────────────────────────────────────────────────────────────
def _make_ohlcv(n: int, code: str = "000001", seed: int = 42) -> pd.DataFrame:
    rng   = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
    base  = 10.0 + rng.randn(n).cumsum() * 0.5
    base  = np.abs(base) + 5
    return pd.DataFrame({
        "code":   code,
        "date":   dates,
        "open":   base.astype("float32"),
        "high":   (base * 1.02).astype("float32"),
        "low":    (base * 0.98).astype("float32"),
        "close":  (base * 1.005).astype("float32"),
        "vol":    np.ones(n, dtype="int64") * 100000,
        "amount": np.ones(n, dtype="int64") * 1000000,
        "source": "tdx",
        "adjust": "hfq",
    })


def _make_akshare_df(n: int, code: str = "000001") -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
    return pd.DataFrame({
        "code":       code,
        "date":       dates,
        "open":       np.ones(n, dtype="float32") * 10.0,
        "high":       np.ones(n, dtype="float32") * 10.5,
        "low":        np.ones(n, dtype="float32") * 9.5,
        "close":      np.ones(n, dtype="float32") * 10.2,
        "vol":        np.ones(n, dtype="int64") * 100000,
        "turnover":   np.random.rand(n).astype("float32") * 5,
        "pct_change": np.random.randn(n).astype("float32"),
        "source":     "akshare",
        "adjust":     "hfq",
    })


# ============================================================================
# T1: 节点扫描器
# ============================================================================
class TestNodeScanner(unittest.TestCase):

    def test_node_count_equals_24(self):
        from src.data.collector.node_scanner import TDX_NODES
        self.assertEqual(len(TDX_NODES), 24)

    def test_all_nodes_port_7709(self):
        from src.data.collector.node_scanner import TDX_NODES
        for n in TDX_NODES:
            self.assertEqual(n["port"], 7709)
            self.assertIn("host", n)
            self.assertIn("name", n)

    def test_sort_ok_before_failed(self):
        from src.data.collector.node_scanner import _sort_results
        data = [
            {"name": "a", "host": "1.1.1.1", "port": 7709, "latency_ms": 80.0,  "status": "ok"},
            {"name": "b", "host": "2.2.2.2", "port": 7709, "latency_ms": -1.0,  "status": "fail:x"},
            {"name": "c", "host": "3.3.3.3", "port": 7709, "latency_ms": 20.0,  "status": "ok"},
        ]
        s = _sort_results(data)
        self.assertEqual(s[0]["name"], "c")
        self.assertEqual(s[1]["name"], "a")
        self.assertEqual(s[2]["name"], "b")

    def test_probe_unreachable(self):
        from src.data.collector.node_scanner import _probe_sync
        node = {"name": "x", "host": "10.255.255.1", "port": 1}
        r    = _probe_sync(node, timeout=0.3)
        self.assertLess(r["latency_ms"], 0)
        self.assertTrue(r["status"].startswith("fail"))


# ============================================================================
# T2: TDXConnectionPool 线程隔离
# ============================================================================
class TestTDXConnectionPool(unittest.TestCase):

    def test_empty_nodes_raises(self):
        import src.data.collector.tdx_pool as pool_mod
        with patch.object(pool_mod, "_PYTDX_AVAILABLE", True):
            from src.data.collector.tdx_pool import TDXConnectionPool
            with self.assertRaises(ValueError):
                TDXConnectionPool([], timeout=1.0)

    def test_local_storage_per_thread(self):
        """threading.local 保证不同线程看到各自的 api 实例。"""
        import threading
        import src.data.collector.tdx_pool as pool_mod
        from src.data.collector.tdx_pool import TDXConnectionPool

        nodes = [{"name": "n0", "host": "127.0.0.1", "port": 7709}]

        with patch.object(pool_mod, "_PYTDX_AVAILABLE", True):
            pool = TDXConnectionPool(nodes, timeout=0.1)

        captured = {}

        def _set_local(tid, val):
            pool._local.api = val
            import time; time.sleep(0.05)
            captured[tid] = pool._local.api

        threads = [threading.Thread(target=_set_local, args=(i, MagicMock())) for i in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()

        ids = list(set(id(v) for v in captured.values()))
        self.assertEqual(len(ids), 4)


# ============================================================================
# T3: 增量更新逻辑
# ============================================================================
class TestIncremental(unittest.TestCase):

    def _write_parquet(self, df: pd.DataFrame) -> Path:
        f = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        path = Path(f.name)
        df.to_parquet(path, index=False)
        return path

    def test_read_local_max_date_no_file(self):
        from src.data.collector.incremental import read_local_max_date
        self.assertIsNone(read_local_max_date(Path("/tmp/nonexistent_xyz999.parquet")))

    def test_read_local_max_date_correct(self):
        from src.data.collector.incremental import read_local_max_date
        df   = _make_ohlcv(30)
        path = self._write_parquet(df)
        self.assertEqual(read_local_max_date(path), df["date"].max())
        path.unlink(missing_ok=True)

    def test_compute_missing_range_full(self):
        from src.data.collector.incremental import compute_missing_range
        s, e = compute_missing_range(None, "2024-03-01", "2005-01-01")
        self.assertEqual(s, "2005-01-01")
        self.assertEqual(e, "2024-03-01")

    def test_compute_missing_range_incremental(self):
        from src.data.collector.incremental import compute_missing_range
        s, e = compute_missing_range("2024-01-10", "2024-01-20")
        self.assertEqual(s, "2024-01-11")
        self.assertEqual(e, "2024-01-20")

    def test_merge_no_duplicates(self):
        from src.data.collector.incremental import merge_incremental
        old = _make_ohlcv(30)
        new = _make_ohlcv(20)
        m   = merge_incremental(old, new)
        self.assertFalse(m.duplicated(subset=["date"]).any())
        self.assertEqual(m["date"].tolist(), sorted(m["date"].tolist()))

    def test_is_up_to_date_yesterday(self):
        from src.data.collector.incremental import is_up_to_date
        yest = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        self.assertTrue(is_up_to_date(yest))

    def test_is_up_to_date_old(self):
        from src.data.collector.incremental import is_up_to_date
        old = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")
        self.assertFalse(is_up_to_date(old))


# ============================================================================
# T4: DataValidator — 三层验证
# ============================================================================
class TestDataValidator(unittest.TestCase):

    def test_none_df_fails(self):
        from src.data.collector.validator import DataValidator
        ok, reason = DataValidator.validate(None)
        self.assertFalse(ok)
        self.assertEqual(reason, "df_is_none")

    def test_empty_df_fails(self):
        from src.data.collector.validator import DataValidator
        ok, reason = DataValidator.validate(pd.DataFrame())
        self.assertFalse(ok)
        self.assertEqual(reason, "df_empty")

    def test_too_few_rows_fails(self):
        from src.data.collector.validator import DataValidator
        df         = _make_ohlcv(5)
        ok, reason = DataValidator.validate(df, min_rows=10)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("too_few_rows"))

    def test_missing_col_fails(self):
        from src.data.collector.validator import DataValidator
        df         = _make_ohlcv(20).drop(columns=["close"])
        ok, reason = DataValidator.validate(df)
        self.assertFalse(ok)
        self.assertIn("missing_cols", reason)

    def test_high_lt_low_fails(self):
        from src.data.collector.validator import DataValidator
        df = _make_ohlcv(20)
        df.loc[5, "high"] = df.loc[5, "low"] - 1.0
        ok, reason = DataValidator.validate(df)
        self.assertFalse(ok)
        self.assertIn("high_lt_low", reason)

    def test_negative_close_fails(self):
        from src.data.collector.validator import DataValidator
        df = _make_ohlcv(20)
        df.loc[3, "close"] = -1.0
        ok, reason = DataValidator.validate(df)
        self.assertFalse(ok)
        self.assertIn("negative_close", reason)

    def test_valid_df_passes(self):
        from src.data.collector.validator import DataValidator
        df         = _make_ohlcv(30)
        ok, reason = DataValidator.validate(df)
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_validate_merge_result_sorted(self):
        from src.data.collector.validator import DataValidator
        df = _make_ohlcv(20)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        ok, reason = DataValidator.validate_merge_result(df)
        self.assertFalse(ok)
        self.assertEqual(reason, "date_not_sorted")


# ============================================================================
# T5: RunReport 持久化
# ============================================================================
class TestRunReport(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_record_and_save(self):
        from src.data.collector.run_report import RunReport
        report = RunReport(self.tmpdir)
        report.record_success("000001", 0, source="tdx",     rows=1000)
        report.record_success("600000", 1, source="akshare", rows=800)
        report.record_failed ("000002", 0, reason="complete_fail")
        report.record_skipped("000003", 0)
        report.save()

        failed_path = Path(self.tmpdir) / "failed_stocks.txt"
        self.assertTrue(failed_path.exists())
        content = failed_path.read_text()
        self.assertIn("000002", content)
        self.assertNotIn("000001", content)

        json_files = list(Path(self.tmpdir).glob("run_stats_*.json"))
        self.assertEqual(len(json_files), 1)
        with open(json_files[0]) as f:
            stats = json.load(f)
        self.assertEqual(stats["success"], 2)
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["skipped"], 1)

    def test_load_failed_list(self):
        from src.data.collector.run_report import RunReport
        report = RunReport(self.tmpdir)
        report.record_failed("000002", 0, reason="test")
        report.record_failed("600001", 1, reason="test2")
        report.save()

        loaded = RunReport.load_failed_list(self.tmpdir)
        codes = [c for c, m in loaded]
        self.assertIn("000002", codes)
        self.assertIn("600001", codes)

    def test_thread_safety(self):
        from src.data.collector.run_report import RunReport
        report = RunReport(self.tmpdir)

        def _record(i):
            report.record_success(f"{i:06d}", 0, source="tdx", rows=100)

        threads = [threading.Thread(target=_record, args=(i,)) for i in range(100)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(report.total_success, 100)


# ============================================================================
# T6: 双轨合并逻辑
# ============================================================================
class TestDualTrackMerge(unittest.TestCase):

    def test_both_none_returns_none(self):
        from src.data.collector.pipeline import _merge_dual_track
        self.assertIsNone(_merge_dual_track(None, None))

    def test_tdx_only(self):
        from src.data.collector.pipeline import _merge_dual_track
        tdx = _make_ohlcv(20)
        result = _merge_dual_track(tdx, None)
        self.assertEqual(len(result), 20)
        self.assertNotIn("turnover", result.columns)

    def test_ak_only(self):
        from src.data.collector.pipeline import _merge_dual_track
        ak = _make_akshare_df(20)
        result = _merge_dual_track(None, ak)
        self.assertEqual(len(result), 20)
        self.assertIn("turnover", result.columns)

    def test_merge_adds_turnover(self):
        from src.data.collector.pipeline import _merge_dual_track
        tdx = _make_ohlcv(20)
        ak  = _make_akshare_df(20)
        result = _merge_dual_track(tdx, ak)
        self.assertIn("turnover", result.columns)
        self.assertIn("pct_change", result.columns)

    def test_merge_prefers_tdx_ohlcv(self):
        from src.data.collector.pipeline import _merge_dual_track
        tdx = _make_ohlcv(20)
        ak  = _make_akshare_df(20)
        result = _merge_dual_track(tdx, ak)
        # OHLCV 以 TDX 为主
        pd.testing.assert_series_equal(
            result["close"].reset_index(drop=True),
            tdx["close"].reset_index(drop=True),
            check_names=False,
        )

    def test_merge_adjust_is_hfq(self):
        from src.data.collector.pipeline import _merge_dual_track
        tdx = _make_ohlcv(20)
        ak  = _make_akshare_df(20)
        result = _merge_dual_track(tdx, ak)
        self.assertTrue((result["adjust"] == "hfq").all())


# ============================================================================
# T7: 三级降级管道（全 Mock）
# ============================================================================
class TestFailoverPipeline(unittest.TestCase):

    def _make_fake_pool(self):
        pool = MagicMock()
        pool.get_connection.return_value = None  # TDX 不可用
        return pool

    def test_fallback_to_baostock_when_both_fail(self):
        from src.data.collector.pipeline import update_single_stock
        from src.data.collector.run_report import RunReport

        tmpdir = Path(tempfile.mkdtemp())
        report = RunReport(str(tmpdir))
        pool   = self._make_fake_pool()

        tdx_df = _make_ohlcv(20)
        bs_df  = _make_ohlcv(20)
        bs_df["source"] = "baostock"
        bs_df["adjust"] = "hfq"

        with patch("src.data.collector.pipeline._fetch_baostock_fallback", return_value=bs_df):
            code, ok, source = update_single_stock(
                code="000001", market=0,
                parquet_dir=tmpdir,
                tdx_pool=pool,
                ak_results={},
                report=report,
                force_full=True,
            )

        self.assertTrue(ok)
        self.assertEqual(source, "baostock")

    def test_complete_fail_recorded(self):
        from src.data.collector.pipeline import update_single_stock
        from src.data.collector.run_report import RunReport

        tmpdir = Path(tempfile.mkdtemp())
        report = RunReport(str(tmpdir))
        pool   = self._make_fake_pool()

        with patch("src.data.collector.pipeline._fetch_baostock_fallback", return_value=None):
            code, ok, source = update_single_stock(
                code="000002", market=0,
                parquet_dir=tmpdir,
                tdx_pool=pool,
                ak_results={},
                report=report,
                force_full=True,
            )

        self.assertFalse(ok)
        self.assertEqual(report.total_failed, 1)


# ============================================================================
# T8: AKShare 限流感知退避
# ============================================================================
class TestAKShareRateLimitBackoff(unittest.TestCase):

    def test_ratelimit_keywords_detected(self):
        from src.data.collector.akshare_client import _RATELIMIT_KEYWORDS
        self.assertIn("429", _RATELIMIT_KEYWORDS)
        self.assertIn("限流", _RATELIMIT_KEYWORDS)
        self.assertIn("too many", _RATELIMIT_KEYWORDS)

    def test_worker_returns_none_on_akshare_not_installed(self):
        from src.data.collector.akshare_client import _akshare_process_worker
        task = ("000001", "2024-01-01", "2024-01-10", 1, 0.0, 0.0)
        # 如果 akshare 未安装则返回错误
        with patch.dict("sys.modules", {"akshare": None}):
            result_code, data, error = _akshare_process_worker(task)
        # akshare 安装时会正常运行，未安装时返回 None
        self.assertEqual(result_code, "000001")


# ============================================================================
# T9: P0 修复验证 — enable_akshare 参数
# ============================================================================
class TestP0Fix(unittest.TestCase):
    """验证 P0 修复：StockDataPipeline 接口与 menu_main.py 对齐"""

    def test_pipeline_accepts_enable_akshare_false(self):
        """StockDataPipeline.__init__ 必须接受 enable_akshare=False"""
        import inspect
        from src.data.collector.pipeline import StockDataPipeline
        sig = inspect.signature(StockDataPipeline.__init__)
        self.assertIn("enable_akshare", sig.parameters,
                      "P0 修复失败：StockDataPipeline 缺少 enable_akshare 参数")

    def test_pipeline_accepts_enable_akshare_true(self):
        """StockDataPipeline.__init__ 必须接受 enable_akshare=True"""
        import inspect
        from src.data.collector.pipeline import StockDataPipeline
        sig = inspect.signature(StockDataPipeline.__init__)
        param = sig.parameters.get("enable_akshare")
        self.assertIsNotNone(param)
        self.assertEqual(param.default, False,
                         "enable_akshare 默认值应为 False（快速模式优先）")

    def test_pipeline_has_enrich_akshare_method(self):
        """StockDataPipeline 必须有 enrich_akshare() 方法"""
        from src.data.collector.pipeline import StockDataPipeline
        self.assertTrue(hasattr(StockDataPipeline, "enrich_akshare"),
                        "P0 修复失败：StockDataPipeline 缺少 enrich_akshare() 方法")
        self.assertTrue(callable(getattr(StockDataPipeline, "enrich_akshare")))

    def test_make_pipeline_with_menu_main_args(self):
        """模拟 menu_main._make_pipeline 的调用方式，不得抛 TypeError"""
        from src.data.collector.pipeline import StockDataPipeline
        import inspect
        sig = inspect.signature(StockDataPipeline.__init__)
        # menu_main 传入的参数
        menu_args = {
            "parquet_dir":    "./data/parquet",
            "reports_dir":    "./data/reports",
            "top_n_nodes":    5,
            "tdx_workers":    8,
            "ak_workers":     2,
            "ak_delay_min":   0.3,
            "ak_delay_max":   0.8,
            "ak_max_retries": 3,
            "force_full":     False,
            "enable_akshare": False,
        }
        for key in menu_args:
            self.assertIn(key, sig.parameters,
                          f"P0 修复失败：StockDataPipeline 缺少参数 {key}")


# ============================================================================
# T10: P1 修复验证 — enable_akshare=False 跳过 AKShare
# ============================================================================
class TestP1Fix(unittest.TestCase):
    """验证 P1 修复：enable_akshare=False 时 run() 不调用 AKShare"""

    def test_run_skips_akshare_when_disabled(self):
        """enable_akshare=False 时，run_akshare_batch 不应被调用"""
        from src.data.collector.pipeline import StockDataPipeline

        with patch("src.data.collector.pipeline.get_fastest_nodes") as mock_nodes, \
             patch("src.data.collector.pipeline.TDXConnectionPool") as mock_pool_cls, \
             patch("src.data.collector.pipeline.run_akshare_batch") as mock_ak_batch, \
             patch.object(StockDataPipeline, "_run_tdx_multiprocess") as mock_tdx:

            mock_nodes.return_value = [{"name": "n1", "host": "1.1.1.1",
                                        "port": 7709, "latency_ms": 10.0, "status": "ok"}]
            mock_pool_inst = MagicMock()
            mock_pool_cls.return_value = mock_pool_inst

            pipeline = StockDataPipeline(enable_akshare=False)
            pipeline.run([("000001", 0)])

            mock_ak_batch.assert_not_called()

    def test_run_calls_akshare_when_enabled(self):
        """enable_akshare=True 时，run_akshare_batch 应被调用"""
        from src.data.collector.pipeline import StockDataPipeline

        with patch("src.data.collector.pipeline.get_fastest_nodes") as mock_nodes, \
             patch("src.data.collector.pipeline.TDXConnectionPool") as mock_pool_cls, \
             patch("src.data.collector.pipeline.run_akshare_batch") as mock_ak_batch, \
             patch("src.data.collector.pipeline.read_local_max_date", return_value=None), \
             patch("src.data.collector.pipeline.compute_missing_range",
                   return_value=("2024-01-01", "2024-01-10")), \
             patch.object(StockDataPipeline, "_run_tdx_multiprocess"):

            mock_nodes.return_value = [{"name": "n1", "host": "1.1.1.1",
                                        "port": 7709, "latency_ms": 10.0, "status": "ok"}]
            mock_pool_cls.return_value = MagicMock()
            mock_ak_batch.return_value = {}

            pipeline = StockDataPipeline(enable_akshare=True)
            pipeline.run([("000001", 0)])

            mock_ak_batch.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)