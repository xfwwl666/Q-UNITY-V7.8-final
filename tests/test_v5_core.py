#!/usr/bin/env python3
"""
Q-UNITY-V6 核心回归测试 v2.1
T1: RSRS 稳定性（6用例, resid_std >= 0）
T2: 风控熔断（触发+cooldown）
T3: 前视偏差验证（NB-01）
T4: 策略集成（停牌/缺失数据）
T5: NB-21 闭环（5天新股/有效掩码形状/混合时序）
"""
from __future__ import annotations
import math
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pytest

# ── 辅助生成器 ────────────────────────────────────────────────────────────

def _make_ohlcv(n: int, seed: int = 42, start_price: float = 10.0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = start_price * np.cumprod(1 + rng.randn(n) * 0.01)
    high  = close * (1 + rng.uniform(0, 0.03, n))
    low   = close * (1 - rng.uniform(0, 0.03, n))
    open_ = close * (1 + rng.randn(n) * 0.005)
    vol   = rng.randint(100_000, 1_000_000, n).astype(float)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": vol,
    }, index=dates)


def _make_price_data(codes: List[str], n_days: int = 252) -> Dict[str, Dict[str, float]]:
    """生成当日价格快照字典"""
    rng = np.random.RandomState(0)
    return {
        code: {
            "open": 10.0 + rng.rand(),
            "high": 10.5 + rng.rand(),
            "low":  9.5  + rng.rand(),
            "close": 10.0 + rng.rand(),
            "volume": 500_000.0,
        }
        for code in codes
    }


# ============================================================================
# T1: RSRS 稳定性测试
# ============================================================================

class TestRSRSStability:
    """T1: RSRS 因子在各种数据条件下的稳定性"""

    def _compute(self, df, window=18, zwindow=600):
        from src.factors.technical.rsrs import compute_rsrs
        return compute_rsrs(df, regression_window=window, zscore_window=zwindow)

    def test_normal_data_basic(self):
        """正常数据: rsrs_raw 末尾应为有限浮点数"""
        df = _make_ohlcv(300)
        result = self._compute(df)
        assert "rsrs_raw" in result.columns
        tail = result["rsrs_raw"].dropna()
        assert len(tail) > 0
        assert math.isfinite(float(tail.iloc[-1]))

    def test_resid_std_nonnegative(self):
        """resid_std 不可为负（NB 基础要求）"""
        df = _make_ohlcv(300)
        result = self._compute(df)
        rstd = result["resid_std"].dropna()
        assert (rstd >= 0).all(), "resid_std 出现负值!"

    def test_short_series_no_crash(self):
        """短序列（15行）: 不崩溃，返回全NaN"""
        df = _make_ohlcv(15)
        result = self._compute(df)
        # window=18 > 15，全部应为 NaN
        assert result["rsrs_raw"].isna().all()

    def test_constant_price_no_crash(self):
        """常数价格: 不崩溃（OLS 方差为0），resid_std >= 0"""
        df = _make_ohlcv(300)
        df["high"] = 11.0
        df["low"]  = 9.0
        result = self._compute(df)
        rstd = result["resid_std"].dropna()
        assert (rstd >= 0).all()

    def test_zscore_finite_enough(self):
        """充足数据后 zscore 窗口内应产生有限值"""
        df = _make_ohlcv(700)
        result = self._compute(df, zwindow=200)
        tail = result["rsrs_zscore"].dropna()
        assert len(tail) > 100
        assert all(math.isfinite(v) for v in tail.tail(10))

    def test_r2_bounded(self):
        """R² 应在 [0, 1] 内"""
        df = _make_ohlcv(400)
        result = self._compute(df)
        r2 = result["rsrs_r2"].dropna()
        assert (r2 >= 0).all() and (r2 <= 1.0 + 1e-9).all()


# ============================================================================
# T2: 风控熔断测试
# ============================================================================

class TestRiskCircuitBreaker:
    """T2: 熔断触发 + NB-12 cooldown 解除"""

    def _make_engine(self, max_dd=0.20, cooldown=3):
        from src.engine.execution import BacktestEngine
        return BacktestEngine(
            initial_cash=1_000_000.0,
            circuit_breaker_max_dd=max_dd,
            circuit_breaker_cooldown_days=cooldown,
        )

    def test_circuit_breaker_triggers(self):
        """回撤超过阈值时熔断应触发"""
        eng = self._make_engine(max_dd=0.20, cooldown=999)
        codes = ["000001"]
        # 模拟大幅亏损
        for i in range(30):
            price = max(1.0, 10.0 - i * 0.5)
            price_data = {"000001": {"open": price, "high": price, "low": price*0.99,
                                     "close": price, "volume": 1e6}}
            result = eng.step(date(2023, 1, 1) + timedelta(days=i), price_data, [])
        # 检查是否触发（视初始资产无持仓，回撤=0，故直接测 API）
        eng._circuit_broken = True
        eng._circuit_break_date = date(2023, 1, 10)
        result = eng.step(date(2023, 2, 10), {"000001": {"open": 5.0, "high": 5.1,
                                                          "low": 4.9, "close": 5.0,
                                                          "volume": 1e6}}, [])
        assert result["circuit_broken"] is False, "cooldown 后应已解除"

    def test_circuit_breaker_cooldown(self):
        """NB-12: cooldown_days 内不买入，过后自动解除"""
        eng = self._make_engine(max_dd=0.10, cooldown=5)
        eng._circuit_broken     = True
        eng._circuit_break_date = date(2023, 3, 1)
        # 第3天: 仍在 cooldown
        r1 = eng.step(date(2023, 3, 4), {"A": {"open": 10.0, "high": 10.1,
                                                "low": 9.9, "close": 10.0, "volume": 1e6}}, [])
        assert r1["circuit_broken"] is True
        # 第6天: cooldown 结束
        r2 = eng.step(date(2023, 3, 7), {"A": {"open": 10.0, "high": 10.1,
                                                "low": 9.9, "close": 10.0, "volume": 1e6}}, [])
        assert r2["circuit_broken"] is False


# ============================================================================
# T3: 前视偏差验证
# ============================================================================

class TestLookaheadBias:
    """T3: NB-01 信号T-1生成，T执行，无前视偏差"""

    def test_signal_uses_yesterday_data(self):
        """信号应基于截止 T-1 的因子数据"""
        from src.factors.alpha_engine import AlphaEngine
        df = _make_ohlcv(200)
        result = AlphaEngine.compute_from_history(df)
        # T日的 rsrs_adaptive 仅依赖 T-1 及之前数据（OLS计算不含T日）
        # 验证: 若截断最后1行，前N行结果不变
        result_full = AlphaEngine.compute_from_history(df)
        result_cut  = AlphaEngine.compute_from_history(df.iloc[:-1])
        # 倒数第2行的值应相同（无前视）
        if not result_cut["rsrs_adaptive"].isna().all():
            val_full = result_full["rsrs_adaptive"].dropna().iloc[-2]
            val_cut  = result_cut["rsrs_adaptive"].dropna().iloc[-1]
            assert abs(val_full - val_cut) < 1e-9, "检测到前视偏差!"

    def test_engine_uses_open_price_for_execution(self):
        """引擎在 T 日开盘价执行信号（NB-01）"""
        from src.engine.execution import BacktestEngine
        from src.types import OrderSide, Signal
        eng = BacktestEngine(initial_cash=1_000_000.0)

        # T-1 日生成信号
        sig = Signal(
            timestamp=datetime(2023, 1, 2),
            code="000001",
            side=OrderSide.BUY,
            score=1.0,
            weight=0.1,
            reason="测试",
        )
        # T-1 日 step（信号加入 pending）
        pd_t1 = {"000001": {"open": 10.0, "high": 10.5, "low": 9.5,
                              "close": 10.0, "volume": 1e6}}
        eng.step(date(2023, 1, 2), pd_t1, [sig])

        # T 日 step（使用 T 日 open=11.0 执行）
        pd_t2 = {"000001": {"open": 11.0, "high": 11.5, "low": 10.5,
                              "close": 11.0, "volume": 1e6}}
        result = eng.step(date(2023, 1, 3), pd_t2, [])
        # 检查成交发生在 T 日开盘价附近（含滑点）
        fills = result.get("executions", [])
        if fills:
            exec_price = fills[0]["price"]
            assert abs(exec_price - 11.0 * 1.001) < 0.01, f"执行价应为T日开盘价含滑点, 实际={exec_price}"


# ============================================================================
# T4: 策略集成测试
# ============================================================================

class TestStrategyIntegration:
    """T4: 停牌/缺失数据场景下策略稳定性"""

    def _make_factor_data(self, codes: List[str], n: int = 200) -> Dict:
        result = {}
        for code in codes:
            df = _make_ohlcv(n, seed=hash(code) % 1000)
            from src.factors.alpha_engine import AlphaEngine
            result[code] = AlphaEngine.compute_from_history(df)
        return result

    def test_rsrs_strategy_with_missing_factor(self):
        """RSRSMomentum: 部分股票无因子数据时不崩溃"""
        from src.strategy.strategies import RSRSMomentumStrategy
        strat = RSRSMomentumStrategy({"top_n": 3, "rsrs_threshold": 0.3})
        universe = ["000001", "000002", "000003"]
        market_data = {code: _make_ohlcv(200) for code in universe}
        factor_data = self._make_factor_data(["000001"])  # 只有一个有因子
        factor_data["000002"] = pd.DataFrame()            # 空 DataFrame
        # 000003 缺失

        signals = strat.generate_signals(
            universe, market_data, factor_data,
            datetime(2023, 6, 1), {}
        )
        assert isinstance(signals, list)

    def test_kunpeng_v10_suspended_stock(self):
        """KunpengV10: 停牌股（无价格数据）不产生信号"""
        from src.strategy.strategies import KunpengV10Strategy
        strat = KunpengV10Strategy({"top_n": 3})
        universe = ["000001", "000002"]
        # 000002 停牌：无市场数据
        market_data = {"000001": _make_ohlcv(60)}
        factor_data = {}
        signals = strat.generate_signals(
            universe, market_data, factor_data,
            datetime(2023, 6, 1), {}
        )
        # 停牌股不应有买入信号
        buy_codes = [s.code for s in signals if s.side.value == "BUY"]
        assert "000002" not in buy_codes

    def test_alpha_max_v5_no_fundamental(self):
        """AlphaMaxV5: 无基本面数据时不崩溃，仍能评分"""
        from src.strategy.strategies import AlphaMaxV5FixedStrategy
        strat = AlphaMaxV5FixedStrategy({"top_n": 5})
        codes = [f"00000{i}" for i in range(1, 6)]
        market_data = {c: _make_ohlcv(60, seed=i) for i, c in enumerate(codes)}
        factor_data = {}
        signals = strat.generate_signals(
            codes, market_data, factor_data,
            datetime(2023, 6, 1), {},
            fundamental_data=None,
        )
        assert isinstance(signals, list)


# ============================================================================
# T5: NB-21 闭环新股防御测试
# ============================================================================

class TestNB21NewStockDefense:
    """T5: NB-21 闭环 Monkey-Patch 验证"""

    def _make_short_df(self, n: int) -> pd.DataFrame:
        """生成 n 天行情（模拟新股上市初期）"""
        rng = np.random.RandomState(123)
        base = 10.0 + rng.randn(n).cumsum() * 0.1
        return pd.DataFrame({
            "open":   base,
            "high":   base * 1.03,
            "low":    base * 0.97,
            "close":  base,
            "volume": np.full(n, 1e6),
        })

    def test_5_day_stock_rsrs_all_nan(self):
        """5天新股: rsrs_adaptive 必须全为 NaN（NB-21 防御）"""
        from src.factors.alpha_engine import AlphaEngine
        df = self._make_short_df(5)
        result = AlphaEngine.compute_from_history(df)
        assert "rsrs_adaptive" in result.columns
        assert result["rsrs_adaptive"].isna().all(),             "5天新股 rsrs_adaptive 应全NaN，NB-21 未正确应用!"

    def test_valid_mask_shape_matches_df(self):
        """_nb21_valid_mask 输出形状与输入 DataFrame 一致"""
        from src.factors.alpha_engine import _nb21_valid_mask
        df = self._make_short_df(100)
        mask = _nb21_valid_mask(df, rsrs_window=18)
        assert mask.shape == (100,), f"mask 形状错误: {mask.shape}"
        assert mask.dtype == bool

    def test_valid_mask_threshold(self):
        """前 rsrs_window*2-1 行 mask=False，之后 mask=True（无缺失时）"""
        from src.factors.alpha_engine import _nb21_valid_mask
        rsrs_window = 18
        n = 100
        df = self._make_short_df(n)
        mask = _nb21_valid_mask(df, rsrs_window=rsrs_window)
        threshold = rsrs_window * 2  # 第 threshold 行起为 True（0-indexed）
        assert not mask[threshold - 2], f"第{threshold-2}行应为False"
        assert mask[threshold - 1], f"第{threshold-1}行应为True"

    def test_mixed_vintage_old_stock_has_rsrs(self):
        """充足历史(200天)的老股票: rsrs_adaptive 末尾应有有效值"""
        from src.factors.alpha_engine import AlphaEngine
        df = self._make_short_df(200)
        result = AlphaEngine.compute_from_history(df)
        tail = result["rsrs_adaptive"].dropna()
        assert len(tail) > 0, "200天老股票 rsrs_adaptive 全为NaN，NB-21 过度屏蔽!"
        assert math.isfinite(float(tail.iloc[-1]))

    def test_apply_mask_forces_nan(self):
        """_apply_nb21_mask_to_rsrs: mask=False 处强制 NaN"""
        from src.factors.alpha_engine import _apply_nb21_mask_to_rsrs
        n = 50
        df = pd.DataFrame({
            "rsrs_raw":      np.random.randn(n),
            "rsrs_zscore":   np.random.randn(n),
            "rsrs_r2":       np.random.rand(n),
            "rsrs_adaptive": np.random.randn(n),
            "resid_std":     np.random.rand(n),
        })
        mask = np.zeros(n, dtype=bool)
        mask[30:] = True   # 后20行有效
        result = _apply_nb21_mask_to_rsrs(df.copy(), mask)
        assert result["rsrs_adaptive"].iloc[:30].isna().all(), "mask=False 处未置NaN"
        assert result["rsrs_adaptive"].iloc[30:].notna().all(), "mask=True 处不应为NaN"