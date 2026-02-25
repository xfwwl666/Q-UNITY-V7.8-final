#!/usr/bin/env python3
"""
Q-UNITY-V6 回测执行引擎 — 全量 Bug 修复版 + Numba 加速 + 并行因子预计算 + 盈亏比统计
"""
from __future__ import annotations
import logging
import uuid
from collections import deque
from datetime import datetime, date, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# ========== Numba 加速导入 ==========
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
# ====================================

from ..types import (
    AccountSnapshot, Fill, Order, OrderSide, OrderStatus, OrderType,
    PositionDirection, PositionState, RiskMetrics, Signal, TradeRecord,
)
from ..constants import (
    DEFAULT_COMMISSION_RATE, DEFAULT_SLIPPAGE_RATE, MIN_COMMISSION,
    STAMP_TAX_RATE, TRADING_DAYS_PER_YEAR,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Position Manager
# ============================================================================

class PositionManager:
    """持仓管理：包含 NB-03 FIFO 配对 + NB-08 追踪止损水位"""

    def __init__(self) -> None:
        self._positions: Dict[str, PositionState] = {}
        # NB-03: FIFO 批次队列  {code: deque[(qty, cost)]}
        self._lots: Dict[str, deque] = {}
        # NB-08: 追踪止损水位 {code: float}
        self._trailing_watermarks: Dict[str, float] = {}

    # ── 基础访问 ─────────────────────────────────────────────────────────

    def get(self, code: str) -> Optional[PositionState]:
        return self._positions.get(code)

    def get_all(self) -> Dict[str, PositionState]:
        return dict(self._positions)

    def has(self, code: str) -> bool:
        return code in self._positions

    # ── 建仓 / 加仓 ───────────────────────────────────────────────────────

    def open_position(
        self, code: str, price: float, volume: int,
        direction: PositionDirection = PositionDirection.LONG,
        entry_date: Optional[datetime] = None,
    ) -> PositionState:
        if code in self._positions:
            pos = self._positions[code]
            pos.add_volume(volume, price)
            self._lots[code].append((volume, price))
            # NB-08: 水位跟进至最新均成本（不降低）
            if pos.current_price > self._trailing_watermarks.get(code, 0):
                self._trailing_watermarks[code] = pos.current_price
            return pos

        pos = PositionState(
            code=code,
            direction=direction,
            volume=volume,
            available_volume=volume,
            frozen_volume=0,
            avg_cost=price,
            current_price=price,
            market_value=price * volume,
            profit_loss=0.0,
            profit_loss_pct=0.0,
            entry_date=entry_date or datetime.now(),   # NB-08
        )
        self._positions[code] = pos
        self._lots[code] = deque([(volume, price)])
        self._trailing_watermarks[code] = price        # NB-08: 初始=建仓价
        return pos

    # ── 平仓 / 减仓 ───────────────────────────────────────────────────────

    def close_position(
        self, code: str, price: float, volume: int
    ) -> Tuple[float, float]:
        """
        FIFO 平仓
        Returns:
            realized_pnl, trade_win (1.0 / 0.0 / -1.0)
        """
        pos = self._positions.get(code)
        if not pos:
            return 0.0, 0.0

        # NB-03: FIFO 配对
        lots = self._lots.get(code, deque())
        remaining = volume
        realized = 0.0
        while remaining > 0 and lots:
            lot_qty, lot_cost = lots[0]
            used = min(remaining, lot_qty)
            realized += (price - lot_cost) * used
            remaining -= used
            if used == lot_qty:
                lots.popleft()
            else:
                lots[0] = (lot_qty - used, lot_cost)

        win = 1.0 if realized > 0 else (0.0 if realized == 0 else -1.0)

        pos.reduce_volume(volume)
        if pos.volume <= 0:
            del self._positions[code]
            self._trailing_watermarks.pop(code, None)
            self._lots.pop(code, None)
        else:
            pos.update_price(price)

        return realized, win

    # ── 价格更新 ──────────────────────────────────────────────────────────

    def update_prices(self, prices: Dict[str, float]) -> None:
        for code, price in prices.items():
            pos = self._positions.get(code)
            if pos and price > 0:
                pos.update_price(price)
                # NB-08: 追踪水位只上升不下降
                if price > self._trailing_watermarks.get(code, 0):
                    self._trailing_watermarks[code] = price

    def get_trailing_watermark(self, code: str) -> float:
        return self._trailing_watermarks.get(code, 0.0)

    # ── 冻结 / 解冻 ───────────────────────────────────────────────────────

    def freeze(self, code: str, volume: int) -> bool:
        pos = self._positions.get(code)
        if not pos or pos.available_volume < volume:
            return False
        pos.available_volume -= volume
        pos.frozen_volume += volume
        return True

    def unfreeze(self, code: str, volume: int) -> None:
        pos = self._positions.get(code)
        if pos:
            freed = min(volume, pos.frozen_volume)
            pos.frozen_volume -= freed
            pos.available_volume += freed

    # ── 估值（含停牌处理 NB-20）──────────────────────────────────────────

    def get_total_market_value(
        self,
        suspended_codes: Optional[set] = None,
        suspension_discount: float = 0.98,
    ) -> float:
        total = 0.0
        for code, pos in self._positions.items():
            mv = pos.market_value
            if suspended_codes and code in suspended_codes:
                mv *= suspension_discount   # NB-20: 停牌折价
            total += mv
        return total


# ============================================================================
# Account
# ============================================================================

class Account:
    """账户：资金管理 + NB-02 available_cash + NB-05 初始快照"""

    def __init__(self, initial_cash: float = 1_000_000.0) -> None:
        self._cash: float = initial_cash
        self._frozen_cash: float = 0.0
        self._initial_cash: float = initial_cash
        self._snapshots: List[AccountSnapshot] = []
        self._total_trades: int = 0

        # NB-05: 立即记录初始快照
        self._record_snapshot(datetime.now(), 0, 0.0)

    # ── NB-02: available_cash 作为 property ──────────────────────────────

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def frozen_cash(self) -> float:
        return self._frozen_cash

    @property
    def available_cash(self) -> float:
        return max(0.0, self._cash - self._frozen_cash)   # NB-02

    # ── 资金操作 ─────────────────────────────────────────────────────────

    def freeze_cash(self, amount: float) -> bool:
        if self.available_cash < amount:
            return False
        self._frozen_cash += amount
        return True

    def unfreeze_cash(self, amount: float) -> None:
        self._frozen_cash = max(0.0, self._frozen_cash - amount)

    def deduct(self, amount: float) -> None:
        self._frozen_cash = max(0.0, self._frozen_cash - amount)
        self._cash -= amount

    def credit(self, amount: float) -> None:
        self._cash += amount

    def record_trade(self) -> None:
        self._total_trades += 1

    # ── 快照 ──────────────────────────────────────────────────────────────

    def _record_snapshot(
        self,
        ts: datetime,
        positions_count: int,
        market_value: float,
    ) -> None:
        total = self._cash + market_value
        self._snapshots.append(AccountSnapshot(
            timestamp=ts,
            total_value=total,
            cash=self._cash,
            market_value=market_value,
            frozen_cash=self._frozen_cash,
            available_cash=self.available_cash,
            positions_count=positions_count,
            total_trades=self._total_trades,
        ))

    def snapshot(self, ts: datetime, pm: PositionManager) -> AccountSnapshot:
        mv = pm.get_total_market_value()
        self._record_snapshot(ts, len(pm.get_all()), mv)
        return self._snapshots[-1]

    def get_snapshots(self) -> List[AccountSnapshot]:
        return list(self._snapshots)


# ============================================================================
# Performance Calculator
# ============================================================================

class PerformanceCalculator:
    """绩效计算器 — NB-03, NB-04, NB-10, NB-11, 添加盈亏比"""

    def __init__(self, initial_cash: float) -> None:
        self.initial_cash = initial_cash
        self._wins: List[float] = []      # 盈利金额（正）
        self._losses: List[float] = []    # 亏损金额（绝对值，正）

    def record_trade_result(self, realized_pnl: float) -> None:
        if realized_pnl >= 0:
            self._wins.append(realized_pnl)
        else:
            self._losses.append(-realized_pnl)

    def profit_loss_ratio(self) -> float:
        avg_win = np.mean(self._wins) if self._wins else 0.0
        avg_loss = np.mean(self._losses) if self._losses else 0.0
        if avg_loss == 0:
            return float('inf') if avg_win > 0 else 0.0
        return avg_win / avg_loss

    def trade_win_rate(self) -> float:
        total = len(self._wins) + len(self._losses)
        if total == 0:
            return 0.0
        return len(self._wins) / total

    def calculate(self, snapshots: List[AccountSnapshot]) -> Dict[str, float]:
        if len(snapshots) < 2:
            return {}
        values = np.array([s.total_value for s in snapshots])
        n = len(values)
        total_return = values[-1] / self.initial_cash - 1.0
        # NB-04: 年化用 n_days - 1 作分母
        n_days = n - 1
        annual_return = (values[-1] / self.initial_cash) ** (TRADING_DAYS_PER_YEAR / max(n_days, 1)) - 1.0

        daily_returns = np.diff(values) / values[:-1]
        volatility = float(np.std(daily_returns) * np.sqrt(TRADING_DAYS_PER_YEAR))
        rf = 0.03 / TRADING_DAYS_PER_YEAR
        excess = daily_returns - rf
        sharpe = float(annual_return / volatility) if volatility > 1e-9 else 0.0

        # NB-11: Sortino — 无负偏离时返回 inf
        neg_excess = excess[excess < 0]
        if len(neg_excess) == 0:
            sortino = float("inf")
        else:
            down_dev = float(np.std(neg_excess) * np.sqrt(TRADING_DAYS_PER_YEAR))
            sortino = float(annual_return / down_dev) if down_dev > 1e-9 else float("inf")

        # NB-10: 最大回撤峰值索引
        peak = values[0]
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 1e-9 else 0.0
            if dd > max_dd:
                max_dd = dd

        return {
            "total_return":    float(total_return),
            "annual_return":   float(annual_return),
            "volatility":      volatility,
            "sharpe_ratio":    sharpe,
            "sortino_ratio":   sortino,
            "max_drawdown":    float(max_dd),
            "trade_win_rate":  self.trade_win_rate(),
            "profit_loss_ratio": self.profit_loss_ratio(),
            "total_trades":    float(len(self._wins) + len(self._losses)),
        }


# ============================================================================
# Order Manager
# ============================================================================

class OrderManager:
    """订单管理：委托 → 成交 → 交割"""

    def __init__(
        self,
        commission_rate: float = DEFAULT_COMMISSION_RATE,
        slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    ) -> None:
        self.commission_rate = commission_rate
        self.slippage_rate   = slippage_rate
        self._orders: Dict[str, Order] = {}
        self._fills: List[Fill] = []

    # NB-06: 双边手续费; NB-07: 滑点方向正确
    def compute_execution_price(self, price: float, side: OrderSide) -> float:
        if side == OrderSide.BUY:
            return price * (1.0 + self.slippage_rate)   # NB-07 买入价上浮
        else:
            return price * (1.0 - self.slippage_rate)   # NB-07 卖出价下浮

    def compute_commission(self, price: float, volume: int) -> float:
        amount = price * volume
        comm = max(MIN_COMMISSION, amount * self.commission_rate)
        return comm  # NB-06 双边均收

    def compute_tax(self, price: float, volume: int, side: OrderSide) -> float:
        if side == OrderSide.SELL:
            return price * volume * STAMP_TAX_RATE
        return 0.0

    def create_order(
        self,
        code: str,
        side: OrderSide,
        price: float,
        volume: int,
        ts: datetime,
        reason: str = "",
    ) -> Order:
        oid = str(uuid.uuid4())[:8]
        order = Order(
            order_id=oid,
            timestamp=ts,
            code=code,
            side=side,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            price=price,
            volume=volume,
            reason=reason,
        )
        self._orders[oid] = order
        return order

    def fill_order(
        self, order: Order, fill_price: float, fill_volume: int, ts: datetime
    ) -> Tuple[Fill, float, float, float]:
        exec_price = self.compute_execution_price(fill_price, order.side)
        comm  = self.compute_commission(exec_price, fill_volume)
        tax   = self.compute_tax(exec_price, fill_volume, order.side)
        amount = exec_price * fill_volume
        net   = amount + comm + tax if order.side == OrderSide.BUY else amount - comm - tax

        order.filled_volume += fill_volume
        order.filled_price = exec_price
        order.commission += comm
        order.status = OrderStatus.FILLED if order.filled_volume >= order.volume else OrderStatus.PARTIAL

        fill = Fill(
            order_id=order.order_id,
            code=order.code,
            side=order.side,
            price=exec_price,
            volume=fill_volume,
            timestamp=ts,
        )
        self._fills.append(fill)
        return fill, amount, comm, tax


# ============================================================================
# Numba 加速辅助函数（必须定义在模块顶层）
# ============================================================================

@njit(cache=True)
def _check_stop_numerical(avg_cost, current_price, stop_loss_pct, take_profit_pct, trailing_stop_pct, trailing_wm):
    """
    纯数值计算止损条件。
    返回值: (event_code, pnl_pct)
        event_code: 0-无触发, 1-止损, 2-止盈, 3-追踪止损
        pnl_pct: 当前盈亏比例
    """
    if avg_cost <= 1e-9:
        return 0, 0.0
    pnl_pct = (current_price - avg_cost) / avg_cost
    if pnl_pct <= -stop_loss_pct:
        return 1, pnl_pct
    if pnl_pct >= take_profit_pct:
        return 2, pnl_pct
    if trailing_wm > avg_cost:
        trail_dd = (trailing_wm - current_price) / trailing_wm
        if trail_dd >= trailing_stop_pct:
            return 3, pnl_pct
    return 0, pnl_pct


@njit(cache=True)
def _calc_buy_budget(weight, total_val, available_cash, max_position_pct, price):
    """
    计算买入股数（100的倍数）
    返回可买入股数，0表示无法买入
    """
    weight = min(weight, max_position_pct)
    budget = total_val * weight
    if budget > available_cash:
        budget = available_cash
    if budget < price * 100:
        return 0
    volume = int(budget / price / 100) * 100
    return volume if volume > 0 else 0


# ============================================================================
# BacktestEngine
# ============================================================================

class BacktestEngine:
    """
    回测引擎主体
    严格 T+1: 信号在T日收盘后基于T-1数据生成，T+1日开盘执行(NB-01)
    """

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        commission_rate: float = DEFAULT_COMMISSION_RATE,
        slippage_rate: float  = DEFAULT_SLIPPAGE_RATE,
        stop_loss_pct: float  = 0.10,
        take_profit_pct: float = 0.20,
        trailing_stop_pct: float = 0.05,
        max_position_pct: float = 0.10,
        circuit_breaker_max_dd: float = 0.20,
        circuit_breaker_cooldown_days: int = 5,   # NB-12
    ) -> None:
        self.initial_cash = initial_cash
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_position_pct = max_position_pct
        self.circuit_breaker_max_dd = circuit_breaker_max_dd
        self.circuit_breaker_cooldown_days = circuit_breaker_cooldown_days

        self.account = Account(initial_cash)
        self.pm      = PositionManager()
        self.om      = OrderManager(commission_rate, slippage_rate)
        self.perf    = PerformanceCalculator(initial_cash)

        # NB-12: 熔断状态
        self._circuit_broken: bool = False
        self._circuit_break_date: Optional[date] = None

        # NB-01: 待执行信号队列（下一个 bar 执行）
        self._pending_signals: List[Signal] = []

        self._trade_records: List[TradeRecord] = []
        self._current_date: Optional[date] = None

    # ── 熔断检测 (NB-12) ─────────────────────────────────────────────────

    def _check_circuit_breaker(self, current_date: date) -> None:
        snapshots = self.account.get_snapshots()
        if len(snapshots) < 2:
            return
        values = [s.total_value for s in snapshots]
        peak = max(values)
        curr = values[-1]
        dd = (peak - curr) / peak if peak > 1e-9 else 0.0
        if not self._circuit_broken and dd >= self.circuit_breaker_max_dd:
            self._circuit_broken = True
            self._circuit_break_date = current_date
            logger.warning(f"熔断触发! 最大回撤 {dd:.2%} 超限 {current_date}")
        # NB-12: cooldown 后自动解除
        elif self._circuit_broken and self._circuit_break_date:
            elapsed = (current_date - self._circuit_break_date).days
            if elapsed >= self.circuit_breaker_cooldown_days:
                self._circuit_broken = False
                self._circuit_break_date = None
                logger.info(f"熔断解除 (冷却 {elapsed}天) {current_date}")

    # ── 止损 / 止盈 (NB-08 NB-15) ────────────────────────────────────────

    def _check_stop_conditions(
        self, code: str, current_price: float, current_date: date
    ) -> Optional[str]:
        pos = self.pm.get(code)
        if not pos:
            return None

        # NB-15: 止损以 avg_cost 为基准（不用缓存价格）
        cost = pos.avg_cost
        if cost <= 1e-9:
            return None

        wm = self.pm.get_trailing_watermark(code)
        event, pnl_pct = _check_stop_numerical(
            cost, current_price,
            self.stop_loss_pct, self.take_profit_pct,
            self.trailing_stop_pct, wm
        )
        if event == 1:
            return f"止损({pnl_pct:.2%})"
        elif event == 2:
            return f"止盈({pnl_pct:.2%})"
        elif event == 3:
            return f"追踪止损(水位{wm:.2f}→{current_price:.2f})"
        return None

    # ── 核心: 单个 bar 推进 ───────────────────────────────────────────────

    def step(
        self,
        bar_date: date,
        price_data: Dict[str, Dict[str, float]],   # {code: {open,high,low,close,volume}}
        new_signals: List[Signal],                  # 策略在 T-1 收盘生成、T 日执行
        suspended_codes: Optional[set] = None,
    ) -> Dict[str, Any]:
        self._current_date = bar_date
        suspended_codes = suspended_codes or set()

        # 1) 更新持仓收盘价
        close_prices = {c: d["close"] for c, d in price_data.items() if "close" in d}
        self.pm.update_prices(close_prices)

        # 2) 熔断检测 (NB-12)
        self._check_circuit_breaker(bar_date)

        # 3) 执行上一 bar 缓存的信号（NB-01：T+1 执行）
        executions = []
        if not self._circuit_broken:
            for sig in self._pending_signals:
                if sig.code in suspended_codes:
                    continue
                # 使用今日 open 价执行
                bar = price_data.get(sig.code, {})
                exec_price = bar.get("open", bar.get("close", 0.0))
                if exec_price <= 0:
                    continue
                result = self._execute_signal(sig, exec_price, bar_date)
                if result:
                    executions.append(result)

        # 4) 缓存新信号，下一 bar 执行（NB-01）
        self._pending_signals = [s for s in new_signals if s.code not in suspended_codes]

        # 5) 止损/止盈检测（用收盘价）
        stops = []
        for code, pos in list(self.pm.get_all().items()):
            if code in suspended_codes:
                continue
            close = close_prices.get(code, 0.0)
            if close <= 0:
                continue
            reason = self._check_stop_conditions(code, close, bar_date)
            if reason:
                result = self._execute_sell(code, close, pos.available_volume,
                                             bar_date, reason)
                if result:
                    stops.append(result)

        # 6) 账户快照
        snap = self.account.snapshot(datetime.combine(bar_date, datetime.min.time()), self.pm)

        return {
            "date": bar_date,
            "executions": executions,
            "stops": stops,
            "snapshot": snap,
            "circuit_broken": self._circuit_broken,
        }

    # ── 执行买卖 ──────────────────────────────────────────────────────────

    def _execute_signal(self, sig: Signal, price: float, bar_date: date) -> Optional[Dict]:
        if sig.side == OrderSide.BUY:
            # NB-16: 仓位权重已在策略层归一化
            total_val = self.account.cash + self.pm.get_total_market_value()
            volume = _calc_buy_budget(
                sig.weight, total_val, self.account.available_cash,
                self.max_position_pct, price
            )
            if volume <= 0:
                return None
            cost = price * volume
            comm = self.om.compute_commission(price, volume)
            total_cost = cost + comm
            if not self.account.freeze_cash(total_cost):
                return None
            ts = datetime.combine(bar_date, datetime.min.time())
            order = self.om.create_order(sig.code, OrderSide.BUY, price, volume, ts, sig.reason)
            fill, amount, comm2, tax = self.om.fill_order(order, price, volume, ts)
            self.account.deduct(total_cost)
            self.pm.open_position(sig.code, fill.price, volume, entry_date=ts)
            self.account.record_trade()
            return {"type": "BUY", "code": sig.code, "price": fill.price, "volume": volume}
        elif sig.side == OrderSide.SELL:
            pos = self.pm.get(sig.code)
            if not pos or pos.available_volume <= 0:
                return None
            return self._execute_sell(sig.code, price, pos.available_volume, bar_date, sig.reason)
        return None

    def _execute_sell(self, code: str, price: float, volume: int,
                      bar_date: date, reason: str) -> Optional[Dict]:
        pos = self.pm.get(code)
        if not pos or volume <= 0:
            return None
        ts = datetime.combine(bar_date, datetime.min.time())
        order = self.om.create_order(code, OrderSide.SELL, price, volume, ts, reason)
        fill, amount, comm, tax = self.om.fill_order(order, price, volume, ts)
        net_proceed = amount - comm - tax
        realized, win = self.pm.close_position(code, fill.price, volume)
        self.account.credit(net_proceed)
        self.perf.record_trade_result(realized)
        self.account.record_trade()
        return {"type": "SELL", "code": code, "price": fill.price, "volume": volume,
                "realized_pnl": realized, "reason": reason}

    # ── 结果汇总 ──────────────────────────────────────────────────────────

    def get_performance(self) -> Dict[str, float]:
        return self.perf.calculate(self.account.get_snapshots())

    def get_equity_curve(self) -> pd.DataFrame:
        snaps = self.account.get_snapshots()
        if not snaps:
            return pd.DataFrame()
        return pd.DataFrame({
            "timestamp":    [s.timestamp for s in snaps],
            "total_value":  [s.total_value for s in snaps],
            "cash":         [s.cash for s in snaps],
            "market_value": [s.market_value for s in snaps],
        }).set_index("timestamp")


# ============================================================================
# V7.7 新增: 并行因子预计算（带进度条）
# ============================================================================

import multiprocessing as _mp
import os as _os
from typing import Tuple as _Tuple

# 尝试导入 tqdm
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    tqdm = lambda x, **kw: x


def _compute_one_factor(args: _Tuple) -> _Tuple:
    """
    multiprocessing.Pool worker 函数。
    因 Pool 使用 pickle 序列化，此函数必须定义在模块顶层。

    参数:
        args: (code, df, date_values)
            code        — 股票代码
            df          — OHLCV DataFrame（全量历史）
            date_values — df["date"] 的 numpy array（字符串），用于设置因子索引

    返回:
        (code, factor_df_or_None, error_str_or_None)
    """
    code, df, date_values = args
    try:
        # 延迟导入，避免在 worker 进程中产生 import 副作用
        from src.factors.alpha_engine import AlphaEngine
        factor_df = AlphaEngine.compute_from_history(df)
        factor_df = factor_df.copy()
        # 将 date 列设为索引，便于后续按日期切片
        factor_df["date"] = date_values[: len(factor_df)]
        factor_df = factor_df.set_index("date")
        return code, factor_df, None
    except Exception as e:
        return code, None, str(e)


def parallel_factor_precomputation(
    codes: list,
    market_data: dict,
    max_workers: int = None,
    desc: str = "并行因子预计算",
) -> dict:
    """
    使用 multiprocessing.Pool 并行计算每只股票的因子，并显示进度条。

    参数:
        codes        — 股票代码列表（仅计算 market_data 中存在的代码）
        market_data  — {code: DataFrame}，DataFrame 含 date/open/high/low/close/vol 列
        max_workers  — 进程数，None 时自动取 CPU 核心数（最多16，避免内存溢出）
        desc         — 进度条描述文字

    返回:
        {code: factor_df}，factor_df 的 index 为 date 字符串

    注意:
        - 依赖 tqdm，若未安装则自动降级为无进度条（仅日志输出）
    """
    _log = logging.getLogger(__name__)

    # 确定进程数：默认取 CPU 核数，但不超过 16 以控制内存
    if max_workers is None:
        cpu_cnt = _os.cpu_count() or 4
        max_workers = min(cpu_cnt, 16)

    # 构建任务列表（仅处理 market_data 中存在的 code）
    tasks = []
    for code in codes:
        df = market_data.get(code)
        if df is not None and not df.empty and "date" in df.columns:
            date_vals = df["date"].values
            tasks.append((code, df, date_vals))

    if not tasks:
        _log.warning("parallel_factor_precomputation: 无有效任务，返回空字典")
        return {}

    factor_data: dict = {}
    success_cnt = error_cnt = 0

    _log.info("并行因子预计算: %d 只股票，进程数=%d", len(tasks), max_workers)

    try:
        ctx = _mp.get_context("spawn")
        with ctx.Pool(processes=max_workers) as pool:
            iterator = pool.imap_unordered(_compute_one_factor, tasks, chunksize=4)
            if _HAS_TQDM:
                iterator = tqdm(iterator, total=len(tasks), desc=desc, unit="股")
            for code, fdf, err in iterator:
                if err:
                    _log.warning("并行因子计算失败 %s: %s", code, err)
                    error_cnt += 1
                else:
                    factor_data[code] = fdf
                    success_cnt += 1
    except Exception as e:
        _log.warning("并行因子预计算异常(%s)，降级为串行", e)
        from src.factors.alpha_engine import AlphaEngine
        task_iter = tasks
        if _HAS_TQDM:
            task_iter = tqdm(tasks, desc="串行因子计算（降级）", unit="股")
        for code, df, date_vals in task_iter:
            try:
                fdf = AlphaEngine.compute_from_history(df)
                fdf = fdf.copy()
                fdf["date"] = date_vals[: len(fdf)]
                fdf = fdf.set_index("date")
                factor_data[code] = fdf
                success_cnt += 1
            except Exception as inner_e:
                _log.warning("串行因子计算失败 %s: %s", code, inner_e)
                error_cnt += 1

    _log.info("并行因子预计算完成: 成功=%d 失败=%d", success_cnt, error_cnt)
    return factor_data