#!/usr/bin/env python3
"""
Q-UNITY-V6 核心数据类型定义
修复: NB-08 PositionState 新增 entry_date 字段
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


# ============================================================================
# 枚举类型
# ============================================================================

class OrderSide(Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"
    STOP   = "STOP"


class OrderStatus(Enum):
    PENDING   = "PENDING"
    FILLED    = "FILLED"
    PARTIAL   = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED  = "REJECTED"


class PositionDirection(Enum):
    LONG  = "LONG"
    SHORT = "SHORT"


# ============================================================================
# 信号
# ============================================================================

@dataclass
class Signal:
    """交易信号"""
    timestamp: Any          # datetime 或 Timestamp 包装类
    code: str
    side: OrderSide
    score: float = 0.0
    reason: str = ""
    price: Optional[float] = None
    volume: Optional[int] = None
    weight: float = 0.0           # 目标仓位权重 [0, 1]
    strategy_name: str = ""       # 生成此信号的策略名称
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# 订单
# ============================================================================

@dataclass
class Order:
    """委托订单"""
    order_id: str
    timestamp: datetime
    code: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    price: float
    volume: int
    filled_volume: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    reason: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class Fill:
    """成交记录（轻量）"""
    order_id: str
    code: str
    side: OrderSide
    price: float
    volume: int
    timestamp: datetime


# ============================================================================
# 持仓
# ============================================================================

@dataclass
class PositionState:
    """持仓状态"""
    code: str
    direction: PositionDirection
    volume: int
    available_volume: int
    frozen_volume: int
    avg_cost: float
    current_price: float
    market_value: float
    profit_loss: float
    profit_loss_pct: float
    holding_days: int = 0
    last_trade_date: Optional[datetime] = None
    entry_date: Optional[datetime] = None     # NB-08 修复：记录建仓日期
    metadata: Dict = field(default_factory=dict)

    def update_price(self, new_price: float) -> None:
        self.current_price = new_price
        self.market_value  = self.volume * new_price
        cost_basis = self.avg_cost * self.volume
        self.profit_loss     = self.market_value - cost_basis
        self.profit_loss_pct = (self.profit_loss / cost_basis) if cost_basis > 1e-9 else 0.0

    def add_volume(self, volume: int, price: float) -> None:
        total_cost  = self.avg_cost * self.volume + price * volume
        self.volume += volume
        self.avg_cost = total_cost / self.volume if self.volume > 0 else price
        self.available_volume = self.volume
        self.update_price(self.current_price)

    def reduce_volume(self, volume: int) -> None:
        self.volume = max(0, self.volume - volume)
        self.available_volume = max(0, self.available_volume - volume)
        self.update_price(self.current_price)


# ============================================================================
# 账户快照
# ============================================================================

@dataclass
class AccountSnapshot:
    """账户快照"""
    timestamp: datetime
    total_value: float
    cash: float
    market_value: float
    frozen_cash: float
    available_cash: float
    positions_count: int
    total_trades: int
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# 成交记录
# ============================================================================

@dataclass
class TradeRecord:
    """成交记录"""
    trade_id: str
    timestamp: datetime
    code: str
    side: OrderSide
    volume: int
    price: float
    amount: float
    commission: float
    slippage: float
    tax: float
    net_amount: float
    order_id: str = ""
    reason: str = ""
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# 风控指标
# ============================================================================

@dataclass
class RiskMetrics:
    """风险指标快照"""
    timestamp: datetime
    total_value: float
    max_drawdown: float
    current_drawdown: float
    volatility: float
    sharpe_ratio: float
    beta: float
    var_95: float
    cvar_95: float
    concentration_ratio: float
    turnover_rate: float
    max_position_pct: float
    industry_exposure: Dict[str, float] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)


__all__ = [
    "OrderSide", "OrderType", "OrderStatus", "PositionDirection",
    "Signal", "Order", "Fill", "PositionState", "AccountSnapshot",
    "TradeRecord", "RiskMetrics",
]