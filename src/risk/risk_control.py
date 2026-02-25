#!/usr/bin/env python3
"""
Q-UNITY-V6 三层风控体系
  L1: 仓位风控（单股+行业，NB-16归一化）
  L2: 止损止盈（由执行引擎处理，见execution.py）
  L3: 组合风控（最大回撤熔断，NB-12 cooldown）
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RiskController:
    """三层风控控制器"""

    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_industry_pct: float = 0.30,   # NB-19 行业限仓
        max_drawdown:     float = 0.20,
        cooldown_days:    int   = 5,       # NB-12
    ) -> None:
        self.max_position_pct = max_position_pct
        self.max_industry_pct = max_industry_pct
        self.max_drawdown     = max_drawdown
        self.cooldown_days    = cooldown_days

    def filter_signals_by_position(
        self,
        signals: list,
        current_positions: Dict[str, object],
        total_value: float,
        available_cash: float,
    ) -> list:
        """L1: 过滤超仓信号 (NB-16: 权重归一化后再过滤)"""
        if not signals:
            return []

        buy_signals = [s for s in signals if hasattr(s, "side") and s.side.value == "BUY"]
        sell_signals = [s for s in signals if hasattr(s, "side") and s.side.value == "SELL"]

        # NB-16: 归一化买入权重
        total_w = sum(s.weight for s in buy_signals) if buy_signals else 0
        if total_w > 1.0:
            for s in buy_signals:
                s.weight /= total_w

        # 过滤超过单股上限
        filtered_buy = [
            s for s in buy_signals
            if s.weight <= self.max_position_pct
        ]
        return filtered_buy + sell_signals

    def filter_signals_by_industry(
        self,
        signals: list,
        current_positions: Dict[str, object],
        industry_map: Optional[Dict[str, str]],
        total_value: float,
    ) -> list:
        """NB-19: 行业暴露跨策略合并检测"""
        if not industry_map or not signals:
            return signals

        # 当前行业持仓比例
        industry_exposure: Dict[str, float] = {}
        for code, pos in current_positions.items():
            ind = industry_map.get(code, "未知")
            mv  = getattr(pos, "market_value", 0.0)
            industry_exposure[ind] = industry_exposure.get(ind, 0.0) + mv / max(total_value, 1)

        out = []
        for sig in signals:
            if not (hasattr(sig, "side") and sig.side.value == "BUY"):
                out.append(sig)
                continue
            ind = industry_map.get(sig.code, "未知")
            proj = industry_exposure.get(ind, 0.0) + sig.weight
            if proj <= self.max_industry_pct:
                out.append(sig)
                industry_exposure[ind] = proj
            else:
                logger.debug(f"行业限仓过滤 {sig.code} ({ind}): {proj:.1%} > {self.max_industry_pct:.1%}")
        return out

    def check_portfolio_risk(
        self,
        equity_curve: "np.ndarray",
    ) -> Dict[str, float]:
        """L3: 组合风险检测"""
        if len(equity_curve) < 2:
            return {"current_drawdown": 0.0, "max_drawdown": 0.0}
        peak = np.maximum.accumulate(equity_curve)
        dd_series = (peak - equity_curve) / np.where(peak > 0, peak, 1)
        return {
            "current_drawdown": float(dd_series[-1]),
            "max_drawdown":     float(dd_series.max()),
        }