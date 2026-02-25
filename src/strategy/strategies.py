#!/usr/bin/env python3
"""
Q-UNITY-V7.8 策略全集（9大策略） + V7.5 实时信号缓存优化 + V7.8 多项修复
  1. RSRSMomentumStrategy     — 基础RSRS动量
  2. AlphaHunterStrategy      — 高频多层锁
  3. RSRSAdvancedStrategy     — R²过滤+量价共振
  4. ShortTermStrategy        — 快进快出+日历止时 (NB-14)
  5. MomentumReversalStrategy — 双模式 60/40
  6. SentimentReversalStrategy— 超卖反转
  7. KunpengV10Strategy       — 微结构(聪明钱+稳定非流动性+缺口惩罚)+宽度熔断 (V7.5 增加use_breadth_check布尔参数)
  8. AlphaMaxV5FixedStrategy  — 机构多因子(EP/成长/动量/质量/REV/流动/残差波动)+行业中性+风险平价
"""
from __future__ import annotations

import logging
import threading                # V7.5 增加线程锁
import time                     # V7.8 修复 B-06: 补充缺失的 time 模块导入
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ..types import OrderSide, Signal

logger = logging.getLogger(__name__)


# ============================================================================
# 基类
# ============================================================================

class BaseStrategy(ABC):
    """策略基类"""

    name: str = "BaseStrategy"

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.config = config or {}
        self._state: Dict[str, Any] = {}

    @abstractmethod
    def generate_signals(
        self,
        universe: List[str],
        market_data: Dict[str, pd.DataFrame],
        factor_data: Dict[str, pd.DataFrame],
        current_date: datetime,
        positions: Dict[str, Any],
        **kwargs,
    ) -> List[Signal]:
        """
        生成交易信号。
        **kwargs 用于接收扩展参数（如 sector_map、sector_data），保持向后兼容。
        """
        ...

    def _make_buy(self, code: str, score: float, weight: float,
                  ts: datetime, reason: str = "") -> Signal:
        return Signal(timestamp=ts, code=code, side=OrderSide.BUY,
                      score=score, weight=weight, reason=reason,
                      strategy_name=self.name)

    def _make_sell(self, code: str, ts: datetime, reason: str = "") -> Signal:
        return Signal(timestamp=ts, code=code, side=OrderSide.SELL,
                      score=0.0, weight=0.0, reason=reason,
                      strategy_name=self.name)


# ============================================================================
# 1. RSRSMomentumStrategy
# ============================================================================

class RSRSMomentumStrategy(BaseStrategy):
    """
    RSRS 动量策略
    - 买入: rsrs_adaptive > threshold（买入前N只）
    - 卖出: rsrs_adaptive < -threshold
    """
    name = "RSRSMomentum"

    def __init__(self, config: Optional[Dict] = None) -> None:
        super().__init__(config)
        self.top_n = self.config.get("top_n", 10)
        self.rsrs_threshold = self.config.get("rsrs_threshold", 0.5)

    def generate_signals(self, universe, market_data, factor_data,
                         current_date, positions, **kwargs):
        """
        生成信号。
        如果传入 precomputed_scores 参数，则使用预计算评分快速生成。
        """
        pre_scores = kwargs.get('precomputed_scores')
        if pre_scores is not None:
            return self._generate_signals_from_scores(pre_scores, positions, current_date)

        # 原有逻辑（逐股票从 factor_data 提取）
        signals = []
        scores = {}
        for code in universe:
            fd = factor_data.get(code)
            if fd is None or fd.empty or "rsrs_adaptive" not in fd.columns:
                continue
            vals = fd["rsrs_adaptive"].dropna()
            if vals.empty:
                continue
            v = float(vals.iloc[-1])
            if v > self.rsrs_threshold:
                scores[code] = v

        # 退出信号
        for code in list(positions.keys()):
            fd = factor_data.get(code)
            if fd is None or fd.empty:
                continue
            vals = fd.get("rsrs_adaptive", pd.Series()).dropna()
            if vals.empty:
                continue
            if float(vals.iloc[-1]) < -self.rsrs_threshold:
                signals.append(self._make_sell(code, current_date, "RSRS跌破下限"))

        top = sorted(scores, key=scores.__getitem__, reverse=True)[: self.top_n]
        weight = 1.0 / max(len(top), 1)
        for code in top:
            if code not in positions:
                signals.append(self._make_buy(code, scores[code], weight,
                                              current_date, "RSRS强势"))
        return signals

    def _generate_signals_from_scores(self, scores_dict, positions, current_date):
        """
        使用预计算评分快速生成信号。
        A-14 Fix: scores_dict 现支持两种格式：
          - 旧格式：{code: float}（兼容原有逻辑）
          - 新格式：{code: {factor: float, ...}}（多因子矩阵）
        """
        signals = []

        def _get_rsrs(entry):
            """从多格式 score entry 中提取 rsrs_adaptive 值"""
            if isinstance(entry, dict):
                return entry.get("rsrs_adaptive")
            if isinstance(entry, (int, float)):
                return float(entry)
            return None

        buy_candidates = {}
        for code, entry in scores_dict.items():
            if code in positions:
                continue
            v = _get_rsrs(entry)
            if v is not None and v > self.rsrs_threshold:
                buy_candidates[code] = v

        top_codes = sorted(buy_candidates, key=buy_candidates.__getitem__, reverse=True)[:self.top_n]
        weight = 1.0 / max(len(top_codes), 1)
        for code in top_codes:
            signals.append(self._make_buy(
                code, buy_candidates[code], weight, current_date,
                f"RSRS强势(预计算)={buy_candidates[code]:.3f}"
            ))

        for code in list(positions.keys()):
            entry = scores_dict.get(code)
            if entry is None:
                continue
            v = _get_rsrs(entry)
            if v is not None and v < -self.rsrs_threshold:
                signals.append(self._make_sell(
                    code, current_date,
                    f"RSRS跌破下限(预计算)={v:.3f}"
                ))
        return signals


# ============================================================================
# 2. AlphaHunterStrategy
# ============================================================================

class AlphaHunterStrategy(BaseStrategy):
    """
    Alpha 猎手策略（多层评分锁定）
    综合 rsrs_adaptive + mom + vol_factor 三因子加权打分
    """
    name = "AlphaHunter"

    def __init__(self, config: Optional[Dict] = None) -> None:
        super().__init__(config)
        self.top_n    = self.config.get("top_n", 15)
        self.min_score = self.config.get("min_score", 0.3)
        self.factor_weights = self.config.get("factor_weights", {
            "rsrs_adaptive": 0.5, "mom": 0.3, "vol_factor": 0.2,
        })

    def _get_score(self, fd: pd.DataFrame) -> float:
        score = 0.0
        total_w = 0.0
        for fn, w in self.factor_weights.items():
            if fn in fd.columns:
                vals = fd[fn].dropna()
                if not vals.empty:
                    score += float(vals.iloc[-1]) * w
                    total_w += w
        return score / total_w if total_w > 0 else 0.0

    def generate_signals(self, universe, market_data, factor_data,
                         current_date, positions, **kwargs) -> List[Signal]:
        signals = []
        scores = {}

        for code in universe:
            fd = factor_data.get(code)
            if fd is not None and not fd.empty:
                s = self._get_score(fd)
                if s >= self.min_score:
                    scores[code] = s

        # 退出低分持仓
        for code in list(positions.keys()):
            fd = factor_data.get(code)
            if fd is not None and not fd.empty:
                s = self._get_score(fd)
                if s < -self.min_score:
                    signals.append(self._make_sell(code, current_date, f"多因子分数{s:.2f}"))

        top = sorted(scores, key=scores.__getitem__, reverse=True)[: self.top_n]
        weight = 1.0 / max(len(top), 1)
        for code in top:
            if code not in positions:
                signals.append(self._make_buy(code, scores[code], weight,
                                              current_date, f"多因子{scores[code]:.2f}"))
        return signals


# ============================================================================
# 3. RSRSAdvancedStrategy
# ============================================================================

class RSRSAdvancedStrategy(BaseStrategy):
    """
    高级RSRS策略: R²过滤 + 量价共振确认
    - 仅在 rsrs_r2 > r2_threshold 时买入
    - 量价共振: turnover 需高于均值
    """
    name = "RSRSAdvanced"

    def __init__(self, config: Optional[Dict] = None) -> None:
        super().__init__(config)
        self.top_n         = self.config.get("top_n", 10)
        self.rsrs_threshold = self.config.get("rsrs_threshold", 0.5)
        self.r2_threshold  = self.config.get("r2_threshold", 0.7)

    def generate_signals(self, universe, market_data, factor_data,
                         current_date, positions, **kwargs) -> List[Signal]:
        signals = []
        candidates = {}

        for code in universe:
            fd = factor_data.get(code)
            if fd is None or fd.empty:
                continue
            ra = fd.get("rsrs_adaptive", pd.Series()).dropna()
            r2 = fd.get("rsrs_r2", pd.Series()).dropna()
            if ra.empty or r2.empty:
                continue
            # R² 过滤
            if float(r2.iloc[-1]) < self.r2_threshold:
                continue
            v = float(ra.iloc[-1])
            if v > self.rsrs_threshold:
                # 量价共振
                if "turnover" in fd.columns:
                    to = fd["turnover"].dropna()
                    if not to.empty and float(to.iloc[-1]) < 1.0:
                        continue   # 换手低，跳过
                candidates[code] = v

        for code in list(positions.keys()):
            fd = factor_data.get(code)
            if fd is not None and not fd.empty:
                ra = fd.get("rsrs_adaptive", pd.Series()).dropna()
                if not ra.empty and float(ra.iloc[-1]) < -self.rsrs_threshold:
                    signals.append(self._make_sell(code, current_date, "RSRS高级退出"))

        top = sorted(candidates, key=candidates.__getitem__, reverse=True)[: self.top_n]
        weight = 1.0 / max(len(top), 1)
        for code in top:
            if code not in positions:
                signals.append(self._make_buy(code, candidates[code], weight,
                                              current_date, f"RSRS+R²+量价共振"))
        return signals


# ============================================================================
# 4. ShortTermStrategy  (NB-14 日历日止时)
# ============================================================================

class ShortTermStrategy(BaseStrategy):
    """
    短线快进快出策略
    NB-14: 时间止损基于日历日（不是交易日）
    """
    name = "ShortTerm"

    def __init__(self, config: Optional[Dict] = None) -> None:
        super().__init__(config)
        self.top_n    = self.config.get("top_n", 5)
        self.hold_calendar_days = self.config.get("hold_calendar_days", 7)  # NB-14
        self.mom_threshold = self.config.get("mom_threshold", 0.03)

    def generate_signals(self, universe, market_data, factor_data,
                         current_date, positions, **kwargs) -> List[Signal]:
        signals = []
        scores = {}

        # NB-14: 日历日时间止损检查
        for code, pos in positions.items():
            entry = getattr(pos, "entry_date", None)
            if entry is not None:
                held_calendar = (current_date - entry).days   # 日历日
                if held_calendar >= self.hold_calendar_days:
                    signals.append(self._make_sell(code, current_date,
                                                   f"时间止损{held_calendar}日历日"))
                    continue
            # 动量退出
            fd = factor_data.get(code)
            if fd is not None and "mom" in fd.columns:
                m = fd["mom"].dropna()
                if not m.empty and float(m.iloc[-1]) < -self.mom_threshold:
                    signals.append(self._make_sell(code, current_date, "动量反转"))

        for code in universe:
            fd = factor_data.get(code)
            if fd is None or fd.empty or "mom" not in fd.columns:
                continue
            m = fd["mom"].dropna()
            if m.empty:
                continue
            v = float(m.iloc[-1])
            if v > self.mom_threshold:
                scores[code] = v

        top = sorted(scores, key=scores.__getitem__, reverse=True)[: self.top_n]
        weight = 1.0 / max(len(top), 1)
        for code in top:
            if code not in positions:
                signals.append(self._make_buy(code, scores[code], weight,
                                              current_date, "短线动量"))
        return signals


# ============================================================================
# 5. MomentumReversalStrategy
# ============================================================================

class MomentumReversalStrategy(BaseStrategy):
    """双模式: 强势市场追动量(60%) / 弱势市场做反转(40%)"""
    name = "MomentumReversal"

    def __init__(self, config: Optional[Dict] = None) -> None:
        super().__init__(config)
        self.top_n = self.config.get("top_n", 10)
        self.market_thresh = self.config.get("market_thresh", 0.0)

    def _get_market_mode(self, market_data: Dict) -> str:
        mom_list = []
        for code, df in market_data.items():
            if "close" in df.columns and len(df) > 20:
                ret = float(df["close"].iloc[-1] / df["close"].iloc[-20] - 1)
                mom_list.append(ret)
        if not mom_list:
            return "neutral"
        avg = np.mean(mom_list)
        return "bull" if avg > self.market_thresh else "bear"

    def generate_signals(self, universe, market_data, factor_data,
                         current_date, positions, **kwargs) -> List[Signal]:
        mode = self._get_market_mode(market_data)
        signals = []
        scores = {}

        for code in universe:
            fd = factor_data.get(code)
            if fd is None or fd.empty:
                continue
            m = fd.get("mom", pd.Series()).dropna()
            if m.empty:
                continue
            v = float(m.iloc[-1])
            if mode == "bull":
                if v > 0:
                    scores[code] = v      # 追动量
            else:
                if v < -0.05:
                    scores[code] = -v     # 做反转（超卖）

        for code in list(positions.keys()):
            if code not in scores:
                signals.append(self._make_sell(code, current_date, f"模式切换{mode}"))

        top = sorted(scores, key=scores.__getitem__, reverse=True)[: self.top_n]
        weight = 1.0 / max(len(top), 1)
        for code in top:
            if code not in positions:
                signals.append(self._make_buy(code, scores[code], weight,
                                              current_date, f"{mode}模式"))
        return signals


# ============================================================================
# 6. SentimentReversalStrategy
# ============================================================================

class SentimentReversalStrategy(BaseStrategy):
    """情绪反转: 超卖买入，超涨卖出"""
    name = "SentimentReversal"

    def __init__(self, config: Optional[Dict] = None) -> None:
        super().__init__(config)
        self.top_n   = self.config.get("top_n", 10)
        self.oversold_z = self.config.get("oversold_z", -1.5)
        self.overbought_z = self.config.get("overbought_z", 1.5)

    def generate_signals(self, universe, market_data, factor_data,
                         current_date, positions, **kwargs) -> List[Signal]:
        signals = []
        scores = {}

        for code in universe:
            fd = factor_data.get(code)
            if fd is None or fd.empty:
                continue
            rs = fd.get("rsrs_zscore", pd.Series()).dropna()
            if rs.empty:
                continue
            z = float(rs.iloc[-1])
            if z < self.oversold_z:
                scores[code] = -z   # 越超卖越高分

        for code in list(positions.keys()):
            fd = factor_data.get(code)
            if fd is not None and not fd.empty:
                rs = fd.get("rsrs_zscore", pd.Series()).dropna()
                if not rs.empty and float(rs.iloc[-1]) > self.overbought_z:
                    signals.append(self._make_sell(code, current_date, "超买退出"))

        top = sorted(scores, key=scores.__getitem__, reverse=True)[: self.top_n]
        weight = 1.0 / max(len(top), 1)
        for code in top:
            if code not in positions:
                signals.append(self._make_buy(code, scores[code], weight,
                                              current_date, "超卖反转"))
        return signals


# ============================================================================
# 7. KunpengV10Strategy — 微结构策略 (V7.5 增加use_breadth_check布尔参数)
# ============================================================================

class KunpengV10Strategy(BaseStrategy):
    """
    鲲鹏V10策略 — 市场微结构因子
    三核心因子:
      SmartMoney  = 大单净流入占比（用 (close-low)/(high-low) * vol 近似）
      StableIlliq = Amihud 非流动性稳定性（低波动非流动 > 稳定持有者存在）
      GapPenalty  = 跳空缺口惩罚（跳空过大降权）
    宽度熔断: 若涨停数/跌停数异常则暂停买入 (可通过use_breadth_check关闭)
    """
    name = "KunpengV10"

    def __init__(self, config: Optional[Dict] = None) -> None:
        super().__init__(config)
        self.top_n           = self.config.get("top_n", 15)
        self.illiq_window    = self.config.get("illiq_window", 20)
        self.smart_window    = self.config.get("smart_window", 10)
        self.breadth_limit   = self.config.get("breadth_limit", 0.15)
        # V7.5 新增布尔参数，用于控制是否启用宽度熔断
        self.use_breadth_check = self.config.get("use_breadth_check", True)

    def _smart_money(self, df: pd.DataFrame) -> float:
        if not all(c in df.columns for c in ["high", "low", "close", "volume"]):
            return 0.0
        w = self.smart_window
        sub = df.tail(w)
        hl  = (sub["high"] - sub["low"]).replace(0, np.nan)
        buy_vol = (sub["close"] - sub["low"]) / hl * sub["volume"]
        sell_vol = (sub["high"] - sub["close"]) / hl * sub["volume"]
        total_vol = sub["volume"].sum()
        if total_vol < 1:
            return 0.0
        return float((buy_vol.sum() - sell_vol.sum()) / total_vol)

    def _amihud_stable(self, df: pd.DataFrame) -> float:
        if not all(c in df.columns for c in ["close", "volume", "amount"]):
            return 0.0
        sub = df.tail(self.illiq_window).copy()
        ret = sub["close"].pct_change().abs()
        amt = sub["amount"].replace(0, np.nan)
        illiq = (ret / amt).dropna()
        if len(illiq) < 5:
            return 0.0
        return float(-illiq.std())  # 稳定=低波动=高分

    def _gap_penalty(self, df: pd.DataFrame) -> float:
        if "open" not in df.columns or "close" not in df.columns or len(df) < 2:
            return 0.0
        gap = abs(float(df["open"].iloc[-1]) - float(df["close"].iloc[-2]))
        ref = float(df["close"].iloc[-2]) if df["close"].iloc[-2] > 0 else 1
        gap_pct = gap / ref
        return -min(gap_pct, 0.1) * 10   # 最大惩罚 -1.0

    def _breadth_check(self, market_data: Dict) -> bool:
        """宽度熔断：涨跌停比例超限返回 True（需暂停买入）"""
        limit_up = limit_dn = total = 0
        for code, df in market_data.items():
            if "close" not in df.columns or "open" not in df.columns or len(df) < 2:
                continue
            chg = float(df["close"].iloc[-1]) / float(df["close"].iloc[-2]) - 1
            total += 1
            if chg >= 0.095:
                limit_up += 1
            elif chg <= -0.095:
                limit_dn += 1
        if total == 0:
            return False
        return (limit_dn / total) > self.breadth_limit

    def generate_signals(self, universe, market_data, factor_data,
                         current_date, positions, **kwargs) -> List[Signal]:
        signals = []

        # 宽度熔断检测 (可开关)
        if self.use_breadth_check and self._breadth_check(market_data):
            logger.info(f"KunpengV10 宽度熔断触发 {current_date}，暂停买入")
            # 仍可卖出
            for code in list(positions.keys()):
                df = market_data.get(code)
                if df is not None and len(df) >= 2:
                    chg = float(df["close"].iloc[-1]) / float(df["close"].iloc[-2]) - 1
                    if chg <= -0.09:
                        signals.append(self._make_sell(code, current_date, "宽度熔断卖出"))
            return signals

        scores = {}
        for code in universe:
            df = market_data.get(code)
            if df is None or len(df) < self.illiq_window:
                continue
            sm  = self._smart_money(df)
            asi = self._amihud_stable(df)
            gp  = self._gap_penalty(df)
            scores[code] = 0.5 * sm + 0.3 * asi + 0.2 * gp

        for code in list(positions.keys()):
            if code not in scores or scores[code] < -0.3:
                signals.append(self._make_sell(code, current_date, "微结构退化"))

        top = sorted(scores, key=scores.__getitem__, reverse=True)[: self.top_n]
        weight = 1.0 / max(len(top), 1)
        for code in top:
            if code not in positions and scores[code] > 0.1:
                signals.append(self._make_buy(code, scores[code], weight,
                                              current_date, f"微结构{scores[code]:.2f}"))
        return signals


# ============================================================================
# 8. AlphaMaxV5FixedStrategy — 机构多因子
# ============================================================================

class AlphaMaxV5FixedStrategy(BaseStrategy):
    """
    AlphaMax V5 (Fixed) — 机构级多因子策略
    七大因子:
      EP          = 盈利收益率 (1/PE_TTM)
      Growth      = 净利润同比增速
      Momentum    = 20日价格动量
      Quality     = ROE_TTM
      Reversal    = 短期反转 (-5日收益)
      Liquidity   = 非流动性 (Amihud)
      ResidualVol = 残差波动率（特质风险）
    特性: 行业中性 + 风险平价权重
    """
    name = "AlphaMaxV5Fixed"

    def __init__(self, config: Optional[Dict] = None) -> None:
        super().__init__(config)
        self.top_n       = self.config.get("top_n", 20)
        self.ep_weight   = self.config.get("ep_weight",   0.20)
        self.growth_w    = self.config.get("growth_w",    0.15)
        self.mom_w       = self.config.get("mom_w",       0.15)
        self.quality_w   = self.config.get("quality_w",  0.20)
        self.rev_w       = self.config.get("rev_w",       0.10)
        self.liq_w       = self.config.get("liq_w",       0.10)
        self.res_vol_w   = self.config.get("res_vol_w",  0.10)

    def _compute_ep(self, fundamental: Optional[Dict]) -> float:
        if not fundamental:
            return 0.0
        pe = fundamental.get("pe_ttm")
        if pe and abs(pe) > 1e-6:
            return 1.0 / pe
        return 0.0

    def _compute_resid_vol(self, df: pd.DataFrame, market_df: Optional[pd.DataFrame]) -> float:
        """残差波动率（特质风险）= std(股票日收益 - beta*市场日收益)"""
        if "close" not in df.columns or len(df) < 20:
            return 0.0
        ret = df["close"].pct_change().tail(60).dropna()
        if market_df is not None and "close" in market_df.columns:
            mkt = market_df["close"].pct_change().reindex(ret.index).dropna()
            common = ret.reindex(mkt.index).dropna()
            mkt = mkt.reindex(common.index)
            if len(common) > 10:
                cov = np.cov(common, mkt)
                beta = cov[0, 1] / (cov[1, 1] + 1e-9) if cov[1, 1] > 1e-9 else 1.0
                resid = common - beta * mkt
                return float(-resid.std())   # 低残差波动=高分
        return float(-ret.std())

    def _zscore_cross_section(self, scores: Dict[str, Dict]) -> Dict[str, float]:
        """截面Z-score + 加权合成"""
        if not scores:
            return {}
        df = pd.DataFrame(scores).T.astype(float)
        for col in df.columns:
            s = df[col]
            std = s.std()
            df[col] = (s - s.mean()) / (std + 1e-9) if std > 1e-9 else 0.0

        weights = {
            "ep": self.ep_weight, "growth": self.growth_w,
            "mom": self.mom_w, "quality": self.quality_w,
            "rev": self.rev_w, "liq": self.liq_w, "resvol": self.res_vol_w,
        }
        total_w = sum(weights.values())
        composite = {}
        for code in df.index:
            s = sum(df.loc[code].get(fn, 0.0) * w for fn, w in weights.items())
            composite[code] = s / total_w
        return composite

    def generate_signals(self, universe, market_data, factor_data,
                         current_date, positions,
                         fundamental_data: Optional[Dict] = None,
                         index_df: Optional[pd.DataFrame] = None,
                         **kwargs) -> List[Signal]:
        signals = []
        raw_scores: Dict[str, Dict] = {}

        for code in universe:
            df  = market_data.get(code)
            fd  = factor_data.get(code)
            fun = (fundamental_data or {}).get(code)

            if df is None or len(df) < 20:
                continue

            close = df["close"]
            pct   = close.pct_change()

            ep      = self._compute_ep(fun)
            growth  = float(fun.get("net_profit_growth", 0.0) or 0.0) if fun else 0.0
            mom     = float(pct.tail(20).add(1).prod() - 1) if len(pct) >= 20 else 0.0
            quality = float(fun.get("roe_ttm", 0.0) or 0.0) if fun else 0.0
            rev     = -float(pct.tail(5).sum()) if len(pct) >= 5 else 0.0

            # Amihud 非流动性（负向，越低越好）
            if "amount" in df.columns:
                amt = df["amount"].tail(20).replace(0, np.nan)
                liq = float(-(pct.tail(20).abs() / amt).mean()) if not amt.isna().all() else 0.0
            else:
                liq = 0.0

            resvol = self._compute_resid_vol(df, index_df)

            raw_scores[code] = {
                "ep": ep, "growth": growth, "mom": mom,
                "quality": quality, "rev": rev, "liq": liq, "resvol": resvol,
            }

        # 截面标准化 + 加权
        composite = self._zscore_cross_section(raw_scores)

        # 退出低分持仓
        for code in list(positions.keys()):
            if composite.get(code, -999) < -0.5:
                signals.append(self._make_sell(code, current_date, "多因子综合分偏低"))

        # 买入 Top-N（风险平价权重需外部传入波动率，此处简化等权）
        top = sorted(composite, key=composite.__getitem__, reverse=True)[: self.top_n]
        weight = 1.0 / max(len(top), 1)
        for code in top:
            if code not in positions and composite[code] > 0.3:
                signals.append(self._make_buy(code, composite[code], weight,
                                              current_date, f"AlphaMax多因子{composite[code]:.2f}"))
        return signals


# ============================================================================
# 实时信号接口 (V7.4 新增 + V7.5 缓存优化)
# ============================================================================

class RealtimeSignalMixin:
    """
    为策略提供实时信号生成接口的混入类 (V7.4)
    generate_realtime_signal(code, df, current_price) -> (signal, score, reason)
      signal: "buy" / "sell" / "hold"
      score:  0.0 ~ 1.0 置信度
      reason: 触发原因描述
    """

    def generate_realtime_signal(
        self,
        code: str,
        df: pd.DataFrame,
        current_price: float,
    ):
        """
        默认实现：基于最新因子值判断方向
        子类可覆盖此方法提供更精确的实时信号
        """
        return "hold", 0.0, "base_default"


# ── 为每个策略追加 generate_realtime_signal (V7.5 增加缓存) ──────────────────

class RSRSMomentumStrategy(RSRSMomentumStrategy, RealtimeSignalMixin):
    """V7.4: 追加实时信号接口 / V7.5: 增加60秒缓存"""

    def __init__(self, config=None):
        super().__init__(config)
        self._cache_lock = threading.Lock()
        self._last_rsrs: Dict[str, tuple] = {}   # code -> (timestamp, value)

    def generate_realtime_signal(self, code: str, df: pd.DataFrame, current_price: float):
        if df is None or len(df) < 40:
            return "hold", 0.0, "数据不足"
        # V7.5: check 60s cache
        now = time.time()
        with self._cache_lock:
            if code in self._last_rsrs:
                ts, cached_v = self._last_rsrs[code]
                if now - ts < 60.0:
                    v = cached_v
                    if v > self.rsrs_threshold:
                        score = min(v / (self.rsrs_threshold * 3), 1.0)
                        return "buy", score, f"RSRS自适应(缓存)={v:.3f}>{self.rsrs_threshold}"
                    elif v < -self.rsrs_threshold:
                        score = min(abs(v) / (self.rsrs_threshold * 3), 1.0)
                        return "sell", score, f"RSRS自适应(缓存)={v:.3f}<-{self.rsrs_threshold}"
                    return "hold", 0.0, f"RSRS自适应(缓存)={v:.3f}中性区间"
        try:
            from ..factors.technical.rsrs import compute_rsrs
            df2 = df.copy()
            df2.loc[len(df2)] = {
                "open": current_price, "high": current_price,
                "low": current_price, "close": current_price,
                "volume": 0, "amount": 0,
            }
            fdf = compute_rsrs(df2, regression_window=18, zscore_window=600)
            ra = fdf["rsrs_adaptive"].dropna()
            if ra.empty:
                return "hold", 0.0, "因子为空"
            v = float(ra.iloc[-1])
            # V7.5: update cache
            with self._cache_lock:
                self._last_rsrs[code] = (time.time(), v)
            if v > self.rsrs_threshold:
                score = min(v / (self.rsrs_threshold * 3), 1.0)
                return "buy", score, f"RSRS自适应={v:.3f}>{self.rsrs_threshold}"
            elif v < -self.rsrs_threshold:
                score = min(abs(v) / (self.rsrs_threshold * 3), 1.0)
                return "sell", score, f"RSRS自适应={v:.3f}<-{self.rsrs_threshold}"
            return "hold", 0.0, f"RSRS自适应={v:.3f}中性区间"
        except Exception as e:
            return "hold", 0.0, f"计算异常: {e}"


class AlphaHunterStrategy(AlphaHunterStrategy, RealtimeSignalMixin):
    """V7.5: 追加实时信号接口（含线程安全缓存，60s TTL）"""

    def __init__(self, config=None):
        super().__init__(config)
        self._cache_lock = threading.Lock()
        self._last_score: Dict[str, tuple] = {}   # code -> (timestamp, score)

    def generate_realtime_signal(self, code: str, df: pd.DataFrame, current_price: float):
        if df is None or len(df) < 60:
            return "hold", 0.0, "数据不足"
        now = time.time()
        with self._cache_lock:
            if code in self._last_score:
                ts, cached_s = self._last_score[code]
                if now - ts < 60.0:
                    score = cached_s
                    if score >= self.min_score:
                        norm = min(score / (self.min_score * 3), 1.0)
                        return "buy", norm, f"多因子综合(缓存)={score:.3f}>={self.min_score}"
                    elif score <= -self.min_score:
                        norm = min(abs(score) / (self.min_score * 3), 1.0)
                        return "sell", norm, f"多因子综合(缓存)={score:.3f}<=-{self.min_score}"
                    return "hold", 0.0, f"多因子综合(缓存)={score:.3f}中性"
        try:
            from ..factors.alpha_engine import AlphaEngine
            fdf = AlphaEngine.compute_from_history(df)
            score = self._get_score(fdf)
            with self._cache_lock:
                self._last_score[code] = (time.time(), score)
            if score >= self.min_score:
                norm = min(score / (self.min_score * 3), 1.0)
                return "buy", norm, f"多因子综合={score:.3f}>={self.min_score}"
            elif score <= -self.min_score:
                norm = min(abs(score) / (self.min_score * 3), 1.0)
                return "sell", norm, f"多因子综合={score:.3f}<=-{self.min_score}"
            return "hold", 0.0, f"多因子综合={score:.3f}中性"
        except Exception as e:
            return "hold", 0.0, f"计算异常: {e}"


class RSRSAdvancedStrategy(RSRSAdvancedStrategy, RealtimeSignalMixin):
    """V7.5: 追加实时信号接口（含线程安全缓存，60s TTL）"""

    def __init__(self, config=None):
        super().__init__(config)
        self._cache_lock = threading.Lock()
        self._last_rsrs_adv: Dict[str, tuple] = {}   # code -> (timestamp, v, r2v)

    def generate_realtime_signal(self, code: str, df: pd.DataFrame, current_price: float):
        if df is None or len(df) < 40:
            return "hold", 0.0, "数据不足"
        now = time.time()
        with self._cache_lock:
            if code in self._last_rsrs_adv:
                ts, cached_v, cached_r2 = self._last_rsrs_adv[code]
                if now - ts < 60.0:
                    v, r2v = cached_v, cached_r2
                    if r2v < self.r2_threshold:
                        return "hold", 0.0, f"R²(缓存)={r2v:.2f}<{self.r2_threshold}过滤"
                    if v > self.rsrs_threshold:
                        score = min(v * r2v / (self.rsrs_threshold * 2), 1.0)
                        return "buy", score, f"RSRS高级(缓存) v={v:.3f} R²={r2v:.2f}"
                    elif v < -self.rsrs_threshold:
                        score = min(abs(v) * r2v / (self.rsrs_threshold * 2), 1.0)
                        return "sell", score, f"RSRS高级卖出(缓存) v={v:.3f}"
                    return "hold", 0.0, "中性(缓存)"
        try:
            from ..factors.technical.rsrs import compute_rsrs
            df2 = df.copy()
            df2.loc[len(df2)] = {
                "open": current_price, "high": current_price,
                "low": current_price, "close": current_price,
                "volume": 0, "amount": 0,
            }
            fdf = compute_rsrs(df2, regression_window=18, zscore_window=600)
            ra = fdf["rsrs_adaptive"].dropna()
            r2 = fdf["rsrs_r2"].dropna()
            if ra.empty or r2.empty:
                return "hold", 0.0, "因子为空"
            v = float(ra.iloc[-1])
            r2v = float(r2.iloc[-1])
            with self._cache_lock:
                self._last_rsrs_adv[code] = (time.time(), v, r2v)
            if r2v < self.r2_threshold:
                return "hold", 0.0, f"R²={r2v:.2f}<{self.r2_threshold}过滤"
            if v > self.rsrs_threshold:
                score = min(v * r2v / (self.rsrs_threshold * 2), 1.0)
                return "buy", score, f"RSRS高级 v={v:.3f} R²={r2v:.2f}"
            elif v < -self.rsrs_threshold:
                score = min(abs(v) * r2v / (self.rsrs_threshold * 2), 1.0)
                return "sell", score, f"RSRS高级卖出 v={v:.3f}"
            return "hold", 0.0, "中性"
        except Exception as e:
            return "hold", 0.0, f"计算异常: {e}"


class ShortTermStrategy(ShortTermStrategy, RealtimeSignalMixin):
    """V7.4: 追加实时信号接口"""

    def generate_realtime_signal(self, code: str, df: pd.DataFrame, current_price: float):
        if df is None or len(df) < 25:
            return "hold", 0.0, "数据不足"
        try:
            close = df["close"].values
            prev_close = close[-1]
            mom = (current_price - prev_close) / prev_close if prev_close > 0 else 0.0
            mom_5 = (current_price / close[-5] - 1) if len(close) >= 5 and close[-5] > 0 else 0.0
            if mom > self.mom_threshold and mom_5 > self.mom_threshold:
                score = min((mom + mom_5) / (self.mom_threshold * 4), 1.0)
                return "buy", score, f"短线动量 mom={mom:.3f} mom5={mom_5:.3f}"
            elif mom_5 < -self.mom_threshold * 2:
                score = min(abs(mom_5) / (self.mom_threshold * 4), 1.0)
                return "sell", score, f"短线反转 mom5={mom_5:.3f}"
            return "hold", 0.0, "动量不足"
        except Exception as e:
            return "hold", 0.0, f"计算异常: {e}"


class MomentumReversalStrategy(MomentumReversalStrategy, RealtimeSignalMixin):
    """V7.4: 追加实时信号接口"""

    def generate_realtime_signal(self, code: str, df: pd.DataFrame, current_price: float):
        if df is None or len(df) < 20:
            return "hold", 0.0, "数据不足"
        try:
            close = df["close"].values
            if len(close) < 20 or close[-20] <= 0:
                return "hold", 0.0, "数据不足"
            mom20 = current_price / close[-20] - 1
            mom5  = current_price / close[-5] - 1 if len(close) >= 5 and close[-5] > 0 else 0.0
            # 市场整体动量用最近均线判断
            ma20 = float(np.mean(close[-20:]))
            if current_price > ma20 and mom20 > 0:
                score = min(mom20 * 5, 1.0)
                return "buy", score, f"牛市动量 20日收益={mom20:.3f}"
            elif current_price < ma20 and mom5 < -0.05:
                score = min(abs(mom5) * 5, 1.0)
                return "buy", score, f"熊市反转 5日收益={mom5:.3f}"
            elif mom20 < -0.08:
                return "sell", min(abs(mom20) * 3, 1.0), f"趋势下行={mom20:.3f}"
            return "hold", 0.0, "无明确方向"
        except Exception as e:
            return "hold", 0.0, f"计算异常: {e}"


class SentimentReversalStrategy(SentimentReversalStrategy, RealtimeSignalMixin):
    """V7.4: 追加实时信号接口"""

    def generate_realtime_signal(self, code: str, df: pd.DataFrame, current_price: float):
        if df is None or len(df) < 30:
            return "hold", 0.0, "数据不足"
        try:
            from ..factors.technical.rsrs import compute_rsrs
            fdf = compute_rsrs(df, regression_window=18, zscore_window=600)
            zs = fdf["rsrs_zscore"].dropna()
            if zs.empty:
                return "hold", 0.0, "zscore为空"
            z = float(zs.iloc[-1])
            if z < self.oversold_z:
                score = min(abs(z - self.oversold_z) / 2.0, 1.0)
                return "buy", score, f"情绪超卖 zscore={z:.2f}<{self.oversold_z}"
            elif z > self.overbought_z:
                score = min((z - self.overbought_z) / 2.0, 1.0)
                return "sell", score, f"情绪超买 zscore={z:.2f}>{self.overbought_z}"
            return "hold", 0.0, f"情绪中性 zscore={z:.2f}"
        except Exception as e:
            return "hold", 0.0, f"计算异常: {e}"


class KunpengV10Strategy(KunpengV10Strategy, RealtimeSignalMixin):
    """V7.4: 追加实时信号接口"""

    def generate_realtime_signal(self, code: str, df: pd.DataFrame, current_price: float):
        if df is None or len(df) < self.illiq_window:
            return "hold", 0.0, "数据不足"
        try:
            df2 = df.copy()
            df2.loc[len(df2)] = {
                "open": current_price, "high": current_price,
                "low": current_price, "close": current_price,
                "volume": df["volume"].tail(5).mean() if "volume" in df.columns else 1e6,
                "amount": df["amount"].tail(5).mean() if "amount" in df.columns else 1e7,
            }
            sm  = self._smart_money(df2)
            asi = self._amihud_stable(df2)
            gp  = self._gap_penalty(df2)
            composite = 0.5 * sm + 0.3 * asi + 0.2 * gp
            if composite > 0.15:
                score = min(composite * 2, 1.0)
                return "buy", score, f"微结构得分={composite:.3f}(sm={sm:.2f} asi={asi:.2f})"
            elif composite < -0.2:
                score = min(abs(composite) * 2, 1.0)
                return "sell", score, f"微结构退化={composite:.3f}"
            return "hold", 0.0, f"微结构中性={composite:.3f}"
        except Exception as e:
            return "hold", 0.0, f"计算异常: {e}"


class AlphaMaxV5FixedStrategy(AlphaMaxV5FixedStrategy, RealtimeSignalMixin):
    """V7.4: 追加实时信号接口"""

    def generate_realtime_signal(self, code: str, df: pd.DataFrame, current_price: float):
        if df is None or len(df) < 20:
            return "hold", 0.0, "数据不足"
        try:
            close = df["close"]
            pct = close.pct_change()
            # 简化版: 仅使用可即时计算的因子
            mom = float(pct.tail(20).add(1).prod() - 1) if len(pct) >= 20 else 0.0
            current_ret = (current_price - float(close.iloc[-1])) / float(close.iloc[-1]) if close.iloc[-1] > 0 else 0.0
            rev = -(float(pct.tail(5).sum()) + current_ret)
            if "amount" in df.columns:
                amt = df["amount"].tail(20).replace(0, float("nan"))
                liq = float(-(pct.tail(20).abs() / amt).mean()) if not amt.isna().all() else 0.0
            else:
                liq = 0.0
            # 简化多因子合成
            score_raw = self.mom_w * mom + self.rev_w * rev + self.liq_w * liq
            score_norm = max(min(score_raw * 5, 1.0), -1.0)
            if score_norm > 0.3:
                return "buy", score_norm, f"AlphaMax实时 mom={mom:.3f} rev={rev:.3f}"
            elif score_norm < -0.3:
                return "sell", abs(score_norm), f"AlphaMax实时偏空 score={score_norm:.3f}"
            return "hold", 0.0, f"AlphaMax实时中性 score={score_norm:.3f}"
        except Exception as e:
            return "hold", 0.0, f"计算异常: {e}"


# ============================================================================
# 策略注册表 (V7.4 使用带实时接口的覆盖版本)
# ============================================================================

STRATEGY_REGISTRY: Dict[str, type] = {
    "rsrs_momentum":       RSRSMomentumStrategy,
    "alpha_hunter":        AlphaHunterStrategy,
    "rsrs_advanced":       RSRSAdvancedStrategy,
    "short_term":          ShortTermStrategy,
    "momentum_reversal":   MomentumReversalStrategy,
    "sentiment_reversal":  SentimentReversalStrategy,
    "kunpeng_v10":         KunpengV10Strategy,
    "alpha_max_v5_fixed":  AlphaMaxV5FixedStrategy,
}

STRATEGY_DISPLAY_NAMES: Dict[str, str] = {
    "rsrs_momentum":       "RSRS动量策略",
    "alpha_hunter":        "Alpha猎手策略",
    "rsrs_advanced":       "高级RSRS策略",
    "short_term":          "短线快进快出",
    "momentum_reversal":   "动量反转双模式",
    "sentiment_reversal":  "情绪反转策略",
    "kunpeng_v10":         "鲲鹏V10微结构",
    "alpha_max_v5_fixed":  "AlphaMaxV5机构多因子",
}

STRATEGY_TUNABLE_PARAMS: Dict[str, Dict] = {
    "rsrs_momentum": {
        "top_n":           {"type": "int",   "default": 10,  "desc": "最大持仓只数"},
        "rsrs_threshold":  {"type": "float", "default": 0.5, "desc": "RSRS自适应阈值"},
    },
    "alpha_hunter": {
        "top_n":      {"type": "int",   "default": 15,  "desc": "最大持仓只数"},
        "min_score":  {"type": "float", "default": 0.3, "desc": "最低综合评分"},
    },
    "rsrs_advanced": {
        "top_n":           {"type": "int",   "default": 10,  "desc": "最大持仓只数"},
        "rsrs_threshold":  {"type": "float", "default": 0.5, "desc": "RSRS阈值"},
        "r2_threshold":    {"type": "float", "default": 0.7, "desc": "R²过滤阈值"},
    },
    "short_term": {
        "top_n":               {"type": "int",   "default": 5,    "desc": "最大持仓只数"},
        "hold_calendar_days":  {"type": "int",   "default": 7,    "desc": "最大持仓日历天数"},
        "mom_threshold":       {"type": "float", "default": 0.03, "desc": "动量触发阈值"},
    },
    "momentum_reversal": {
        "top_n":         {"type": "int",   "default": 10,  "desc": "最大持仓只数"},
        "market_thresh": {"type": "float", "default": 0.0, "desc": "市场牛熊判断阈值"},
    },
    "sentiment_reversal": {
        "top_n":         {"type": "int",   "default": 10,   "desc": "最大持仓只数"},
        "oversold_z":    {"type": "float", "default": -1.5, "desc": "超卖Z-score阈值"},
        "overbought_z":  {"type": "float", "default": 1.5,  "desc": "超买Z-score阈值"},
    },
    "kunpeng_v10": {
        "top_n":          {"type": "int",   "default": 15,   "desc": "最大持仓只数"},
        "illiq_window":   {"type": "int",   "default": 20,   "desc": "非流动性计算窗口"},
        "smart_window":   {"type": "int",   "default": 10,   "desc": "聪明钱计算窗口"},
        "breadth_limit":  {"type": "float", "default": 0.15, "desc": "宽度熔断跌停比例阈值"},
        "use_breadth_check": {"type": "bool", "default": True, "desc": "是否启用宽度熔断"},  # V7.5 新增布尔参数示例
    },
    "alpha_max_v5_fixed": {
        "top_n":       {"type": "int",   "default": 20,   "desc": "最大持仓只数"},
        "ep_weight":   {"type": "float", "default": 0.20, "desc": "EP因子权重"},
        "growth_w":    {"type": "float", "default": 0.15, "desc": "成长因子权重"},
        "mom_w":       {"type": "float", "default": 0.15, "desc": "动量因子权重"},
        "quality_w":   {"type": "float", "default": 0.20, "desc": "质量因子权重"},
        "rev_w":       {"type": "float", "default": 0.10, "desc": "反转因子权重"},
        "liq_w":       {"type": "float", "default": 0.10, "desc": "流动性因子权重"},
        "res_vol_w":   {"type": "float", "default": 0.10, "desc": "残差波动率权重"},
    },
}


def create_strategy(name: str, config: Optional[Dict] = None) -> BaseStrategy:
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"未知策略: {name}，可用: {list(STRATEGY_REGISTRY.keys())}")
    return cls(config)

# ============================================================================
# V7.7 新增: SectorMomentumStrategy (板块共振策略示例)
# ============================================================================

class SectorMomentumStrategy(RSRSMomentumStrategy):
    """
    结合板块动量的 RSRS 策略 (V7.7 示例)
    =====================================
    评分公式:
        final_score = (1 - sector_weight) * rsrs_score
                    + sector_weight * sector_momentum

    所需额外参数（通过 **kwargs 传入 generate_signals）:
        sector_map  : Dict[str, str]        — {stock_code: sector_name}
        sector_data : Dict[str, pd.DataFrame] — {sector_name: 板块日线 DataFrame}
                      板块日线 DataFrame 需包含 "close" 列，index 为日期字符串

    配置参数 (config 字典):
        sector_weight          (float, 默认 0.3)  — 板块动量权重
        sector_momentum_window (int,   默认 20)   — 板块动量计算窗口（交易日）
        top_n                  (int,   默认 10)   — 最大买入只数
        rsrs_threshold         (float, 默认 0.5)  — RSRS 买入阈值

    使用示例（在 run_single_backtest 中注入板块数据）:
        signals = strategy.generate_signals(
            universe, market_data, factor_data, current_date, positions,
            sector_map=sector_map, sector_data=sector_data,
        )

    注意:
        若未传入 sector_map / sector_data，则退化为标准 RSRSMomentumStrategy，
        板块动量得分默认为 0，即全部使用 RSRS 分数排序。
    """

    name = "SectorMomentum"

    def __init__(self, config: Optional[Dict] = None) -> None:
        super().__init__(config)
        # 板块动量权重（0~1），剩余权重分配给 RSRS 分数
        self.sector_weight = float(self.config.get("sector_weight", 0.3))
        # 板块动量计算窗口（N日收益率）
        self.sector_momentum_window = int(self.config.get("sector_momentum_window", 20))

    def _calc_sector_momentum(
        self,
        sector_name: str,
        sector_data: Dict[str, pd.DataFrame],
        current_date: datetime,
    ) -> float:
        """
        计算指定板块在 current_date 前 sector_momentum_window 个交易日的动量。

        动量定义: (最新收盘 / N日前收盘) - 1
        返回 0.0 如果数据不足或异常。
        """
        try:
            df = sector_data.get(sector_name)
            if df is None or df.empty or "close" not in df.columns:
                return 0.0

            # 确保按日期排序，过滤截止 current_date 前的数据（防前视偏差）
            cur_str = current_date.strftime("%Y-%m-%d")
            if isinstance(df.index, pd.DatetimeIndex):
                df = df[df.index.strftime("%Y-%m-%d") < cur_str]
            else:
                df = df[df.index.astype(str) < cur_str]

            close = df["close"].dropna()
            if len(close) < self.sector_momentum_window + 1:
                return 0.0

            latest = float(close.iloc[-1])
            past   = float(close.iloc[-self.sector_momentum_window - 1])
            if past <= 0:
                return 0.0

            momentum = latest / past - 1.0
            # 将动量归一化到 [-1, 1]（简单截断）
            return float(max(min(momentum * 5, 1.0), -1.0))

        except Exception as e:
            logger.debug("板块动量计算异常 [%s]: %s", sector_name, e)
            return 0.0

    def generate_signals(
        self,
        universe: List[str],
        market_data: Dict[str, pd.DataFrame],
        factor_data: Dict[str, pd.DataFrame],
        current_date: datetime,
        positions: Dict[str, Any],
        **kwargs,
    ) -> List[Signal]:
        """
        生成信号：融合 RSRS 分数与板块动量分数。

        **kwargs 支持:
            sector_map  (Dict[str, str])          — 股票→板块名称映射
            sector_data (Dict[str, pd.DataFrame]) — 板块名称→日线 DataFrame
        """
        # 从 kwargs 中提取板块数据，支持降级（无板块数据时退化为纯 RSRS）
        sector_map:  Dict[str, str]          = kwargs.get("sector_map",  {})
        sector_data: Dict[str, pd.DataFrame] = kwargs.get("sector_data", {})

        signals = []
        composite_scores: Dict[str, float] = {}

        # ── 计算各股票的综合评分 ──────────────────────────────────────────
        for code in universe:
            fd = factor_data.get(code)
            if fd is None or fd.empty or "rsrs_adaptive" not in fd.columns:
                continue

            # RSRS 分数（T-1 数据，防前视偏差）
            vals = fd["rsrs_adaptive"].dropna()
            if vals.empty:
                continue
            rsrs_score = float(vals.iloc[-1])

            # 若低于阈值且不在持仓中，直接跳过（优化：减少无效板块查询）
            if rsrs_score <= self.rsrs_threshold and code not in positions:
                continue

            # 板块动量分数
            sector_name     = sector_map.get(code, "")
            sector_mom_score = 0.0
            if sector_name and sector_data:
                sector_mom_score = self._calc_sector_momentum(
                    sector_name, sector_data, current_date
                )

            # 综合评分（加权融合）
            w_sector = self.sector_weight if sector_name else 0.0
            w_rsrs   = 1.0 - w_sector
            composite = w_rsrs * rsrs_score + w_sector * sector_mom_score

            composite_scores[code] = composite

        # ── 退出信号: 综合评分跌破负阈值的持仓 ──────────────────────────
        for code in list(positions.keys()):
            score = composite_scores.get(code)
            if score is None:
                # 因子数据缺失的持仓：不强制卖出，保留
                continue
            if score < -self.rsrs_threshold:
                signals.append(
                    self._make_sell(code, current_date,
                                    f"SectorMomentum退出 综合分={score:.3f}")
                )

        # ── 买入信号: Top-N 综合分最高且未持仓 ───────────────────────────
        buy_candidates = {
            code: score
            for code, score in composite_scores.items()
            if score > self.rsrs_threshold and code not in positions
        }
        top_codes = sorted(buy_candidates, key=buy_candidates.__getitem__,
                           reverse=True)[: self.top_n]
        weight = 1.0 / max(len(top_codes), 1)

        for code in top_codes:
            signals.append(
                self._make_buy(
                    code, composite_scores[code], weight, current_date,
                    f"SectorMomentum买入 综合分={composite_scores[code]:.3f} "
                    f"板块={sector_map.get(code, '未知')}"
                )
            )

        return signals


# ── 更新注册表（V7.7 新增板块共振策略）───────────────────────────────────
STRATEGY_REGISTRY["sector_momentum"] = SectorMomentumStrategy

STRATEGY_DISPLAY_NAMES["sector_momentum"] = "板块共振RSRS策略"

STRATEGY_TUNABLE_PARAMS["sector_momentum"] = {
    "top_n":                  {"type": "int",   "default": 10,  "desc": "最大持仓只数"},
    "rsrs_threshold":         {"type": "float", "default": 0.5, "desc": "RSRS买入阈值"},
    "sector_weight":          {"type": "float", "default": 0.3, "desc": "板块动量权重(0~1)"},
    "sector_momentum_window": {"type": "int",   "default": 20,  "desc": "板块动量计算窗口(交易日)"},
}