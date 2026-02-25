#!/usr/bin/env python3
"""
Q-UNITY-V6 基本面数据获取模块 v2.1
特性:
  - AKShare stock_financial_analysis_indicator 真正 TTM 数据
  - 字段级 try/except 隔离，单字段失败不影响其他
  - 季度→TTM 滚动降级路径
  - 缓存版本校验 (CACHE_VERSION=2)
  - 输出字段: pe_ttm, pb_lf, roe_ttm, net_profit_ttm,
              revenue_growth, net_profit_growth, circ_mv
"""
from __future__ import annotations
import logging
import json
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

CACHE_VERSION = 2          # 升版本号可强制刷新旧缓存
CACHE_TTL     = 24 * 3600  # 24小时


class FundamentalDataProvider:
    """基本面数据提供者（v2.1 TTM精准化）"""

    def __init__(
        self,
        cache_dir: str = "./data/cache/fundamental",
        cache_ttl: int = CACHE_TTL,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = cache_ttl
        self._mem: Dict[str, Tuple[float, Any]] = {}  # 内存二级缓存

    # ── 缓存 ──────────────────────────────────────────────────────────────

    def _cache_key(self, code: str) -> str:
        raw = f"v{CACHE_VERSION}:{code}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_path(self, code: str) -> Path:
        return self.cache_dir / f"{self._cache_key(code)}.json"

    def _load_cache(self, code: str) -> Optional[Dict]:
        key = self._cache_key(code)
        # 内存缓存
        if key in self._mem:
            ts, data = self._mem[key]
            if time.time() - ts < self.cache_ttl:
                return data
        # 磁盘缓存
        path = self._cache_path(code)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            # 版本校验
            if raw.get("__version__") != CACHE_VERSION:
                path.unlink(missing_ok=True)
                return None
            if time.time() - raw.get("__ts__", 0) > self.cache_ttl:
                path.unlink(missing_ok=True)
                return None
            data = {k: v for k, v in raw.items() if not k.startswith("__")}
            self._mem[key] = (raw["__ts__"], data)
            return data
        except Exception:
            return None

    def _save_cache(self, code: str, data: Dict) -> None:
        key = self._cache_key(code)
        ts  = time.time()
        self._mem[key] = (ts, data)
        path = self._cache_path(code)
        try:
            out = {"__version__": CACHE_VERSION, "__ts__": ts}
            out.update(data)
            path.write_text(json.dumps(out, ensure_ascii=False, default=float), encoding="utf-8")
        except Exception as e:
            logger.debug(f"写入缓存失败 {code}: {e}")

    # ── 主接口 ────────────────────────────────────────────────────────────

    def get_fundamental(self, code: str) -> Optional[Dict]:
        """
        获取股票基本面指标（TTM口径）
        返回字段:
            pe_ttm            — 市盈率(TTM)
            pb_lf             — 市净率(LF)
            roe_ttm           — 净资产收益率(TTM)
            net_profit_ttm    — 净利润TTM (万元)
            revenue_growth    — 营收同比增长率
            net_profit_growth — 净利润同比增长率
            circ_mv           — 流通市值(万元)
        """
        # 查缓存
        cached = self._load_cache(code)
        if cached:
            return cached

        result: Dict[str, Optional[float]] = {
            "pe_ttm": None, "pb_lf": None, "roe_ttm": None,
            "net_profit_ttm": None, "revenue_growth": None,
            "net_profit_growth": None, "circ_mv": None,
        }

        # ── 路径一: AKShare 真正TTM ───────────────────────────────────────
        try:
            result = self._fetch_via_akshare_ttm(code, result)
        except Exception as e:
            logger.debug(f"AKShare TTM 路径失败 {code}: {e}")

        # ── 路径二: 季度→TTM 滚动降级 ────────────────────────────────────
        missing = [k for k, v in result.items() if v is None]
        if missing:
            try:
                result = self._fetch_via_quarterly_fallback(code, result)
            except Exception as e:
                logger.debug(f"季度降级路径失败 {code}: {e}")

        # ── 路径三: 实时行情补充市值/PE ──────────────────────────────────
        if result.get("pe_ttm") is None or result.get("circ_mv") is None:
            try:
                result = self._fetch_realtime_supplement(code, result)
            except Exception as e:
                logger.debug(f"实时行情补充失败 {code}: {e}")

        if any(v is not None for v in result.values()):
            self._save_cache(code, result)
        return result

    def get_batch(self, codes: list) -> Dict[str, Optional[Dict]]:
        """批量获取基本面数据"""
        out = {}
        for code in codes:
            out[code] = self.get_fundamental(code)
            time.sleep(0.05)
        return out

    # ── 内部采集方法 ─────────────────────────────────────────────────────

    def _fetch_via_akshare_ttm(self, code: str, result: Dict) -> Dict:
        """
        从 AKShare stock_financial_analysis_indicator 获取真正TTM指标
        此接口直接返回 TTM/LF 口径数据，无需自行滚动计算
        """
        import akshare as ak
        df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2020")
        if df is None or df.empty:
            return result
        # 取最新一行
        row = df.iloc[-1]

        # pe_ttm — 字段名可能因版本变化
        for col_pe in ["市盈率(TTM)", "PE(TTM)", "pe_ttm", "市盈率TTM"]:
            if col_pe in row.index:
                try:
                    result["pe_ttm"] = float(row[col_pe])
                    break
                except Exception:
                    pass

        # pb_lf
        for col_pb in ["市净率", "PB", "pb", "pb_lf"]:
            if col_pb in row.index:
                try:
                    result["pb_lf"] = float(row[col_pb])
                    break
                except Exception:
                    pass

        # roe_ttm
        for col_roe in ["净资产收益率(TTM)", "ROE(TTM)", "roe_ttm", "加权净资产收益率"]:
            if col_roe in row.index:
                try:
                    v = float(row[col_roe])
                    result["roe_ttm"] = v / 100.0 if v > 1.0 else v
                    break
                except Exception:
                    pass

        # net_profit_ttm (万元)
        for col_np in ["净利润(TTM)", "归母净利润(TTM)", "净利润TTM"]:
            if col_np in row.index:
                try:
                    result["net_profit_ttm"] = float(row[col_np])
                    break
                except Exception:
                    pass

        # revenue / profit growth
        for col_rg in ["营收同比", "营业收入同比增长率", "revenue_growth"]:
            if col_rg in row.index:
                try:
                    v = float(row[col_rg])
                    result["revenue_growth"] = v / 100.0 if abs(v) > 1.5 else v
                    break
                except Exception:
                    pass

        for col_pg in ["净利润同比", "归母净利润同比增长率", "net_profit_growth"]:
            if col_pg in row.index:
                try:
                    v = float(row[col_pg])
                    result["net_profit_growth"] = v / 100.0 if abs(v) > 1.5 else v
                    break
                except Exception:
                    pass

        return result

    def _fetch_via_quarterly_fallback(self, code: str, result: Dict) -> Dict:
        """季度报表→手工滚动TTM（降级路径）"""
        try:
            import akshare as ak
            # 利润表季度数据
            df = ak.stock_profit_statement_by_report_em(symbol=code)
            if df is None or df.empty:
                return result
            # 按报告期排序
            date_col = [c for c in df.columns if "报告期" in c or "date" in c.lower()]
            if date_col:
                df[date_col[0]] = pd.to_datetime(df[date_col[0]], errors="coerce")
                df = df.sort_values(date_col[0])
            # 取最近4期滚动求和 = TTM净利润
            np_col = [c for c in df.columns if "净利润" in c and "归母" in c]
            if np_col and result.get("net_profit_ttm") is None:
                vals = pd.to_numeric(df[np_col[0]].tail(4), errors="coerce").dropna()
                if len(vals) >= 4:
                    result["net_profit_ttm"] = float(vals.sum()) / 1e4  # 转万元
        except Exception as e:
            logger.debug(f"季度降级-净利润 {code}: {e}")

        # 同比增长（两年对比）
        try:
            import akshare as ak
            df = ak.stock_financial_benefit_ths(symbol=code, indicator="按年度")
            if df is not None and not df.empty and len(df) >= 2:
                rev_col = [c for c in df.columns if "营业总收入" in c or "营收" in c]
                npf_col = [c for c in df.columns if "归母净利润" in c or "净利润" in c]
                if rev_col and result.get("revenue_growth") is None:
                    vals = pd.to_numeric(df[rev_col[0]].head(2), errors="coerce")
                    if not vals.isna().any() and vals.iloc[1] > 0:
                        result["revenue_growth"] = float(vals.iloc[0] / vals.iloc[1] - 1)
                if npf_col and result.get("net_profit_growth") is None:
                    vals = pd.to_numeric(df[npf_col[0]].head(2), errors="coerce")
                    if not vals.isna().any() and vals.iloc[1] > 0:
                        result["net_profit_growth"] = float(vals.iloc[0] / vals.iloc[1] - 1)
        except Exception as e:
            logger.debug(f"季度降级-增速 {code}: {e}")

        return result

    def _fetch_realtime_supplement(self, code: str, result: Dict) -> Dict:
        """实时行情补充 PE / 流通市值"""
        try:
            import akshare as ak
            df = ak.stock_individual_info_em(symbol=code)
            if df is None or df.empty:
                return result
            # 展开 key-value 结构
            info = {}
            if "item" in df.columns and "value" in df.columns:
                info = dict(zip(df["item"].astype(str), df["value"].astype(str)))
            elif len(df.columns) == 2:
                info = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))

            def safe_float(s: str) -> Optional[float]:
                try:
                    s = s.replace(",", "").replace("亿", "").strip()
                    return float(s)
                except Exception:
                    return None

            for k in ["市盈率(动)", "PE(TTM)", "市盈率TTM"]:
                if k in info and result.get("pe_ttm") is None:
                    v = safe_float(info[k])
                    if v is not None:
                        result["pe_ttm"] = v

            for k in ["流通市值", "市值"]:
                if k in info and result.get("circ_mv") is None:
                    v = safe_float(info[k])
                    if v is not None:
                        result["circ_mv"] = v * 1e4  # 亿→万元
        except Exception as e:
            logger.debug(f"实时行情补充失败 {code}: {e}")
        return result