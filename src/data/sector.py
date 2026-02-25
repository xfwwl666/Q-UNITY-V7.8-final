#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/data/sector.py — Q-UNITY-V7.6 板块数据采集模块
===================================================
提供行业板块、概念板块的列表、日线行情、成分股三类数据采集功能。
数据来源: AKShare（东方财富）

【模块结构】
  fetch_sector_list()          — 获取板块列表 (行业/概念)
  fetch_sector_daily()         — 获取单个板块日线数据
  fetch_sector_constituents()  — 获取板块成分股列表
  SectorDataPipeline           — 管道类，支持批量采集、增量更新、报告

【限流处理】
  与 akshare_client.py 保持一致的退避策略:
    - 检测限流关键词: 429 / 限流 / 频繁 / too many
    - 限流时等待 30s/60s/90s（远长于普通错误的 1s/2s/4s）

【存储结构】
  data/sector/
    list/
      industry.parquet          — 行业板块列表
      concept.parquet           — 概念板块列表
    daily/
      industry/{板块名}.parquet — 行业板块日线数据
      concept/{板块名}.parquet  — 概念板块日线数据
    constituents/
      industry_map.parquet      — 行业成分股映射表 (sector_name, sector_type, code)
      industry_map.csv
      concept_map.parquet       — 概念成分股映射表
      concept_map.csv
"""
from __future__ import annotations

import logging
import random
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ─── 限流关键词（与 akshare_client.py 保持一致）──────────────────────────
_RATELIMIT_KEYWORDS = ("429", "限流", "频繁", "too many", "Too Many", "rate limit")

# ─── AKShare 板块日线列名映射 ────────────────────────────────────────────
# 东方财富行业/概念板块历史行情列名（stock_board_industry_hist_em）
_SECTOR_HIST_COL_MAP = {
    "日期":   "date",
    "开盘":   "open",
    "收盘":   "close",
    "最高":   "high",
    "最低":   "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅":   "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover",
}

# ─── 板块列表列名映射 ────────────────────────────────────────────────────
# stock_board_industry_name_em / stock_board_concept_name_em 返回列名
_SECTOR_LIST_COL_MAP_INDUSTRY = {
    "板块名称": "name",
    "板块代码": "code",
}
_SECTOR_LIST_COL_MAP_CONCEPT = {
    "板块名称": "name",
    "板块代码": "code",
}

# ─── 成分股列名映射 ──────────────────────────────────────────────────────
_CONS_CODE_COLS = ["代码", "股票代码", "证券代码", "code", "symbol"]


def _is_ratelimit_error(exc: Exception) -> bool:
    """判断异常是否由限流引起"""
    msg = str(exc).lower()
    return any(kw.lower() in msg for kw in _RATELIMIT_KEYWORDS)


def _backoff_sleep(attempt: int, ratelimited: bool,
                   delay_min: float, delay_max: float) -> None:
    """
    限流感知退避等待:
      普通错误:  1s/2s/4s (指数退避)
      限流错误: 30s/60s/90s
    """
    if ratelimited:
        wait = 30 * (attempt + 1)
        logger.warning("触发限流，等待 %ds 后重试 (attempt=%d)...", wait, attempt + 1)
    else:
        base = 1.0 * (2 ** attempt)
        wait = base + random.uniform(delay_min, delay_max)
    time.sleep(wait)


def _sanitize_name(name: str) -> str:
    """
    将板块名称转换为安全文件名:
      - 替换 / 为 _
      - 替换其他特殊字符
    """
    name = name.replace("/", "_").replace("\\", "_")
    name = re.sub(r'[<>:"|?*]', "_", name)
    return name.strip()


# ============================================================================
# 独立采集函数
# ============================================================================

def fetch_sector_list(
    sector_type: str,
    max_retries: int = 3,
    delay_min: float = 0.5,
    delay_max: float = 1.0,
) -> Optional[pd.DataFrame]:
    """
    获取板块列表（行业或概念）。

    Args:
        sector_type: "industry" 或 "concept"
        max_retries: 最大重试次数
        delay_min:   随机延迟最小秒数
        delay_max:   随机延迟最大秒数

    Returns:
        DataFrame，列: name, code, sector_type
        失败返回 None
    """
    if sector_type not in ("industry", "concept"):
        raise ValueError(f"sector_type 必须为 'industry' 或 'concept'，实际: {sector_type}")

    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare 未安装，无法采集板块数据")
        return None

    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(delay_min, delay_max))

            if sector_type == "industry":
                df = ak.stock_board_industry_name_em()
            else:
                df = ak.stock_board_concept_name_em()

            if df is None or df.empty:
                logger.warning("板块列表返回空数据 (type=%s)", sector_type)
                return None

            # 标准化列名
            # 先尝试中文列名映射，再尝试原样保留
            rename_map = {}
            for cn, en in _SECTOR_LIST_COL_MAP_INDUSTRY.items():
                if cn in df.columns:
                    rename_map[cn] = en
            if rename_map:
                df = df.rename(columns=rename_map)

            # 确保必要列存在（兼容不同 AKShare 版本）
            if "name" not in df.columns:
                # 尝试第一列作为名称
                first_col = df.columns[0]
                df = df.rename(columns={first_col: "name"})
            if "code" not in df.columns:
                # 尝试第二列作为代码
                if len(df.columns) >= 2:
                    second_col = df.columns[1]
                    df = df.rename(columns={second_col: "code"})
                else:
                    df["code"] = ""

            df = df[["name", "code"]].copy()
            df["sector_type"] = sector_type
            df["name"] = df["name"].astype(str).str.strip()
            df["code"] = df["code"].astype(str).str.strip()
            df = df[df["name"].str.len() > 0].reset_index(drop=True)

            logger.info("获取 %s 板块列表: %d 个", sector_type, len(df))
            return df

        except Exception as e:
            ratelimited = _is_ratelimit_error(e)
            logger.warning(
                "获取板块列表失败 (type=%s, attempt=%d/%d): %s",
                sector_type, attempt + 1, max_retries, e
            )
            if attempt < max_retries - 1:
                _backoff_sleep(attempt, ratelimited, delay_min, delay_max)

    logger.error("获取板块列表失败，已达最大重试次数 (type=%s)", sector_type)
    return None


def fetch_sector_daily(
    sector_name: str,
    sector_type: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    delay_min: float = 0.5,
    delay_max: float = 1.0,
) -> Optional[pd.DataFrame]:
    """
    获取单个板块的日线历史行情。

    Args:
        sector_name: 板块名称（如 "银行"、"半导体"）
        sector_type: "industry" 或 "concept"
        start_date:  起始日期 "YYYYMMDD" 或 "YYYY-MM-DD"
        end_date:    结束日期 "YYYYMMDD" 或 "YYYY-MM-DD"
        max_retries: 最大重试次数
        delay_min/max: 随机延迟范围

    Returns:
        DataFrame，列: date, open, high, low, close, volume, amount,
                       pct_change, turnover, amplitude, sector_name, sector_type
        失败返回 None
    """
    if sector_type not in ("industry", "concept"):
        raise ValueError(f"sector_type 必须为 'industry' 或 'concept'")

    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare 未安装")
        return None

    # 统一日期格式为 YYYYMMDD（部分 AKShare 接口要求）
    start_fmt = start_date.replace("-", "")
    end_fmt   = end_date.replace("-", "")

    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(delay_min, delay_max))

            if sector_type == "industry":
                df = ak.stock_board_industry_hist_em(
                    symbol=sector_name,
                    start_date=start_fmt,
                    end_date=end_fmt,
                    period="日k",
                    adjust="",
                )
            else:
                df = ak.stock_board_concept_hist_em(
                    symbol=sector_name,
                    start_date=start_fmt,
                    end_date=end_fmt,
                    period="日k",
                    adjust="",
                )

            if df is None or df.empty:
                logger.debug("板块日线返回空数据: %s (%s)", sector_name, sector_type)
                return None

            # 标准化列名
            rename_map = {}
            for cn, en in _SECTOR_HIST_COL_MAP.items():
                if cn in df.columns:
                    rename_map[cn] = en
            if rename_map:
                df = df.rename(columns=rename_map)

            # 确保 date 列存在并格式化
            if "date" not in df.columns:
                # 尝试找到日期列
                date_candidates = [c for c in df.columns
                                   if "日期" in c or "date" in c.lower() or "time" in c.lower()]
                if date_candidates:
                    df = df.rename(columns={date_candidates[0]: "date"})
                else:
                    logger.warning("未找到日期列: %s", df.columns.tolist())
                    return None

            # 转换日期为字符串 YYYY-MM-DD
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            df = df.dropna(subset=["date"])

            # 数值列转换
            numeric_cols = ["open", "high", "low", "close", "volume", "amount",
                            "pct_change", "turnover", "amplitude", "change"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # 添加元信息列
            df["sector_name"] = sector_name
            df["sector_type"] = sector_type

            # 按日期排序
            df = df.sort_values("date").reset_index(drop=True)

            logger.debug("获取板块日线: %s (%s) %d 行", sector_name, sector_type, len(df))
            return df

        except Exception as e:
            ratelimited = _is_ratelimit_error(e)
            logger.warning(
                "板块日线获取失败 (%s/%s, attempt=%d/%d): %s",
                sector_name, sector_type, attempt + 1, max_retries, e
            )
            if attempt < max_retries - 1:
                _backoff_sleep(attempt, ratelimited, delay_min, delay_max)

    logger.error("板块日线获取失败，已达最大重试次数: %s (%s)", sector_name, sector_type)
    return None


def fetch_sector_constituents(
    sector_name: str,
    sector_type: str,
    max_retries: int = 3,
    delay_min: float = 0.5,
    delay_max: float = 1.0,
) -> Optional[List[str]]:
    """
    获取板块成分股代码列表。

    Args:
        sector_name: 板块名称
        sector_type: "industry" 或 "concept"
        max_retries: 最大重试次数
        delay_min/max: 随机延迟范围

    Returns:
        6位股票代码字符串列表（如 ["000001", "600519", ...]）
        失败返回 None
    """
    if sector_type not in ("industry", "concept"):
        raise ValueError(f"sector_type 必须为 'industry' 或 'concept'")

    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare 未安装")
        return None

    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(delay_min, delay_max))

            if sector_type == "industry":
                df = ak.stock_board_industry_cons_em(symbol=sector_name)
            else:
                df = ak.stock_board_concept_cons_em(symbol=sector_name)

            if df is None or df.empty:
                logger.debug("板块成分股返回空: %s (%s)", sector_name, sector_type)
                return []

            # 查找代码列
            code_col = None
            for candidate in _CONS_CODE_COLS:
                if candidate in df.columns:
                    code_col = candidate
                    break

            if code_col is None:
                # 如果没找到标准列名，取第一个包含数字的列
                for col in df.columns:
                    sample = df[col].astype(str).head(5).tolist()
                    if any(re.match(r"^\d{6}$", s) for s in sample):
                        code_col = col
                        break

            if code_col is None:
                logger.warning("未找到股票代码列: %s, 列名=%s",
                               sector_name, df.columns.tolist())
                return []

            codes = (
                df[code_col]
                .astype(str)
                .str.strip()
                .str.zfill(6)  # 补足6位
                .tolist()
            )
            # 过滤有效代码（6位数字）
            codes = [c for c in codes if re.match(r"^\d{6}$", c)]

            logger.debug("获取板块成分股: %s (%s) %d 只", sector_name, sector_type, len(codes))
            return codes

        except Exception as e:
            ratelimited = _is_ratelimit_error(e)
            logger.warning(
                "板块成分股获取失败 (%s/%s, attempt=%d/%d): %s",
                sector_name, sector_type, attempt + 1, max_retries, e
            )
            if attempt < max_retries - 1:
                _backoff_sleep(attempt, ratelimited, delay_min, delay_max)

    logger.error("板块成分股获取失败，已达最大重试次数: %s (%s)", sector_name, sector_type)
    return None


# ============================================================================
# SectorDataPipeline
# ============================================================================

class SectorDataPipeline:
    """
    板块数据采集管道 (V7.6)

    类似 StockDataPipeline，提供批量采集、增量更新、报告功能。

    使用示例:
        pipeline = SectorDataPipeline(sector_dir="./data/sector")
        pipeline.fetch_all_lists(force=True)
        pipeline.update_sector_daily("industry")
        pipeline.build_constituents_map("industry")
    """

    def __init__(
        self,
        sector_dir: str = "./data/sector",
        reports_dir: str = "./data/reports",
        ak_workers: int = 2,
        ak_delay_min: float = 0.5,
        ak_delay_max: float = 1.0,
        ak_max_retries: int = 3,
    ) -> None:
        self.sector_dir    = Path(sector_dir)
        self.reports_dir   = Path(reports_dir)
        self.ak_workers    = ak_workers
        self.ak_delay_min  = ak_delay_min
        self.ak_delay_max  = ak_delay_max
        self.ak_max_retries = ak_max_retries

        # 创建必要目录
        for sub in ("list", "daily/industry", "daily/concept", "constituents"):
            (self.sector_dir / sub).mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # ── 工具方法 ─────────────────────────────────────────────────────────

    def load_sector_list(self, sector_type: str) -> Optional[pd.DataFrame]:
        """加载已保存的板块列表"""
        path = self.sector_dir / "list" / f"{sector_type}.parquet"
        if not path.exists():
            logger.warning("板块列表文件不存在: %s", path)
            return None
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.error("读取板块列表失败 (%s): %s", sector_type, e)
            return None

    def load_sector_map(self, sector_type: str) -> Optional[pd.DataFrame]:
        """加载已保存的成分股映射表"""
        path = self.sector_dir / "constituents" / f"{sector_type}_map.parquet"
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.error("读取成分股映射表失败 (%s): %s", sector_type, e)
            return None

    def _get_local_max_date(self, sector_name: str, sector_type: str) -> Optional[str]:
        """获取本地板块数据的最大日期"""
        safe_name = _sanitize_name(sector_name)
        path = self.sector_dir / "daily" / sector_type / f"{safe_name}.parquet"
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path, columns=["date"])
            if df.empty or "date" not in df.columns:
                return None
            return str(df["date"].max())
        except Exception:
            return None

    def _merge_and_save_daily(
        self,
        sector_name: str,
        sector_type: str,
        new_df: pd.DataFrame,
    ) -> bool:
        """将新数据与本地数据合并并保存"""
        safe_name = _sanitize_name(sector_name)
        path = self.sector_dir / "daily" / sector_type / f"{safe_name}.parquet"

        try:
            if path.exists():
                old_df = pd.read_parquet(path)
                # 合并，去重，排序
                merged = pd.concat([old_df, new_df], ignore_index=True)
                merged = merged.drop_duplicates(subset=["date"]).sort_values("date")
                merged = merged.reset_index(drop=True)
            else:
                merged = new_df.sort_values("date").reset_index(drop=True)

            merged.to_parquet(path, index=False, compression="zstd")
            return True
        except Exception as e:
            logger.error("保存板块日线失败 (%s/%s): %s", sector_name, sector_type, e)
            return False

    # ── 主方法: 获取板块列表 ─────────────────────────────────────────────

    def fetch_all_lists(self, force: bool = False) -> Dict:
        """
        获取行业和概念板块列表，保存为 Parquet。

        Args:
            force: 强制重新采集（否则如文件已存在则跳过）

        Returns:
            统计字典: {industry_success, industry_failed, concept_success, concept_failed}
        """
        stats = {
            "industry_success": 0, "industry_failed": 0,
            "concept_success":  0, "concept_failed":  0,
        }

        for sector_type in ("industry", "concept"):
            list_path = self.sector_dir / "list" / f"{sector_type}.parquet"

            if not force and list_path.exists():
                logger.info("板块列表已存在，跳过: %s", list_path)
                try:
                    df = pd.read_parquet(list_path)
                    stats[f"{sector_type}_success"] = len(df)
                except Exception:
                    pass
                continue

            logger.info("正在获取 %s 板块列表...", sector_type)
            df = fetch_sector_list(
                sector_type=sector_type,
                max_retries=self.ak_max_retries,
                delay_min=self.ak_delay_min,
                delay_max=self.ak_delay_max,
            )

            if df is not None and not df.empty:
                try:
                    df.to_parquet(list_path, index=False, compression="zstd")
                    stats[f"{sector_type}_success"] = len(df)
                    logger.info("已保存 %s 板块列表: %d 个 -> %s", sector_type, len(df), list_path)
                except Exception as e:
                    logger.error("保存板块列表失败 (%s): %s", sector_type, e)
                    stats[f"{sector_type}_failed"] = 1
            else:
                stats[f"{sector_type}_failed"] = 1
                logger.error("获取 %s 板块列表失败", sector_type)

        return stats

    # ── 主方法: 更新板块日线 ─────────────────────────────────────────────

    def update_sector_daily(
        self,
        sector_type: str,
        force_full: bool = False,
        limit: Optional[int] = None,
    ) -> Dict:
        """
        批量增量更新指定类型的所有板块日线数据。

        Args:
            sector_type: "industry" 或 "concept"
            force_full:  强制全量重采（忽略本地最大日期）
            limit:       仅处理前 N 个板块（调试用）

        Returns:
            统计字典: {success, failed, skipped, total, elapsed_s, reports_dir}
        """
        if sector_type not in ("industry", "concept"):
            raise ValueError(f"sector_type 必须为 'industry' 或 'concept'")

        # 尝试导入 RunReport，不可用时用简单统计
        try:
            from src.data.collector.run_report import RunReport
            report = RunReport(str(self.reports_dir))
            _use_report = True
        except ImportError:
            report = None
            _use_report = False

        # 加载板块列表
        sector_list_df = self.load_sector_list(sector_type)
        if sector_list_df is None or sector_list_df.empty:
            logger.warning("%s 板块列表为空，请先执行 fetch_all_lists()", sector_type)
            # 尝试自动拉取
            logger.info("尝试自动拉取 %s 板块列表...", sector_type)
            self.fetch_all_lists(force=False)
            sector_list_df = self.load_sector_list(sector_type)
            if sector_list_df is None or sector_list_df.empty:
                return {"success": 0, "failed": 0, "skipped": 0, "total": 0,
                        "elapsed_s": 0, "error": "板块列表为空"}

        sector_names = sector_list_df["name"].tolist()
        if limit and limit > 0:
            sector_names = sector_names[:limit]

        total = len(sector_names)
        success = failed = skipped = 0
        today_str = date.today().strftime("%Y-%m-%d")
        start_all  = "20100101"   # 全量采集起始日期

        t_start = time.time()
        logger.info("开始更新 %s 板块日线: %d 个板块", sector_type, total)

        # 尝试导入 tqdm
        try:
            from tqdm import tqdm
            sector_iter = tqdm(sector_names, desc=f"更新{sector_type}板块日线")
        except ImportError:
            sector_iter = sector_names

        for sector_name in sector_iter:
            try:
                # 确定采集起止日期
                if force_full:
                    start_date = start_all
                else:
                    local_max = self._get_local_max_date(sector_name, sector_type)
                    if local_max and local_max >= today_str:
                        skipped += 1
                        if _use_report:
                            report.record_skipped(sector_name, 0, reason="already_up_to_date")
                        continue
                    elif local_max:
                        # 增量：从下一天开始
                        from datetime import datetime as dt2, timedelta
                        next_day = (dt2.strptime(local_max, "%Y-%m-%d") + timedelta(days=1))
                        start_date = next_day.strftime("%Y-%m-%d")
                    else:
                        start_date = start_all

                # 采集
                df = fetch_sector_daily(
                    sector_name=sector_name,
                    sector_type=sector_type,
                    start_date=start_date,
                    end_date=today_str,
                    max_retries=self.ak_max_retries,
                    delay_min=self.ak_delay_min,
                    delay_max=self.ak_delay_max,
                )

                if df is None or df.empty:
                    failed += 1
                    if _use_report:
                        report.record_failed(sector_name, 0, reason="empty_data")
                    logger.debug("板块日线为空: %s (%s)", sector_name, sector_type)
                    continue

                # 合并保存
                ok = self._merge_and_save_daily(sector_name, sector_type, df)
                if ok:
                    success += 1
                    if _use_report:
                        report.record_success(sector_name, 0, source="akshare", rows=len(df))
                else:
                    failed += 1
                    if _use_report:
                        report.record_failed(sector_name, 0, reason="save_failed")

            except Exception as e:
                failed += 1
                logger.warning("板块日线处理异常 (%s): %s", sector_name, e)
                if _use_report:
                    report.record_failed(sector_name, 0, reason=f"exception:{e}")

        elapsed = time.time() - t_start

        # 保存报告
        if _use_report and report:
            try:
                report.save()
            except Exception as e:
                logger.warning("保存报告失败: %s", e)

        stats = {
            "success":     success,
            "failed":      failed,
            "skipped":     skipped,
            "total":       total,
            "elapsed_s":   round(elapsed, 2),
            "reports_dir": str(self.reports_dir),
            "sector_type": sector_type,
        }
        logger.info(
            "板块日线更新完成 (%s): 成功=%d 失败=%d 跳过=%d 耗时=%.1fs",
            sector_type, success, failed, skipped, elapsed
        )
        return stats

    # ── 主方法: 生成成分股映射表 ─────────────────────────────────────────

    def build_constituents_map(
        self,
        sector_type: str,
        force_refresh: bool = False,
        limit: Optional[int] = None,
    ) -> Dict:
        """
        为指定板块类型生成成分股映射表。

        Args:
            sector_type:   "industry" 或 "concept"
            force_refresh: 强制重新采集所有板块的成分股
            limit:         仅处理前 N 个板块（调试用）

        Returns:
            统计字典: {success, failed, skipped, total, total_stocks}

        输出文件:
            {sector_dir}/constituents/{sector_type}_map.parquet
            {sector_dir}/constituents/{sector_type}_map.csv
        """
        if sector_type not in ("industry", "concept"):
            raise ValueError(f"sector_type 必须为 'industry' 或 'concept'")

        # 加载板块列表
        sector_list_df = self.load_sector_list(sector_type)
        if sector_list_df is None or sector_list_df.empty:
            logger.warning("%s 板块列表为空，请先执行 fetch_all_lists()", sector_type)
            self.fetch_all_lists(force=False)
            sector_list_df = self.load_sector_list(sector_type)
            if sector_list_df is None or sector_list_df.empty:
                return {"success": 0, "failed": 0, "skipped": 0, "total": 0, "total_stocks": 0}

        sector_names = sector_list_df["name"].tolist()
        if limit and limit > 0:
            sector_names = sector_names[:limit]

        total = len(sector_names)
        success = failed = skipped = 0
        all_records: List[Dict] = []

        # 检查已有映射表（增量更新支持）
        map_path_parquet = self.sector_dir / "constituents" / f"{sector_type}_map.parquet"
        existing_sectors: set = set()
        if not force_refresh and map_path_parquet.exists():
            try:
                existing_df = pd.read_parquet(map_path_parquet)
                if "sector_name" in existing_df.columns:
                    existing_sectors = set(existing_df["sector_name"].unique())
                    # 保留已有记录
                    all_records = existing_df.to_dict("records")
                    logger.info("已有 %d 个板块的成分股映射，将增量补充", len(existing_sectors))
            except Exception as e:
                logger.warning("读取已有映射表失败，重新全量采集: %s", e)
                all_records = []
                existing_sectors = set()

        # 尝试 tqdm
        try:
            from tqdm import tqdm
            sector_iter = tqdm(sector_names, desc=f"采集{sector_type}成分股")
        except ImportError:
            sector_iter = sector_names

        for sector_name in sector_iter:
            if sector_name in existing_sectors and not force_refresh:
                skipped += 1
                continue

            try:
                codes = fetch_sector_constituents(
                    sector_name=sector_name,
                    sector_type=sector_type,
                    max_retries=self.ak_max_retries,
                    delay_min=self.ak_delay_min,
                    delay_max=self.ak_delay_max,
                )

                if codes is None:
                    failed += 1
                    logger.debug("成分股获取失败: %s (%s)", sector_name, sector_type)
                    continue

                if len(codes) == 0:
                    # 空成分股也算成功（某些板块可能暂无成分股）
                    success += 1
                    continue

                # 生成记录
                for code in codes:
                    all_records.append({
                        "sector_name": sector_name,
                        "sector_type": sector_type,
                        "code":        code,
                    })
                success += 1

            except Exception as e:
                failed += 1
                logger.warning("成分股处理异常 (%s): %s", sector_name, e)

        # 保存映射表
        total_stocks = 0
        if all_records:
            map_df = pd.DataFrame(all_records)
            map_df = map_df.drop_duplicates(subset=["sector_name", "code"])
            map_df = map_df.sort_values(["sector_type", "sector_name", "code"])
            map_df = map_df.reset_index(drop=True)
            total_stocks = len(map_df)

            try:
                map_df.to_parquet(map_path_parquet, index=False, compression="zstd")
                logger.info("成分股映射表已保存 (parquet): %s (%d 条)", map_path_parquet, len(map_df))
            except Exception as e:
                logger.error("保存成分股映射表(parquet)失败: %s", e)

            try:
                map_path_csv = self.sector_dir / "constituents" / f"{sector_type}_map.csv"
                map_df.to_csv(map_path_csv, index=False, encoding="utf-8-sig")
                logger.info("成分股映射表已保存 (csv): %s", map_path_csv)
            except Exception as e:
                logger.warning("保存成分股映射表(csv)失败: %s", e)
        else:
            logger.warning("未获取到任何成分股记录 (%s)", sector_type)

        stats = {
            "success":      success,
            "failed":       failed,
            "skipped":      skipped,
            "total":        total,
            "total_stocks": total_stocks,
            "sector_type":  sector_type,
        }
        logger.info(
            "成分股映射完成 (%s): 成功=%d 失败=%d 跳过=%d 总记录=%d",
            sector_type, success, failed, skipped, total_stocks
        )
        return stats

    # ── 辅助查询方法 ─────────────────────────────────────────────────────

    def get_sector_for_stock(self, code: str, sector_type: str = "industry") -> List[str]:
        """
        查询某股票所属的板块名称列表。

        Args:
            code:        6位股票代码
            sector_type: "industry" 或 "concept"

        Returns:
            板块名称列表（可能为空）
        """
        map_df = self.load_sector_map(sector_type)
        if map_df is None or map_df.empty:
            return []
        if "code" not in map_df.columns or "sector_name" not in map_df.columns:
            return []
        return map_df.loc[map_df["code"] == code, "sector_name"].tolist()

    def get_stocks_in_sector(self, sector_name: str, sector_type: str = "industry") -> List[str]:
        """
        查询某板块的所有成分股代码。

        Args:
            sector_name: 板块名称
            sector_type: "industry" 或 "concept"

        Returns:
            6位股票代码列表
        """
        map_df = self.load_sector_map(sector_type)
        if map_df is None or map_df.empty:
            return []
        if "code" not in map_df.columns or "sector_name" not in map_df.columns:
            return []
        return map_df.loc[map_df["sector_name"] == sector_name, "code"].tolist()

    def load_sector_daily(
        self,
        sector_name: str,
        sector_type: str = "industry",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        加载指定板块的日线数据。

        Args:
            sector_name: 板块名称
            sector_type: "industry" 或 "concept"
            start_date:  过滤起始日期（可选）
            end_date:    过滤结束日期（可选）

        Returns:
            DataFrame 或 None
        """
        safe_name = _sanitize_name(sector_name)
        path = self.sector_dir / "daily" / sector_type / f"{safe_name}.parquet"
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if start_date:
                df = df[df["date"] >= start_date]
            if end_date:
                df = df[df["date"] <= end_date]
            return df.reset_index(drop=True)
        except Exception as e:
            logger.error("读取板块日线失败 (%s/%s): %s", sector_name, sector_type, e)
            return None

    def summary(self) -> Dict:
        """返回当前板块数据存储摘要"""
        result = {}
        for sector_type in ("industry", "concept"):
            list_path  = self.sector_dir / "list" / f"{sector_type}.parquet"
            daily_dir  = self.sector_dir / "daily" / sector_type
            cons_path  = self.sector_dir / "constituents" / f"{sector_type}_map.parquet"

            list_count  = 0
            daily_count = 0
            cons_count  = 0

            if list_path.exists():
                try:
                    list_count = len(pd.read_parquet(list_path))
                except Exception:
                    pass

            if daily_dir.exists():
                daily_count = len(list(daily_dir.glob("*.parquet")))

            if cons_path.exists():
                try:
                    cons_count = len(pd.read_parquet(cons_path))
                except Exception:
                    pass

            result[sector_type] = {
                "list_sectors":   list_count,
                "daily_files":    daily_count,
                "constituents_records": cons_count,
            }

        return result