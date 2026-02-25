#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_report.py — 运行报告持久化 (patch_v9)
==========================================
每次采集运行结束后生成：
  - failed_stocks.txt      — 失败股票列表（可用于 --retry-failed 补采）
  - run_stats_{ts}.json    — 完整运行统计（供后续审计）

设计原则:
  - 线程安全（内部使用 threading.Lock）
  - 幂等安全（同一 code 被记录多次以最后一次为准）
  - 失败分类：tdx_fail / akshare_fail / baostock_fail / validate_fail / complete_fail
"""

import json
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RunReport:
    """
    线程安全的运行报告收集器。

    使用示例:
        report = RunReport(reports_dir="./data/reports")
        report.record_success(code, market, source="tdx")
        report.record_failed(code, market, reason="complete_fail")
        report.record_skipped(code, market)
        report.save()   # 写出 txt + json
    """

    def __init__(self, reports_dir: str = "./data/reports") -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self._lock      = threading.Lock()
        self._ts        = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 核心统计
        self._success: Dict[str, dict] = {}   # code -> {market, source, rows}
        self._failed:  Dict[str, dict] = {}   # code -> {market, reason}
        self._skipped: Dict[str, dict] = {}   # code -> {market, reason}

        # 验证失败（单独分类，可能部分字段失败但仍有数据）
        self._validation_failures: Dict[str, str] = {}  # code -> reason

        # 来源统计
        self._source_counts = {"tdx": 0, "akshare": 0, "baostock": 0, "merged": 0}

        # 运行时元信息
        self._start_time = datetime.now()
        self._end_time: Optional[datetime] = None

    # ------------------------------------------------------------------
    # 记录接口（线程安全）
    # ------------------------------------------------------------------

    def record_success(
        self,
        code: str,
        market: int,
        source: str,
        rows: int = 0,
    ) -> None:
        """记录采集成功（含双轨合并场景）。"""
        with self._lock:
            self._success[code] = {"market": market, "source": source, "rows": rows}
            if source in self._source_counts:
                self._source_counts[source] += 1
            else:
                self._source_counts["merged"] += 1

    def record_failed(
        self,
        code: str,
        market: int,
        reason: str,
    ) -> None:
        """记录采集彻底失败（三级全失败）。"""
        with self._lock:
            self._failed[code] = {"market": market, "reason": reason}

    def record_skipped(
        self,
        code: str,
        market: int,
        reason: str = "already_up_to_date",
    ) -> None:
        """记录跳过（本地数据已最新）。"""
        with self._lock:
            self._skipped[code] = {"market": market, "reason": reason}

    def record_validation_failure(self, code: str, reason: str) -> None:
        """记录数据验证失败（数据采集到了，但校验不通过）。"""
        with self._lock:
            self._validation_failures[code] = reason

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    @property
    def total_success(self) -> int:
        return len(self._success)

    @property
    def total_failed(self) -> int:
        return len(self._failed)

    @property
    def total_skipped(self) -> int:
        return len(self._skipped)

    def get_failed_list(self) -> List[Tuple[str, int, str]]:
        """返回 [(code, market, reason), ...] 失败列表。"""
        with self._lock:
            return [
                (code, info["market"], info["reason"])
                for code, info in self._failed.items()
            ]

    def summary_str(self) -> str:
        """单行摘要字符串（供 tqdm postfix 显示）。"""
        with self._lock:
            return (
                f"\u2713{self.total_success}"
                f" tdx={self._source_counts['tdx']}"
                f" ak={self._source_counts['akshare']}"
                f" bs={self._source_counts['baostock']}"
                f" \u2717{self.total_failed}"
                f" skip={self.total_skipped}"
            )

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self) -> None:
        """
        写出两个文件：
          failed_stocks.txt  — 可直接用于 --retry-failed
          run_stats_{ts}.json — 完整审计日志
        """
        self._end_time = datetime.now()
        elapsed = (self._end_time - self._start_time).total_seconds()

        with self._lock:
            # -- failed_stocks.txt --
            failed_path = self.reports_dir / "failed_stocks.txt"
            # A-15 Fix: 使用标准 \n / \t 字面量替代 chr() 构造
            out_lines = []
            out_lines.append("# 采集失败列表 — 生成时间: " + self._ts)
            out_lines.append("# 格式: code\tmarket\treason")
            for code, info in self._failed.items():
                out_lines.append(code + "\t" + str(info["market"]) + "\t" + str(info["reason"]))
            failed_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            logger.info("失败列表已写出: %s (%d 只)", failed_path, len(self._failed))

            # -- run_stats_{ts}.json --
            stats_path = self.reports_dir / f"run_stats_{self._ts}.json"
            stats = {
                "run_id":        self._ts,
                "start_time":    self._start_time.isoformat(),
                "end_time":      self._end_time.isoformat(),
                "elapsed_s":     round(elapsed, 2),
                "total":         len(self._success) + len(self._failed) + len(self._skipped),
                "success":       len(self._success),
                "failed":        len(self._failed),
                "skipped":       len(self._skipped),
                "validation_failures": len(self._validation_failures),
                "source_counts": dict(self._source_counts),
                "failed_detail": {
                    code: info for code, info in list(self._failed.items())[:200]
                },
                "validation_failure_detail": dict(
                    list(self._validation_failures.items())[:100]
                ),
            }
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            logger.info("运行统计已写出: %s", stats_path)

    # ------------------------------------------------------------------
    # 静态工具：从 txt 加载失败列表
    # ------------------------------------------------------------------

    @staticmethod
    def load_failed_list(
        reports_dir: str = "./data/reports",
    ) -> List[Tuple[str, int]]:
        """
        从 failed_stocks.txt 加载失败股票列表，供 --retry-failed 使用。

        Returns:
            [(code, market), ...]
        """
        failed_path = Path(reports_dir) / "failed_stocks.txt"
        if not failed_path.exists():
            logger.warning("未找到失败列表: %s", failed_path)
            return []

        result: List[Tuple[str, int]] = []
        with open(failed_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    try:
                        result.append((parts[0], int(parts[1])))
                    except ValueError:
                        logger.debug("跳过无效行: %s", line)
        logger.info("加载失败列表: %d 只股票", len(result))
        return result