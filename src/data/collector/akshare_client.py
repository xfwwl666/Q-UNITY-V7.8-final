#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
akshare_client.py — AKShare 进程隔离采集客户端 (patch_v10)
==========================================================

【patch_v10 修复说明】

BUG-1（致命）: 限流关键词未覆盖实际报错
  原版: _RATELIMIT_KEYWORDS 仅含 "429"/"限流"/"too many" 等文字
  实际: 东方财富限流时直接断开TCP，报 RemoteDisconnected / Connection aborted
  → 关键词匹配全部失败 → 判定为"普通错误" → 退避仅 1s/2s/4s → 继续猛打 → 越来越快被封
  修复: 新增 "RemoteDisconnected" / "Connection aborted" / "ConnectionResetError" 等关键词

BUG-2（严重）: 无跨股票全局冷却
  原版: 单只股票内三次重试均失败后，直接 return 失败，主进程立即调度下一只
  实际: 东方财富封IP后，连续数十只都会立即失败，进程池持续高频请求徒增封禁时长
  修复: run_akshare_batch 中追踪 consecutive_fail 连续失败计数
        连续失败 ≥ 5 时触发 全局冷却（120s），等服务端解封后再继续

BUG-3（明显）: delay 配置太短
  原版: delay_min=0.3, delay_max=0.8，2进程并发 → 合并请求速率 2~4次/秒
  修复: 默认改为 delay_min=1.5, delay_max=3.5，单进程 → 约 1次/1.5~3.5s

BUG-4（隐患）: ProcessPool 一次性提交全部5186个任务
  原版: 所有任务全部 submit 到 pool，失控时无法暂停/插入等待
  修复: 改为分批提交（BATCH_SIZE=30），每批完成后检查失败率，按需全局冷却

【其余逻辑保持不变】
"""

import time
import random
import logging
from typing import Optional, Tuple, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed, Future

logger = logging.getLogger(__name__)

# ── BUG-1 修复：补充东方财富TCP断连特征 ─────────────────────────────────────
# 东方财富限流有两种表现：
#   1. 返回 HTTP 429（含"too many"等文字）—— 原版已覆盖
#   2. 直接断开TCP连接，requests 报 RemoteDisconnected / Connection aborted
_RATELIMIT_KEYWORDS = (
    # 原版关键词
    "429", "限流", "频繁", "too many", "Too Many", "rate limit",
    # BUG-1 补充：TCP断连类错误（东方财富限流的实际表现）
    "RemoteDisconnected",
    "Connection aborted",
    "ConnectionResetError",
    "ConnectionRefusedError",
    "Remote end closed connection",
    "Failed to establish a new connection",
    "Max retries exceeded",
)

# AKShare → 标准列名映射
_AK_COL_MAP = {
    "日期":  "date",
    "开盘":  "open",
    "收盘":  "close",
    "最高":  "high",
    "最低":  "low",
    "成交量": "vol",
    "成交额": "amount",
    "振幅":  "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover",
}

# 扩展字段（TDX 不提供，AKShare 专属）
AK_EXTENDED_FIELDS = {"amplitude", "pct_change", "change", "turnover"}

# BUG-4 修复：分批提交参数
_SUBMIT_BATCH_SIZE       = 30    # 每批提交任务数
_CONSECUTIVE_FAIL_THRESH = 5     # 触发全局冷却的连续失败阈值
_GLOBAL_COOLDOWN_BASE    = 120.0 # 全局冷却基础时长（秒）
_GLOBAL_COOLDOWN_MAX     = 300.0 # 全局冷却上限（秒）


# ============================================================================
# 子进程工作函数（必须是顶层函数，才能被 pickle 序列化传入子进程）
# ============================================================================

def _akshare_process_worker(
    task: Tuple[str, str, str, int, float, float]
) -> Tuple[str, Optional[Any], Optional[str]]:
    """
    在独立子进程中采集单只股票的 AKShare 数据。

    Args:
        task: (code, start_date, end_date, max_retries, delay_min, delay_max)

    Returns:
        (code, df_dict_or_None, error_msg_or_None)
        注意：DataFrame 不能直接 pickle，通过 to_dict("records") 传回主进程。
    """
    try:
        import akshare as ak
    except ImportError:
        return (task[0], None, "akshare_not_installed")

    import pandas as pd

    code, start_date, end_date, max_retries, delay_min, delay_max = task

    # 统一日期格式为 YYYYMMDD（AKShare 要求）
    start_fmt = start_date.replace("-", "")
    end_fmt   = end_date.replace("-", "")

    for attempt in range(max_retries):
        try:
            # 随机延迟（防封）—— BUG-3修复后此值由调用方传入更合理的范围
            time.sleep(random.uniform(delay_min, delay_max))

            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_fmt,
                end_date=end_fmt,
                adjust="hfq",   # 后复权，量化回测标准
            )

            if df is None or df.empty:
                raise ValueError("返回空数据")

            # 列名标准化
            df = df.rename(columns=_AK_COL_MAP)
            df["code"]   = code
            df["source"] = "akshare"
            df["adjust"] = "hfq"

            # date 格式统一
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

            # 数值类型
            for col in ("open", "high", "low", "close"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            for col in ("vol", "amount"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
            for col in ("pct_change", "turnover", "amplitude", "change"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

            return (code, df.to_dict("records"), None)

        except Exception as exc:
            err_str = str(exc)
            # BUG-1修复：使用扩充后的关键词集合检测限流（含TCP断连）
            is_ratelimit = any(kw in err_str for kw in _RATELIMIT_KEYWORDS)

            if is_ratelimit:
                # 限流退避：30s / 60s / 90s（指数递增）
                wait = 30.0 * (attempt + 1)
                logger.warning(
                    "[子进程] 限流/断连 %s (attempt=%d), 等待 %.0fs: %s",
                    code, attempt, wait, err_str[:80]
                )
            else:
                # 普通错误：1s / 2s / 4s + 随机抖动
                wait = (2 ** attempt) * (1 + random.random() * 0.3)

            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                return (code, None, f"all_retries_failed:{err_str[:120]}")

    return (code, None, "unknown_error")


# ============================================================================
# 批量调度接口（主进程调用）
# ============================================================================

def run_akshare_batch(
    stock_list: List[Tuple[str, str, str]],
    max_workers: int = 1,          # BUG-3修复：默认改为1进程，降低并发压力
    max_retries: int = 3,
    delay_min: float = 1.5,        # BUG-3修复：最小延迟从0.3提升到1.5s
    delay_max: float = 3.5,        # BUG-3修复：最大延迟从0.8提升到3.5s
    progress_callback=None,
) -> Dict[str, Optional[Any]]:
    """
    使用 ProcessPoolExecutor 批量采集 AKShare 数据。

    BUG-2/4 修复：
      - 分批提交（BATCH_SIZE=30）而非一次性提交全部任务
      - 主进程追踪连续失败计数，触发阈值时插入全局冷却

    Args:
        stock_list:        [(code, start_date, end_date), ...]
        max_workers:       进程数（默认1，东方财富对并发极敏感）
        max_retries:       单股最大重试次数
        delay_min/max:     子进程内随机 sleep 区间（秒）
        progress_callback: fn(code, success, error) 进度回调

    Returns:
        {code: df_or_None}
    """
    import pandas as pd

    if not stock_list:
        return {}

    tasks = [
        (code, start, end, max_retries, delay_min, delay_max)
        for code, start, end in stock_list
    ]

    results: Dict[str, Optional[Any]] = {}
    consecutive_fail  = 0   # BUG-2修复：跨股票连续失败计数
    cooldown_count    = 0   # 已触发全局冷却次数（用于递增冷却时长）

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        # BUG-4修复：分批提交，而非一次性 submit 全部
        for batch_start in range(0, len(tasks), _SUBMIT_BATCH_SIZE):
            batch = tasks[batch_start: batch_start + _SUBMIT_BATCH_SIZE]

            future_map: Dict[Future, str] = {
                pool.submit(_akshare_process_worker, task): task[0]
                for task in batch
            }

            for future in as_completed(future_map):
                code = future_map[future]
                try:
                    result_code, data, error = future.result()
                    if data is not None:
                        df = pd.DataFrame(data)
                        if "date" in df.columns:
                            df = df.sort_values("date").reset_index(drop=True)
                        results[code] = df
                        consecutive_fail = 0  # 成功则重置连续失败计数
                        if progress_callback:
                            progress_callback(code, True, None)
                    else:
                        results[code] = None
                        consecutive_fail += 1
                        logger.warning("AKShare 采集失败 %s: %s", code, error)
                        if progress_callback:
                            progress_callback(code, False, error)

                        # ── BUG-2修复：连续失败达阈值 → 全局冷却 ─────────────
                        if consecutive_fail >= _CONSECUTIVE_FAIL_THRESH:
                            cooldown_count += 1
                            wait = min(
                                _GLOBAL_COOLDOWN_BASE * cooldown_count,
                                _GLOBAL_COOLDOWN_MAX
                            )
                            logger.warning(
                                "⚠️  连续失败 %d 次，判定 IP 被限流！"
                                "全局冷却 %.0fs（第%d次）...",
                                consecutive_fail, wait, cooldown_count
                            )
                            time.sleep(wait)
                            consecutive_fail = 0  # 冷却后重置

                except Exception as exc:
                    results[code] = None
                    consecutive_fail += 1
                    logger.error("AKShare worker 异常 %s: %s", code, exc)
                    if progress_callback:
                        progress_callback(code, False, str(exc))

    return results


# ============================================================================
# 单股同步接口（主进程直接调用，供降级 fallback 和测试用）
# ============================================================================

def fetch_akshare_single(
    code: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    delay_min: float = 1.5,
    delay_max: float = 3.5,
) -> Optional[Any]:
    """
    同步采集单只股票（直接在调用进程中运行）。
    注意：此接口仅供单线程场景使用，多线程并发调用有 session 竞争风险。

    Returns:
        pd.DataFrame 或 None
    """
    import pandas as pd

    task = (code, start_date, end_date, max_retries, delay_min, delay_max)
    result_code, data, error = _akshare_process_worker(task)
    if data is None:
        return None
    df = pd.DataFrame(data)
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    return df