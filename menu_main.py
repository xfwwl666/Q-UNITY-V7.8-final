#!/usr/bin/env python3
"""
Q-UNITY-V7.8 主菜单系统

V7.6 新增:
  - 回测系统: 选项1 完整单策略回测 / 选项2 多策略对比
  - 数据管理: 选项8 板块数据采集 (行业/概念)
"""
from __future__ import annotations
import json
import logging
import socket
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# 工具函数 (保持 V7.5 原有)
# ============================================================================

def _check_data_source_heartbeat() -> None:
    print("" + "-" * 50)
    print("  数据源心跳检测")
    print("-" * 50)

    print("[1] AKShare:")
    try:
        import akshare as ak
        t0 = time.time()
        df = ak.stock_zh_index_spot_em(symbol="\u4e0a\u8bc1\u6307\u6570")
        elapsed = time.time() - t0
        if df is not None and not df.empty:
            print(f"  \u2713 连通正常 ({elapsed:.2f}s)")
        else:
            print(f"  \u2717 返回空数据 ({elapsed:.2f}s)")
    except ImportError:
        print("  \u2717 akshare 未安装")
    except Exception as e:
        print(f"  \u2717 异常: {e}")

    print("[2] TDX 节点（前5优选）:")
    try:
        from src.data.collector.node_scanner import get_fastest_nodes
        nodes = get_fastest_nodes(top_n=5, timeout=3.0)
        for n in nodes:
            icon = "\u2713" if n["status"] == "ok" else "\u2717"
            lat  = f"{n['latency_ms']:.0f}ms" if n["latency_ms"] >= 0 else "timeout"
            print(f"  {icon} {n['name']} {n['host']}:{n['port']}  {lat}")
    except Exception as e:
        print(f"  \u2717 节点扫描失败: {e}")

    print("[3] BaoStock:")
    try:
        import baostock as bs
        t0 = time.time()
        lg = bs.login()
        elapsed = time.time() - t0
        if lg.error_code == "0":
            print(f"  \u2713 登录成功 ({elapsed:.2f}s)")
            bs.logout()
        else:
            print(f"  \u2717 登录失败: {lg.error_msg}")
    except ImportError:
        print("  \u2717 baostock 未安装")
    except Exception as e:
        print(f"  \u2717 异常: {e}")

    # A-11 Fix: 移除 Tushare DNS 心跳检测 — 系统不依赖 Tushare，保留会误导用户
    print("-" * 50)
    input("按 Enter 返回...")


def _load_collector_config() -> dict:
    defaults = {
        "parquet_dir":    "./data/parquet",
        "reports_dir":    "./data/reports",
        "top_n_nodes":    5,
        "tdx_workers":    8,
        "ak_workers":     2,
        "ak_delay_min":   0.3,
        "ak_delay_max":   0.8,
        "ak_max_retries": 3,
    }
    try:
        cfg_path = Path("config.json")
        if cfg_path.exists():
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            c = raw.get("collector", {})
            d = raw.get("data", {})
            defaults.update({
                "parquet_dir":    d.get("parquet_dir",         defaults["parquet_dir"]),
                "reports_dir":    d.get("reports_dir",         defaults["reports_dir"]),
                "top_n_nodes":    c.get("tdx_top_nodes",       defaults["top_n_nodes"]),
                "tdx_workers":    c.get("tdx_workers",         defaults["tdx_workers"]),
                "ak_workers":     c.get("akshare_workers",     defaults["ak_workers"]),
                "ak_delay_min":   c.get("akshare_delay_min",   defaults["ak_delay_min"]),
                "ak_delay_max":   c.get("akshare_delay_max",   defaults["ak_delay_max"]),
                "ak_max_retries": c.get("akshare_max_retries", defaults["ak_max_retries"]),
            })
    except Exception:
        pass
    return defaults


def _load_backtest_config() -> dict:
    """加载回测配置（含 realtime.strategy_params）"""
    defaults = {
        "initial_cash":     1_000_000.0,
        "commission_rate":  0.0003,
        "slippage_rate":    0.001,
        "tax_rate":         0.001,
        "position_limit":   20,
        "max_position_pct": 0.2,
        "stop_loss_pct":    0.10,
        "take_profit_pct":  0.20,
        "trailing_stop_pct":0.05,
        "circuit_breaker_max_dd":          0.20,
        "circuit_breaker_cooldown_days":    5,
        "strategy_params":  {},
    }
    try:
        cfg_path = Path("config.json")
        if cfg_path.exists():
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            bt  = raw.get("backtest", {})
            rt  = raw.get("realtime", {})
            rsk = raw.get("risk", {})
            defaults.update({
                "initial_cash":     bt.get("initial_cash",     defaults["initial_cash"]),
                "commission_rate":  bt.get("commission_rate",  defaults["commission_rate"]),
                "slippage_rate":    bt.get("slippage_rate",    defaults["slippage_rate"]),
                "tax_rate":         bt.get("tax_rate",         defaults["tax_rate"]),
                "position_limit":   bt.get("position_limit",   defaults["position_limit"]),
                "max_position_pct": bt.get("max_position_pct", defaults["max_position_pct"]),
                "stop_loss_pct":    rsk.get("stop_loss_pct",   defaults["stop_loss_pct"]),
                "take_profit_pct":  rsk.get("take_profit_pct", defaults["take_profit_pct"]),
                "trailing_stop_pct":rsk.get("trailing_stop_pct", defaults["trailing_stop_pct"]),
                "circuit_breaker_max_dd":
                    rsk.get("max_drawdown", defaults["circuit_breaker_max_dd"]),
                "circuit_breaker_cooldown_days":
                    rsk.get("circuit_breaker_cooldown_days",
                            defaults["circuit_breaker_cooldown_days"]),
                "strategy_params":  rt.get("strategy_params", {}),
            })
    except Exception as e:
        logger.warning("加载回测配置失败: %s，使用默认值", e)
    return defaults


def _load_sector_config() -> dict:
    """加载板块采集配置"""
    defaults = {
        "sector_dir":     "./data/sector",
        "reports_dir":    "./data/reports",
        "ak_workers":     2,
        "ak_delay_min":   0.5,
        "ak_delay_max":   1.0,
        "ak_max_retries": 3,
    }
    try:
        cfg_path = Path("config.json")
        if cfg_path.exists():
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            sc  = raw.get("sector", {})
            d   = raw.get("data", {})
            defaults.update({
                "sector_dir":     sc.get("sector_dir",     defaults["sector_dir"]),
                "reports_dir":    d.get("reports_dir",     defaults["reports_dir"]),
                "ak_workers":     sc.get("ak_workers",     defaults["ak_workers"]),
                "ak_delay_min":   sc.get("ak_delay_min",   defaults["ak_delay_min"]),
                "ak_delay_max":   sc.get("ak_delay_max",   defaults["ak_delay_max"]),
                "ak_max_retries": sc.get("ak_max_retries", defaults["ak_max_retries"]),
            })
    except Exception:
        pass
    return defaults


def _make_pipeline(force_full: bool = False, enable_akshare: bool = False):
    from src.data.collector.pipeline import StockDataPipeline
    cfg = _load_collector_config()
    return StockDataPipeline(
        parquet_dir=cfg["parquet_dir"],
        reports_dir=cfg["reports_dir"],
        top_n_nodes=cfg["top_n_nodes"],
        tdx_workers=cfg["tdx_workers"],
        ak_workers=cfg["ak_workers"],
        ak_delay_min=cfg["ak_delay_min"],
        ak_delay_max=cfg["ak_delay_max"],
        ak_max_retries=cfg["ak_max_retries"],
        force_full=force_full,
        enable_akshare=enable_akshare,
    )


def _print_stats(stats: dict) -> None:
    mode = "双轨(TDX+AKShare)" if stats.get("akshare_enabled") else "快速(仅TDX)"
    print(f"{'='*54}")
    print(f"  采集完成  [{mode}]")
    print(f"{'='*54}")
    print(f"  总计:   {stats.get('total',   0):>6} 只")
    print(f"  成功:   {stats.get('success', 0):>6} 只")
    print(f"  失败:   {stats.get('failed',  0):>6} 只")
    print(f"  跳过:   {stats.get('skipped', 0):>6} 只（已最新）")
    print(f"  耗时:   {stats.get('elapsed_s', 0):>6.1f} 秒")
    print(f"  速度:   {stats.get('speed',    0):>6.1f} 股/秒")
    print(f"  报告:   {stats.get('reports_dir', 'N/A')}")
    print(f"{'='*54}")


# ============================================================================
# 数据管理子命令 (保持 V7.5 原有)
# ============================================================================

def _cmd_tdx_fast_collect() -> None:
    print("\u26a1 TDX 快速全量采集（仅 TDX，后复权 HFQ，multiprocessing.Pool）")
    print("  模式: TDX 多进程并发，不启动 AKShare 进程")
    print("  预计: 5000+ 只 A 股约 15~25 分钟")
    print("  字段: open/high/low/close/vol/amount（不含 turnover/pct_change）")
    print("  后续: 如需扩展字段，完成后选「选项2」单独补充")
    confirm = input("  确认开始? [y/N]: ").strip().lower()
    if confirm != "y":
        print("  已取消。")
        return
    try:
        pipeline = _make_pipeline(force_full=True, enable_akshare=False)
        stats = pipeline.download_all_a_stocks()
        _print_stats(stats)
    except ImportError as e:
        print(f"  \u2717 导入失败: {e}")
    except Exception as e:
        logger.exception("TDX 快速采集异常")
        print(f"  \u2717 失败: {e}")


def _cmd_enrich_akshare() -> None:
    print("\U0001f52c AKShare 扩展字段补充（独立步骤）")
    print("  说明: 对已有 Parquet 文件补充 turnover、pct_change、amplitude 字段")
    print("  模式: 进程隔离（2进程），包含限流感知退避（30/60/90s）")
    print("  预计: 5000只约 2~4 小时（受东方财富接口限流影响）")
    print("  前提: 请先完成「选项1: TDX 快速全量采集」")

    cfg = _load_collector_config()
    parquet_dir = Path(cfg["parquet_dir"])
    existing = list(parquet_dir.glob("*.parquet"))
    if not existing:
        print(f"  \u2717 Parquet 目录为空: {parquet_dir}")
        print("  请先执行「选项1: TDX 快速全量采集」。")
        return

    print(f"  当前本地 Parquet: {len(existing)} 只")
    import pandas as pd
    has_ext = no_ext = 0
    for f in existing[:30]:
        try:
            cols = pd.read_parquet(f).columns.tolist()
            if "turnover" in cols:
                has_ext += 1
            else:
                no_ext += 1
        except Exception:
            no_ext += 1
    print(f"  抽样（前30只）: {has_ext} 只已有 turnover，{no_ext} 只未含扩展字段")

    confirm = input("  确认开始? [y/N]: ").strip().lower()
    if confirm != "y":
        print("  已取消。")
        return
    try:
        pipeline = _make_pipeline(force_full=False, enable_akshare=True)
        stats = pipeline.enrich_akshare()
        print(f"  \u2713 完成: 处理 {stats.get('total',0)} 只，"
              f"成功 {stats.get('success',0)} 只，失败 {stats.get('failed',0)} 只")
    except ImportError as e:
        print(f"  \u2717 导入失败: {e}")
    except Exception as e:
        logger.exception("AKShare 扩展字段补充异常")
        print(f"  \u2717 失败: {e}")


def _cmd_incremental_update() -> None:
    print("\U0001f504 增量数据更新（TDX，仅补采缺失日期）")
    try:
        pipeline = _make_pipeline(force_full=False, enable_akshare=False)
        stock_list = pipeline._get_all_a_stock_list()
        if not stock_list:
            print("  \u2717 获取股票列表失败")
            return
        print(f"  共 {len(stock_list)} 只 A 股，开始增量更新...")
        stats = pipeline.run(stock_list)
        _print_stats(stats)
    except ImportError as e:
        print(f"  \u2717 导入失败: {e}")
    except Exception as e:
        logger.exception("增量更新异常")
        print(f"  \u2717 失败: {e}")


def _cmd_retry_failed() -> None:
    print("\U0001f501 补采失败股票")
    cfg = _load_collector_config()
    failed_txt = Path(cfg["reports_dir"]) / "failed_stocks.txt"
    if not failed_txt.exists():
        print(f"  \u26a0\ufe0f  未找到失败列表: {failed_txt}")
        print("  请先执行全量采集或增量更新。")
        return
    try:
        pipeline = _make_pipeline(force_full=True, enable_akshare=False)
        stats = pipeline.retry_failed(reports_dir=cfg["reports_dir"])
        _print_stats(stats)
    except Exception as e:
        logger.exception("补采异常")
        print(f"  \u2717 失败: {e}")


def _cmd_check_integrity(storage=None) -> None:
    print("\U0001f50d 数据完整性检查")
    cfg = _load_collector_config()
    parquet_dir = Path(cfg["parquet_dir"])
    if not parquet_dir.exists():
        print(f"  \u26a0\ufe0f  Parquet 目录不存在: {parquet_dir}")
        return
    files = list(parquet_dir.glob("*.parquet"))
    print(f"  已存储 {len(files)} 只股票的 Parquet 文件")
    if files:
        import pandas as pd
        print("  抽样检查（前5只）:")
        for f in files[:5]:
            try:
                df = pd.read_parquet(f)
                ext = [c for c in ("turnover", "pct_change", "adjust") if c in df.columns]
                d_min = df["date"].min() if "date" in df.columns else "?"
                d_max = df["date"].max() if "date" in df.columns else "?"
                print(f"    \u2713 {f.stem}: {len(df)} 行 | {d_min} ~ {d_max} | 扩展={ext}")
            except Exception as ex:
                print(f"    \u2717 {f.stem}: {ex}")
    if storage:
        codes = storage.get_all_codes()
        print(f"  ColumnarStorage: {len(codes)} 只股票")
    input("按 Enter 返回...")


def _cmd_node_race() -> None:
    print("\U0001f3c1 TDX 节点赛马测试")
    try:
        from src.data.collector.node_scanner import race_nodes, TDX_NODES
        print(f"  测试 {len(TDX_NODES)} 个候选节点...")
        results = race_nodes(timeout=3.0)
        ok_nodes = [r for r in results if r["status"] == "ok"]
        print(f"  结果（{len(ok_nodes)}/{len(results)} 可达）:")
        for i, r in enumerate(results[:10], 1):
            icon = "\u2713" if r["status"] == "ok" else "\u2717"
            lat  = f"{r['latency_ms']:>7.2f} ms" if r["latency_ms"] >= 0 else "  timeout"
            print(f"    {i:>2}. {icon} {r['name']:<10} {r['host']:>18}:{r['port']}  {lat}")
        if len(results) > 10:
            print(f"    ... 余 {len(results)-10} 个节点")
    except Exception as e:
        print(f"  \u2717 节点测试失败: {e}")
    input("按 Enter 返回...")


def _cmd_clean_cache() -> None:
    print("\U0001f5d1\ufe0f  清理缓存")
    cleaned = 0
    cache_dir = Path("./data/cache")
    if cache_dir.exists():
        for p in cache_dir.rglob("*.json"):
            try:
                p.unlink()
                cleaned += 1
            except Exception:
                pass
    print(f"  已清理 {cleaned} 个缓存文件（fundamental JSON 缓存）")
    print("  注: Parquet 行情文件和运行报告不会被清除。")


# ============================================================================
# V7.6 新增：板块采集子菜单
# ============================================================================

def _cmd_sector_collect() -> None:
    """板块数据采集子菜单 (V7.6 新增)"""
    try:
        from src.data.sector import SectorDataPipeline
    except ImportError as e:
        print(f"  \u2717 无法导入 SectorDataPipeline: {e}")
        print("  请确认 src/data/sector.py 已存在（build_v7.6.py 应已写出该文件）。")
        input("按 Enter 返回...")
        return

    cfg = _load_sector_config()
    pipeline = SectorDataPipeline(
        sector_dir=cfg["sector_dir"],
        reports_dir=cfg["reports_dir"],
        ak_workers=cfg["ak_workers"],
        ak_delay_min=cfg["ak_delay_min"],
        ak_delay_max=cfg["ak_delay_max"],
        ak_max_retries=cfg["ak_max_retries"],
    )

    while True:
        print("\n" + "=" * 44)
        print("  板块数据采集  (V7.6)")
        print("=" * 44)
        print("  1. 更新板块列表 (行业/概念)")
        print("  2. 更新行业板块日线 (增量)")
        print("  3. 更新概念板块日线 (增量)")
        print("  4. 生成行业成分股映射表")
        print("  5. 生成概念成分股映射表")
        print("  6. 强制全量重采行业日线")
        print("  7. 强制全量重采概念日线")
        print("  0. 返回上级菜单")
        print()
        choice = input("请选择 [0-7]: ").strip()

        if choice == "0":
            break

        elif choice == "1":
            print("\n  正在更新板块列表（行业 + 概念）...")
            try:
                stats = pipeline.fetch_all_lists(force=True)
                print(f"  \u2713 行业板块: 成功={stats.get('industry_success', 0)}"
                      f"  失败={stats.get('industry_failed', 0)}")
                print(f"  \u2713 概念板块: 成功={stats.get('concept_success', 0)}"
                      f"  失败={stats.get('concept_failed', 0)}")
            except Exception as e:
                logger.exception("板块列表更新异常")
                print(f"  \u2717 失败: {e}")
            input("按 Enter 继续...")

        elif choice == "2":
            print("\n  正在增量更新行业板块日线...")
            print("  提示：首次运行可能需要较长时间，受东方财富接口限流影响。")
            try:
                stats = pipeline.update_sector_daily("industry", force_full=False)
                _print_sector_stats(stats)
            except Exception as e:
                logger.exception("行业板块日线更新异常")
                print(f"  \u2717 失败: {e}")
            input("按 Enter 继续...")

        elif choice == "3":
            print("\n  正在增量更新概念板块日线...")
            print("  提示：概念板块数量较多，首次运行可能需要数小时。")
            try:
                stats = pipeline.update_sector_daily("concept", force_full=False)
                _print_sector_stats(stats)
            except Exception as e:
                logger.exception("概念板块日线更新异常")
                print(f"  \u2717 失败: {e}")
            input("按 Enter 继续...")

        elif choice == "4":
            print("\n  正在生成行业成分股映射表...")
            try:
                stats = pipeline.build_constituents_map("industry")
                print(f"  \u2713 成功: {stats.get('success', 0)} 个板块")
                print(f"  \u2717 失败: {stats.get('failed', 0)} 个板块")
                print(f"  跳过: {stats.get('skipped', 0)} 个板块（已存在）")
                out_path = Path(cfg["sector_dir"]) / "constituents" / "industry_map.parquet"
                if out_path.exists():
                    import pandas as pd
                    df = pd.read_parquet(out_path)
                    print(f"  映射表: {len(df)} 条记录，已保存至 {out_path}")
            except Exception as e:
                logger.exception("行业成分股映射异常")
                print(f"  \u2717 失败: {e}")
            input("按 Enter 继续...")

        elif choice == "5":
            print("\n  正在生成概念成分股映射表...")
            try:
                stats = pipeline.build_constituents_map("concept")
                print(f"  \u2713 成功: {stats.get('success', 0)} 个板块")
                print(f"  \u2717 失败: {stats.get('failed', 0)} 个板块")
                print(f"  跳过: {stats.get('skipped', 0)} 个板块（已存在）")
                out_path = Path(cfg["sector_dir"]) / "constituents" / "concept_map.parquet"
                if out_path.exists():
                    import pandas as pd
                    df = pd.read_parquet(out_path)
                    print(f"  映射表: {len(df)} 条记录，已保存至 {out_path}")
            except Exception as e:
                logger.exception("概念成分股映射异常")
                print(f"  \u2717 失败: {e}")
            input("按 Enter 继续...")

        elif choice == "6":
            confirm = input("  确认强制全量重采行业日线? 这将覆盖已有数据。[y/N]: ").strip().lower()
            if confirm != "y":
                print("  已取消。")
                continue
            print("\n  正在强制全量重采行业板块日线...")
            try:
                stats = pipeline.update_sector_daily("industry", force_full=True)
                _print_sector_stats(stats)
            except Exception as e:
                logger.exception("行业板块全量重采异常")
                print(f"  \u2717 失败: {e}")
            input("按 Enter 继续...")

        elif choice == "7":
            confirm = input("  确认强制全量重采概念日线? 这将覆盖已有数据。[y/N]: ").strip().lower()
            if confirm != "y":
                print("  已取消。")
                continue
            print("\n  正在强制全量重采概念板块日线...")
            try:
                stats = pipeline.update_sector_daily("concept", force_full=True)
                _print_sector_stats(stats)
            except Exception as e:
                logger.exception("概念板块全量重采异常")
                print(f"  \u2717 失败: {e}")
            input("按 Enter 继续...")

        else:
            print("  \u2717 无效选项")


def _print_sector_stats(stats: dict) -> None:
    """打印板块采集统计"""
    print(f"  {'='*40}")
    print(f"  板块采集结果")
    print(f"  {'='*40}")
    print(f"  成功:   {stats.get('success', 0):>5} 个板块")
    print(f"  失败:   {stats.get('failed',  0):>5} 个板块")
    print(f"  跳过:   {stats.get('skipped', 0):>5} 个板块（已最新）")
    print(f"  总计:   {stats.get('total',   0):>5} 个板块")
    elapsed = stats.get('elapsed_s', 0)
    if elapsed:
        print(f"  耗时:   {elapsed:>5.1f} 秒")
    if stats.get('reports_dir'):
        print(f"  报告:   {stats['reports_dir']}")


# ============================================================================
# 数据管理菜单 (V7.6 新增选项8)
# ============================================================================

def data_management_menu(config=None, storage=None) -> None:
    while True:
        print("=" * 56)
        print("  数据管理  (V7.8 TDX快速 + AKShare扩展 + 板块数据)")
        print("=" * 56)
        print("  1. TDX 快速全量采集   (~15~25min，HFQ，多进程)")
        print("  2. AKShare 扩展字段补充 (turnover/pct_change，独立步骤)")
        print("  3. 增量数据更新        (仅补采缺失日期)")
        print("  4. 补采失败股票        (failed_stocks.txt)")
        print("  5. 数据完整性检查")
        print("  6. 节点赛马测试")
        print("  7. 清理缓存")
        print("  8. 板块数据采集        (行业/概念板块日线+成分股)")  # V7.6 新增
        print("  0. 返回主菜单")
        print()
        print("  \u2139  完整工作流: 先选1(~20min) -> 再选2(可选,~2~4h)")
        choice = input("请选择 [0-8]: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            _cmd_tdx_fast_collect()
        elif choice == "2":
            _cmd_enrich_akshare()
        elif choice == "3":
            _cmd_incremental_update()
        elif choice == "4":
            _cmd_retry_failed()
        elif choice == "5":
            _cmd_check_integrity(storage)
        elif choice == "6":
            _cmd_node_race()
        elif choice == "7":
            _cmd_clean_cache()
        elif choice == "8":
            _cmd_sector_collect()  # V7.6 新增
        else:
            print("  \u2717 无效选项")


# ============================================================================
# V7.6 核心新增：回测执行引擎
# ============================================================================

def _input_date_range():
    """提示用户输入回测日期范围，支持直接回车使用默认值"""
    from datetime import date, timedelta
    default_end   = date.today()
    default_start = default_end - timedelta(days=365)

    print(f"  日期范围 (格式 YYYY-MM-DD，直接回车使用默认值)")
    print(f"  默认起始日期: {default_start}  默认结束日期: {default_end}")

    while True:
        start_str = input(f"  起始日期 [{default_start}]: ").strip()
        if not start_str:
            start_str = str(default_start)
        try:
            from datetime import datetime
            start_dt = datetime.strptime(start_str, "%Y-%m-%d").date()
            break
        except ValueError:
            print("  \u2717 格式错误，请输入 YYYY-MM-DD")

    while True:
        end_str = input(f"  结束日期 [{default_end}]: ").strip()
        if not end_str:
            end_str = str(default_end)
        try:
            from datetime import datetime
            end_dt = datetime.strptime(end_str, "%Y-%m-%d").date()
            if end_dt <= start_dt:
                print("  \u2717 结束日期必须晚于起始日期")
                continue
            break
        except ValueError:
            print("  \u2717 格式错误，请输入 YYYY-MM-DD")

    return str(start_dt), str(end_dt)


def _select_strategy_interactive() -> str:
    """交互式策略选择，返回策略名称（key）"""
    from src.strategy.strategies import STRATEGY_REGISTRY, STRATEGY_DISPLAY_NAMES
    keys  = list(STRATEGY_REGISTRY.keys())
    names = STRATEGY_DISPLAY_NAMES

    print("\n  可用策略:")
    for i, k in enumerate(keys, 1):
        print(f"    {i:>2}. {k:<22} — {names.get(k, '')}")
    print()

    while True:
        raw = input("  请输入策略名称或编号: ").strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(keys):
                return keys[idx]
            else:
                print(f"  \u2717 编号超出范围 (1~{len(keys)})")
        elif raw in STRATEGY_REGISTRY:
            return raw
        else:
            # 模糊匹配
            matches = [k for k in keys if raw.lower() in k.lower()]
            if len(matches) == 1:
                print(f"  \u2139  自动匹配到: {matches[0]}")
                return matches[0]
            elif len(matches) > 1:
                print(f"  \u2717 模糊匹配到多个: {matches}，请精确输入")
            else:
                print(f"  \u2717 未知策略: {raw}")


def _get_stock_codes(parquet_dir: str, limit: int = 0) -> list:
    """从 parquet 目录扫描所有股票代码"""
    p = Path(parquet_dir)
    if not p.exists():
        logger.warning("Parquet 目录不存在: %s", parquet_dir)
        return []
    codes = [f.stem for f in sorted(p.glob("*.parquet"))]
    if limit > 0:
        codes = codes[:limit]
    return codes


def run_single_backtest(
    strategy_name: str,
    start_date: str,
    end_date: str,
    codes: list,
    bt_cfg: dict,
) -> dict:
    import pandas as pd
    from datetime import date, datetime
    import numpy as np

    try:
        from src.engine.execution import BacktestEngine
        from src.factors.alpha_engine import AlphaEngine
        from src.strategy.strategies import create_strategy
    except ImportError as e:
        return {"performance": {}, "equity_curve": [], "error": f"模块导入失败: {e}"}

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        def _tqdm(iterable, **kwargs):
            desc = kwargs.get("desc", "")
            total = kwargs.get("total", None)
            for i, item in enumerate(iterable):
                if desc:
                    print(f"\r  {desc}: {i+1}/{total or '?'}", end="", flush=True)
                yield item
            print()

    # 读取策略参数
    strategy_params = bt_cfg.get("strategy_params", {}).get(strategy_name, {})
    try:
        strategy = create_strategy(strategy_name, strategy_params)
    except Exception as e:
        return {"performance": {}, "equity_curve": [], "error": f"创建策略失败: {e}"}

    # 初始化回测引擎
    engine = BacktestEngine(
        initial_cash=bt_cfg["initial_cash"],
        commission_rate=bt_cfg["commission_rate"],
        slippage_rate=bt_cfg["slippage_rate"],
        stop_loss_pct=bt_cfg["stop_loss_pct"],
        take_profit_pct=bt_cfg["take_profit_pct"],
        trailing_stop_pct=bt_cfg["trailing_stop_pct"],
        max_position_pct=bt_cfg["max_position_pct"],
        circuit_breaker_max_dd=bt_cfg["circuit_breaker_max_dd"],
        circuit_breaker_cooldown_days=bt_cfg["circuit_breaker_cooldown_days"],
    )

    parquet_dir = Path(bt_cfg.get("parquet_dir", "./data/parquet"))
    MIN_HISTORY_DAYS = 100

    # ---------- 1. 加载历史数据 ----------
    print(f"  [{strategy_name}] 加载历史数据并预计算因子...")
    market_data: dict = {}
    loaded_count = skipped_count = 0

    for code in _tqdm(codes, desc="加载数据", total=len(codes)):
        parquet_path = parquet_dir / f"{code}.parquet"
        if not parquet_path.exists():
            skipped_count += 1
            continue
        try:
            df = pd.read_parquet(parquet_path)
            if "date" not in df.columns:
                skipped_count += 1
                continue
            # V7.8 B-05 Fix: 统一列名 vol -> volume
            df.rename(columns={"vol": "volume"}, inplace=True)
            df["date"] = df["date"].astype(str)
            df = df[df["date"] <= end_date].copy()
            if len(df) < MIN_HISTORY_DAYS:
                skipped_count += 1
                continue
            df = df.sort_values("date").reset_index(drop=True)
            market_data[code] = df
            loaded_count += 1
        except Exception as e:
            logger.warning("加载 %s 失败: %s", code, e)
            skipped_count += 1

    if not market_data:
        return {"performance": {}, "equity_curve": [], "error": "无有效股票数据"}

    # ---------- 2. 构建交易日历 ----------
    all_dates = set()
    for code, df in market_data.items():
        dates_in_range = df.loc[(df["date"] >= start_date) & (df["date"] <= end_date), "date"]
        all_dates.update(dates_in_range.tolist())
    trading_calendar = sorted(all_dates)
    if not trading_calendar:
        return {"performance": {}, "equity_curve": [], "error": "区间内无交易日数据"}
    date_to_idx = {date: i for i, date in enumerate(trading_calendar)}
    print(f"  [{strategy_name}] 交易日历: {len(trading_calendar)} 个交易日")

    # ---------- 3. 并行因子预计算 ----------
    print(f"  [{strategy_name}] 开始因子预计算（并行加速）...")
    try:
        from src.engine.execution import parallel_factor_precomputation
        factor_data = parallel_factor_precomputation(
            codes=list(market_data.keys()),
            market_data=market_data,
            max_workers=None,
        )
    except Exception as e:
        logger.warning("并行因子预计算失败，使用串行: %s", e)
        factor_data = {}
        for code, df in market_data.items():
            try:
                factor_data[code] = AlphaEngine.compute_from_history(df)
            except Exception as inner_e:
                logger.warning("串行因子计算失败 %s: %s", code, inner_e)

    # ---------- 4. 预计算每日评分矩阵 ----------
    # A-14 Fix: 扩展为多因子日评分矩阵，使所有策略都能使用预计算路径。
    #   daily_scores[day_idx][code] = {factor_name: value, ...}
    #   precomputed_scores 传给策略时为 {code: factor_dict}，
    #   RSRSMomentum 优先读取 rsrs_adaptive，其他策略可读取所需因子。
    print(f"  [{strategy_name}] 预计算每日多因子评分矩阵...")
    # 需提取的因子列（按策略需求收集）
    _SCORE_FACTORS = [
        "rsrs_adaptive",  # RSRSMomentum / RSRSAdvanced
        "mom",            # AlphaHunter / MomentumReversal
        "vol_factor",     # AlphaHunter
        "rsrs_zscore",    # RSRSAdvanced
        "r2",             # RSRSAdvanced
        "illiq",          # KunpengV10
        "smart_money",    # KunpengV10
    ]
    daily_scores = [{} for _ in trading_calendar]
    for code, fdf in factor_data.items():
        if fdf is None or fdf.empty:
            continue
        # 统一索引为字符串后对齐交易日历
        fdf = fdf.copy()
        if "date" in fdf.columns:
            fdf = fdf.set_index("date")
        fdf_aligned = fdf.reindex(trading_calendar)
        for day_idx in range(len(trading_calendar)):
            row_scores = {}
            for fn in _SCORE_FACTORS:
                if fn in fdf_aligned.columns:
                    v = fdf_aligned[fn].iloc[day_idx]
                    if not (isinstance(v, float) and np.isnan(v)):
                        row_scores[fn] = float(v)
            if row_scores:
                daily_scores[day_idx][code] = row_scores
    # 兼容旧的 RSRSMomentum 预计算路径（precomputed_scores 期望 {code: float}）
    # 主循环中 pre_scores = daily_scores[idx-1]，格式已是 {code: {factor:val}}
    # RSRSMomentumStrategy._generate_signals_from_scores 需要 {code: float(rsrs)}
    # 此处保持 {code: dict} 传入，策略侧自适应处理（已在 strategies.py 中兼容）

    # ---------- 5. 预计算价格数组 ----------
    # A-13 Fix: 内存说明
    #   price_arrays 预加载全部 OHLCV 为 NumPy 数组以加速主循环。
    #   内存估算：N_stocks × N_days × 5字段 × 4B(float32)
    #   例：5000只 × 2500日 × 5 × 4B ≈ 250MB RSS。
    #   内存受限时可通过 limit 参数（_get_stock_codes 的第二个参数）限制股票数量。
    _estimated_mb = len(market_data) * len(trading_calendar) * 5 * 4 / 1024 / 1024
    if _estimated_mb > 200:
        logger.warning(
            "price_arrays 预计占用 %.0fMB（%d只×%d天）；"
            "内存受限时请使用 codes[:N] 限制股票数量",
            _estimated_mb, len(market_data), len(trading_calendar)
        )
    price_arrays = {}
    for code, df in market_data.items():
        df_aligned = df.set_index("date").reindex(trading_calendar)
        price_arrays[code] = {
            "open":   df_aligned["open"].values.astype(np.float32),
            "high":   df_aligned["high"].values.astype(np.float32),
            "low":    df_aligned["low"].values.astype(np.float32),
            "close":  df_aligned["close"].values.astype(np.float32),
            "volume": df_aligned["volume"].fillna(0).values.astype(np.float32),  # A-01 Fix
        }

    # ---------- 6. 优化后的主循环 ----------
    equity_curve_records = []
    for idx, t_str in enumerate(_tqdm(trading_calendar, desc=f"回测 {strategy_name}", total=len(trading_calendar))):
        try:
            # 构建当日价格数据（使用 NumPy 数组快速获取）
            price_data = {}
            for code, parr in price_arrays.items():
                open_price = parr["open"][idx]
                if pd.isna(open_price):
                    continue
                price_data[code] = {
                    "open": float(open_price),
                    "high": float(parr["high"][idx]),
                    "low": float(parr["low"][idx]),
                    "close": float(parr["close"][idx]),
                    "volume": float(parr["volume"][idx]),
                }

            # 获取前一天的评分（第一天无评分）
            if idx == 0:
                pre_scores = {}
            else:
                pre_scores = daily_scores[idx - 1]

            # 获取当前持仓
            positions = engine.pm.get_all()

            # V7.8 B-01 Fix: 构建截止T-1日的 factor_data 切片（防前视偏差）
            # A-03 Fix: 切片前统一 fdf.index 为字符串类型，防止 DatetimeIndex/整数索引
            #           导致 searchsorted 类型不匹配，静默返回错误位置
            cur_factor_slice = {}
            for code, fdf in factor_data.items():
                if fdf is not None and not (hasattr(fdf, 'empty') and fdf.empty):
                    try:
                        fdf_s = fdf
                        # 统一索引为字符串
                        if "date" in fdf_s.columns:
                            fdf_s = fdf_s.set_index("date")
                        elif not (hasattr(fdf_s.index, 'dtype') and
                                  str(fdf_s.index.dtype) in ('object', 'string')):
                            fdf_s = fdf_s.copy()
                            fdf_s.index = fdf_s.index.astype(str)
                        idx_cut = fdf_s.index.searchsorted(t_str)
                        if idx_cut > 0:
                            cur_factor_slice[code] = fdf_s.iloc[:idx_cut]
                    except Exception:
                        cur_factor_slice[code] = fdf

            # V7.8 B-01 Fix: 构建截止T-1日的 market_data 切片
            cur_market_slice = {}
            for code, df in market_data.items():
                if "date" in df.columns:
                    sub = df[df["date"] < t_str]
                    if not sub.empty:
                        cur_market_slice[code] = sub
                else:
                    cur_market_slice[code] = df

            # 生成信号（传入真实历史数据切片 + 预计算评分作为快速路径）
            try:
                current_dt = datetime.strptime(t_str, "%Y-%m-%d")
                signals = strategy.generate_signals(
                    universe=list(price_data.keys()),
                    market_data=cur_market_slice,   # V7.8 B-01 Fix: 真实数据切片
                    factor_data=cur_factor_slice,    # V7.8 B-01 Fix: 真实因子切片
                    current_date=current_dt,
                    positions=positions,
                    precomputed_scores=pre_scores,   # RSRSMomentum快速路径保留
                )
            except Exception as e:
                logger.warning("策略 %s 在 %s 生成信号异常: %s", strategy_name, t_str, e)
                signals = []

            # 执行 step
            bar_date = date.fromisoformat(t_str)
            result = engine.step(bar_date, price_data, signals)
            snap = result.get("snapshot")
            if snap:
                equity_curve_records.append({
                    "timestamp": t_str,
                    "total_value": float(snap.total_value),
                    "cash": float(snap.cash),
                    "market_value": float(snap.market_value),
                })
        except Exception as e:
            logger.warning("回测主循环 %s 异常: %s", t_str, e)
            continue

    # 计算绩效
    try:
        performance = engine.get_performance()
    except Exception as e:
        logger.warning("计算绩效失败: %s", e)
        performance = {}

    return {
        "performance": performance,
        "equity_curve": equity_curve_records,
        "strategy_name": strategy_name,
        "strategy_params": strategy_params,
        "start_date": start_date,
        "end_date": end_date,
        "codes_count": loaded_count,
        "error": None,
    }


def _print_performance(perf: dict, strategy_name: str = "") -> None:
    """格式化打印绩效指标，包含盈亏比和卡玛比率"""
    if not perf:
        print("  （无绩效数据）")
        return

    label = f" [{strategy_name}]" if strategy_name else ""
    print(f"\n  {'='*46}")
    print(f"  回测绩效指标{label}")
    print(f"  {'='*46}")

    def fmt_pct(v):
        return f"{v*100:+.2f}%" if isinstance(v, (int, float)) else "N/A"

    def fmt_f(v, decimals=4):
        return f"{v:.{decimals}f}" if isinstance(v, (int, float)) else "N/A"

    # 计算卡玛比率（如果年化收益率和最大回撤都存在）
    annual = perf.get("annual_return")
    mdd = perf.get("max_drawdown")
    if isinstance(annual, (int, float)) and isinstance(mdd, (int, float)) and mdd != 0:
        calmar = annual / mdd
    else:
        calmar = None

    # 从 perf 中获取盈亏比（可能不存在）
    profit_loss_ratio = perf.get("profit_loss_ratio")

    rows = [
        ("总收益率",   fmt_pct(perf.get("total_return"))),
        ("年化收益率", fmt_pct(perf.get("annual_return"))),
        ("夏普比率",   fmt_f(perf.get("sharpe_ratio"), 3)),
        ("最大回撤",   fmt_pct(perf.get("max_drawdown"))),
        ("胜率",       fmt_pct(perf.get("trade_win_rate"))),
        ("总交易次数", str(int(perf.get("total_trades", 0)))),
        ("盈亏比",     fmt_f(profit_loss_ratio, 3) if profit_loss_ratio is not None else "N/A"),
        ("卡玛比率",   fmt_f(calmar, 3) if calmar is not None else "N/A"),
    ]
    for name, val in rows:
        print(f"  {name:<10}: {val}")
    print(f"  {'='*46}")


def _save_backtest_result(result: dict, results_dir: str = "./results") -> str:
    """保存回测结果到 JSON 文件，返回文件路径"""
    import json
    from datetime import datetime

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = result.get("strategy_name", "unknown")
    sd   = result.get("start_date", "").replace("-", "")
    ed   = result.get("end_date",   "").replace("-", "")
    fname = f"{name}_{sd}_{ed}_{ts}.json"
    fpath = Path(results_dir) / fname

    out = {
        "strategy_name":   result.get("strategy_name"),
        "strategy_params": result.get("strategy_params", {}),
        "start_date":      result.get("start_date"),
        "end_date":        result.get("end_date"),
        "codes_count":     result.get("codes_count", 0),
        "performance":     result.get("performance", {}),
        "equity_curve":    result.get("equity_curve", []),
        "generated_at":    datetime.now().isoformat(),
    }
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    return str(fpath)


def _ascii_bar_chart(items: list, value_key: str, label_key: str, width: int = 30) -> None:
    """简单 ASCII 条形图（用于多策略对比）"""
    vals = [item.get(value_key, 0) or 0 for item in items]
    max_abs = max((abs(v) for v in vals), default=1) or 1
    print(f"\n  {'策略':<24} {'值':>9}  图示")
    print("  " + "-" * (24 + 9 + width + 5))
    for item, v in zip(items, vals):
        label = item.get(label_key, "?")[:22]
        bar_len = int(abs(v) / max_abs * width)
        bar_char = "█" if v >= 0 else "▒"
        bar = bar_char * bar_len
        print(f"  {label:<24} {v*100:>+8.2f}%  {bar}")


# ============================================================================
# 回测菜单 (V7.6 选项1、2完整实现)
# ============================================================================

def backtest_menu(config=None) -> None:
    """回测系统主菜单 (V7.6)"""
    bt_cfg = _load_backtest_config()
    # 注入 parquet_dir
    collector_cfg = _load_collector_config()
    bt_cfg["parquet_dir"] = collector_cfg["parquet_dir"]

    while True:
        print("\n" + "=" * 44)
        print("  回测系统  (V7.8)")
        print("=" * 44)
        print("  1. 运行单策略回测")
        print("  2. 多策略对比")
        print("  3. 查看历史回测结果")
        print("  4. 生成 HTML 报告 (从已有回测结果)")  # V7.7 新增
        print("  0. 返回主菜单")
        choice = input("请选择 [0-4]: ").strip()

        if choice == "0":
            break

        # ── 选项 1: 单策略回测 ──────────────────────────────────────────
        elif choice == "1":
            print("\n" + "=" * 44)
            print("  单策略回测  (V7.6)")
            print("=" * 44)

            # 1a. 选择策略
            try:
                strategy_name = _select_strategy_interactive()
            except Exception as e:
                print(f"  \u2717 策略选择失败: {e}")
                input("按 Enter 继续...")
                continue

            print(f"\n  已选择策略: {strategy_name}")
            sp = bt_cfg.get("strategy_params", {}).get(strategy_name, {})
            if sp:
                print(f"  策略参数:   {sp}")
            else:
                print("  策略参数:   使用默认值")

            # 1b. 日期范围
            start_date, end_date = _input_date_range()
            print(f"  回测区间:   {start_date} ~ {end_date}")

            # 1c. 股票池
            codes = _get_stock_codes(bt_cfg["parquet_dir"])
            if not codes:
                print(f"  \u2717 未找到股票数据，请先执行数据采集")
                input("按 Enter 继续...")
                continue

            # 询问是否限制股票数量
            limit_str = input(f"\n  共 {len(codes)} 只股票，是否限制数量？"
                              f"（直接回车=全部，输入数字=前N只）: ").strip()
            if limit_str.isdigit() and int(limit_str) > 0:
                codes = codes[:int(limit_str)]
                print(f"  已限制为前 {len(codes)} 只股票")

            print(f"\n  回测配置:")
            print(f"    初始资金: {bt_cfg['initial_cash']:,.0f}")
            print(f"    佣金率:   {bt_cfg['commission_rate']*100:.3f}%")
            print(f"    滑点率:   {bt_cfg['slippage_rate']*100:.3f}%")
            print(f"    单股上限: {bt_cfg['max_position_pct']*100:.1f}%")
            print(f"    止损:     {bt_cfg['stop_loss_pct']*100:.1f}%  "
                  f"止盈: {bt_cfg['take_profit_pct']*100:.1f}%")
            print(f"    股票数量: {len(codes)} 只")

            confirm = input("\n  确认开始回测? [y/N]: ").strip().lower()
            if confirm != "y":
                print("  已取消。")
                continue

            # 1d. 执行回测
            import time as _time
            t0 = _time.time()
            result = run_single_backtest(
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                codes=codes,
                bt_cfg=bt_cfg,
            )
            elapsed = _time.time() - t0

            if result.get("error"):
                print(f"  \u2717 回测失败: {result['error']}")
                input("按 Enter 继续...")
                continue

            # 1e. 展示绩效
            print(f"\n  \u2713 回测完成（耗时 {elapsed:.1f}s）")
            _print_performance(result["performance"], strategy_name)

            # 1f. 保存结果
            try:
                fpath = _save_backtest_result(result)
                print(f"\n  结果已保存: {fpath}")
            except Exception as e:
                print(f"  \u2717 保存失败: {e}")

            input("按 Enter 继续...")

        # ── 选项 2: 多策略对比 ──────────────────────────────────────────
        elif choice == "2":
            print("\n" + "=" * 44)
            print("  多策略对比  (V7.6)")
            print("=" * 44)

            try:
                from src.strategy.strategies import STRATEGY_REGISTRY, STRATEGY_DISPLAY_NAMES
            except ImportError as e:
                print(f"  \u2717 导入策略失败: {e}")
                input("按 Enter 继续...")
                continue

            keys  = list(STRATEGY_REGISTRY.keys())
            names = STRATEGY_DISPLAY_NAMES

            print("  可用策略:")
            for i, k in enumerate(keys, 1):
                print(f"    {i:>2}. {k:<24} — {names.get(k, '')}")
            print()
            print("  输入策略名称（逗号分隔），或输入 all 选择全部策略")
            raw = input("  请输入: ").strip()

            if raw.lower() == "all":
                selected = list(keys)
            else:
                parts = [p.strip() for p in raw.split(",")]
                selected = []
                for p in parts:
                    if p.isdigit() and 1 <= int(p) <= len(keys):
                        selected.append(keys[int(p) - 1])
                    elif p in STRATEGY_REGISTRY:
                        selected.append(p)
                    else:
                        print(f"  \u26a0\ufe0f  未知策略: {p}，已跳过")

            if not selected:
                print("  \u2717 未选择任何有效策略")
                input("按 Enter 继续...")
                continue

            print(f"\n  已选择 {len(selected)} 个策略: {selected}")

            # 日期范围
            start_date, end_date = _input_date_range()
            print(f"  回测区间: {start_date} ~ {end_date}")

            # 股票池
            codes = _get_stock_codes(bt_cfg["parquet_dir"])
            if not codes:
                print(f"  \u2717 未找到股票数据")
                input("按 Enter 继续...")
                continue

            limit_str = input(f"\n  共 {len(codes)} 只股票，是否限制数量？"
                              f"（直接回车=全部，输入数字=前N只）: ").strip()
            if limit_str.isdigit() and int(limit_str) > 0:
                codes = codes[:int(limit_str)]
                print(f"  已限制为前 {len(codes)} 只股票")

            print(f"\n  \u26a0\ufe0f  将串行运行 {len(selected)} 个策略的回测，可能需要较长时间。")
            print(f"  每个策略处理 {len(codes)} 只股票 × {start_date}~{end_date}。")
            confirm = input("  确认开始多策略对比? [y/N]: ").strip().lower()
            if confirm != "y":
                print("  已取消。")
                continue

            # A-09 Fix: 使用 ProcessPoolExecutor 并行执行多策略
            # 注意：每个策略在独立子进程中运行，无 GIL 瓶颈；
            #       max_workers 上限 4 防止内存溢出（每策略~250MB）
            all_results = []
            import time as _time
            from concurrent.futures import ProcessPoolExecutor, as_completed as _as_completed
            total_t0 = _time.time()

            _max_parallel = min(len(selected), 4)
            print(f"\n  ⚡ 并行运行 {len(selected)} 个策略（最多 {_max_parallel} 并发）...")

            def _run_one_strategy(args):
                sname, sd, ed, cds, cfg = args
                return sname, run_single_backtest(
                    strategy_name=sname,
                    start_date=sd, end_date=ed, codes=cds, bt_cfg=cfg,
                )

            _tasks = [(sname, start_date, end_date, codes, bt_cfg)
                      for sname in selected]
            _result_map = {}

            try:
                with ProcessPoolExecutor(max_workers=_max_parallel) as _pool:
                    _fmap = {_pool.submit(_run_one_strategy, t): t[0] for t in _tasks}
                    for _fut in _as_completed(_fmap):
                        _sn = _fmap[_fut]
                        try:
                            _, _res = _fut.result()
                            _result_map[_sn] = _res
                            if _res.get("error"):
                                print(f"    \u2717 {_sn}: {_res['error']}")
                            else:
                                print(f"    \u2713 {_sn} 完成")
                        except Exception as _exc:
                            _result_map[_sn] = {"error": str(_exc), "performance": {}, "equity_curve": []}
                            print(f"    \u2717 {_sn} 异常: {_exc}")
            except Exception as _pool_err:
                # 并行失败时降级到串行
                print(f"  ⚠ 并行执行失败（{_pool_err}），降级为串行...")
                for sname in selected:
                    print(f"  [{selected.index(sname)+1}/{len(selected)}] 运行策略: {sname} ...")
                    _result_map[sname] = run_single_backtest(
                        strategy_name=sname, start_date=start_date,
                        end_date=end_date, codes=codes, bt_cfg=bt_cfg,
                    )

            for sname in selected:
                result = _result_map.get(sname, {"error": "未执行", "performance": {}, "equity_curve": []})
                if result.get("error"):
                    all_results.append({
                        "strategy_name": sname,
                        "display_name":  names.get(sname, sname),
                        "error":         result["error"],
                        "performance":   {},
                        "equity_curve":  [],
                    })
                else:
                    all_results.append({
                        "strategy_name": sname,
                        "display_name":  names.get(sname, sname),
                        "error":         None,
                        "performance":   result["performance"],
                        "equity_curve":  result["equity_curve"],
                        "start_date":    start_date,
                        "end_date":      end_date,
                        "codes_count":   result.get("codes_count", 0),
                    })

            total_elapsed = _time.time() - total_t0
            print(f"\n  \u2713 多策略对比完成（总耗时 {total_elapsed:.1f}s）")

            # 打印对比表格
            print("\n" + "=" * 74)
            print(f"  {'策略名称':<22} {'总收益':>8} {'年化':>8} {'夏普':>7} {'最大回撤':>9} {'胜率':>7} {'交易次数':>8}")
            print("  " + "-" * 72)

            chart_items = []
            for r in all_results:
                perf = r.get("performance", {})
                sn   = r.get("strategy_name", "?")[:20]
                if r.get("error"):
                    print(f"  {sn:<22} {'ERROR: ' + str(r['error'])[:40]}")
                    continue
                tr  = perf.get("total_return",  0) or 0
                ar  = perf.get("annual_return",  0) or 0
                sr  = perf.get("sharpe_ratio",   0) or 0
                mdd = perf.get("max_drawdown",   0) or 0
                wr  = perf.get("trade_win_rate", 0) or 0  # A-02 Fix: 统一键名
                tt  = int(perf.get("total_trades", 0) or 0)
                print(
                    f"  {sn:<22} {tr*100:>+7.2f}% {ar*100:>+7.2f}% "
                    f"{sr:>7.3f} {mdd*100:>8.2f}% {wr*100:>6.1f}% {tt:>8}"
                )
                chart_items.append({
                    "strategy_name": sn,
                    "annual_return":  ar,
                    "sharpe_ratio":   sr,
                })

            print("=" * 74)

            # ASCII 条形图（年化收益率）
            if chart_items:
                print("\n  年化收益率对比 (ASCII 图示):")
                _ascii_bar_chart(chart_items, "annual_return", "strategy_name", width=28)

            # 保存多策略对比结果
            try:
                import json
                from datetime import datetime
                Path("results").mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = Path("results") / f"multi_compare_{ts}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "generated_at": datetime.now().isoformat(),
                        "start_date":   start_date,
                        "end_date":     end_date,
                        "codes_count":  len(codes),
                        "strategies":   all_results,
                    }, f, ensure_ascii=False, indent=2, default=str)
                print(f"\n  详细结果已保存: {out_path}")
            except Exception as e:
                print(f"  \u2717 保存失败: {e}")

            input("按 Enter 继续...")

        # ── 选项 3: 查看历史回测结果 ────────────────────────────────────
        elif choice == "3":
            results_dir = Path("results")
            if results_dir.exists():
                files = sorted(results_dir.glob("*.json"), reverse=True)
                print(f"\n  共 {len(files)} 个历史结果（最新在前）:")
                for i, f in enumerate(files[:15], 1):
                    size_kb = f.stat().st_size / 1024
                    print(f"    {i:>2}. {f.name}  ({size_kb:.1f} KB)")
                if len(files) > 15:
                    print(f"    ... 余 {len(files)-15} 个")
                if files:
                    print("\n  输入序号查看详情（直接回车跳过）:")
                    sel = input("  > ").strip()
                    if sel.isdigit() and 1 <= int(sel) <= min(15, len(files)):
                        chosen = files[int(sel) - 1]
                        try:
                            import json
                            with open(chosen, encoding="utf-8") as fp:
                                data = json.load(fp)
                            perf = data.get("performance", {})
                            print(f"\n  文件: {chosen.name}")
                            print(f"  策略: {data.get('strategy_name', '?')}")
                            print(f"  区间: {data.get('start_date','?')} ~ {data.get('end_date','?')}")
                            print(f"  股票: {data.get('codes_count', '?')} 只")
                            _print_performance(perf, data.get("strategy_name", ""))
                        except Exception as e:
                            print(f"  \u2717 读取失败: {e}")
            else:
                print("  暂无历史回测结果")
            input("按 Enter 返回...")

        # ── 选项 4: 生成 HTML 报告 ──────────────────────────────────────────
        elif choice == "4":
            print("\n" + "=" * 44)
            print("  生成 HTML 报告  (V7.7)")
            print("=" * 44)

            results_dir = Path("results")
            if not results_dir.exists():
                print("  ✗ results 目录不存在，请先运行回测")
                input("按 Enter 继续...")
                continue

            # 列出所有 JSON 文件
            json_files = sorted(results_dir.glob("*.json"), reverse=True)
            if not json_files:
                print("  ✗ results 目录中没有 JSON 回测结果文件")
                input("按 Enter 继续...")
                continue

            print(f"\n  共 {len(json_files)} 个回测结果（最新在前）:")
            for i, f in enumerate(json_files[:20], 1):
                size_kb = f.stat().st_size / 1024
                print(f"    {i:>2}. {f.name}  ({size_kb:.1f} KB)")
            if len(json_files) > 20:
                print(f"    ... 余 {len(json_files) - 20} 个")

            print()
            raw = input("  请输入序号或完整文件名（直接回车取消）: ").strip()
            if not raw:
                print("  已取消。")
                continue

            # 解析用户输入
            chosen_json: Path = None
            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(json_files):
                    chosen_json = json_files[idx]
                else:
                    print(f"  ✗ 序号超出范围")
                    input("按 Enter 继续...")
                    continue
            else:
                # 按文件名匹配
                p = results_dir / raw
                if not raw.endswith(".json"):
                    p = results_dir / (raw + ".json")
                if p.exists():
                    chosen_json = p
                else:
                    print(f"  ✗ 文件不存在: {p}")
                    input("按 Enter 继续...")
                    continue

            # 生成 HTML 路径（同目录，替换后缀）
            html_name = chosen_json.stem + "_report.html"
            default_html = results_dir / html_name
            html_input = input(f"  输出 HTML 路径 [{default_html}]: ").strip()
            out_html = Path(html_input) if html_input else default_html

            # 调用报告生成函数
            try:
                from src.utils.report import generate_html_report
                generate_html_report(str(chosen_json), str(out_html))
                print(f"\n  ✓ HTML 报告已生成: {out_html}")
                print(f"    (用浏览器打开即可查看)")
            except FileNotFoundError as e:
                print(f"  ✗ 文件未找到: {e}")
            except ValueError as e:
                print(f"  ✗ 数据解析失败: {e}")
            except ImportError:
                print("  ✗ 导入 src.utils.report 失败，请确认 build_v7.7.py 已成功运行")
            except Exception as e:
                logger.exception("HTML 报告生成异常")
                print(f"  ✗ 报告生成失败: {e}")
            input("按 Enter 继续...")

        else:
            print("  \u2717 无效选项")


# ============================================================================
# 系统管理菜单 (保持 V7.5 原有)
# ============================================================================

def system_management_menu(config=None) -> None:
    while True:
        print("" + "=" * 40)
        print("  系统管理")
        print("=" * 40)
        print("  1. 健康检查")
        print("  2. 查看日志")
        print("  3. 数据源心跳检测")
        print("  0. 返回主菜单")
        choice = input("请选择 [0-3]: ").strip()
        if choice == "0":
            break
        elif choice == "1":
            from main import run_health_check
            run_health_check()
        elif choice == "2":
            log_path = Path("logs/q-unity.log")
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8").splitlines()
                print(f"最近 20 行日志:")
                for line in lines[-20:]:
                    print(f"  {line}")
            else:
                print("  暂无日志文件")
        elif choice == "3":
            _check_data_source_heartbeat()
        else:
            print("  \u2717 无效选项")


def _strategy_select_menu(config_path="config.json") -> None:
    """Strategy selection submenu (V7.4)"""
    import json
    from pathlib import Path

    # V7.8 B-07 Fix: 动态获取策略注册表，自动包含新策略
    try:
        from src.strategy.strategies import STRATEGY_REGISTRY, STRATEGY_DISPLAY_NAMES
        STRAT_KEYS = list(STRATEGY_REGISTRY.keys())
        STRAT_NAMES = STRATEGY_DISPLAY_NAMES
    except ImportError:
        STRAT_KEYS = [
            "rsrs_momentum", "alpha_hunter", "rsrs_advanced", "short_term",
            "momentum_reversal", "sentiment_reversal", "kunpeng_v10", "alpha_max_v5_fixed",
            # A-06 Fix: 移除幽灵策略 sector_momentum（类不存在，运行时 KeyError）
        ]
        STRAT_NAMES = {
            "rsrs_momentum":      "RSRS动量策略",
            "alpha_hunter":       "Alpha猎手策略",
            "rsrs_advanced":      "高级RSRS策略",
            "short_term":         "短线快进快出",
            "momentum_reversal":  "动量反转双模式",
            "sentiment_reversal": "情绪反转策略",
            "kunpeng_v10":        "鲲鹏V10微结构",
            "alpha_max_v5_fixed": "AlphaMaxV5机构多因子",
            # A-06 Fix: 已移除 sector_momentum（幽灵策略）
        }

    p = Path(config_path)
    if p.exists():
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    else:
        cfg = {}

    rt = cfg.setdefault("realtime", {})
    active = list(rt.get("active_strategies", ["rsrs_momentum", "kunpeng_v10"]))

    while True:
        print(chr(10) + "=" * 56)
        print("  策略选择 (V7.4)  -- 选择用于实时预警的策略")
        print("=" * 56)
        for i, k in enumerate(STRAT_KEYS, 1):
            mark = "[\u2713]" if k in active else "[ ]"
            print(f"  {i}. {mark} {STRAT_NAMES[k]:<24} ({k})")
        merge = rt.get("signal_merge_rule", "any")
        print("  当前合并规则: " + str(merge))
        print("  操作:")
        print("  1-8  切换策略启用/停用")
        print("  r    修改合并规则 (any/majority/weighted)")
        print("  s    保存并返回")
        print("  q    不保存返回")
        print()
        choice = input("请选择: ").strip().lower()

        if choice == "q":
            break
        elif choice == "s":
            rt["active_strategies"] = active
            try:
                p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"  \u2713 已保存 active_strategies={active}")
            except Exception as e:
                print(f"  \u2717 保存失败: {e}")
            break
        elif choice == "r":
            print("  合并规则:")
            print("    any      -- 任一策略触发即预警（默认）")
            print("    majority -- 超半数策略触发才预警")
            print("    weighted -- 加权评分合并，分数超阈值预警")
            rule = input("  输入规则 [any/majority/weighted]: ").strip().lower()
            if rule in ("any", "majority", "weighted"):
                rt["signal_merge_rule"] = rule
                print(f"  \u2713 合并规则已设为: {rule}")
            else:
                print("  \u2717 无效规则，未修改")
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(STRAT_KEYS):
                k = STRAT_KEYS[idx]
                if k in active:
                    active.remove(k)
                    print(f"  -- 已停用: {STRAT_NAMES[k]}")
                else:
                    active.append(k)
                    print(f"  ++ 已启用: {STRAT_NAMES[k]}")
            else:
                print("  \u2717 无效编号")
        else:
            print("  \u2717 无效输入")


def _strategy_params_menu(config_path="config.json") -> None:
    """Strategy parameter tuning submenu (V7.4)"""
    import json
    from pathlib import Path

    TUNABLE = {
        "rsrs_momentum": {
            "top_n":           ("int",   10,  "最大持仓只数"),
            "rsrs_threshold":  ("float", 0.5, "RSRS自适应阈值"),
        },
        "alpha_hunter": {
            "top_n":      ("int",   15,  "最大持仓只数"),
            "min_score":  ("float", 0.3, "最低综合评分"),
        },
        "rsrs_advanced": {
            "top_n":           ("int",   10,  "最大持仓只数"),
            "rsrs_threshold":  ("float", 0.5, "RSRS阈值"),
            "r2_threshold":    ("float", 0.7, "R^2过滤阈值"),
        },
        "short_term": {
            "top_n":              ("int",   5,    "最大持仓只数"),
            "hold_calendar_days": ("int",   7,    "最大持仓日历天数"),
            "mom_threshold":      ("float", 0.03, "动量触发阈值"),
        },
        "momentum_reversal": {
            "top_n":         ("int",   10,  "最大持仓只数"),
            "market_thresh": ("float", 0.0, "市场牛熊判断阈值"),
        },
        "sentiment_reversal": {
            "top_n":         ("int",   10,   "最大持仓只数"),
            "oversold_z":    ("float", -1.5, "超卖Z-score阈值"),
            "overbought_z":  ("float", 1.5,  "超买Z-score阈值"),
        },
        "kunpeng_v10": {
            "top_n":         ("int",   15,   "最大持仓只数"),
            "illiq_window":  ("int",   20,   "非流动性计算窗口"),
            "smart_window":  ("int",   10,   "聪明钱计算窗口"),
            "breadth_limit": ("float", 0.15, "宽度熔断跌停比例阈值"),
        },
        "alpha_max_v5_fixed": {
            "top_n":       ("int",   20,   "最大持仓只数"),
            "ep_weight":   ("float", 0.20, "EP因子权重"),
            "growth_w":    ("float", 0.15, "成长因子权重"),
            "mom_w":       ("float", 0.15, "动量因子权重"),
            "quality_w":   ("float", 0.20, "质量因子权重"),
            "rev_w":       ("float", 0.10, "反转因子权重"),
            "liq_w":       ("float", 0.10, "流动性因子权重"),
            "res_vol_w":   ("float", 0.10, "残差波动率权重"),
        },
    }
    STRAT_NAMES = {
        "rsrs_momentum":      "RSRS动量策略",
        "alpha_hunter":       "Alpha猎手策略",
        "rsrs_advanced":      "高级RSRS策略",
        "short_term":         "短线快进快出",
        "momentum_reversal":  "动量反转双模式",
        "sentiment_reversal": "情绪反转策略",
        "kunpeng_v10":        "鲲鹏V10微结构",
        "alpha_max_v5_fixed": "AlphaMaxV5机构多因子",
    }

    p = Path(config_path)
    if p.exists():
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    else:
        cfg = {}
    rt = cfg.setdefault("realtime", {})
    sp = rt.setdefault("strategy_params", {})
    strat_keys = list(TUNABLE.keys())

    while True:
        print(chr(10) + "=" * 56)
        print("  策略参数调优 (V7.4)")
        print("=" * 56)
        for i, k in enumerate(strat_keys, 1):
            print(f"  {i}. {STRAT_NAMES[k]} ({k})")
        print("  0. 返回")
        print()
        choice = input("请选择策略 [0-8]: ").strip()
        if choice == "0":
            break
        if not choice.isdigit() or not (1 <= int(choice) <= len(strat_keys)):
            print("  \u2717 无效选项")
            continue

        strat_key = strat_keys[int(choice) - 1]
        params_def = TUNABLE[strat_key]
        cur_params = sp.get(strat_key, {})

        while True:
            print("  策略: " + STRAT_NAMES[strat_key])
            print(f"  {'参数名':<22} {'当前值':<12} {'默认值':<12} {'说明'}")
            print("  " + "-" * 65)
            param_keys = list(params_def.keys())
            for j, pname in enumerate(param_keys, 1):
                ptype, pdefault, pdesc = params_def[pname]
                cur_val = cur_params.get(pname, pdefault)
                print(f"  {j}. {pname:<22} {str(cur_val):<12} {str(pdefault):<12} {pdesc}")
            print("  输入参数编号修改，s=保存并返回，r=重置默认，q=不保存返回")
            pchoice = input("  请选择: ").strip().lower()
            if pchoice == "q":
                break
            elif pchoice == "r":
                sp.pop(strat_key, None)
                cur_params = {}
                print(f"  \u2713 {strat_key} 参数已重置为默认值")
            elif pchoice == "s":
                if cur_params:
                    sp[strat_key] = cur_params
                try:
                    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"  \u2713 已保存 {strat_key} 参数: {cur_params}")
                except Exception as e:
                    print(f"  \u2717 保存失败: {e}")
                break
            elif pchoice.isdigit() and 1 <= int(pchoice) <= len(param_keys):
                pidx = int(pchoice) - 1
                pname = param_keys[pidx]
                ptype, pdefault, pdesc = params_def[pname]
                cur_val = cur_params.get(pname, pdefault)
                new_val_str = input(f"  输入 {pname} 新值 (当前={cur_val}, 类型={ptype}): ").strip()
                try:
                    if ptype == "int":
                        new_val = int(new_val_str)
                    else:
                        new_val = float(new_val_str)
                    cur_params[pname] = new_val
                    print(f"  \u2713 {pname} = {new_val}")
                except ValueError:
                    print(f"  \u2717 输入无效，需要 {ptype} 类型")
            else:
                print("  \u2717 无效输入")


def realtime_menu(config=None) -> None:
    """实时交易菜单 (V7.4)"""
    _engine_ref = [None]

    def _get_engine():
        if _engine_ref[0] is None:
            try:
                from src.realtime.monitor import MonitorEngine
                _engine_ref[0] = MonitorEngine()
                print("  \u2713 MonitorEngine 已初始化")
            except Exception as e:
                print(f"  \u2717 无法初始化 MonitorEngine: {e}")
        return _engine_ref[0]

    while True:
        print(chr(10) + "=" * 56)
        print("  实时交易  (V7.4)")
        print("=" * 56)
        running_str = ""
        eng = _engine_ref[0]
        if eng is not None:
            running_str = " [运行中]" if eng.is_running() else " [已停止]"
        print(f"  1. 启动实时监控{running_str}")
        print("  2. 停止实时监控")
        print("  3. 立即执行一次扫描")
        print("  4. 查看最近信号")
        print("  5. 查看模拟持仓")
        print("  6. 账户摘要")
        print("  7. 预警设置")
        print("  8. 发送测试预警")
        print("  9. 查看实时日志")
        print("  a. 策略选择 (V7.4)")
        print("  b. 策略参数调优 (V7.4)")
        print("  0. 返回主菜单")
        print()
        choice = input("请选择 [0-9/a/b]: ").strip().lower()

        if choice == "0":
            eng = _engine_ref[0]
            if eng is not None and eng.is_running():
                confirm = input("  实时监控仍在运行，确认停止并退出? [y/N]: ").strip().lower()
                if confirm == "y":
                    eng.stop()
                    print("  \u2713 监控已停止")
                else:
                    continue
            break

        elif choice == "1":
            eng = _get_engine()
            if eng is None:
                continue
            if eng.is_running():
                print("  \u26a0\ufe0f  监控已在运行中")
            else:
                try:
                    eng.start()
                    print("  \u2713 实时监控已启动（后台线程）")
                    strats = list(eng._strategies.keys()) if hasattr(eng, "_strategies") else []
                    print(f"  已加载策略: {strats if strats else '均线回退模式'}")
                except Exception as e:
                    print(f"  \u2717 启动失败: {e}")

        elif choice == "2":
            eng = _engine_ref[0]
            if eng is None or not eng.is_running():
                print("  \u26a0\ufe0f  监控未在运行")
            else:
                eng.stop()
                print("  \u2713 实时监控已停止")

        elif choice == "3":
            eng = _get_engine()
            if eng is None:
                continue
            print("  \u23f3 执行单次扫描...")
            try:
                signals = eng.scan_once()
                if signals:
                    print(f"  \u2713 本次扫描发现 {len(signals)} 个信号:")
                    print(f"  {'时间':<20} {'代码':<8} {'策略':<28} {'方向':<6} {'评分':<8}")
                    print("  " + "-" * 75)
                    for s in signals[:20]:
                        print(f"  {s.get('time','?'):<20} {s.get('code','?'):<8} "
                              f"{s.get('strategy','?'):<28} {s.get('signal','?'):<6} "
                              f"{s.get('score',0):<8.3f}")
                    if len(signals) > 20:
                        print(f"  ... 共 {len(signals)} 个信号，仅显示前20个")
                else:
                    print("  本次扫描无信号")
            except Exception as e:
                print(f"  \u2717 扫描失败: {e}")
            input("按 Enter 继续...")

        elif choice == "4":
            eng = _engine_ref[0]
            if eng is None:
                print("  监控尚未初始化，请先选择「1.启动实时监控」")
                continue
            signals = eng.get_recent_signals(n=30)
            if not signals:
                print("  暂无最近信号")
            else:
                print(f"  最近 {len(signals)} 条信号:")
                print(f"  {'时间':<20} {'代码':<8} {'策略':<28} {'方向':<6} {'评分':<8}")
                print("  " + "-" * 75)
                for s in signals:
                    print(f"  {s.get('time','?'):<20} {s.get('code','?'):<8} "
                          f"{s.get('strategy','?'):<28} {s.get('signal','?'):<6} "
                          f"{s.get('score',0):<8.3f}")
            input("按 Enter 继续...")

        elif choice == "5":
            try:
                from src.realtime.trader import SimulatedTrader
                trader = SimulatedTrader()
                positions = trader.get_positions()
                if not positions:
                    print("  暂无持仓")
                else:
                    print(f"  当前持仓 ({len(positions)} 只):")
                    print(f"  {'代码':<8} {'成本':<10} {'现价':<10} {'数量':<8} {'盈亏%':<10} {'持天':<6}")
                    print("  " + "-" * 55)
                    for pos in positions:
                        code  = pos.get('code', '?')
                        cost  = pos.get('avg_cost', 0)
                        price = pos.get('current_price', cost)
                        qty   = pos.get('shares', 0)
                        days  = pos.get('holding_days', 0)
                        pnl_pct = (price - cost) / cost * 100 if cost > 0 else 0
                        print(f"  {code:<8} {cost:<10.3f} {price:<10.3f} {qty:<8} "
                              f"{pnl_pct:+.2f}%{'':>2} {days:<6}")
            except Exception as e:
                print(f"  \u2717 获取持仓失败: {e}")
            input("按 Enter 继续...")

        elif choice == "6":
            try:
                from src.realtime.trader import SimulatedTrader
                trader = SimulatedTrader()
                summary = trader.get_account_summary()
                print("  " + "=" * 42)
                print("  账户摘要（模拟交易）")
                print("  " + "=" * 42)
                print(f"  当前现金:   {summary.get('cash',0):>15,.2f}")
                print(f"  持仓市值:   {summary.get('market_value',0):>15,.2f}")
                print(f"  总资产:     {summary.get('total_assets',0):>15,.2f}")
                pnl = summary.get('pnl', 0)
                print(f"  总盈亏:     {pnl:>+15,.2f}")
                pnl_pct = summary.get('pnl_pct', 0) * 100
                print(f"  总收益率:   {pnl_pct:>14.2f}%")
                print(f"  持仓数量:   {summary.get('position_count',0):>15}")
                print("  " + "=" * 42)
            except Exception as e:
                print(f"  \u2717 获取账户摘要失败: {e}")
            input("按 Enter 继续...")

        elif choice == "7":
            try:
                cfg_path = Path("config.json")
                if cfg_path.exists():
                    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
                    rt_cfg = raw.get("realtime", {})
                    alert_cfg = rt_cfg.get("alert", {})
                    trade_cfg = rt_cfg.get("trading", {})
                    risk_cfg  = rt_cfg.get("risk", {})
                    feed_cfg  = rt_cfg.get("feed", {})
                    print("  当前实时预警配置:")
                    print(f"  扫描间隔:    {rt_cfg.get('scan_interval_seconds', 300)} 秒")
                    print(f"  监控范围:    {rt_cfg.get('universe', 'all')}")
                    print(f"  实时行情:    enabled={feed_cfg.get('enabled', True)}"
                          f" interval={feed_cfg.get('interval_seconds', 3)}s")
                    print(f"  启用策略:    {rt_cfg.get('active_strategies', [])}")
                    print(f"  合并规则:    {rt_cfg.get('signal_merge_rule', 'any')}")
                    print(f"  邮件预警:    {alert_cfg.get('enable_email', False)}")
                    print(f"  钉钉预警:    {alert_cfg.get('enable_dingtalk', False)}")
                    print(f"  Telegram:    {alert_cfg.get('enable_telegram', False)}")
                    print(f"  企业微信:    {alert_cfg.get('enable_wechat_work', False)}")
                    print(f"  初始资金:    {trade_cfg.get('initial_cash', 1000000)}")
                    print(f"  止损比例:    {risk_cfg.get('stop_loss_pct', 0.08)*100:.1f}%")
                    print(f"  止盈比例:    {risk_cfg.get('take_profit_pct', 0.20)*100:.1f}%")
                    print()
                    print("  提示: 编辑 config.json 中 'realtime' 节来修改配置")
                else:
                    print("  \u26a0\ufe0f  config.json 不存在")
            except Exception as e:
                print(f"  \u2717 读取配置失败: {e}")
            input("按 Enter 继续...")

        elif choice == "8":
            try:
                from src.realtime.alerter import Alerter
                alerter = Alerter()
                alerter.send_alert(
                    level="info",
                    subject="Q-UNITY-V7.6 测试预警",
                    body="这是一条测试预警消息，系统运行正常。V7.6 实时监控已就绪。",
                    dedup_key="test_alert_manual_v76"
                )
                print("  \u2713 测试预警已发送（请查看日志文件）")
                log_path = Path("logs/realtime_alerts.log")
                if log_path.exists():
                    lines = log_path.read_text(encoding="utf-8").splitlines()
                    print("  最近预警日志:")
                    for line in lines[-5:]:
                        print(f"    {line}")
            except Exception as e:
                print(f"  \u2717 发送测试预警失败: {e}")
            input("按 Enter 继续...")

        elif choice == "9":
            log_path = Path("logs/realtime_alerts.log")
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8").splitlines()
                print(f"  实时预警日志（最近30条，共 {len(lines)} 条）:")
                for line in lines[-30:]:
                    print(f"  {line}")
            else:
                print("  暂无实时预警日志")
            input("按 Enter 继续...")

        elif choice == "a":
            _strategy_select_menu(config_path="config.json")
            eng = _engine_ref[0]
            if eng is not None:
                try:
                    eng.reload_strategies()
                    print("  \u2713 策略配置已热更新")
                except Exception:
                    pass

        elif choice == "b":
            _strategy_params_menu(config_path="config.json")
            eng = _engine_ref[0]
            if eng is not None:
                try:
                    eng.reload_strategies()
                    print("  \u2713 策略参数已热更新")
                except Exception:
                    pass

        else:
            print("  \u2717 无效选项")


# ============================================================================
# 主菜单
# ============================================================================

def main_menu() -> None:
    config = storage = None
    try:
        from src.config import ConfigManager
        from src.data.storage import ColumnarStorageManager
        config   = ConfigManager()
        data_dir = config.get("data", {}).get("base_dir", "./data")
        storage  = ColumnarStorageManager(data_dir)
    except Exception as e:
        print(f"\u26a0\ufe0f  初始化失败: {e}")

    while True:
        print("=" * 56)
        print("       Q-UNITY-V7.8 量化交易系统 v7.8.0")
        print("       TDX多进程 + AKShare + 板块数据 + 完整回测 + 实时预警 [V7.8]")
        print("=" * 56)
        print("  1. 数据管理  (采集/扩展/增量/节点/板块)")
        print("  2. 回测系统  (单策略/多策略对比)")
        print("  3. 系统管理  (健康检查/日志/心跳)")
        print("  4. 实时交易  (多策略预警/实时行情/参数调优)")
        print("  0. 退出")
        print("-" * 56)
        choice = input("请选择 [0-4]: ").strip()
        if choice == "0":
            print("再见! Q-UNITY-V7.8 已退出。")
            sys.exit(0)
        elif choice == "1":
            data_management_menu(config, storage)
        elif choice == "2":
            backtest_menu(config)
        elif choice == "3":
            system_management_menu(config)
        elif choice == "4":
            realtime_menu(config)
        else:
            print("  \u2717 无效选项，请重新选择")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main_menu()