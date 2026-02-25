#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_op.py — Q-UNITY-V7.8 双轨采集引擎 CLI (patch_v7.8-final)  # A-10 Fix
==============================================================
使用方法:
    python main_op.py --nodes              # 节点赛马测试
    python main_op.py --sample 20          # 测试前 20 只
    python main_op.py --code 000001 0      # 单只股票
    python main_op.py                      # 全量采集（仅 TDX，快速）
    python main_op.py --full               # 强制全量重下载
    python main_op.py --retry-failed       # 补采失败股票
    python main_op.py --workers 8          # 指定 TDX 进程数
    python main_op.py --ak-workers 2       # 指定 AKShare 进程数
    python main_op.py --enable-akshare     # 启用 AKShare 双轨模式
    python main_op.py --enrich-akshare     # 仅补充 AKShare 扩展字段
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main_op")


def cmd_node_race() -> None:
    from src.data.collector.node_scanner import race_nodes, TDX_NODES
    print(f"\n{'='*62}")
    print(f"TDX 节点赛马测试 ({len(TDX_NODES)} 个候选节点)")
    print(f"{'='*62}")
    results = race_nodes(timeout=3.0)
    for i, r in enumerate(results, 1):
        icon    = "\u2713" if r["status"] == "ok" else "\u2717"
        lat_str = f"{r['latency_ms']:>7.2f} ms" if r["latency_ms"] >= 0 else "  timeout"
        print(f"  {i:>2}. {icon} {r['name']:<10} {r['host']:>18}:{r['port']}  {lat_str}")
    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        print(f"\n可达: {len(ok)}/{len(results)} | "
              f"最优: {ok[0]['name']} ({ok[0]['host']}) - {ok[0]['latency_ms']:.2f}ms")
    else:
        print("\n无可达节点")
    print()


def _make_pipeline(args) -> "StockDataPipeline":
    from src.data.collector.pipeline import StockDataPipeline
    return StockDataPipeline(
        parquet_dir=args.output,
        reports_dir=args.reports,
        top_n_nodes=5,
        tdx_workers=args.workers,
        ak_workers=args.ak_workers,
        force_full=args.full,
        enable_akshare=getattr(args, "enable_akshare", False),
        batch_sleep_every=getattr(args, "batch_sleep_every", 50),
        batch_sleep_min=getattr(args, "batch_sleep_min", 1.0),
        batch_sleep_max=getattr(args, "batch_sleep_max", 3.0),
    )


def cmd_collect(args) -> None:
    pipeline   = _make_pipeline(args)
    stock_list = pipeline._get_all_a_stock_list()
    if not stock_list:
        logger.error("获取股票列表失败，退出")
        sys.exit(1)
    logger.info("共 %d 只 A 股", len(stock_list))
    if args.sample > 0:
        stock_list = stock_list[:args.sample]
        logger.info("Sample 模式: 前 %d 只", len(stock_list))
    stats = pipeline.run(stock_list)
    _print_stats(stats)


def cmd_enrich_akshare(args) -> None:
    pipeline = _make_pipeline(args)
    stats    = pipeline.enrich_akshare()
    print(f"\nAKShare 扩展字段补充完成: 成功={stats.get('success',0)} 失败={stats.get('failed',0)}")


def cmd_retry_failed(args) -> None:
    pipeline = _make_pipeline(args)
    stats    = pipeline.retry_failed(reports_dir=args.reports)
    _print_stats(stats)


def cmd_single(args) -> None:
    import pandas as pd
    from src.data.collector.pipeline import StockDataPipeline, update_single_stock
    from src.data.collector.run_report import RunReport
    from src.data.collector.akshare_client import fetch_akshare_single
    from src.data.collector.incremental import compute_missing_range, read_local_max_date
    from datetime import date

    pipeline     = _make_pipeline(args)
    report       = RunReport(args.reports)
    code, mkt    = args.code[0], int(args.code[1])
    parquet_path = Path(args.output) / f"{code}.parquet"
    local_max    = read_local_max_date(parquet_path)
    start, end   = compute_missing_range(local_max, date.today().strftime("%Y-%m-%d"))
    ak_df        = fetch_akshare_single(code, start, end)
    ak_results   = {code: ak_df}

    _, ok, source = update_single_stock(
        code=code, market=mkt,
        parquet_dir=Path(args.output),
        tdx_pool=pipeline.tdx_pool,
        ak_results=ak_results,
        report=report,
        force_full=True,
    )
    report.save()

    if ok:
        df = pd.read_parquet(parquet_path)
        print(f"\n\u2713 {code} 采集成功 (来源: {source})")
        print(f"  行数: {len(df)} | 日期: {df['date'].min()} ~ {df['date'].max()}")
        ext = [c for c in ("turnover", "pct_change") if c in df.columns]
        if ext:
            print(f"  扩展字段: {ext}")
        print(df.tail(5).to_string(index=False))
    else:
        print(f"\n\u2717 {code} 采集失败（详见 {args.reports}/run_stats_*.json）")


def _print_stats(stats: dict) -> None:
    mode = "双轨(TDX+AKShare)" if stats.get("akshare_enabled") else "快速(仅TDX)"
    print(f"\n{'='*60}")
    print(f"采集完成统计  [{mode}]")
    print(f"{'='*60}")
    print(f"  总计:    {stats.get('total', 0):>6} 只")
    print(f"  成功:    {stats.get('success', 0):>6} 只")
    print(f"  失败:    {stats.get('failed', 0):>6} 只")
    print(f"  跳过:    {stats.get('skipped', 0):>6} 只（已最新）")
    print(f"  耗时:    {stats.get('elapsed_s', 0):>6.1f} 秒")
    print(f"  速度:    {stats.get('speed', 0):>6.1f} 股/秒")
    print(f"  报告目录: {stats.get('reports_dir', 'N/A')}")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Q-UNITY-V7.8 双轨采集引擎 (patch_v7.8-final)"  # A-10 Fix)
    parser.add_argument("--nodes",            action="store_true", help="节点赛马测试")
    parser.add_argument("--sample",           type=int, default=0, help="仅采集前 N 只（测试）")
    parser.add_argument("--full",             action="store_true", help="强制全量重下载")
    parser.add_argument("--retry-failed",     action="store_true", help="补采失败股票")
    parser.add_argument("--enable-akshare",   action="store_true", help="启用 AKShare 双轨采集")
    parser.add_argument("--enrich-akshare",   action="store_true", help="仅补充 AKShare 扩展字段")
    parser.add_argument("--workers",          type=int, default=8, help="TDX 进程数（默认 8）")
    parser.add_argument("--ak-workers",       type=int, default=2, help="AKShare 进程数（默认 2）")
    parser.add_argument("--batch-sleep-every",type=int, default=50, help="每 N 只 sleep 一次（默认 50）")
    parser.add_argument("--batch-sleep-min",  type=float, default=1.0, help="批次 sleep 最小秒数")
    parser.add_argument("--batch-sleep-max",  type=float, default=3.0, help="批次 sleep 最大秒数")
    parser.add_argument("--output",           type=str, default="./data/parquet", help="Parquet 目录")
    parser.add_argument("--reports",          type=str, default="./data/reports", help="报告目录")
    parser.add_argument("--code",             type=str, nargs=2, help="单股: --code 000001 0")
    args = parser.parse_args()

    if args.nodes:
        cmd_node_race()
    elif args.code:
        cmd_single(args)
    elif args.retry_failed:
        cmd_retry_failed(args)
    elif args.enrich_akshare:
        cmd_enrich_akshare(args)
    else:
        cmd_collect(args)


if __name__ == "__main__":
    main()