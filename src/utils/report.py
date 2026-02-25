#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/utils/report.py — Q-UNITY-V7.8 HTML 回测报告生成器
=======================================================
从回测结果 JSON 文件生成带图表和绩效表格的独立 HTML 报告。

主要函数:
    generate_html_report(json_path, html_path)
        — 从 JSON 生成 HTML（包含权益曲线图 + 绩效指标表格）

依赖:
    matplotlib >= 3.5.0  (pip install matplotlib)

使用示例:
    from src.utils.report import generate_html_report
    generate_html_report("results/my_backtest.json", "results/my_report.html")
"""
from __future__ import annotations

import base64
import json
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# 内部工具函数
# ============================================================================

def _load_json(json_path: str) -> dict:
    """加载并解析回测结果 JSON 文件，返回数据字典。异常时抛出 ValueError。"""
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {json_path}")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败 [{json_path}]: {e}") from e


def _equity_curve_to_base64(equity_curve: List[Dict]) -> Optional[str]:
    """
    将权益曲线数据绘制为折线图，返回 base64 编码的 PNG 字符串。
    若 matplotlib 未安装或数据为空，返回 None。

    参数:
        equity_curve — [{"timestamp": "YYYY-MM-DD", "total_value": float, ...}, ...]
    """
    if not equity_curve:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")   # 非交互后端，避免需要显示器
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib 未安装，跳过图表生成。请执行: pip install matplotlib")
        return None

    try:
        # 提取数据
        timestamps  = [row.get("timestamp", "") for row in equity_curve]
        total_vals  = [float(row.get("total_value",  0)) for row in equity_curve]
        cash_vals   = [float(row.get("cash",          0)) for row in equity_curve]

        # 解析日期
        dates = []
        for ts in timestamps:
            try:
                dates.append(datetime.strptime(ts[:10], "%Y-%m-%d"))
            except ValueError:
                dates.append(datetime.now())

        # 绘图
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(dates, total_vals, label="总资产", color="#1a73e8", linewidth=1.8)
        ax.plot(dates, cash_vals,  label="现金",   color="#34a853", linewidth=1.2,
                linestyle="--", alpha=0.7)

        # 初始资金参考线
        if total_vals:
            init_val = total_vals[0]
            ax.axhline(y=init_val, color="#ea4335", linewidth=0.8, linestyle=":",
                       alpha=0.8, label=f"初始资金 {init_val:,.0f}")

        # 最大值/最小值标注
        if len(total_vals) > 1:
            max_idx = total_vals.index(max(total_vals))
            min_idx = total_vals.index(min(total_vals))
            ax.annotate(f"峰值\n{total_vals[max_idx]:,.0f}",
                        xy=(dates[max_idx], total_vals[max_idx]),
                        fontsize=8, color="#1a73e8",
                        xytext=(10, 10), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color="#1a73e8", lw=0.8))
            ax.annotate(f"谷值\n{total_vals[min_idx]:,.0f}",
                        xy=(dates[min_idx], total_vals[min_idx]),
                        fontsize=8, color="#ea4335",
                        xytext=(10, -20), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color="#ea4335", lw=0.8))

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(dates) // 100)))
        fig.autofmt_xdate(rotation=30)
        ax.set_title("权益曲线", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("日期", fontsize=10)
        ax.set_ylabel("资产总值 (元)", fontsize=10)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e4:.1f}万")
            if max(total_vals) > 1e5 else matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
        )
        plt.tight_layout()

        # 转为 base64
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return encoded

    except Exception as e:
        logger.warning("权益曲线绘制失败: %s", e)
        return None


def _drawdown_chart_to_base64(equity_curve: List[Dict]) -> Optional[str]:
    """绘制回撤曲线图，返回 base64 PNG。"""
    if not equity_curve:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import numpy as np
    except ImportError:
        return None

    try:
        timestamps = [row.get("timestamp", "") for row in equity_curve]
        total_vals = [float(row.get("total_value", 0)) for row in equity_curve]

        dates = []
        for ts in timestamps:
            try:
                dates.append(datetime.strptime(ts[:10], "%Y-%m-%d"))
            except ValueError:
                dates.append(datetime.now())

        arr  = np.array(total_vals, dtype=float)
        peak = np.maximum.accumulate(arr)
        dd   = np.where(peak > 0, (peak - arr) / peak, 0.0) * 100  # 百分比

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.fill_between(dates, -dd, 0, alpha=0.4, color="#ea4335", label="回撤")
        ax.plot(dates, -dd, color="#ea4335", linewidth=1.0)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(dates) // 100)))
        fig.autofmt_xdate(rotation=30)
        ax.set_title("回撤曲线", fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel("回撤 (%)", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.1f}%")
        )
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        logger.warning("回撤图绘制失败: %s", e)
        return None


def _fmt_pct(v: Any) -> str:
    """格式化为百分比字符串，N/A 若无效。"""
    try:
        return f"{float(v) * 100:+.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_float(v: Any, decimals: int = 3) -> str:
    """格式化为浮点字符串，N/A 若无效。"""
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def _color_value(v: Any, is_pct: bool = True, reverse: bool = False) -> str:
    """
    根据数值正负返回带颜色的 HTML span。
    reverse=True 时，负值为绿色（如最大回撤越小越好）。
    """
    try:
        fv = float(v)
        text = _fmt_pct(v) if is_pct else _fmt_float(v)
        if reverse:
            color = "#34a853" if fv <= 0 else "#ea4335"
        else:
            color = "#34a853" if fv > 0 else ("#ea4335" if fv < 0 else "#555")
        return f'<span style="color:{color};font-weight:600">{text}</span>'
    except (TypeError, ValueError):
        return "N/A"


# ============================================================================
# HTML 模板构建
# ============================================================================

def _build_html(data: dict, equity_img_b64: Optional[str],
                dd_img_b64: Optional[str]) -> str:
    """
    根据回测数据构建完整 HTML 字符串。
    不依赖外部模板引擎，所有 HTML 直接在 Python 字符串中生成。
    """
    perf         = data.get("performance", {})
    strategy     = data.get("strategy_name", "未知策略")
    start_date   = data.get("start_date", "")
    end_date     = data.get("end_date", "")
    codes_count  = data.get("codes_count", 0)
    params       = data.get("strategy_params", {})
    generated_at = data.get("generated_at", datetime.now().isoformat())
    equity_curve = data.get("equity_curve", [])

    # ── 图表 HTML ──────────────────────────────────────────────────────────
    equity_html = (
        f'<img src="data:image/png;base64,{equity_img_b64}" '
        f'style="width:100%;max-width:900px;border-radius:8px;" alt="权益曲线">'
        if equity_img_b64 else
        '<p style="color:#888;text-align:center;padding:30px">权益曲线图生成失败（请安装 matplotlib）</p>'
    )
    dd_html = (
        f'<img src="data:image/png;base64,{dd_img_b64}" '
        f'style="width:100%;max-width:900px;border-radius:8px;" alt="回撤曲线">'
        if dd_img_b64 else ""
    )

    # ── 绩效指标行 ─────────────────────────────────────────────────────────
    perf_rows = [
        ("总收益率",   _color_value(perf.get("total_return"))),
        ("年化收益率", _color_value(perf.get("annual_return"))),
        ("夏普比率",   _fmt_float(perf.get("sharpe_ratio"))),
        ("最大回撤",   _color_value(perf.get("max_drawdown"), reverse=True)),
        ("胜率",       _fmt_pct(perf.get("win_rate"))),
        ("总交易次数", str(int(perf.get("total_trades", 0) or 0))),
        ("盈亏比",     _fmt_float(perf.get("profit_loss_ratio"))),
        ("卡玛比率",   _fmt_float(perf.get("calmar_ratio"))),
        ("Sortino",   _fmt_float(perf.get("sortino_ratio", perf.get("sortino")))),
    ]

    perf_table_rows = "\n".join(
        f"<tr><td>{name}</td><td>{val}</td></tr>"
        for name, val in perf_rows
    )

    # ── 策略参数表格 ────────────────────────────────────────────────────────
    if params:
        param_rows = "\n".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in params.items()
        )
        params_html = f"""
        <div class="section">
          <h2>策略参数</h2>
          <table>
            <thead><tr><th>参数名</th><th>值</th></tr></thead>
            <tbody>{param_rows}</tbody>
          </table>
        </div>"""
    else:
        params_html = ""

    # ── 权益曲线数据表格（最近20条）─────────────────────────────────────────
    if equity_curve:
        sample = equity_curve[-20:]  # 最近20行
        ec_rows = "\n".join(
            f'<tr><td>{r.get("timestamp","")}</td>'
            f'<td>{r.get("total_value",0):,.0f}</td>'
            f'<td>{r.get("cash",0):,.0f}</td>'
            f'<td>{r.get("market_value",0):,.0f}</td></tr>'
            for r in sample
        )
        ec_html = f"""
        <div class="section">
          <h2>权益曲线（最近 {len(sample)} 条记录）</h2>
          <table>
            <thead>
              <tr><th>日期</th><th>总资产</th><th>现金</th><th>持仓市值</th></tr>
            </thead>
            <tbody>{ec_rows}</tbody>
          </table>
        </div>"""
    else:
        ec_html = ""

    # ── 组装完整 HTML ──────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Q-UNITY 回测报告 — {strategy}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
      background: #f0f2f5;
      color: #333;
      padding: 24px;
    }}
    .container {{
      max-width: 960px;
      margin: 0 auto;
    }}
    .header {{
      background: linear-gradient(135deg, #1a73e8, #0d47a1);
      color: #fff;
      border-radius: 12px;
      padding: 28px 32px;
      margin-bottom: 24px;
      box-shadow: 0 4px 16px rgba(26,115,232,.25);
    }}
    .header h1 {{ font-size: 22px; font-weight: 700; margin-bottom: 6px; }}
    .header .meta {{ font-size: 13px; opacity: .85; }}
    .section {{
      background: #fff;
      border-radius: 10px;
      padding: 22px 26px;
      margin-bottom: 20px;
      box-shadow: 0 2px 8px rgba(0,0,0,.06);
    }}
    .section h2 {{
      font-size: 16px;
      font-weight: 600;
      color: #1a73e8;
      margin-bottom: 14px;
      padding-bottom: 8px;
      border-bottom: 2px solid #e8f0fe;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid #f0f0f0;
    }}
    th {{
      background: #f8f9fa;
      font-weight: 600;
      color: #555;
    }}
    tr:hover td {{ background: #fafbff; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 14px;
      margin-bottom: 4px;
    }}
    .kpi-card {{
      background: #f8f9fa;
      border-radius: 8px;
      padding: 14px 16px;
      text-align: center;
      border-left: 4px solid #1a73e8;
    }}
    .kpi-card .kpi-label {{
      font-size: 11px;
      color: #888;
      text-transform: uppercase;
      letter-spacing: .5px;
      margin-bottom: 6px;
    }}
    .kpi-card .kpi-value {{
      font-size: 20px;
      font-weight: 700;
    }}
    .chart-wrap {{ text-align: center; padding: 8px 0; }}
    .footer {{
      text-align: center;
      font-size: 12px;
      color: #aaa;
      margin-top: 24px;
      padding-top: 16px;
      border-top: 1px solid #eee;
    }}
    @media (max-width: 600px) {{
      body {{ padding: 12px; }}
      .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
  </style>
</head>
<body>
<div class="container">

  <!-- 报告头 -->
  <div class="header">
    <h1>📊 Q-UNITY 回测报告</h1>
    <div class="meta">
      策略: <strong>{strategy}</strong> &nbsp;|&nbsp;
      区间: {start_date} ~ {end_date} &nbsp;|&nbsp;
      股票池: {codes_count} 只 &nbsp;|&nbsp;
      生成: {generated_at[:19]}
    </div>
  </div>

  <!-- KPI 卡片 -->
  <div class="section">
    <h2>核心绩效指标</h2>
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">总收益率</div>
        <div class="kpi-value">{_color_value(perf.get("total_return"))}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">年化收益率</div>
        <div class="kpi-value">{_color_value(perf.get("annual_return"))}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">夏普比率</div>
        <div class="kpi-value">{_fmt_float(perf.get("sharpe_ratio"))}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">最大回撤</div>
        <div class="kpi-value">{_color_value(perf.get("max_drawdown"), reverse=True)}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">胜率</div>
        <div class="kpi-value">{_fmt_pct(perf.get("win_rate"))}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">总交易次数</div>
        <div class="kpi-value">{int(perf.get("total_trades", 0) or 0)}</div>
      </div>
    </div>
  </div>

  <!-- 详细绩效表格 -->
  <div class="section">
    <h2>详细绩效指标</h2>
    <table>
      <thead><tr><th>指标</th><th>值</th></tr></thead>
      <tbody>{perf_table_rows}</tbody>
    </table>
  </div>

  {params_html}

  <!-- 权益曲线图 -->
  <div class="section">
    <h2>权益曲线</h2>
    <div class="chart-wrap">{equity_html}</div>
  </div>

  <!-- 回撤曲线图 -->
  {'<div class="section"><h2>回撤曲线</h2><div class="chart-wrap">' + dd_html + '</div></div>' if dd_html else ''}

  {ec_html}

  <div class="footer">
    Q-UNITY V7.8 · 本报告由 src/utils/report.py 自动生成 · 仅供研究参考，不构成投资建议
  </div>

</div>
</body>
</html>"""
    return html


# ============================================================================
# 公开接口
# ============================================================================

def generate_html_report(json_path: str, html_path: str) -> None:
    """
    从回测结果 JSON 文件生成 HTML 报告。

    参数:
        json_path  — 输入 JSON 文件路径（由 _save_backtest_result 生成）
        html_path  — 输出 HTML 文件路径（若父目录不存在将自动创建）

    异常处理:
        FileNotFoundError — JSON 文件不存在
        ValueError        — JSON 解析失败
        其他异常          — 记录日志并向上抛出

    示例:
        generate_html_report(
            "results/rsrs_momentum_20240101_20241231.json",
            "results/rsrs_momentum_report.html",
        )
    """
    logger.info("生成 HTML 报告: %s -> %s", json_path, html_path)

    # 1. 加载数据
    data = _load_json(json_path)

    # 2. 生成图表（可能返回 None，若 matplotlib 未安装）
    equity_curve = data.get("equity_curve", [])
    equity_img   = _equity_curve_to_base64(equity_curve)
    dd_img       = _drawdown_chart_to_base64(equity_curve)

    # 3. 构建 HTML
    html_content = _build_html(data, equity_img, dd_img)

    # 4. 写出文件
    out_path = Path(html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")

    logger.info("HTML 报告已生成: %s (%.1f KB)", html_path, out_path.stat().st_size / 1024)
    print(f"  ✓ HTML 报告已生成: {html_path}  ({out_path.stat().st_size / 1024:.1f} KB)")