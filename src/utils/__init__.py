#!/usr/bin/env python3
"""Q-UNITY src/utils 工具包"""
# V7.7 新增: HTML 回测报告生成
try:
    from .report import generate_html_report
    __all__ = ["generate_html_report"]
except ImportError:
    # 若 matplotlib 未安装，report 模块仍可导入，图表生成会降级处理
    __all__ = []