#!/usr/bin/env python3
"""
Q-UNITY-V6 行业数据提供者
"""
from __future__ import annotations
import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


def fetch_and_save_industry_data(storage) -> bool:
    """从 AKShare 获取申万行业分类并保存"""
    try:
        import akshare as ak
        df = ak.stock_board_industry_name_em()
        if df is None or df.empty:
            logger.error("行业数据为空")
            return False
        storage.save_industry_data(df)
        logger.info(f"行业数据已保存: {len(df)} 条")
        return True
    except ImportError:
        logger.error("akshare 未安装")
        return False
    except Exception as e:
        logger.error(f"行业数据获取失败: {e}")
        return False


def load_industry_map(storage) -> Optional[pd.DataFrame]:
    """加载行业映射表"""
    return storage.load_industry_data()