#!/usr/bin/env python3
"""
Q-UNITY-V6 配置管理模块
"""
from __future__ import annotations
import json
import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器：支持 JSON 配置文件和默认值"""

    _DEFAULT = {
        "data": {
            "base_dir": "./data",
            "parquet_dir": "./data/parquet",
            "cache_dir": "./data/cache",
            "industry_dir": "./data/industry",
        },
        "backtest": {
            "initial_cash": 1_000_000.0,
            "commission_rate": 0.0003,
            "slippage_rate": 0.001,
            "tax_rate": 0.001,
            "position_limit": 20,
            "max_position_pct": 0.2,
        },
        "risk": {
            "max_drawdown": 0.2,
            "max_position_pct": 0.1,
            "industry_limit": 0.3,
            "stop_loss_pct": 0.1,
            "take_profit_pct": 0.2,
            "trailing_stop_pct": 0.05,
            "circuit_breaker_cooldown_days": 5,
        },
        "factors": {
            "rsrs": {"regression_window": 18, "zscore_window": 600, "enable": True},
            "alpha": {"momentum_window": 20, "volatility_window": 20, "enable": True},
        },
        "strategy": {"rebalance_freq": "daily", "top_n": 20, "min_score": 0.0},
        "logging": {"level": "INFO", "file": "./logs/q-unity.log"},
    }

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config: Dict[str, Any] = copy.deepcopy(self._DEFAULT)  # A-05 Fix: 深拷贝防止 _DEFAULT 被污染
        if config_path:
            self._load(config_path)
        else:
            default_path = Path("config.json")
            if default_path.exists():
                self._load(str(default_path))

    def _load(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            self._deep_merge(self.config, loaded)
            logger.info(f"配置已加载: {path}")
        except FileNotFoundError:
            logger.warning(f"配置文件不存在: {path}，使用默认值")
        except json.JSONDecodeError as e:
            logger.error(f"配置文件解析失败: {e}，使用默认值")

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> None:
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                ConfigManager._deep_merge(base[k], v)
            else:
                base[k] = v

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)