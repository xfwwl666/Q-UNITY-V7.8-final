#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""storage.py — Parquet 存取层"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ParquetStorage:
    """Parquet 存储管理器。"""

    def __init__(self, base_dir: str = "./data/parquet") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, code: str) -> Path:
        return self.base_dir / f"{code}.parquet"

    def exists(self, code: str) -> bool:
        return self.get_path(code).exists()

    def load(self, code: str) -> Optional[pd.DataFrame]:
        path = self.get_path(code)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning("读取失败 %s: %s", code, exc)
            return None

    def save(self, code: str, df: pd.DataFrame, compression: str = "zstd") -> bool:
        path = self.get_path(code)
        try:
            df.to_parquet(path, index=False, compression=compression)
            return True
        except Exception as exc:
            logger.error("保存失败 %s: %s", code, exc)
            return False

    def list_codes(self) -> List[str]:
        return [f.stem for f in self.base_dir.glob("*.parquet")]

    def get_max_date(self, code: str) -> Optional[str]:
        from .collector.incremental import read_local_max_date
        return read_local_max_date(self.get_path(code))


class ColumnarStorageManager:
    """ColumnarStorage 兼容包装（供 menu_main.py 使用）。"""

    def __init__(self, data_dir: str = "./data") -> None:
        self._parquet = ParquetStorage(str(Path(data_dir) / "parquet"))

    def get_all_codes(self) -> List[str]:
        return self._parquet.list_codes()

    def load(self, code: str) -> Optional[pd.DataFrame]:
        return self._parquet.load(code)

    def save(self, code: str, df: pd.DataFrame) -> bool:
        return self._parquet.save(code, df)