#!/usr/bin/env python3
"""
Q-UNITY-V6 采集器缓存管理器
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Optional


class CacheManager:
    """简单的本地文件缓存管理器（JSON格式，含TTL）"""

    def __init__(self, cache_dir: str = "./data/cache", ttl: int = 3600) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl

    def _path(self, key: str) -> Path:
        safe_key = key.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key: str) -> Optional[Any]:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - data.get("ts", 0) > self.ttl:
                path.unlink(missing_ok=True)
                return None
            return data.get("value")
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        try:
            path.write_text(
                json.dumps({"ts": time.time(), "value": value}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def clear(self) -> int:
        count = 0
        for p in self.cache_dir.glob("*.json"):
            p.unlink(missing_ok=True)
            count += 1
        return count