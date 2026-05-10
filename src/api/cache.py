from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, read_json, write_json
from src.utils.json_utils import stable_hash


class JsonCache:
    def __init__(self, cache_dir: str | Path = ".cache/openai", enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        ensure_dir(self.cache_dir)

    def key_for(self, payload: Any) -> str:
        return stable_hash(payload)

    def path_for_key(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, payload: Any) -> Any | None:
        if not self.enabled:
            return None
        path = self.path_for_key(self.key_for(payload))
        if path.exists():
            return read_json(path)
        return None

    def set(self, payload: Any, value: Any) -> None:
        if not self.enabled:
            return
        write_json(value, self.path_for_key(self.key_for(payload)))
