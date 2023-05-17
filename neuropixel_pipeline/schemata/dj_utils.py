from __future__ import annotations

from pydantic import BaseModel
from typing import Dict
from enum import Enum
from pathlib import Path
import json


class DatajointStoreProtocol(str, Enum):
    FILE = "file"
    BLOB = "blob"


class StoresConfig(BaseModel):
    stores: Dict[str, Store]

    class Store(BaseModel):
        protocol: DatajointStoreProtocol = "file"
        location: Path
        stage: Path

    def from_tuples() -> Self:  # noqa: F821
        pass

    def export(self) -> Dict[str, Dict[str, str]]:
        return json.loads(self.model_dump_json())

    def set_dj_config(self):
        import datajoint as dj

        stores_config = self._export()
        if "stores" not in dj.config:
            dj.config["stores"] = stores_config
        else:
            dj.config["stores"].update(stores_config)
