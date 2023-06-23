from __future__ import annotations

from pydantic import BaseModel
from typing import Dict
from enum import Enum
from pathlib import Path
import json


class TODO:
    def __init__(self, msg: str = None):
        error_msg = "This is a placeholder value"
        if msg is not None:
            error_msg = f'{error_msg}: "{msg}"'
        raise NotImplementedError(error_msg)


class DatajointStoreProtocol(str, Enum):
    FILE = "file"
    BLOB = "blob"


class StoresConfig(BaseModel):
    stores: Dict[str, Store]

    class Store(BaseModel):
        protocol: DatajointStoreProtocol = "file"
        location: Path
        stage: Path

    def from_tuples():
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


# from datajoint element-interface utils
def dict_to_uuid(key: dict):
    """Given a dictionary `key`, returns a hash string as UUID

    Args:
        key (dict): Any python dictionary"""
    import hashlib
    import uuid

    hashed = hashlib.md5()
    for k, v in sorted(key.items()):
        hashed.update(str(k).encode())
        hashed.update(str(v).encode())
    return uuid.UUID(hex=hashed.hexdigest())

def check_for_first_bin_with_prefix(session_dir: Path, prefix: str):
    for path in session_dir.glob("*bin"):
        if prefix in path.stem:
            return path
    else:
        raise IOError(
            f"No bin with {prefix} in the prefix in directory"
        )