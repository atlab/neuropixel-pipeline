from __future__ import annotations

from pydantic import BaseModel, validate_call
from typing import Dict
from enum import Enum
from pathlib import Path
import numpy as np
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


@validate_call
def check_for_first_bin_with_prefix(session_dir: Path, prefix: str):
    for path in session_dir.glob("*bin"):
        if prefix in path.stem:
            return path
    else:
        raise IOError(f"No bin with {prefix} in the prefix in directory: {session_dir}")


@validate_call
def extract_data_from_bin(
    bin_file: Path, num_channels: int, has_sync_channel: bool = False
) -> np.ndarray:
    bin_file = Path(bin_file)
    raw_data = np.memmap(bin_file, dtype="int16", mode="r")
    if not has_sync_channel:
        data = np.reshape(raw_data, (int(raw_data.size / num_channels), num_channels))
    else:
        total_channels = num_channels + 1
        data = np.reshape(
            raw_data, (int(raw_data.size / total_channels), total_channels)
        )
        data = data[:, 0:num_channels]
    return data
