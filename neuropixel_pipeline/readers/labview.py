"""
Custom labview neuropixel aquisition format reader
"""

from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field, field_validator, constr, computed_field
from typing import List, Tuple, Any, Optional, Dict
import numpy as np

from ..api.metadata import NeuropixelConfig
from .. import utils


class LabviewNeuropixelMeta(BaseModel, arbitrary_types_allowed=True):
    # probe serial number
    serial_number: constr(max_length=32) = Field(alias="SerialNum")

    # probe version
    version: float = Field(alias="Version")

    # sampling_rate as Fs
    sampling_rate: float = Field(alias="Fs")

    #
    channel_names: List[str] = Field(alias="channelNames")

    #
    class_name: str = Field(alias="class")

    #
    scale: Tuple[float, float]

    #
    t0: float = Field(alias="t0")

    #
    config_params: List[str] = Field(alias="ConfigParams")

    #
    config_data: Optional[Any] = Field(alias="Config")

    @field_validator("config_params", "channel_names", mode="before")
    def convert_to_list(cls, v):
        if isinstance(v, bytes):
            return v.decode().strip().split(",")
        elif isinstance(v, str):
            return v.strip().split(",")
        else:
            return v

    @field_validator("scale", mode="before")
    def check_scale_shape(cls, v):
        a, b = v
        return (a, b)

    @staticmethod
    def _validate_probe_naming_convention(
        meta: dict, original_key_name: str, normalized_key_name: str
    ):
        if normalized_key_name in original_key_name:
            meta[normalized_key_name] = meta.pop(original_key_name)

    @classmethod
    def from_h5(
        cls, directory: Path, family: str = "NPElectrophysiology%d.h5"
    ) -> LabviewNeuropixelMeta:
        """
        Uses an h5 family driver if necessary
        """
        import h5py

        directory = Path(directory)
        with h5py.File(directory / family, driver="family", memb_size=0) as f:
            meta = dict(f.attrs)

            # need eager keys evaluation here, therefore list is used
            for key in list(meta.keys()):
                cls._validate_probe_naming_convention(meta, key, "SerialNum")
                cls._validate_probe_naming_convention(meta, key, "t0")

            for key in f.keys():
                if "Config" in key:
                    meta["Config"] = np.array(f[key])

        return cls.model_validate(meta)

    @classmethod
    def from_metafile(cls) -> LabviewNeuropixelMeta:
        """
        This will be implemented when the metadata from labview is separated from the h5.
        """
        raise NotImplementedError(
            "This will be implemented when the metadata from labview is separated from the h5"
        )

    def channels(self) -> List[int]:
        # can use self.config_data instead, with config_params's channel and port
        return list(int(channel_name[-4:]) for channel_name in self.channel_names)

    def electrode_config(self) -> Dict[str, Any]:
        return dict(zip(self.config_params, self.config_data.T))

    def electrode_config_hash(self) -> str:
        return utils.dict_to_uuid(self.model_dump())

    def to_metadata(self) -> NeuropixelConfig:
        raise NotImplementedError(
            "This isn't implemented but is needed for neuropixel config generation"
        )
