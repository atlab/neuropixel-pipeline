"""
Custom labview neuropixel aquisition format reader
"""

from __future__ import annotations

import h5py
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, constr
from typing import List
import numpy as np

from ..api.metadata import NeuropixelConfig

from .kilosort import Kilosort


class LabviewNeuropixelMeta(Kilosort, BaseModel, arbitrary_types_allowed=True):
    # probe serial number
    serial_number: constr(max_length=32) = Field(alias="SerialNum")

    # probe version
    version: float = Field(alias="Version")

    #
    sampling_rate: float = Field(alias="Fs")

    #
    channel_names: List[str] = Field(alias="channelNames")

    #
    class_name: str = Field(alias="class")

    #
    scale: np.ndarray

    #
    t0: float

    #
    config_params: List[str] = Field(alias="ConfigParams")

    @field_validator("config_params", "channel_names", mode="before")
    def convert_to_list(cls, v):
        if isinstance(v, bytes):
            return v.decode().strip().split(",")
        elif isinstance(v, str):
            return v.strip().split(",")
        else:
            return v

    @classmethod
    def from_h5(
        cls, directory: Path, family: str = "NPElectrophysiology%d.h5"
    ) -> LabviewNeuropixelMeta:
        """
        Uses an h5 family driver if necessary
        """
        with h5py.File(directory / family, driver="family", memb_size=0) as f:
            meta = dict(f.attrs)

        return cls(**meta)

    @classmethod
    def from_metafile(cls) -> LabviewNeuropixelMeta:
        """
        This will be implemented when the metadata from labview is separated from the h5.
        """
        raise NotImplementedError(
            "This will be implemented when the metadata from labview is separated from the h5"
        )

    def channels(self) -> List[int]:
        return list(int(channel_name[-4:]) for channel_name in self.channel_names)

    def to_metadata(self) -> NeuropixelConfig:
        pass
