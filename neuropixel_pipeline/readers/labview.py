"""
Custom labview neuropixel aquisition format reader
"""

from __future__ import annotations

import h5py
from pathlib import Path
from pydantic import BaseModel, Field, validator, constr
from typing import List
import numpy as np

from .kilosort import Kilosort


class LabviewNeuropixelMetadata(Kilosort, BaseModel, arbitrary_types_allowed=True):
    # probe serial number
    serial_number: constr(max_length=32) = Field(alias="SerialNum")

    # probe version
    version: float = Field(alias="Version")

    #
    frequency: float = Field(alias="Fs")
    scale: np.ndarray
    t0: float
    config_params: List[str] = Field(alias="ConfigParams")

    @validator("config_params", pre=True)
    def convert_to_list(cls, v):
        if isinstance(v, bytes):
            return v.split(b",")
        elif isinstance(v, str):
            return v.split(",")
        else:
            return v

    @classmethod
    def from_h5(
        cls, directory: Path, family: str = "NPElectrophysiology%d.h5"
    ) -> LabviewNeuropixelMetadata:
        """
        Uses an h5 family driver if necessary
        """
        with h5py.File(directory / family, driver="family", memb_size=0) as f:
            meta = dict(f.attrs)

        return cls(**meta)

    @classmethod
    def from_metafile(cls) -> LabviewNeuropixelMetadata:
        """
        This will be implemented when the metadata from labview is separated from the h5.
        """
        raise NotImplementedError(
            "This will be implemented when the metadata from labview is separated from the h5"
        )
