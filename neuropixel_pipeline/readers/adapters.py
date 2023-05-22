"""Pydantic adapters for working with file-loaded datajoint store data.

Connected to the stores configs in schemata (only ephys currently).

Intended for use with the filepath datajoint protocol (as opposed to blob protocol).
"""

from __future__ import annotations

from pydantic import BaseModel
from typing import Dict
from pathlib import Path
from abc import ABC, abstractmethod

from neuropixel_pipeline.utils import TODO, StoresConfig

# TODO: Need a utility class for file loading, because this will be common.

# TODO: Adapters will read the data from a filepath (using utility class) and
#       convert to a similar but slightly more useful/descriptive pydantic model.


class FilepathAdapter(BaseModel, ABC):
    dirpath: Path

    @abstractmethod
    def read_from_file(self, filepath):
        """some utility to help reading from files less boilerplatey"""
    
    @abstractmethod
    def load(self, filepath: Path):
        pass

class LFPTimeStamps(FilepathAdapter):
    def load(self, filepath: Path):
        self.read_from_file(filepath)
        TODO()

class Adapters(BaseModel):
    _stores: StoresConfig

    loaders: Dict[str, FilepathAdapter]

    # move these to just instantiation for loaders
    lfp_time_stamps: FilepathAdapter
    lfp_mean: FilepathAdapter
    lfp: FilepathAdapter
    params: FilepathAdapter
    spike_times: FilepathAdapter
    spike_sites: FilepathAdapter
    spike_depths: FilepathAdapter
    peak_electrode_waveform: FilepathAdapter
    waveform_mean: FilepathAdapter
    waveforms: FilepathAdapter

    def export_stores(self) -> Dict[str, Dict[str, str]]:
        self._stores.export()

    def set_dj_config(self):
        self._stores.set_dj_config()
