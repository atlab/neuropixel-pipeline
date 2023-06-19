from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from typing import Any
from pathlib import Path


class EphysParams(BaseModel):
    sample_rate: float = Field(
        default=30000.0,
        description="Sample rate of Neuropixels AP band continuous data",
    )
    lfp_sample_rate: float = Field(
        default=2500.0,
        description="Sample rate of Neuropixels LFP band continuous data",
    )
    bit_volts: float = Field(
        default=0.195,
        description="Scalar required to convert int16 values into microvolts",
    )
    num_channels: int = Field(
        default=384, description="Total number of channels in binary data files"
    )
    reference_channels: Any = Field(
        default=[36, 75, 112, 151, 188, 227, 264, 303, 340, 379],
        description="Reference channels on Neuropixels probe (numbering starts at 0)",
    )
    template_zero_padding: int = Field(
        default=21, description="Zero-padding on templates output by Kilosort"
    )
    vertical_site_spacing: float = Field(
        default=20e-6, description="Vertical site spacing in meters"
    )
    probe_type: str = Field(default="NP1", description="3A, 3B2, NP1")
    # lfp_band_file: str = Field(description="Location of LFP band binary file")
    # ap_band_file: str = Field(description="Location of AP band binary file")
    reorder_lfp_channels: bool = Field(
        default=True,
        description="Should we fix the ordering of LFP channels (necessary for 3a probes following extract_from_npx modules)",
    )
    cluster_group_file_name: str = Field(default="cluster_group.tsv")

    @model_validator(mode="after")
    def coerce_numpy(cls, values):
        import numpy as np

        values.reference_channels = np.array(values.reference_channels)
        return values


class Directories(BaseModel):
    ecephys_directory: Path = Field(
        description="Location of the ecephys_spike_sorting directory containing modules directory"
    )
    npx_directory: Path = Field(description="Location of raw neuropixels binary files")
    kilosort_output_directory: Path = Field(
        description="Location of Kilosort output files"
    )
    extracted_data_directory: Path = Field(
        description="Location for NPX/CatGT processed files"
    )
    kilosort_output_tmp: Path = Field(description="Location for temporary KS output")


# class CommonFiles(BaseModel):
#     probe_json: Path = Field(description="Location of probe JSON file")
#     settings_json: Path = Field(
#         description="Location of settings JSON written by extract_from_npx module"
#     )


class WaveformMetricsFile(BaseModel):
    waveform_metrics_file: Path = Field(description="Location of waveform metrics CSV")


class ClusterMetricsFile(BaseModel):
    cluster_metrics_file: Path = Field(description="Location of cluster metrics CSV")
