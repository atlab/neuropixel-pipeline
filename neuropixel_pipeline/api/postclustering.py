from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, validate_call
from pydantic.dataclasses import dataclass
from typing import Any
import numpy as np
import os

# i.e. Waveforms and QualityMetrics
# runs ecephys_spike_sorting to produce the waveform analysis and quality metrics files


@validate_call
def extract_data_from_bin(
    bin_file: Path, num_channels: int, has_sync_channel=False
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


# https://github.com/jenniferColonell/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/mean_waveforms/_schemas.py
class WaveformSetRunner(BaseModel):
    sample_rate: float = 30000.0
    num_channels: int = 384
    bit_volts: float
    params: dict = {
        "samples_per_spike": 82,
        "pre_samples": 20,
        "num_epochs": 4,
        "spikes_per_epoch": 5,
    }

    @dataclass
    class Output:
        data: Any
        spike_counts: Any
        coords: Any
        labesl: Any

    def calculate(
        self, bin_file: Path, clustering_output_dir: Path, has_sync_channel=False
    ):
        from ecephys_spike_sorting.modules.mean_waveforms.extract_waveforms import (
            extract_waveforms,
        )
        import ecephys_spike_sorting.common.utils as utils

        bin_file = Path(bin_file)
        clustering_output_dir = Path(clustering_output_dir)
        data = extract_data_from_bin(
            bin_file=bin_file,
            num_channels=self.num_channels,
            has_sync_channel=has_sync_channel,
        )

        (
            spike_times,
            spike_clusters,
            amplitudes,
            templates,
            channel_map,
            cluster_ids,
            cluster_quality,
        ) = utils.load_kilosort_data(
            clustering_output_dir, self.sample_rate, convert_to_seconds=False
        )

        return WaveformSetRunner.Output(
            *extract_waveforms(
                data,
                spike_times,
                spike_clusters,
                cluster_ids,
                cluster_quality,
                self.bit_volts,
                self.sample_rate,
                self.params,
            )
        )


# https://github.com/jenniferColonell/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/_schemas.py
class QualityMetricsRunner(BaseModel):
    sample_rate: float = 30000.0
    num_channels: int = 384
    params: dict = {
        "samples_per_spike": 82,
        "pre_samples": 20,
        "snr_spike_count": 500,
        "isi_threshold": 0.015,
    }

    @dataclass
    class Output:
        metrics: Any

    def calculate(
        self, bin_file: Path, clustering_output_dir: Path, has_sync_channel=False
    ):
        from ecephys_spike_sorting.modules.quality_metrics.metrics import (
            calculate_metrics,
        )
        import ecephys_spike_sorting.common.utils as utils

        bin_file = Path(bin_file)
        clustering_output_dir = Path(clustering_output_dir)
        data = extract_data_from_bin(
            bin_file=bin_file,
            num_channels=self.num_channels,
            has_sync_channel=has_sync_channel,
        )

        (
            spike_times,
            spike_clusters,
            amplitudes,
            templates,
            channel_map,
            cluster_ids,
            cluster_quality,
        ) = utils.load_kilosort_data(
            clustering_output_dir, self.sample_rate, convert_to_seconds=False
        )

        return QualityMetricsRunner.Output(
            calculate_metrics(
                data,
                spike_times,
                spike_clusters,
                amplitudes,
                self.sample_rate,
                self.params,
            )
        )
