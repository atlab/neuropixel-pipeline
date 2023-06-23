from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from typing import Any
import numpy as np
import os

# i.e. Waveforms and QualityMetrics
# runs ecephys_spike_sorting to produce the waveform analysis and quality metrics files


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

    def calculate(self, data_dir: Path, bin_name: Path):
        from ecephys_spike_sorting.modules.mean_waveforms.extract_waveforms import (
            extract_waveforms,
        )
        import ecephys_spike_sorting.common.utils as utils

        data_dir = Path(data_dir)
        rawData = np.memmap(data_dir / bin_name, dtype="int16", mode="r")
        data = np.reshape(
            rawData, (int(rawData.size / self.num_channels), self.num_channels)
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
            data_dir, self.sample_rate, convert_to_seconds=False
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

    def calculate(self, data_dir: Path, bin_name: Path):
        from ecephys_spike_sorting.modules.quality_metrics.metrics import (
            calculate_metrics,
        )
        import ecephys_spike_sorting.common.utils as utils

        data_dir = Path(data_dir)
        rawData = np.memmap(data_dir / bin_name, dtype="int16", mode="r")
        data = np.reshape(
            rawData, (int(rawData.size / self.num_channels), self.num_channels)
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
            data_dir, self.sample_rate, convert_to_seconds=False
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
