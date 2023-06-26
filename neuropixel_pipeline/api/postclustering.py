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
    vertical_site_spacing: Any # TODO: Figure out what this does
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
        self, bin_file: Path, kilosort_output_dir: Path, has_sync_channel=False
    ):
        from ecephys_spike_sorting.modules.mean_waveforms.extract_waveforms import (
            extract_waveforms,
        )
        import ecephys_spike_sorting.common.utils as utils

        bin_file = Path(bin_file)
        kilosort_output_dir = Path(kilosort_output_dir)
        data = extract_data_from_bin(
            bin_file=bin_file,
            num_channels=self.num_channels,
            has_sync_channel=has_sync_channel,
        )

        (
            spike_times,
            spike_clusters,
            spike_templates,
            amplitudes,
            unwhitened_temps,
            channel_map,
            channel_pos,
            cluster_ids,
            cluster_quality,
            cluster_amplitude,
        ) = utils.load_kilosort_data(
            kilosort_output_dir, self.sample_rate, convert_to_seconds=False
        )
        templates = np.load(kilosort_output_dir / 'templates.npy')

        # TODO: calculate_mean_waveforms
        # might want to use calculate_mean_waveforms because that produces the mean_waveforms.npy file that gets ingested
        return WaveformSetRunner.Output(
            *extract_waveforms(
                data,
                spike_times,
                spike_clusters,
                templates,
                channel_map,
                self.bit_volts,
                self.sample_rate,
                self.vertical_site_spacing,
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
        "include_pcs": False,
    }

    @dataclass
    class Output:
        metrics: Any

    def calculate(
        self, bin_file: Path, kilosort_output_dir: Path, has_sync_channel=False
    ):
        from ecephys_spike_sorting.modules.quality_metrics.metrics import (
            calculate_metrics,
        )
        import ecephys_spike_sorting.common.utils as utils

        bin_file = Path(bin_file)
        kilosort_output_dir = Path(kilosort_output_dir)
        # data = extract_data_from_bin(
        #     bin_file=bin_file,
        #     num_channels=self.num_channels,
        #     has_sync_channel=has_sync_channel,
        # )

        (
            spike_times,
            spike_clusters,
            spike_templates,
            amplitudes,
            unwhitened_temps,
            channel_map,
            channel_pos,
            cluster_ids,
            cluster_quality,
            cluster_amplitude,
        ) = utils.load_kilosort_data(
            kilosort_output_dir, self.sample_rate, convert_to_seconds=False
        )
        templates = np.load(kilosort_output_dir / 'templates.npy')

        # might want to use calculate_quality_metrics because that produces the metrics.csv file that gets ingested
        return QualityMetricsRunner.Output(
            calculate_metrics(
                spike_times,
                spike_clusters,
                spike_templates,
                amplitudes,
                channel_map,
                channel_pos,
                templates,
                None, # pc_features
                None, # pc_feature_ind
                self.params,
            )
        )
