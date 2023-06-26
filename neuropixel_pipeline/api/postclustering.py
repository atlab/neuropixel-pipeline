from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field, validate_call
from pydantic.dataclasses import dataclass
from typing import Optional, Any
import numpy as np


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
    sample_rate: float = Field(
        default=30000.0,
        description="Sample rate of Neuropixels AP band continuous data",
    )
    num_channels: int = Field(
        default=384, description="Total number of channels in binary data files"
    )
    bit_volts: float = Field(
        default=0.195, # This might not want to have a default?
        description="Scalar required to convert int16 values into microvolts",
    )
    vertical_site_spacing: float = Field(
        default=20e-6, description="Vertical site spacing in meters"
    )
    params: WaveformSetRunner.Params = Field(
        default_factory=lambda: WaveformSetRunner.Params()
    )

    class Params(BaseModel):
        samples_per_spike: int = Field(
            default=82, help="Number of samples to extract for each spike"
        )
        pre_samples: int = Field(
            default=20, help="Number of samples between start of spike and the peak"
        )
        num_epochs: int = Field(
            default=1, help="Number of epochs to compute mean waveforms"
        )
        spikes_per_epoch: int = Field(
            default=100, help="Max number of spikes per epoch"
        )
        upsampling_factor: float = Field(
            default=200 / 82, help="Upsampling factor for calculating waveform metrics"
        )
        spread_threshold: float = Field(
            default=0.12, help="Threshold for computing channel spread of 2D waveform"
        )
        site_range: int = Field(
            default=16, help="Number of sites to use for 2D waveform metrics"
        )
        cWaves_path: Optional[Path] = Field(
            None, help="directory containing the TPrime executable."
        )
        use_C_Waves: bool = Field(
            default=False, help="Use faster C routine to calculate mean waveforms"
        )
        snr_radius: int = Field(
            default=8,
            help="disk radius (chans) about pk-chan for snr calculation in C_waves",
        )
        snr_radius_um: int = Field(
            default=8,
            help="disk radius (um) about pk-chan for snr calculation in C_waves",
        )
        mean_waveforms_file: Path = Field(help="Path to mean waveforms file (.npy)") # Is this for the output file??

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
        templates = np.load(kilosort_output_dir / "templates.npy")

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
                self.params.model_dump(),
            )
        )


# https://github.com/jenniferColonell/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/_schemas.py
class QualityMetricsRunner(BaseModel):
    sample_rate: float = Field(
        default=30000.0,
        description="Sample rate of Neuropixels AP band continuous data",
    )
    num_channels: int = Field(
        default=384, description="Total number of channels in binary data files"
    )
    params: QualityMetricsRunner.Params = Field(
        default_factory=lambda: QualityMetricsRunner.Params()
    )

    class Params(BaseModel):
        isi_threshold: float = Field(
            default=0.0015, help="Maximum time (in seconds) for ISI violation"
        )
        min_isi: float = Field(
            default=0.00, help="Minimum time (in seconds) for ISI violation"
        )
        tbin_sec: float = Field(
            default=0.001, help="time bin in seconds for ccg in contam_rate calculation"
        )
        max_radius_um: int = Field(
            default=68, help="Maximum radius for computing PC metrics, in um"
        )
        max_spikes_for_unit: int = Field(
            default=500, help="Number of spikes to subsample for computing PC metrics"
        )
        max_spikes_for_nn: int = Field(
            default=10000, help="Further subsampling for NearestNeighbor calculation"
        )
        n_neighbors: int = Field(
            default=4, help="Number of neighbors to use for NearestNeighbor calculation"
        )
        n_silhouette: int = Field(
            default=10000,
            help="Number of spikes to use for calculating silhouette score",
        )
        drift_metrics_min_spikes_per_interval: int = Field(
            default=10, help="Minimum number of spikes for computing depth"
        )
        drift_metrics_interval_s: float = Field(
            default=100.0, help="Interval length is seconds for computing spike depth"
        )
        include_pcs: bool = Field(
            default=False, help="Set to false if principal component analysis is not available"
        )

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
        templates = np.load(kilosort_output_dir / "templates.npy")

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
                None,  # pc_features
                None,  # pc_feature_ind
                self.params.model_dump(),
            )
        )
