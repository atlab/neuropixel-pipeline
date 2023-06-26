from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field, validate_call
import numpy as np
import pandas as pd
import time
import os

MEAN_WAVEFORM_FILE = Path("mean_waveforms.npy")
WAVEFORM_METRICS_FILE = Path("waveform_metrics.csv")
CLUSTER_METRICS_FILE = Path("metrics.csv")


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
    generic_params: WaveformSetRunner.GenericParams = Field(
        alias="ephys_params", default_factory=lambda: WaveformSetRunner.GenericParams()
    )
    params: WaveformSetRunner.Params = Field(
        alias="mean_waveform_params", default_factory=lambda: WaveformSetRunner.Params()
    )

    class GenericParams(BaseModel):
        sample_rate: float = Field(
            default=30000.0,
            description="Sample rate of Neuropixels AP band continuous data",
        )
        num_channels: int = Field(
            default=384, description="Total number of channels in binary data files"
        )
        bit_volts: float = Field(
            default=0.195,  # FIXME: This might not want to have a default?
            description="Scalar required to convert int16 values into microvolts",
        )
        vertical_site_spacing: float = Field(
            default=20e-6, description="Vertical site spacing in meters"
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
        snr_radius: int = Field(
            default=8,
            help="disk radius (chans) about pk-chan for snr calculation in C_waves",
        )
        snr_radius_um: int = Field(
            default=8,
            help="disk radius (um) about pk-chan for snr calculation in C_waves",
        )

    # This cannot use calculate_mean_waveforms to directly produce the mean_waveforms.npy file
    # We need to support stripping the sync_channel from the recording session .bin
    # It also doesn't support C_waves
    def calculate(
        self, bin_file: Path, kilosort_output_dir: Path, has_sync_channel=False
    ):
        from ecephys_spike_sorting.common.utils import load_kilosort_data
        from ecephys_spike_sorting.modules.mean_waveforms.extract_waveforms import (
            extract_waveforms,
            writeDataAsNpy,
        )

        bin_file = Path(bin_file)
        kilosort_output_dir = Path(kilosort_output_dir)

        print("ecephys spike sorting: mean waveforms module")

        start = time.time()

        print("Calculating mean waveforms using python.")
        print("Loading data...")

        data = extract_data_from_bin(
            bin_file=bin_file,
            num_channels=self.generic_params.num_channels,
            has_sync_channel=has_sync_channel,
        )

        (
            spike_times,
            spike_clusters,
            spike_templates,
            amplitudes,
            templates,
            channel_map,
            channel_pos,
            clusterIDs,
            cluster_quality,
            cluster_amplitude,
        ) = load_kilosort_data(
            kilosort_output_dir,
            self.generic_params.sample_rate,
            convert_to_seconds=False,
        )

        print("Calculating mean waveforms...")

        waveforms, spike_counts, coords, labels, metrics = extract_waveforms(
            data,
            spike_times,
            spike_clusters,
            templates,
            channel_map,
            self.generics_params.bit_volts,
            self.generics_params.sample_rate,
            self.generics_params.vertical_site_spacing,
            self.params.model_dump(),
        )

        writeDataAsNpy(waveforms, kilosort_output_dir / MEAN_WAVEFORM_FILE)
        metrics.to_csv(kilosort_output_dir / WAVEFORM_METRICS_FILE, index=False)

        # if the cluster metrics have already been run, merge the waveform metrics into that file
        # build file path with current version
        # FIXME: Doesn't support multiple versions in the same way that the ecephys_spike_sorting package does
        metrics_file = CLUSTER_METRICS_FILE
        metrics_curr = os.path.join(
            Path(metrics_file).parent, Path(metrics_file).stem + "_.csv"
        )

        if os.path.exists(metrics_curr):
            qmetrics = pd.read_csv(metrics_curr)
            qmetrics = qmetrics.drop(qmetrics.columns[0], axis="columns")
            qmetrics = qmetrics.merge(
                pd.read_csv(kilosort_output_dir / WAVEFORM_METRICS_FILE, index_col=0),
                on="cluster_id",
                suffixes=("_quality_metrics", "_waveform_metrics"),
            )
            print("Saving merged quality metrics ...")
            qmetrics.to_csv(metrics_curr, index=False)

        execution_time = time.time() - start

        print("total time: " + str(np.around(execution_time, 2)) + " seconds")
        print()

        return {"execution_time": execution_time}  # output manifest


# https://github.com/jenniferColonell/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/_schemas.py
class QualityMetricsRunner(BaseModel):
    generic_params: QualityMetricsRunner.GenericParams = Field(
        alias="ephys_params",
        default_factory=lambda: QualityMetricsRunner.GenericParams(),
    )
    params: QualityMetricsRunner.Params = Field(
        alias="quality_metrics_params",
        default_factory=lambda: QualityMetricsRunner.Params(),
    )

    class GenericParams(BaseModel):
        sample_rate: float = Field(
            default=30000.0,
            description="Sample rate of Neuropixels AP band continuous data",
        )
        num_channels: int = Field(
            default=384, description="Total number of channels in binary data files"
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
            default=False,
            help="Set to false if principal component analysis is not available",
        )

    def calculate(self, kilosort_output_dir: Path):
        from ecephys_spike_sorting.modules.quality_metrics.__main__ import (
            calculate_quality_metrics,
        )

        kilosort_output_dir = Path(kilosort_output_dir)

        args = self.model_dump(by_alias=True)
        args["cluster_metrics"] = {
            "cluster_metrics_file": kilosort_output_dir / "metrics.csv"
        }
        args["directories"] = {"kilosort_output_directory": kilosort_output_dir}
        return calculate_quality_metrics(args)
