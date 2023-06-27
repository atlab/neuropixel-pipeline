from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field

WAVEFORM_METRICS_FILE = "waveform_metrics.csv"
QUALITY_METRICS_FILE = "metrics.csv"


# i.e. Waveforms and QualityMetrics
# runs ecephys_spike_sorting to produce the waveform analysis and quality metrics files


# https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/_schemas.py
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
        num_channels_to_compare: int = Field(
            default=13,
            help="Number of channels to use for computing PC metrics; must be odd",
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
        include_pc_metrics: bool = Field(
            default=False, help="Compute features that require principal components"
        )

    def calculate(self, kilosort_output_dir: Path):
        from ecephys_spike_sorting.modules.quality_metrics.__main__ import (
            calculate_quality_metrics,
        )

        kilosort_output_dir = Path(kilosort_output_dir)

        args = self.model_dump(by_alias=True)
        args["quality_metrics_params"]["quality_metrics_output_file"] = (
            kilosort_output_dir / QUALITY_METRICS_FILE
        )
        args["directories"] = {"kilosort_output_directory": kilosort_output_dir}
        args["waveform_metrics"] = {
            "waveform_metrics_file": kilosort_output_dir / WAVEFORM_METRICS_FILE
        }
        return calculate_quality_metrics(args)
