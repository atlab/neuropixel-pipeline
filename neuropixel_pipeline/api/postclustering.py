from __future__ import annotations

from pydantic import BaseModel
from .ecephys_common import (
    EphysParams,
    Directories,
    WaveformMetricsFile,
    ClusterMetricsFile,
)
from ecephys_spike_sorting.modules import quality_metrics, mean_waveforms

# i.e. Waveforms and QualityMetrics

# runs ecephys_spike_sorting to produce the waveform analysis and quality metrics files


class EcephysSpikeSorting(BaseModel):
    # input_json: str # likely don't want to rely on input json here
    args: dict


# https://github.com/jenniferColonell/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/mean_waveforms/_schemas.py
class WaveformSetRunner(EcephysSpikeSorting):
    # mean_waveform_params = Nested(MeanWaveformParams)
    waveform_metrics: WaveformMetricsFile
    cluster_metrics: ClusterMetricsFile
    ephys_params: EphysParams
    directories: Directories

    def calculate(self):
        import ecephys_spike_sorting.modules.mean_waveforms.__main__  # noqa: F401

        return mean_waveforms.__main__.calculate_mean_waveforms(self.args)


# https://github.com/jenniferColonell/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/_schemas.py
class QualityMetricsRunner(EcephysSpikeSorting):
    # quality_metrics_params = Nested(QualityMetricsParams)
    ephys_params: EphysParams
    directories: Directories
    waveform_metrics: WaveformMetricsFile
    cluster_metrics: ClusterMetricsFile

    def calculate(self):
        import ecephys_spike_sorting.modules.quality_metrics.__main__  # noqa: F401

        return quality_metrics.__main__.calculate_quality_metrics(self.args)
