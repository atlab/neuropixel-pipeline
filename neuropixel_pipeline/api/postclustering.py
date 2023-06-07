from __future__ import annotations

from pydantic import BaseModel
from typing import Optional
from ecephys_spike_sorting.modules import quality_metrics, mean_waveforms

# i.e. Waveforms and QualityMetrics

# runs ecephys_spike_sorting to produce the waveform analysis and quality metrics files


class EcephysSpikeSorting(BaseModel):
    # input_json: str # likely don't want to rely on input json here
    args: dict
    output_json: Optional[str] = None


class WaveformSetRunner(EcephysSpikeSorting):
    def calculate(self):
        return mean_waveforms.calculate_mean_waveforms(self.args)


class QualityMetricsRunner(EcephysSpikeSorting):
    def calculate(self):
        return quality_metrics.calculate_quality_metrics(self.args)
