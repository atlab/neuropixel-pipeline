from __future__ import annotations

from pydantic import BaseModel, constr
from datetime import datetime


class EphysFilePath(BaseModel):
    file_path: constr(max_length=255)


class EphysRecordingData(BaseModel):
    sampling_rate: float
    recording_datetime: datetime
    recording_duration: float
