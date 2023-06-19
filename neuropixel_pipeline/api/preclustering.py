from __future__ import annotations

from . import metadata
from pydantic import BaseModel, constr, condecimal
from typing import Optional
from datetime import datetime


class InsertionData(BaseModel):
    # (um) anterior-posterior; ref is 0; more anterior is more positive
    ap_location: condecimal(max_digits=6, decimal_places=2)

    # (um) medial axis; ref is 0 ; more right is more positive
    ml_location: condecimal(max_digits=6, decimal_places=2)

    # (um) manipulator depth relative to surface of the brain (0); more ventral is more
    # negative
    depth: condecimal(max_digits=6, decimal_places=2)

    # SkullReference, can be coerced from a str
    skull_reference: metadata.SkullReferenceValue = "Bregma"

    # (deg) - elevation - rotation about the ml-axis [0, 180] - w.r.t the z+ axis
    theta: Optional[condecimal(max_digits=5, decimal_places=2)] = None

    # (deg) - azimuth - rotation about the dv-axis [0, 360] - w.r.t the x+ axis
    phi: Optional[condecimal(max_digits=5, decimal_places=2)] = None

    # (deg) rotation about the shank of the probe [-180, 180] - clockwise is increasing
    # in degree - 0 is the probe-front facing anterior
    beta: Optional[condecimal(max_digits=5, decimal_places=2)] = None


class EphysFilePath(BaseModel):
    file_path: constr(max_length=255)


class EphysRecordingData(BaseModel):
    sampling_rate: float
    recording_datetime: datetime
    recording_duration: float
