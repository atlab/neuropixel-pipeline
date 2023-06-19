from __future__ import annotations

from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
from enum import Enum


class ClusteringTaskMode(str, Enum):
    LOAD = "load"
    TRIGGER = "trigger"


class CurationInput(BaseModel):
    curation_id: int
    curation_time: datetime = datetime.now()
    curation_output_dir: Path = None
    quality_control: bool = False
    curation: str = "no curation"
    curation_note: str = ""
