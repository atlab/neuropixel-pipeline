from __future__ import annotations

from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from pathlib import Path
from enum import Enum


class ClusteringTaskMode(str, Enum):
    LOAD = "load"
    TRIGGER = "trigger"


class CurationInput(BaseModel):
    # curation_id: Optional[int] = None
    curation_time: datetime = datetime.now()
    curation_output_dir: Optional[Path] = None
    curation: str = "no curation"
    curation_note: str = ""
