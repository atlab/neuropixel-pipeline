from pydantic import BaseModel
from enum import Enum
from pathlib import Path

class ClusteringTaskMode(str, Enum):
    LOAD = 'load'
    TRIGGER = 'trigger'

class ClusteringTaskRunner(BaseModel):
    file_path: Path
    clustering_output_dir: Path
    task_mode: ClusteringTaskMode

    def load_time_finished(self):
        if self.task_mode is ClusteringTaskMode.TRIGGER:
            self.trigger_clustering()
        # then load results
        pass

    def trigger_clustering(self):
        # Using docker or kubernetes
        pass