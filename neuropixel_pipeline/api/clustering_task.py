from pydantic import BaseModel
from enum import Enum


class ClusteringTaskMode(str, Enum):
    LOAD = "load"
    TRIGGER = "trigger"


class ClusteringTaskRunner(BaseModel):
    clustering_params: dict
    task_mode: ClusteringTaskMode

    def trigger_clustering(self):
        # Locally or using an HTTP request to a REST server
        if self.task_mode is ClusteringTaskMode.TRIGGER:
            try:
                from kilosort_runner.run import KilosortParams

                KilosortParams.model_validate(self.clustering_params).run_kilosort()
            except Exception as e:
                print(f"Caught exception when trying to trigger Kilosort:\n{e}")
        elif self.task_mode is ClusteringTaskMode.LOAD:
            print("task mode set to 'load', not doing anything")
        else:
            raise NotImplementedError(
                f"This task mode '{self.task_mode}' is not supported"
            )
