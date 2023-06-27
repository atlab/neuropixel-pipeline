from pydantic import BaseModel
from typing import Optional
from enum import Enum
from pathlib import Path

from ..readers.kilosort import Kilosort


class ClusteringTaskMode(str, Enum):
    LOAD = "load"
    TRIGGER = "trigger"


class ClusteringTaskRunner(BaseModel):
    data_dir: Path = None
    results_dir: Path = None
    filename: Path = None
    clustering_params: Optional[dict] = None

    def trigger_clustering(self, check_for_existing_results=False):
        def run_kilosort(args):
            # Locally or eventually maybe using an HTTP request to a REST server
            from kilosort_runner.run import KilosortRunner, KilosortParams

            params = KilosortParams.model_validate(self.clustering_params)
            runner = KilosortRunner(
                data_dir=self.data_dir,
                results_dir=self.results_dir,
                filename=self.filename,
                params=params,
            )
            runner.run_kilosort()

        if check_for_existing_results:
            try:
                Kilosort(self.results_dir)
                print(
                    f"kilosort results already exist in this directory: {self.results_dir}"
                )
                print(
                    "skipping triggering kilosort because `check_for_existing_results` is set to True"
                )
            except FileNotFoundError:
                print("kilosort results do not exist yet, triggering kilosort")
                run_kilosort(self)
        else:
            run_kilosort(self)
