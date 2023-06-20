import time
import logging

from pydantic import validate_call
from pydantic.dataclasses import dataclass
from typing import Optional
from pathlib import Path
from enum import Enum

from . import (
    ACQ_SOFTWARE,
    DEFAULT_CLUSTERING_METHOD,
    DEFAULT_CLUSTERING_OUTPUT_RELATIVE,
)
from .probe_setup import probe_setup
from .session_search import ScanKey, get_session_path
from .rig_search import get_rig
from .kilosort_params import default_kilosort_parameters
from ...api import metadata
from ...api.clustering import ClusteringTaskMode, CurationInput
from ...api.clustering_task import ClusteringTaskRunner
from ...readers.labview import LabviewNeuropixelMeta
from ...schemata import probe, ephys


# Related to how to use the pipeline, not yet used
class PipelineMode(str, Enum):
    MINION = "minion"
    NO_CURATION = "no curation"
    CURATION = "curation"


@dataclass
class AtlabParams:
    scan_key: ScanKey
    base_dir: Optional[Path] = None
    acq_software: str = ACQ_SOFTWARE
    # Will ephys.InsertionLocation just be inserted into directly from 2pmaster?
    insertion_location: Optional[metadata.InsertionLocation] = None
    clustering_method: str = DEFAULT_CLUSTERING_METHOD
    clustering_task_mode: ClusteringTaskMode = ClusteringTaskMode.TRIGGER
    clustering_output_dir: Optional[Path] = None
    curation_input: CurationInput = CurationInput()
    setup: bool = False

    def run_pipeline(self):
        logging.info("starting neuropixel pipeline")
        start_time = time.time()

        ### Setup
        if self.setup:
            logging.info("starting setup section")
            probe.ProbeType.fill_neuropixel_probes()
            probe_setup()
            logging.info("done with setup section")

        ### PreClustering
        logging.info("starting preclustering section")
        session_meta = dict(**self.scan_key)
        session_meta["rig"] = get_rig(self.scan_key)
        ephys.Session.add_session(session_meta, error_on_duplicate=False)

        session_path = get_session_path(self.scan_key, base_dir=self.base_dir)
        INSERTION_NUMBER = 0  # TODO: Insertion number should be auto_increment?

        labview_metadata = LabviewNeuropixelMeta.from_h5(session_path)

        session_id = (ephys.Session & session_meta).fetch1("session_id")
        insertion_key = dict(session_id=session_id, insertion_number=INSERTION_NUMBER)

        ephys.ProbeInsertion.insert1(
            dict(
                **insertion_key,
                probe=labview_metadata.serial_number,
            ),
            skip_duplicates=True,
        )

        if self.insertion_location is not None:
            ephys.InsertionLocation.insert(self.insertion_location.model_dict())

        ephys.EphysFile.insert1(
            dict(
                **insertion_key,
                session_path=session_path,
                acq_software=ACQ_SOFTWARE,
            ),
            skip_duplicates=True,
        )
        ephys.EphysRecording.populate()

        ephys.LFP.populate()  # This isn't implemented yet

        logging.info("done with preclustering section")

        ### Clustering
        logging.info("starting clustering section")
        # This currently only supports the default kilosort parameters, which might be alright for atlab
        if self.clustering_method == DEFAULT_CLUSTERING_METHOD:
            ephys.ClusteringParamSet.fill(
                params=default_kilosort_parameters(),
                clustering_method="kilosort4",
                description="default kilosort4 params",
                skip_duplicates=True,
            )

        paramset_rel = ephys.ClusteringParamSet & self.clustering_method

        if self.clustering_output_dir is not None:
            self.clustering_output_dir = (
                session_path / DEFAULT_CLUSTERING_OUTPUT_RELATIVE
            )

        task_source_rel = (ephys.EphysRecording & insertion_key).proj() * (
            ephys.ClusteringParamSet() & paramset_rel
        ).proj()
        task_source_key = task_source_rel.fetch1()

        task_source_key["clustering_output_dir"] = self.clustering_output_dir
        task_source_key["task_mode"] = str(self.clustering_task_mode)
        ephys.ClusteringTask.insert1(task_source_key, skip_duplicates=True)

        if self.clustering_task_mode is ClusteringTaskMode.TRIGGER:
            task_runner = ClusteringTaskRunner.model_validate(task_source_key)
            logging.info("attempting to trigger kilosort clustering")
            task_runner.trigger_clustering()
            logging.info("one with kilosort clustering")
        ephys.Clustering.populate()

        ##### Next roadblock is deciding how to handle curation
        ##### Currently it could go Input -> Trigger Kilosort -> Ingest into CuratedClustering.Unit
        ##### with "no curation"
        #####
        ##### but to add curation it would always have to come after kilosort triggering.
        if self.curation_input.curation_output_dir is None:
            clustering_source_key = (
                (ephys.Clustering() & task_source_key).proj().fetch1()
            )
            self.curation_input.curation_output_dir = clustering_source_key[
                "clustering_output_dir"
            ]
        ephys.Curation.create1_from_clustering_task(
            dict(
                **clustering_source_key,
                **self.curation_input.model_dump(),
            )
        )

        logging.info("done with clustering section")

        logging.info("starting post-clustering section")
        ephys.WaveformSet.populate()
        ephys.QualityMetrics.populate()
        logging.info("done with post-clustering section")

        elapsed_time = round(time.time() - start_time, 2)
        logging.info(f"done with neuropixel pipeline, elapsed_time: {elapsed_time}")


@validate_call
def setup_logging(log_level=logging.DEBUG):
    import sys

    root = logging.getLogger()
    root.setLevel(log_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s: %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    return root


@validate_call
def main(args: AtlabParams):
    setup_logging()

    args.run_pipeline()

    raise NotImplementedError(
        "Curation mode is not supported yet (no curation is though)"
    )


if __name__ == "__main__":
    ### TODO: Should have a minion mode that checks for any scans to push through the pipeline.
    ###     Will use the --mode=minion flag.
    main()
