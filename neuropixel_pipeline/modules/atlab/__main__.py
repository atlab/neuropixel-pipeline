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
from ...api.clustering import ClusteringTaskMode
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
    clustering_output_directory: Optional[Path] = None
    setup: bool = False


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
    start_time = time.time()
    logging.info("starting neuropixel pipeline")

    ### Setup
    if args.setup:
        logging.info("starting setup section")
        probe.ProbeType.fill_neuropixel_probes()
        probe_setup()
        logging.info("done with setup section")

    ### PreClustering
    logging.info("starting preclustering section")
    session_meta = dict(**args.scan_key)
    session_meta["rig"] = get_rig(args.scan_key)
    ephys.Session.add_session(session_meta, error_on_duplicate=False)

    session_path = get_session_path(args.scan_key, base_dir=args.base_dir)
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

    if args.insertion_location is not None:
        ephys.InsertionLocation.insert(args.insertion_location.model_dict())

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
    if args.clustering_method == DEFAULT_CLUSTERING_METHOD:
        ephys.ClusteringParamSet.fill(
            params=default_kilosort_parameters(),
            clustering_method="kilosort4",
            description="default kilosort4 params",
            skip_duplicates=True,
        )

    paramset_rel = ephys.ClusteringParamSet & args.clustering_method

    if args.clustering_output_dir is not None:
        args.clustering_output_directory = (
            session_path / DEFAULT_CLUSTERING_OUTPUT_RELATIVE
        )

    task_source_rel = (ephys.EphysRecording & insertion_key).proj() * (
        ephys.ClusteringParamSet() & paramset_rel
    ).proj()
    task_source_key = task_source_rel.fetch1()

    task_source_key["clustering_output_dir"] = args.clustering_output_directory
    task_source_key["task_mode"] = str(args.clustering_task_mode)
    ephys.ClusteringTask.insert1(task_source_key, skip_duplicates=True)

    if args.clustering_task_mode is ClusteringTaskMode.TRIGGER:
        task_runner = ClusteringTaskRunner.model_validate(task_source_key)
        logging.info("attempting to trigger kilosort clustering")
        task_runner.trigger_clustering()
        logging.info("one with kilosort clustering")

    ##### Next roadblock is deciding how to handle curation
    ##### Currently it could go Input -> Trigger Kilosort -> Ingest into CuratedClustering.Unit
    ##### with "no curation"
    #####
    ##### but to add curation it would always have to come after kilosort triggering.

    raise NotImplementedError("Not implemented to this point yet")
    logging.info("done with clustering section")

    logging.info("starting post-clustering section")

    logging.info("done with post-clustering section")

    elapsed_time = round(time.time() - start_time, 2)
    logging.info(f"done with neuropixel pipeline, elapsed_time: {elapsed_time}")


if __name__ == "__main__":
    ### TODO: Should have a minion mode that checks for any scans to push through the pipeline.
    ###     Will use the --mode=minion flag.
    main()
