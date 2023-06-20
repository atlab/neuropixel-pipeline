from pydantic import validate_call
from pydantic.dataclasses import dataclass
from typing import Optional
from pathlib import Path
from neuropixel_pipeline.api.clustering import ClusteringTaskMode

from neuropixel_pipeline.readers.labview import LabviewNeuropixelMeta

from . import ACQ_SOFTWARE, CLUSTERING_METHOD, CLUSTERING_OUTPUT_RELATIVE
from .probe_setup import probe_setup
from .session_search import ScanKey, get_session_path
from .rig_search import get_rig
from .kilosort_params import default_kilosort_parameters
from ...api import metadata
from ...schemata import probe, ephys

@dataclass
class AtlabParams:
    scan_key: ScanKey
    base_dir: Optional[Path] = None
    acq_software: str = ACQ_SOFTWARE
    # Will ephys.InsertionLocation just be inserted into directly from 2pmaster?
    insertion_location: Optional[metadata.InsertionLocation] = None
    clustering_method: str = CLUSTERING_METHOD
    clustering_task_mode: ClusteringTaskMode = ClusteringTaskMode.TRIGGER
    clustering_output_directory: Optional[Path] = None
    setup: bool = False

@validate_call
def main(args: AtlabParams):
    ### Setup
    if args.setup:
        probe.ProbeType.fill_neuropixel_probes()
        probe_setup()

    ### PreClustering
    session_meta = dict(**args.scan_key)
    session_meta['rig'] = get_rig(args.scan_key)
    ephys.Session.add_session(session_meta, error_on_duplicate=False)

    session_path = get_session_path(args.scan_key, base_dir=args.base_dir)
    INSERTION_NUMBER = 0 # TODO: Insertion number should be auto_increment?

    labview_metadata = LabviewNeuropixelMeta.from_h5(session_path)

    session_id = (ephys.Session & session_meta).fetch1('session_id')
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

    ephys.LFP.populate() # This isn't implemented yet

    ### Clustering
    # This currently only supports the default kilosort parameters, which might be alright
    if args.clustering_method == CLUSTERING_METHOD:
        ephys.ClusteringParamSet.fill(
            params=default_kilosort_parameters(),
            clustering_method="kilosort4",
            description="default kilosort4 params",
            skip_duplicates=True,
        )

    paramset_rel = (ephys.ClusteringParamSet & args.clustering_method)

    if args.clustering_output_dir is not None:
        args.clustering_output_directory = session_path / CLUSTERING_OUTPUT_RELATIVE

    task_source_rel = (ephys.EphysRecording & insertion_key).proj() * (
        ephys.ClusteringParamSet() & paramset_rel
    ).proj()
    task_source_key = task_source_rel.fetch1()

    task_source_key["clustering_output_dir"] = args.clustering_output_directory
    task_source_key["task_mode"] = "load"
    ephys.ClusteringTask.insert1(task_source_key, skip_duplicates=True)


    ##### TRIGGER KILOSORT HERE IF task_mode = 'trigger'

    ##### Next roadblock is deciding how to handle curation
    ##### Currently it could go Input -> Trigger Kilosort -> Ingest into CuratedClustering.Unit
    ##### with "no curation"
    #####
    ##### but to add curation it would always have to come after kilosort triggering.


    raise NotImplementedError("Not implemented to this point yet")

if __name__ == '__main__':
    ### TODO: Should have a minion mode that checks for any scans to push through the pipeline.
    ###     Will use the --mode=minion flag.
    main()