from pydantic import validate_call
from pydantic.dataclasses import dataclass
from typing import Optional
from pathlib import Path

from neuropixel_pipeline.readers.labview import LabviewNeuropixelMeta

from . import ACQ_SOFTWARE
from .probe_setup import probe_setup
from .session_search import ScanKey, get_session_path
from .rig_search import get_rig
from ...api import metadata
from ...schemata import probe, ephys

@dataclass
class AtlabParams:
    scan_key: ScanKey
    base_dir: Optional[Path] = None
    acq_software: str = ACQ_SOFTWARE
    # Will ephys.InsertionLocation just be inserted into directly from 2pmaster?
    insertion_location: Optional[metadata.InsertionLocation] = None
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
    def this_needs_work(): # The contents inside this temp function need to be fixed
        settings = {
            "NchanTOT": 385,
            "fs": 30000,
            "nt": 61,
            "Th": 8,
            "spkTh": 8,
            "Th_detect": 9,
            "nwaves": 6,
            "nskip": 25,
            "nblocks": 5,
            "binning_depth": 5,
            "sig_interp": 20,
            "probe_name": "neuropixPhase3B1_kilosortChanMap.mat",
        }
        settings["nt0min"] = int(20 * settings["nt"] / 61)
        settings["NT"] = 2 * settings["fs"]
        settings["n_chan_bin"] = settings["NchanTOT"]


        # # lookup table, more commonly used (need a simple interface to insert into this occasionally)
        ephys.ClusteringParamSet.fill(
            params=settings,
            clustering_method="kilosort4",
            description="default kilosort4 params",
            skip_duplicates=True,
        )

    raise NotImplementedError("Not implemented to this point yet")

if __name__ == '__main__':
    main()