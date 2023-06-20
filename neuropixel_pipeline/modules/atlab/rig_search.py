from pydantic import validate_call

from .common import ScanKey


@validate_call
def get_rig(scan_key: ScanKey):
    import datajoint as dj

    experiment = dj.create_virtual_module("experiment", "pipeline_experiment")

    session_rel = experiment.Session() & scan_key
    experiment_rig = session_rel.fetch1("rig")

    return experiment_rig
