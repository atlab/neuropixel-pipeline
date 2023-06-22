from pydantic import BaseModel, validate_call
from pathlib import Path

from .common import ScanKey


class SessionSearch(BaseModel):
    scan_key: ScanKey

    def get_generic_session_path(self):
        import datajoint as dj

        scan_key = {"animal_id": 29187, "session": 8, "scan_idx": 1}

        experiment = dj.create_virtual_module("experiment", "pipeline_experiment")
        acq = dj.create_virtual_module("acq", "acq")

        ephys_start_time_rel = dj.U("ephys_start_time") & (
            experiment.ScanEphysLink & scan_key
        )
        acq_ephys_rel = acq.Ephys - acq.EphysIgnore
        ephys_path = (acq_ephys_rel & ephys_start_time_rel).fetch1("ephys_path")
        return Path(ephys_path)

    @staticmethod
    def normalize_session_path(generic_session_path: Path, base_dir: Path) -> Path:
        parts = generic_session_path.parts
        index = parts.index("raw")
        session_path = Path(base_dir).joinpath(*parts[index + 1 :])
        return session_path.parent


@validate_call
def get_session_path(scan_key: dict, base_dir: Path = None) -> Path:
    scan_key = ScanKey.model_validate(scan_key)
    generic_session_path = SessionSearch(scan_key=scan_key).get_generic_session_path()
    if base_dir is not None:
        return SessionSearch.normalize_session_path(generic_session_path, base_dir)
    return generic_session_path
