import os
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, conint, validate_call


PROBE_CALIBRATION_DIR = Path("/mnt/lab/Neuropixel Electrode Calib Files")
DEFAULT_PROBE_TYPE = "neuropixels 1.0 - 3B"
DEFAULT_PROBE_COMMENT = "Filled using calibration files"

class ScanKey(BaseModel, from_attributes=True):
    animal_id: conint(ge=0, le=2_147_483_647)
    session: conint(ge=0, le=32_767)
    scan_idx: conint(ge=0, le=32_767)

class SessionSearch(BaseModel):
    scan_key: ScanKey

    def get_generic_session_path(self):
        import datajoint as dj

        scan_key = {"animal_id": 29187, "session": 8, "scan_idx": 1}

        experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
        acq = dj.create_virtual_module('acq', 'acq')

        ephys_start_time_rel = dj.U('ephys_start_time') & (experiment.ScanEphysLink & scan_key)
        acq_ephys_rel = (acq.Ephys - acq.EphysIgnore)
        ephys_path = (acq_ephys_rel & ephys_start_time_rel).fetch1('ephys_path')
        return Path(ephys_path)
    
    @staticmethod
    def normalize_session_path(generic_session_path: Path, base_dir: Path) -> Path:
        parts = generic_session_path.parts
        index = parts.index('raw')
        session_path = Path(base_dir).joinpath(*parts[index+1:])
        return session_path.parent

@validate_call
def get_session_path(scan_key: ScanKey, base_dir: Path = None) -> Path:
    generic_session_path = SessionSearch(scan_key=scan_key).get_generic_session_path()
    if base_dir is not None:
        return SessionSearch.normalize_session_path(generic_session_path, base_dir)
    return generic_session_path

@validate_call
def get_probe_serial_numbers(
    probe_calibration_dir: Path = PROBE_CALIBRATION_DIR,
) -> List[int]:
    assert probe_calibration_dir.exists()

    calibration_suffix = "_ADCCalibration.csv"

    probe_serial_nums = []
    for p in os.listdir(probe_calibration_dir):
        if calibration_suffix in p:
            probe_serial_nums.append(int(p.split("_")[0]))

    return probe_serial_nums


@validate_call
def probe_setup(
    probe_calibration_dir: Path = PROBE_CALIBRATION_DIR, insert=True
) -> List[Dict[str, Any]]:
    """Requires datajoint access if insert is True (the default)"""

    probe_serial_nums = get_probe_serial_numbers(
        probe_calibration_dir=probe_calibration_dir
    )
    probes = [
        {
            "probe": serial_num,
            "probe_type": DEFAULT_PROBE_TYPE,
            "probe_comment": DEFAULT_PROBE_COMMENT,
        }
        for serial_num in probe_serial_nums
    ]

    if insert:
        from ..schemata import probe

        probe.Probe.insert(probes, skip_duplicates=True)

    return probes
