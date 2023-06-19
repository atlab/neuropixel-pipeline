import os
from pathlib import Path
from typing import List, Dict, Any
from pydantic import validate_call


PROBE_CALIBRATION_DIR = Path("/mnt/lab/Neuropixel Electrode Calib Files")
DEFAULT_PROBE_TYPE = "neuropixels 1.0 - 3B"
DEFAULT_PROBE_COMMENT = "Filled using calibration files"


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
