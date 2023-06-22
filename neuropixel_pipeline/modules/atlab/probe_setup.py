from pathlib import Path
from typing import List, Dict, Any
import os

from . import (
    PROBE_CALIBRATION_DIR,
    PROBE_CALIBRATION_SUFFIX,
    DEFAULT_PROBE_COMMENT,
    DEFAULT_PROBE_TYPE,
)


def get_probe_serial_numbers(
    probe_calibration_dir: Path = PROBE_CALIBRATION_DIR,
    probe_calibration_suffix: str = PROBE_CALIBRATION_SUFFIX,
) -> List[int]:
    assert probe_calibration_dir.exists()

    probe_serial_nums = []
    for p in os.listdir(probe_calibration_dir):
        if probe_calibration_suffix in p:
            probe_serial_nums.append(int(p.split("_")[0]))

    return probe_serial_nums


def probe_setup(
    probe_calibration_dir: Path = PROBE_CALIBRATION_DIR,
    probe_calibration_suffix: str = PROBE_CALIBRATION_SUFFIX,
    insert=True,
) -> List[Dict[str, Any]]:
    """Requires datajoint access if insert is True (the default)"""

    probe_serial_nums = get_probe_serial_numbers(
        probe_calibration_dir=probe_calibration_dir,
        probe_calibration_suffix=probe_calibration_suffix,
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
        from ...schemata import probe

        probe.Probe.insert(probes, skip_duplicates=True)

    return probes
