from __future__ import annotations

from pydantic import BaseModel, constr, condecimal
from typing import Optional
from typing import List
from enum import Enum
import numpy as np


class NeuropixelConfig(BaseModel):
    # probe type (e.g., "neuropixels 1.0 - 3A").
    probe_type: str
    # site count per shank.
    site_count_per_shank: int
    # (um) horizontal spacing between sites. Defaults to None (single column).
    col_spacing: float = None
    # (um) vertical spacing between columns. Defaults to None (single row).
    row_spacing: float = None
    # (um) offset spacing. Defaults to None.
    white_spacing: float = None
    # number of column per shank. Defaults to 1 (single column).
    col_count_per_shank: int = 1
    # number of shank. Defaults to 1 (single shank).
    shank_count: int = 1
    # (um) spacing between shanks. Defaults to None (single shank).
    shank_spacing: float = None
    # {"bottom", "top"}. y value decrements if "top". Defaults to "bottom".
    y_origin: str = "bottom"

    def build_electrode_layout(self):
        """Will likely want to change this to a (different) Model instead"""
        row_count = int(self.site_count_per_shank / self.col_count_per_shank)
        # self._spacing or 1 is not a good idea with floats
        x_coords = np.tile(
            np.arange(
                0,
                (self.col_spacing or 1) * self.col_count_per_shank,
                (self.col_spacing or 1),
            ),
            row_count,
        )
        y_coords = np.repeat(
            np.arange(row_count) * (self.row_spacing or 1), self.col_count_per_shank
        )

        if self.white_spacing:
            x_white_spaces = np.tile(
                [self.white_spacing, self.white_spacing, 0, 0], int(row_count / 2)
            )
            x_coords = x_coords + x_white_spaces

        shank_cols = np.tile(range(self.col_count_per_shank), row_count)
        shank_rows = np.repeat(range(row_count), self.col_count_per_shank)

        return [
            {
                "probe_type": self.probe_type,
                "electrode": (self.site_count_per_shank * shank_no) + e_id,
                "shank": shank_no,
                "shank_col": c_id,
                "shank_row": r_id,
                "x_coord": x + (shank_no * (self.shank_spacing or 1)),
                "y_coord": {"top": -y, "bottom": y}[self.y_origin],
            }
            for shank_no in range(self.shank_count)
            for e_id, (c_id, r_id, x, y) in enumerate(
                zip(shank_cols, shank_rows, x_coords, y_coords)
            )
        ]

    @classmethod
    def configs(cls) -> List[NeuropixelConfig]:
        return [
            cls(
                probe_type="neuropixels 1.0 - 3A",
                site_count_per_shank=960,
                col_spacing=32,
                row_spacing=20,
                white_spacing=16,
                col_count_per_shank=2,
                shank_count=1,
                shank_spacing=0,
            ),
            cls(
                probe_type="neuropixels 1.0 - 3B",
                site_count_per_shank=960,
                col_spacing=32,
                row_spacing=20,
                white_spacing=16,
                col_count_per_shank=2,
                shank_count=1,
                shank_spacing=0,
            ),
            cls(
                probe_type="neuropixels UHD",
                site_count_per_shank=384,
                col_spacing=6,
                row_spacing=6,
                white_spacing=0,
                col_count_per_shank=8,
                shank_count=1,
                shank_spacing=0,
            ),
            cls(
                probe_type="neuropixels 2.0 - SS",
                site_count_per_shank=1280,
                col_spacing=32,
                row_spacing=15,
                white_spacing=0,
                col_count_per_shank=2,
                shank_count=1,
                shank_spacing=250,
            ),
            cls(
                probe_type="neuropixels 2.0 - MS",
                site_count_per_shank=1280,
                col_spacing=32,
                row_spacing=15,
                white_spacing=0,
                col_count_per_shank=2,
                shank_count=4,
                shank_spacing=250,
            ),
        ]


class SessionKey(BaseModel):
    animal_id: int
    session: int
    scan_idx: int = None


class AcquisitionSoftware(BaseModel):
    acq_software: constr(max_length=24)


class ProbeData(BaseModel):
    # unique indentifier for this model of probe, serial number
    probe: constr(max_length=32)
    probe_type: constr(max_length=32)
    probe_comment: constr(max_length=1000) = None


class SkullReferenceValue(str, Enum):
    BREGMA = "Bregma"
    LAMBDA = "Lambda"


class InsertionData(BaseModel):
    # (um) anterior-posterior; ref is 0; more anterior is more positive
    ap_location: condecimal(max_digits=6, decimal_places=2)

    # (um) medial axis; ref is 0 ; more right is more positive
    ml_location: condecimal(max_digits=6, decimal_places=2)

    # (um) manipulator depth relative to surface of the brain (0); more ventral is more
    # negative
    depth: condecimal(max_digits=6, decimal_places=2)

    # SkullReference, can be coerced from a str
    skull_reference: SkullReferenceValue = "Bregma"

    # (deg) - elevation - rotation about the ml-axis [0, 180] - w.r.t the z+ axis
    theta: Optional[condecimal(max_digits=5, decimal_places=2)] = None

    # (deg) - azimuth - rotation about the dv-axis [0, 360] - w.r.t the x+ axis
    phi: Optional[condecimal(max_digits=5, decimal_places=2)] = None

    # (deg) rotation about the shank of the probe [-180, 180] - clockwise is increasing
    # in degree - 0 is the probe-front facing anterior
    beta: Optional[condecimal(max_digits=5, decimal_places=2)] = None