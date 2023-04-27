# from typing import 
from pydantic import BaseModel
import numpy as np

class NeuropixelConfig(BaseModel): # Add the docstrings from build_electrode_layouts method
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
        """Electrode layout building will be moved here, while likely also return a different Model"""
        row_count = int(self.site_count_per_shank / self.col_count_per_shank)
        x_coords = np.tile(
            np.arange(0, (self.col_spacing or 1) * self.col_count_per_shank, (self.col_spacing or 1)),
            row_count,
        )
        y_coords = np.repeat(np.arange(row_count) * (self.row_spacing or 1), self.col_count_per_shank)

        if self.white_spacing is not None:
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


neuropixels_probes_config = [
    NeuropixelConfig(
        probe_type="neuropixels 1.0 - 3A",
        site_count_per_shank=960,
        col_spacing=32,
        row_spacing=20,
        white_spacing=16,
        col_count_per_shank=2,
        shank_count=1,
        shank_spacing=0,
    ),
    NeuropixelConfig(
        probe_type="neuropixels 1.0 - 3B",
        site_count_per_shank=960,
        col_spacing=32,
        row_spacing=20,
        white_spacing=16,
        col_count_per_shank=2,
        shank_count=1,
        shank_spacing=0,
    ),
    NeuropixelConfig(
        probe_type="neuropixels UHD",
        site_count_per_shank=384,
        col_spacing=6,
        row_spacing=6,
        white_spacing=0,
        col_count_per_shank=8,
        shank_count=1,
        shank_spacing=0,
    ),
    NeuropixelConfig(
        probe_type="neuropixels 2.0 - SS",
        site_count_per_shank=1280,
        col_spacing=32,
        row_spacing=15,
        white_spacing=0,
        col_count_per_shank=2,
        shank_count=1,
        shank_spacing=250,
    ),
    NeuropixelConfig(
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

if __name__ == "__main__":
    from devtools import debug
    debug(neuropixels_probes_config)