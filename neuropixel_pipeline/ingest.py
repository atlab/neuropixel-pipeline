# from typing import 
from pydantic import BaseModel

class NeuropixelConfig(BaseModel):
    probe_type: str
    site_count_per_shank: int
    col_spacing: float = None
    row_spacing: float = None
    white_spacing: float = None
    col_count_per_shank: int = 1
    shank_count: int = 1
    shank_spacing: float = None

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