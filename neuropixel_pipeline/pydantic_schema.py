# Test file

## flake8: noqa

from pydantic import BaseModel, field_validator

class NeuropixelConfig(BaseModel):
    probe_type: str
    site_count_per_shank: int
    col_spacing: float = None
    row_spacing: float = None
    white_spacing: float = None
    col_count_per_shank: int = 1
    shank_count: int = 1
    shank_spacing: float = None

    @field_validator(*(
        'site_count_per_shank',
        'col_spacing',
        'row_spacing',
        'white_spacing',
        'col_count_per_shank',
        'shank_count',
        'shank_spacing'
        ))
    def positive_value(cls, v):
        if v is not None and v < 0.0:
            raise ValueError('value must be strictly positive')
        return v


probe_config = NeuropixelConfig(
    probe_type="neuropixels 1.0 - 3A",
    site_count_per_shank=960,
    col_spacing=32,
    row_spacing=20,
    white_spacing=16,
    col_count_per_shank=2,
    shank_count=1,
    shank_spacing=0,
)

from devtools import debug
debug(probe_config);