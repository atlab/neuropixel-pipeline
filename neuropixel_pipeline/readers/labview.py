"""
Custom labview neuropixel aquisition format reader
"""

from __future__ import annotations

import h5py

from pathlib import Path
from pydantic import BaseModel

class LabviewNeuropixelMetadata(BaseModel):
    pass

    def from_h5(directory: Path, family: str = 'NPElectrophysiology%d.h5') -> LabviewNeuropixelMetadata:
        """
        Uses an h5 family driver if necessary
        """

        meta_file = directory / 'NPElectrophysiology%d.h5'
        with h5py.File(meta_file, driver='family', memb_size=0) as f:
            meta = dict(f.attrs)

        print(meta)

        raise NotImplementedError

    def from_metafile() -> LabviewNeuropixelMetadata:
        """
        This will be implemented when the metadata from labview is separated from the h5.
        """
        raise NotImplementedError('This will be implemented when the metadata from labview is separated from the h5')