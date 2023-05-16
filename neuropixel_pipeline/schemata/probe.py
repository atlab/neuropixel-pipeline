# flake8: noqa

"""
Neuropixels Probes
"""

from __future__ import annotations

import datajoint as dj
import numpy as np
# from ..api.metadata import NeuropixelConfig

schema = dj.schema("neuropixel_probe")


def run_populate():
    # possibly temporary way of populating these values
    ProbeType.fill_neuropixel_probes()

    # Probe and ElectrodeConfig(/.Electrode) don't currently have fill methods
    pass


@schema
class ProbeType(dj.Lookup):
    """Type of probe."""

    definition = """
    # Type of probe, with specific electrodes geometry defined
    probe_type: varchar(32)  # e.g. neuropixels_1.0
    """

    class Electrode(dj.Part):
        """Electrode information for a given probe."""

        definition = """
        -> master
        electrode: int       # electrode index, starts at 0
        ---
        shank: int           # shank index, starts at 0, advance left to right
        shank_col: int       # column index, starts at 0, advance left to right
        shank_row: int       # row index, starts at 0.
        x_coord=NULL: float  # (um) x coordinate of the electrode within the probe.
        y_coord=NULL: float  # (um) y coordinate of the electrode within the probe.
        """

    @staticmethod
    def fill_neuropixel_probes():
        """
        Create `ProbeType` and `Electrode` for neuropixels probes:
        + neuropixels 1.0 - 3A
        + neuropixels 1.0 - 3B
        + neuropixels UHD
        + neuropixels 2.0 - SS
        + neuropixels 2.0 - MS
        For electrode location, the (0, 0) is the
         bottom left corner of the probe (ignore the tip portion)
        Electrode numbering is 1-indexing
        """

        for probe_config in NeuropixelConfig.configs():
            electrode_layouts = probe_config.build_electrode_layouts()

            with ProbeType.connection.transaction:
                ProbeType.insert1((probe_config.probe_type,), skip_duplicates=True)
                ProbeType.Electrode.insert(electrode_layouts, skip_duplicates=True)


@schema
class Probe(dj.Lookup):
    """Represent a physical probe with unique ID"""

    definition = """
    # Represent a physical probe with unique identification
    probe: varchar(32)  # unique identifier for this model of probe (e.g. serial number)
    ---
    -> ProbeType
    probe_comment='' :  varchar(1000)
    """


@schema
class ElectrodeConfig(dj.Lookup):
    """Electrode configuration setting on a probe."""

    definition = """
    # The electrode configuration setting on a given probe
    electrode_config_hash: uuid
    ---
    -> ProbeType
    electrode_config_name: varchar(4000)  # user friendly name
    """

    class Electrode(dj.Part):
        """Electrode included in the recording."""

        definition = """  # Electrodes selected for recording
        -> master
        -> ProbeType.Electrode
        """
