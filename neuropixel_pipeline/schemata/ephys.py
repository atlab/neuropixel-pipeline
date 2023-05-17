# flake8: noqa

import gc

from decimal import Decimal

import datajoint as dj
from . import base
from . import probe
from ..api import metadata
from . import dj_utils
from pathlib import Path
from typing import List
from pydantic import BaseModel, PositiveInt

schema = dj.schema("neuropixel_ephys")

# TODO: Define config stores (pydantic serialization is a better alternative to the adapter side of things though)
# TODO: Also, decide whether the external blob is fine rather than file paths (or find an abstraction over filepaths & the data types we're interested in, basically everthing with longblob)
#       Maybe just some identifier based on Class (Table) name + field name?
stores = {"": {}}
dj_utils.StoresConfig(stores).set_dj_config()
# TODO: YOOOOOOOO, I should have the pydantic adapters either match (or at least have names that point to) their field names, like adapters.lfp_mean or adapters['lfp_mean']
#       this would make it really easy to know how to fetch an filepath datatype! If two different fields have the same adapter internally, doesn't matter they can have separate
#       names that point to the same internal adapter. So like adapters.lfp_mean and adapters.lfp might just be pointing to the same adapter, but have two different ways to get there.


class Populate: # TODO: Add a discriminant (pydantic supports these) or enum for changing how Populate gets run
    def run(data: dict):
        AcquisitionSoftware  # no populate necessary

        probe_insertion = dict(
            session_id=0,
            insertion_number=0,
            shank_number=None,  # needs to be added as a key?
            probe=None,
        )
        ProbeInsertion.insert1(probe_insertion)

        InsertionLocation.insert1(
            dict(
                **probe_insertion,
                skull_reference="Bregma",  # will it always be Bregma?
                ap_location=None,  # from kilosort metadata
                ml_location=None,  # from kilosort metadata
                depth=None,  # from kilosort metadata
                theta=None,  # nullable?
                phi=None,  # nullable?
                beta=None,  # nullable?
            )
        )


class PreClusteringData(BaseModel, from_attributes=True):
    session_id: PositiveInt
    probe: metadata.ProbeData


### ----------------------------- Table declarations ----------------------


# ------------ Pre-Clustering --------------


@schema
class AcquisitionSoftware(dj.Lookup):
    """Name of software used for recording electrophysiological data."""

    definition = """  # Software used for recording of neuropixels probes
    acq_software: varchar(24)
    """
    contents = zip(["LabviewV1", "SpikeGLX", "Open Ephys"])


@schema
class ProbeInsertion(dj.Manual):
    """Information about probe insertion across subjects and sessions."""

    definition = """
    # Probe insertion implanted into an animal for a given session.
    -> base.Session
    insertion_number: tinyint unsigned
    ---
    -> probe.Probe
    """


@schema
class InsertionLocation(dj.Manual):
    """Stereotaxic location information for each probe insertion."""

    definition = """
    # Brain Location of a given probe insertion.
    -> ProbeInsertion
    ---
    -> base.SkullReference
    ap_location: decimal(6, 2) # (um) anterior-posterior; ref is 0; more anterior is more positive
    ml_location: decimal(6, 2) # (um) medial axis; ref is 0 ; more right is more positive
    depth:       decimal(6, 2) # (um) manipulator depth relative to surface of the brain (0); more ventral is more negative
    theta=null:  decimal(5, 2) # (deg) - elevation - rotation about the ml-axis [0, 180] - w.r.t the z+ axis
    phi=null:    decimal(5, 2) # (deg) - azimuth - rotation about the dv-axis [0, 360] - w.r.t the x+ axis
    beta=null:   decimal(5, 2) # (deg) rotation about the shank of the probe [-180, 180] - clockwise is increasing in degree - 0 is the probe-front facing anterior
    """


@schema
class EphysRecording(dj.Imported):
    """Automated table with electrophysiology recording information for each probe inserted during an experimental session."""

    definition = """
    # Ephys recording from a probe insertion for a given session.
    -> ProbeInsertion
    ---
    -> probe.ElectrodeConfig
    -> AcquisitionSoftware
    sampling_rate: float # (Hz)
    recording_datetime: datetime # datetime of the recording from this probe
    recording_duration: float # (seconds) duration of the recording from this probe
    """

    class EphysFile(dj.Part):
        """Paths of electrophysiology recording files for each insertion."""

        definition = """
        # Paths of files of a given EphysRecording round.
        -> master
        file_path: varchar(255)  # filepath relative to root data directory
        """

    @classmethod
    def read_metadatas(
        cls, directories: List[Path]
    ) -> List[metadata.LabviewNeuropixelMetadata]:
        labview_metadatas = []
        for session_dir in directories:
            labview_metadatas.append(
                metadata.LabviewNeuropixelMetadata.from_h5(session_dir)
            )
        return labview_metadatas

    @classmethod
    def fill(cls, probe_insertion, directories: List[Path]):
        record = dict(
            **probe_insertion,
        )

    def make(self, key):
        """Populates table with electrophysiology recording information."""
        raise NotImplementedError(
            "currently the way to get session filepaths is too flexible"
        )


@schema
class LFP(dj.Imported):
    """Extracts local field potentials (LFP) from an electrophysiology recording."""

    definition = """
    # Acquired local field potential (LFP) from a given Ephys recording.
    -> EphysRecording
    ---
    lfp_sampling_rate: float   # (Hz)
    lfp_time_stamps: longblob  # (s) timestamps with respect to the start of the recording (recording_timestamp)
    lfp_mean: longblob         # (uV) mean of LFP across electrodes - shape (time,)
    """

    class Electrode(dj.Part):
        """Saves local field potential data for each electrode."""

        definition = """
        -> master
        -> probe.ElectrodeConfig.Electrode
        ---
        lfp: longblob               # (uV) recorded lfp at this electrode
        """

    def make(self, key):
        pass


# ------------ Clustering --------------


@schema
class ClusteringMethod(dj.Lookup):
    """Kilosort clustering method."""

    definition = """
    # Method for clustering
    clustering_method: varchar(16)
    ---
    clustering_method_desc: varchar(1000)
    """

    contents = [
        ("kilosort2", "kilosort2 clustering method"),
        ("kilosort2.5", "kilosort2.5 clustering method"),
        ("kilosort3", "kilosort3 clustering method"),
        ("kilosort4", "kilosort4 clustering method"),
    ]


@schema
class ClusteringParamSet(dj.Lookup):
    """Parameters to be used in clustering procedure for spike sorting."""

    definition = """
    # Parameter set to be used in a clustering procedure
    paramset_idx:  smallint
    ---
    -> ClusteringMethod
    paramset_desc: varchar(128)
    param_set_hash: uuid
    unique index (param_set_hash)
    params: longblob  # dictionary of all applicable parameters
    """


# TODO: Will revisit the necessity of this, or put as a separate table
@schema
class ClusterQualityLabel(dj.Lookup):
    """Quality label for each spike sorted cluster."""

    definition = """
    # Quality
    cluster_quality_label:  varchar(100)  # cluster quality type - e.g. 'good', 'MUA', 'noise', etc.
    ---
    cluster_quality_description:  varchar(4000)
    """
    contents = [
        ("good", "single unit"),
        ("ok", "probably a single unit, but could be contaminated"),
        ("mua", "multi-unit activity"),
        ("noise", "bad unit"),
    ]


@schema
class ClusteringTask(dj.Manual):
    """A clustering task to spike sort electrophysiology datasets."""

    definition = """
    # Manual table for defining a clustering task ready to be run
    -> EphysRecording
    -> ClusteringParamSet
    ---
    clustering_output_dir='': varchar(255)  #  clustering output directory relative to the clustering root data directory
    task_mode='load': enum('load', 'trigger')  # 'load': load computed analysis results, 'trigger': trigger computation
    """


@schema
class Clustering(dj.Imported):
    """A processing table to handle each clustering task."""

    definition = """
    # Clustering Procedure
    -> ClusteringTask
    ---
    clustering_time: datetime  # time of generation of this set of clustering results
    package_version='': varchar(16)
    """


# Probably more layers above this are useful (for multiple users per curation, auto-curation maybe, etc.)
# Also further downstream to keep in mind what would be necessary to fully ingest phy (https://github.com/cortex-lab/phy)
#   "The [phy] GUI keeps track of all decisions in a file called phy.log"
@schema
class CurationType(dj.Lookup):  # Table definition subject to change
    definition = """
    # Type of curation performed on the clustering
    curation: varchar(16)
    ---
    """

    contents = zip(["no curation"])


# TODO: Would 0 mean no curation, or is a different design for the key better
@schema
class Curation(dj.Manual):
    """Curation procedure table."""

    definition = """
    # Manual curation procedure
    -> Clustering
    curation_id: int
    ---
    curation_time: datetime             # time of generation of this set of curated clustering results
    curation_output_dir: varchar(255)   # output directory of the curated results, relative to root data directory
    quality_control: bool               # has this clustering result undergone quality control?
    -> CurationType                     # what type of curation has been performed on this clustering result?
    curation_note='': varchar(2000)
    """


# TODO: Remove longblob types, replace with external-attach (or some form of this)
@schema
class CuratedClustering(dj.Imported):
    """Clustering results after curation."""

    definition = """
    # Clustering results of a curation.
    -> Curation
    """

    class Unit(dj.Part):
        """Single unit properties after clustering and curation."""

        definition = """
        # Properties of a given unit from a round of clustering (and curation)
        -> master
        unit: int
        ---
        -> probe.ElectrodeConfig.Electrode  # electrode with highest waveform amplitude for this unit
        -> ClusterQualityLabel
        spike_count : int         # how many spikes in this recording for this unit
        spike_times : longblob    # (s) spike times of this unit, relative to the start of the EphysRecording
        spike_sites : longblob   # array of electrode associated with each spike
        spike_depths=null : longblob  # (um) array of depths associated with each spike, relative to the (0, 0) of the probe
        """


@schema
class WaveformSet(dj.Imported):
    """A set of spike waveforms for units out of a given CuratedClustering."""

    definition = """
    # A set of spike waveforms for units out of a given CuratedClustering
    -> CuratedClustering
    """

    class PeakWaveform(dj.Part):
        """Mean waveform across spikes for a given unit."""

        definition = """
        # Mean waveform across spikes for a given unit at its representative electrode
        -> master
        -> CuratedClustering.Unit
        ---
        peak_electrode_waveform: longblob  # (uV) mean waveform for a given unit at its representative electrode
        """

    class Waveform(dj.Part):
        """Spike waveforms for a given unit."""

        definition = """
        # Spike waveforms and their mean across spikes for the given unit
        -> master
        -> CuratedClustering.Unit
        -> probe.ElectrodeConfig.Electrode
        ---
        waveform_mean: longblob   # (uV) mean waveform across spikes of the given unit
        waveforms=null: longblob  # (uV) (spike x sample) waveforms of a sampling of spikes at the given electrode for the given unit
        """


# important to note the original source of these quality metrics:
#   https://allensdk.readthedocs.io/en/latest/
#   https://github.com/AllenInstitute/ecephys_spike_sorting
#
@schema
class QualityMetrics(dj.Imported):
    """Clustering and waveform quality metrics."""

    definition = """
    # Clusters and waveforms metrics
    -> CuratedClustering
    """

    class Cluster(dj.Part):
        """Cluster metrics for a unit."""

        definition = """
        # Cluster metrics for a particular unit
        -> master
        -> CuratedClustering.Unit
        ---
        firing_rate=null: float # (Hz) firing rate for a unit
        snr=null: float  # signal-to-noise ratio for a unit
        presence_ratio=null: float  # fraction of time in which spikes are present
        isi_violation=null: float   # rate of ISI violation as a fraction of overall rate
        number_violation=null: int  # total number of ISI violations
        amplitude_cutoff=null: float  # estimate of miss rate based on amplitude histogram
        isolation_distance=null: float  # distance to nearest cluster in Mahalanobis space
        l_ratio=null: float  #
        d_prime=null: float  # Classification accuracy based on LDA
        nn_hit_rate=null: float  # Fraction of neighbors for target cluster that are also in target cluster
        nn_miss_rate=null: float # Fraction of neighbors outside target cluster that are in target cluster
        silhouette_score=null: float  # Standard metric for cluster overlap
        max_drift=null: float  # Maximum change in spike depth throughout recording
        cumulative_drift=null: float  # Cumulative change in spike depth throughout recording
        contamination_rate=null: float #
        """

    class Waveform(dj.Part):
        """Waveform metrics for a particular unit."""

        definition = """
        # Waveform metrics for a particular unit
        -> master
        -> CuratedClustering.Unit
        ---
        amplitude: float  # (uV) absolute difference between waveform peak and trough
        duration: float  # (ms) time between waveform peak and trough
        halfwidth=null: float  # (ms) spike width at half max amplitude
        pt_ratio=null: float  # absolute amplitude of peak divided by absolute amplitude of trough relative to 0
        repolarization_slope=null: float  # the repolarization slope was defined by fitting a regression line to the first 30us from trough to peak
        recovery_slope=null: float  # the recovery slope was defined by fitting a regression line to the first 30us from peak to tail
        spread=null: float  # (um) the range with amplitude above 12-percent of the maximum amplitude along the probe
        velocity_above=null: float  # (s/m) inverse velocity of waveform propagation from the soma toward the top of the probe
        velocity_below=null: float  # (s/m) inverse velocity of waveform propagation from the soma toward the bottom of the probe
        """
