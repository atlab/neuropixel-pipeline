# flake8: noqa

import gc

from decimal import Decimal

import datajoint as dj
import numpy as np
import datetime

from neuropixel_pipeline.api.clustering import ClusteringTaskRunner
from . import probe
from ..api import metadata
from .. import utils
from ..readers import labview, kilosort
from pathlib import Path
from typing import List
from pydantic import BaseModel, PositiveInt, constr, validate_call
from pydantic.dataclasses import dataclass

from neuropixel_pipeline import readers


schema = dj.schema("neuropixel_ephys")

# TODO: Define config stores (pydantic serialization is a better alternative to the adapter side of things though)
# TODO: Also, decide whether the external blob is fine rather than file paths (or find an abstraction over filepaths & the data types we're interested in, basically everthing with longblob)
#       Maybe just some identifier based on Class (Table) name + field name?
# stores = {"": {}}
# dj_utils.StoresConfig(stores).set_dj_config()
# TODO: YOOOOOOOO, I should have the pydantic adapters either match (or at least have names that point to) their field names, like adapters.lfp_mean or adapters['lfp_mean']
#       this would make it really easy to know how to fetch an filepath datatype! If two different fields have the same adapter internally, doesn't matter they can have separate
#       names that point to the same internal adapter. So like adapters.lfp_mean and adapters.lfp might just be pointing to the same adapter, but have two different ways to get there.


class PopulateHelper: # TODO: Add a discriminant (pydantic supports these) or enum for changing how Populate gets run
    def run(data: dict):
        AcquisitionSoftware  # no populate necessary

        probe_insertion = dict(
            session_id=0,
            insertion_number=0,
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


    class InitialSetup(BaseModel, from_attributes=True):
        acq_software: constr(max_length=24)

    class PreClusteringData(BaseModel, from_attributes=True):
        session_id: PositiveInt
        probe: metadata.ProbeData

    class ClusteringData(BaseModel, from_attributes=True):
        pass

    class QualityMetricsData(BaseModel, from_attributes=True):
        pass

    


### ----------------------------- Table declarations ----------------------


# ------------ Pre-Clustering --------------


@schema
class Session(dj.Manual):
    """Session key"""

    definition = """
    # Session: table connection
    session_id : int auto_increment # Session primary key hash
    ---
    animal_id=null: int unsigned # animal id
    session=null: smallint unsigned # original session id
    scan_idx=null: smallint unsigned # scan idx
    """

    @classmethod
    def add_session(cls, session_meta, skip_duplicates=True):
        if not cls & session_meta:
            cls.insert1(session_meta)


@schema
class SkullReference(dj.Lookup):
    """Reference area from the skull"""

    definition = """
    skull_reference   : varchar(60)
    """
    contents = zip(["Bregma", "Lambda"])


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
    -> Session
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
    -> SkullReference
    ap_location: decimal(6, 2) # (um) anterior-posterior; ref is 0; more anterior is more positive
    ml_location: decimal(6, 2) # (um) medial axis; ref is 0 ; more right is more positive
    depth:       decimal(6, 2) # (um) manipulator depth relative to surface of the brain (0); more ventral is more negative
    theta=null:  decimal(5, 2) # (deg) - elevation - rotation about the ml-axis [0, 180] - w.r.t the z+ axis
    phi=null:    decimal(5, 2) # (deg) - azimuth - rotation about the dv-axis [0, 360] - w.r.t the x+ axis
    beta=null:   decimal(5, 2) # (deg) rotation about the shank of the probe [-180, 180] - clockwise is increasing in degree - 0 is the probe-front facing anterior
    """


@schema
class EphysFile(dj.Manual):
    """File directory for ephys sessions"""

    definition = """
    # File directories for ephys sessions
    file_path: varchar(255) # file path or directory for an ephys session
    ---
    -> AcquisitionSoftware
    """


@schema
class EphysRecording(dj.Imported):
    """Automated table with electrophysiology recording information for each probe inserted during an experimental session."""

    definition = """
    # Ephys recording from a probe insertion for a given session.
    -> ProbeInsertion
    -> EphysFile
    ---
    -> probe.ElectrodeConfig
    sampling_rate: float # (Hz)
    recording_datetime=null: datetime # datetime of the recording from this probe
    recording_duration=null: float # (seconds) duration of the recording from this probe
    """

    def make(self, key):
        """Populates table with electrophysiology recording information."""
        session_dir = Path(key['file_path'])
        acq_software = (EphysFile & key).fetch1('acq_software')

        inserted_probe_serial_number = (ProbeInsertion * probe.Probe & key).fetch1(
            "probe"
        )

        # search session dir and determine acquisition software

        # supported_probe_types = probe.ProbeType.fetch("probe_type")

        if acq_software == "LabviewV1":
            labview_meta = labview.LabviewNeuropixelMeta.from_h5(session_dir)
            if not str(labview_meta.serial_number) == inserted_probe_serial_number:
                raise FileNotFoundError(
                    "No Labview data found for probe insertion: {}".format(key)
                )

            probe_type = (probe.Probe & dict(probe=labview_meta.serial_number)).fetch1("probe_type")
            
            electrode_config_hash_key = probe.ElectrodeConfig.add_new_config(labview_meta, probe_type=probe_type)

            self.insert1(
                {
                    **key,
                    **electrode_config_hash_key,
                    "acq_software": acq_software,
                    "sampling_rate": labview_meta.sampling_rate,
                    "recording_datetime": None,
                    "recording_duration": None,
                }, ignore_extra_fields=True
            )
        else:
            raise NotImplementedError(
                f"Processing ephys files from"
                f" acquisition software of type {acq_software} is"
                f" not yet implemented"
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
        # based on acq_software switch to 
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
    paramset_idx:  smallint auto_increment
    ---
    -> ClusteringMethod
    paramset_desc: varchar(128)
    paramset_hash: uuid
    unique index (paramset_hash)
    params: longblob  # dictionary of all applicable parameters
    """

    @classmethod
    def fill(cls, params:dict, clustering_method:str, description:str="", skip_duplicates=False):
        params_uuid = utils.dict_to_uuid(params)
        cls.insert1(dict(
            clustering_method=clustering_method,
            paramset_desc=description,
            paramset_hash=params_uuid,
            params=params,
        ), skip_duplicates=skip_duplicates)


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

    def make(self, key):
        source_key = (ClusteringTask & key).fetch1()
        task_runner = ClusteringTaskRunner(**source_key)
        time_finish = task_runner.trigger_clustering() # .load_time_finished()
        current_time = datetime.datetime.now() # FIXME: this should be time of the clustering, which isn't always triggered

        self.insert1(dict(
            **source_key,
            clustering_time=current_time,
        ), ignore_extra_fields=True)


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
        unit_id: int
        ---
        -> probe.ElectrodeConfig.Electrode  # electrode with highest waveform amplitude for this unit
        -> ClusterQualityLabel
        spike_count : int         # how many spikes in this recording for this unit
        spike_times : longblob    # (s) spike times of this unit, relative to the start of the EphysRecording
        spike_sites : longblob   # array of electrode associated with each spike
        spike_depths=null : longblob  # (um) array of depths associated with each spike, relative to the (0, 0) of the probe
        """
    
    def make(self, key):
        """Automated population of Unit information."""
        
        kilosort_path = (ClusteringTask & key).fetch1('clustering_output_dir')
        # kilosort_path = (Curation & key).fetch1('curation_output_dir')
        kilosort_dataset = kilosort.Kilosort(kilosort_path)
        sample_rate = (EphysRecording & key).fetch1("sampling_rate")

        sample_rate = kilosort_dataset.data["params"].get("sample_rate", sample_rate)

        # ---------- Unit ----------
        # -- Remove 0-spike units
        withspike_idx = [
            i
            for i, u in enumerate(kilosort_dataset.data["cluster_ids"])
            if (kilosort_dataset.data["spike_clusters"] == u).any()
        ]
        valid_units = kilosort_dataset.data["cluster_ids"][withspike_idx]
        valid_unit_labels = kilosort_dataset.data["cluster_groups"][withspike_idx]

        # -- Spike-times --
        # spike_times_sec_adj > spike_times_sec > spike_times
        spike_time_key = (
            "spike_times_sec_adj"
            if "spike_times_sec_adj" in kilosort_dataset.data
            else "spike_times_sec"
            if "spike_times_sec" in kilosort_dataset.data
            else "spike_times"
        )
        spike_times = kilosort_dataset.data[spike_time_key]
        kilosort_dataset.extract_spike_depths()

        # -- Spike-sites and Spike-depths --
        spike_sites = kilosort_dataset.data["spike_sites"]
        spike_depths = kilosort_dataset.data["spike_depths"]

        labview_metadata = labview.LabviewNeuropixelMeta.from_h5(key['file_path'])
        electrode_config_hash = labview_metadata.electrode_config_hash()

        probe_type = (probe.Probe & dict(serial_number=labview_metadata.serial_number)).fetch1('probe_type')

        # -- Insert unit, label, peak-chn
        units = []
        for unit, unit_lbl in zip(valid_units, valid_unit_labels):
            if (kilosort_dataset.data["spike_clusters"] == unit).any():
                unit_channel, _ = kilosort_dataset.get_best_channel(unit)
                unit_spike_times = (
                    spike_times[kilosort_dataset.data["spike_clusters"] == unit]
                    / sample_rate
                )
                spike_count = len(unit_spike_times)

                units.append(
                    {
                        "unit_id": unit,
                        "cluster_quality_label": unit_lbl,
                        "electrode_config_hash": electrode_config_hash,
                        "probe_type": probe_type,
                        "electrode": unit_channel,
                        "spike_times": unit_spike_times,
                        "spike_count": spike_count,
                        "spike_sites": spike_sites[
                            kilosort_dataset.data["spike_clusters"] == unit
                        ],
                        "spike_depths": spike_depths[
                            kilosort_dataset.data["spike_clusters"] == unit
                        ]
                        if spike_depths is not None
                        else None,
                    }
                )

        self.insert1(key)
        self.Unit.insert([{**key, **u} for u in units])

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

    def make(self, key):
        """Populates waveform tables."""
        output_dir = (Curation & key).fetch1("curation_output_dir")
        kilosort_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        kilosort_dataset = kilosort.Kilosort(kilosort_dir)

        acq_software, probe_serial_number = (
            EphysRecording * ProbeInsertion & key
        ).fetch1("acq_software", "probe")

        # -- Get channel and electrode-site mapping
        recording_key = (EphysRecording & key).fetch1("KEY")
        channel2electrodes = get_neuropixels_channel2electrode_map(
            recording_key, acq_software
        )

        is_qc = (Curation & key).fetch1("quality_control")

        # Get all units
        units = {
            u["unit"]: u
            for u in (CuratedClustering.Unit & key).fetch(as_dict=True, order_by="unit")
        }

        if is_qc:
            unit_waveforms = np.load(
                kilosort_dir / "mean_waveforms.npy"
            )  # unit x channel x sample

            def yield_unit_waveforms():
                for unit_no, unit_waveform in zip(
                    kilosort_dataset.data["cluster_ids"], unit_waveforms
                ):
                    unit_peak_waveform = {}
                    unit_electrode_waveforms = []
                    if unit_no in units:
                        for channel, channel_waveform in zip(
                            kilosort_dataset.data["channel_map"], unit_waveform
                        ):
                            unit_electrode_waveforms.append(
                                {
                                    **units[unit_no],
                                    **channel2electrodes[channel],
                                    "waveform_mean": channel_waveform,
                                }
                            )
                            if (
                                channel2electrodes[channel]["electrode"]
                                == units[unit_no]["electrode"]
                            ):
                                unit_peak_waveform = {
                                    **units[unit_no],
                                    "peak_electrode_waveform": channel_waveform,
                                }
                    yield unit_peak_waveform, unit_electrode_waveforms

        else:
            if acq_software == "SpikeGLX":
                spikeglx_meta_filepath = get_spikeglx_meta_filepath(key)
                neuropixels_recording = spikeglx.SpikeGLX(spikeglx_meta_filepath.parent)
            elif acq_software == "Open Ephys":
                session_dir = find_full_path(
                    get_ephys_root_data_dir(), get_session_directory(key)
                )
                openephys_dataset = openephys.OpenEphys(session_dir)
                neuropixels_recording = openephys_dataset.probes[probe_serial_number]

            def yield_unit_waveforms():
                for unit_dict in units.values():
                    unit_peak_waveform = {}
                    unit_electrode_waveforms = []

                    spikes = unit_dict["spike_times"]
                    waveforms = neuropixels_recording.extract_spike_waveforms(
                        spikes, kilosort_dataset.data["channel_map"]
                    )  # (sample x channel x spike)
                    waveforms = waveforms.transpose(
                        (1, 2, 0)
                    )  # (channel x spike x sample)
                    for channel, channel_waveform in zip(
                        kilosort_dataset.data["channel_map"], waveforms
                    ):
                        unit_electrode_waveforms.append(
                            {
                                **unit_dict,
                                **channel2electrodes[channel],
                                "waveform_mean": channel_waveform.mean(axis=0),
                                "waveforms": channel_waveform,
                            }
                        )
                        if (
                            channel2electrodes[channel]["electrode"]
                            == unit_dict["electrode"]
                        ):
                            unit_peak_waveform = {
                                **unit_dict,
                                "peak_electrode_waveform": channel_waveform.mean(
                                    axis=0
                                ),
                            }

                    yield unit_peak_waveform, unit_electrode_waveforms

        # insert waveform on a per-unit basis to mitigate potential memory issue
        self.insert1(key)
        for unit_peak_waveform, unit_electrode_waveforms in yield_unit_waveforms():
            if unit_peak_waveform:
                self.PeakWaveform.insert1(unit_peak_waveform, ignore_extra_fields=True)
            if unit_electrode_waveforms:
                self.Waveform.insert(unit_electrode_waveforms, ignore_extra_fields=True)

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
    
    def make(self, key):
        """Populates tables with quality metrics data."""
        import pandas as pd

        kilosort_dir = (ClusteringTask & key).fetch1("file_path")

        metric_fp = kilosort_dir / "metrics.csv"
        rename_dict = {
            "isi_viol": "isi_violation",
            "num_viol": "number_violation",
            "contam_rate": "contamination_rate",
        }

        if not metric_fp.exists():
            raise FileNotFoundError(f"QC metrics file not found: {metric_fp}")

        metrics_df = pd.read_csv(metric_fp)
        metrics_df.set_index("cluster_id", inplace=True)
        metrics_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        metrics_df.columns = metrics_df.columns.str.lower()
        metrics_df.rename(columns=rename_dict, inplace=True)
        metrics_list = [
            dict(metrics_df.loc[unit_key["unit"]], **unit_key)
            for unit_key in (CuratedClustering.Unit & key).fetch("KEY")
        ]

        self.insert1(key)
        self.Cluster.insert(metrics_list, ignore_extra_fields=True)
        self.Waveform.insert(metrics_list, ignore_extra_fields=True)