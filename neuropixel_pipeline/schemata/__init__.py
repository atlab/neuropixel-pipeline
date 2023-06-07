import datajoint as dj

dj.config["enable_python_native_blobs"] = True

from . import probe  # noqa: E402
from . import ephys  # noqa: E402

from pydantic import BaseModel, constr, computed_field  # noqa: E402
from typing import List, Optional  # noqa: E402
from pathlib import Path  # noqa: E402
from ..readers import labview  # noqa: E402

from .. import api  # noqa: E402


class PopulateHelper:
    class Setup(BaseModel):
        probes: List[api.metadata.ProbeData] = None
        skip_duplicates: bool = False

        def run(self):
            probe.ProbeType.fill_neuropixel_probes()
            probe.Probe.insert(
                [probe.model_dump() for probe in self.probes],
                skip_duplicates=self.skip_duplicates,
            )
            PopulateHelper.Clustering.new_parameters(
                PopulateHelper.Clustering.default_parameters(),
                clustering_method="kilosort4",
                description="default kilosort4 params",
            )

    class PreClustering(BaseModel):
        session_key: api.metadata.SessionKey
        insertion_number: int
        session_dir: Path
        acq_software: constr(max_length=24)
        insertion_location: Optional[api.preclustering.InsertionData] = None
        skip_duplicates: bool = False

        def run(self):
            # Autoincrement new session
            ephys.Session.add_session(self.session_key.model_dump())
            session_id = ephys.Session.get_session_id(
                self.session_key.model_dump()
            ).fetch1("session_id")

            labview_metadata = labview.LabviewNeuropixelMeta.from_h5(self.session_dir)

            insertion_key = dict(
                session_id=session_id, insertion_number=self.insertion_number
            )
            ephys.ProbeInsertion.insert1(
                dict(
                    **insertion_key,
                    probe=labview_metadata.serial_number,
                ),
                skip_duplicates=self.skip_duplicates,
            )

            if self.insertion_location is not None:
                ephys.InsertionLocation.insert1(self.insertion_location.model_dump())

            ephys.EphysFile.insert1(
                dict(
                    file_path=self.session_dir,
                    acq_software=self.acq_software,
                ),
                skip_duplicates=self.skip_duplicates,
            )

            ephys.EphysRecording.populate()

            return insertion_key

    class Clustering(BaseModel):
        insertion_key: dict
        session_dir: Path
        paramset_idx: int
        task_mode: api.clustering.ClusteringTaskMode = "load"

        @computed_field
        @property
        def task_source_key(self) -> dict:
            return dict(
                **self.insertion_key,
                file_path=str(self.session_dir),
                paramset_idx=self.paramset_idx,
            )

        @staticmethod
        def new_parameters(
            settings: dict,
            clustering_method: str,
            description: str,
            skip_duplicates=True,
        ):
            ephys.ClusteringParamSet.fill(
                params=settings,
                clustering_method=clustering_method,
                description=description,
                skip_duplicates=skip_duplicates,
            )

        @staticmethod
        def default_parameters():
            settings = {
                "NchanTOT": 385,
                "fs": 30000,
                "nt": 61,
                "Th": 8,
                "spkTh": 8,
                "Th_detect": 9,
                "nwaves": 6,
                "nskip": 25,
                "nblocks": 5,
                "binning_depth": 5,
                "sig_interp": 20,
                "probe_name": "neuropixPhase3B1_kilosortChanMap.mat",
            }
            settings["nt0min"] = int(20 * settings["nt"] / 61)
            settings["NT"] = 2 * settings["fs"]
            settings["n_chan_bin"] = settings["NchanTOT"]
            return settings

        def run_clustering(self, clustering_output_dir=Path("spikes_kilosort4")):
            task_source_rel = (ephys.EphysRecording & self.insertion_key).proj() * (
                ephys.ClusteringParamSet() & {"paramset_idx": self.paramset_idx}
            ).proj()
            task_source_key = task_source_rel.fetch1()
            task_source_key["clustering_output_dir"] = (
                clustering_output_dir
                if clustering_output_dir.is_absolute()
                else self.session_dir / clustering_output_dir
            )
            ephys.ClusteringTask.insert1(task_source_key, skip_duplicates=True)

            ephys.Clustering.populate()

        def run_curation(self, curation_input: api.clustering.CurationInput):
            curation_input = api.clustering.CurationInput.model_validate(curation_input)
            curation_source_key = (
                (ephys.Clustering() & self.task_source_key).proj().fetch1()
            )
            if curation_input.curation_output_dir is None:
                curation_input.curation_output_dir = curation_source_key["file_path"]
            ephys.Curation.insert1(
                dict(
                    **curation_source_key,
                    **curation_input.model_dump(),
                ),
                skip_duplicates=True,
            )

            ephys.CuratedClustering.populate()

    class PostClustering(BaseModel):
        def run(self, *restrictions):
            ephys.QualityMetrics.populate(*restrictions)
            ephys.WaveformSet.populate(*restrictions)

    def run(self):
        raise NotImplementedError


__all__ = ["probe", "ephys"]
