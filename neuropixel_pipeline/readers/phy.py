import re
from datetime import datetime

import numpy as np
import pandas as pd


def extract_clustering_info(cluster_output_dir):
    creation_time = None

    phy_curation_indicators = [
        "Merge clusters",
        "Split cluster",
        "Change metadata_group",
    ]
    # ---- Manual curation? ----
    phylog_filepath = cluster_output_dir / "phy.log"
    if phylog_filepath.exists():
        phylog = pd.read_fwf(phylog_filepath, colspecs=[(6, 40), (41, 250)])
        phylog.columns = ["meta", "detail"]
        curation_row = [
            bool(re.match("|".join(phy_curation_indicators), str(s)))
            for s in phylog.detail
        ]
        is_curated = bool(np.any(curation_row))
        if creation_time is None and is_curated:
            row_meta = phylog.meta[np.where(curation_row)[0].max()]
            datetime_str = re.search("\d{2}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}", row_meta)
            if datetime_str:
                creation_time = datetime.strptime(
                    datetime_str.group(), "%Y-%m-%d %H:%M:%S"
                )
            else:
                creation_time = datetime.fromtimestamp(phylog_filepath.stat().st_ctime)
                time_str = re.search("\d{2}:\d{2}:\d{2}", row_meta)
                if time_str:
                    creation_time = datetime.combine(
                        creation_time.date(),
                        datetime.strptime(time_str.group(), "%H:%M:%S").time(),
                    )
    else:
        is_curated = False

    # ---- Quality control? ----
    metric_filepath = cluster_output_dir / "metrics.csv"
    is_qc = metric_filepath.exists()
    if is_qc:
        if creation_time is None:
            creation_time = datetime.fromtimestamp(metric_filepath.stat().st_ctime)

    if creation_time is None:
        spiketimes_filepath = next(cluster_output_dir.glob("spike_times.npy"))
        creation_time = datetime.fromtimestamp(spiketimes_filepath.stat().st_ctime)

    return creation_time, is_curated, is_qc