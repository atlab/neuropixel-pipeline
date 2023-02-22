"""
pykilosort is currently only available as a git repo and requires cuda
https://github.com/MouseLand/pykilosort
"""

import pathlib
import pykilosort

def run_pykilosort(
    continuous_file,
    kilosort_output_directory,
    params,
    channel_ind,
    x_coords,
    y_coords,
    shank_ind,
    connected,
    sample_rate,
):
    dat_path = pathlib.Path(continuous_file)

    probe = pykilosort.Bunch()
    channel_count = len(channel_ind)
    probe.Nchan = channel_count
    probe.NchanTOT = channel_count
    probe.chanMap = np.arange(0, channel_count, dtype="int")
    probe.xc = x_coords
    probe.yc = y_coords
    probe.kcoords = shank_ind

    pykilosort.run(
        dat_path=continuous_file,
        dir_path=dat_path.parent,
        output_dir=kilosort_output_directory,
        probe=probe,
        params=params,
        n_channels=probe.Nchan,
        dtype=np.int16,
        sample_rate=sample_rate,
    )