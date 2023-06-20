def default_kilosort_parameters():
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