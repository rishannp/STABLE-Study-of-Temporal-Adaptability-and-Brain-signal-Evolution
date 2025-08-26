# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 07:08:05 2024

parallelMain.py

Rishan Patel
Bioelectronics Group and Aspire Create

UCL

### Resting State Network Analysis ###
"""

import matplotlib.pyplot as plt   # fixed
import numpy as np
import os
import glob
import scipy
import pickle

# Import the necessary functions from your parallel helpers file.
from parallelrsnhelpers import findSession, PreProcess_parallel, process_S_and_save_matrix_session

def main():
    folder2 = r'C:\Users\uceerjp\Desktop\G-W Data\eegGH'
    folder1 = r'C:\Users\uceerjp\Desktop\G-W Data\eegLS'

    # Get list of files (excluding directories)
    filesInFolder1 = [f for f in glob.glob(os.path.join(folder1, '*')) if os.path.isfile(f)]
    filesInFolder2 = [f for f in glob.glob(os.path.join(folder2, '*')) if os.path.isfile(f)]

    # Extract the file names from the full paths
    LS = [os.path.basename(f) for f in filesInFolder1]
    GH = [os.path.basename(f) for f in filesInFolder2]

    # Clear unnecessary variables if needed
    del filesInFolder1, filesInFolder2, folder1, folder2

    # ===== CONFIG =====
    # NOTE: 'resample' is the processing sampling rate you want.
    config = {
        'fs': 500,                # raw sampling frequency
        'resample': 200,          # processing sampling rate (downsample target)
        'class': 2,               # keep if you want for future; not used in RSN calc
        'alias': 'LS',            # 'LS' or 'GH'
        'patient': LS,            # list of filenames for selected alias
        'trainingduration': 9999, # how many sessions (use 9999 for all)
        'channel': 'All',         # '10-20'|'10-10'|'Parietal'|'All'
        'locsdir': r"C:\Users\uceerjp\Desktop\G-W Data\Understanding Non-starionarity in GW Dataset\Understanding-Non-stationarity-over-2-Years-with-ALS",
        'classselect': 'Rest',
        'plots': 1,

        # Preprocessing
        'preproc': {
            'bad_channel_reject': True,
            'bad_chan_zthr': 5.0,
            'bandpass_full': [1, 97],   # full-range BPF before band splits
            'apply_car': True           # CAR on by default
        },

        # Windowing (non-overlapping 5s)
        'window': {
            'length_sec': 5.0,
            'overlap': 0.0
        },

        # Features to compute
        'features': {
            'fc': ['plv', 'icoh'],   # only PLV + iCoh
            'icoh_method' : 'fft', # fft fast or welch slow
            'icoh_freq_decimantion': 1,
            'power': True,           # band power per band
            'criticality': { 'lzc': True, 'ple': True}          # broadband 2–40 Hz per window
        },

        # Bands (global control)
        'bands': {
            '2-4':  [2, 4],
            '4-7':  [4, 7],
            '7-13': [7, 13],
            '13-30':[13, 30],
            '30-47':[30, 47],
            '53-97':[53, 97]
        },

        # Placeholder for future source localization (kept minimal)
        'source_localization': {
            'enable': False,
            'method': 'LCMV'
        }
    }

    if config['alias'] == 'LS':
        config['bpf'] = [55, 85]
        config['dir'] = r"C:\Users\uceerjp\Desktop\G-W Data\eegLS"
        config['patient'] = LS
    elif config['alias'] == 'GH':
        config['bpf'] = [1, 5]
        config['dir'] = r"C:\Users\uceerjp\Desktop\G-W Data\eegGH"
        config['patient'] = GH

    # Use the parallel processing version to process the data.
    S, config = PreProcess_parallel(config)

    # Check that channel information was successfully retrieved.
    if config.get('channels') is None:
        raise ValueError("No channel information was returned. Please verify that the files are valid and processed correctly.")

    # Save the processed data and config into a file
    with open(f"RSN_{config['alias']}.pkl", 'wb') as f:
        pickle.dump((S, config), f)

    # Plots (includes: FC mean heatmaps + per-window heatmaps; and dataset-level time-series plots)
    process_S_and_save_matrix_session(S, output_dir="RSN", channel_labels=config['channels']['Channel'], config=config)


if __name__ == '__main__':
    main()
