# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 07:08:05 2024

Rishan Patel
Bioelectronics Group and Aspire Create

UCL

### Resting State Network Analysis ###
"""

import matplotlib as plt
import numpy as np 
import os
import glob
import scipy
import pickle

# Import the necessary functions from your parallel helpers file.
from parallelrsnhelpers import findSession, selectEEGChannels, PreProcess_parallel

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

    config = {}
    config['fs'] =  500      # Sampling frequency
    config['resample'] = 200 # Downsample Fs (not used in the parallel version)
    config['workingfs'] =  200
    config['plen'] = 400     # Processing Epoch
    config['class'] = 2      # Number of Classes 
    config['alias'] = 'GH'   # Patient Alias
    config['patient'] = GH   # List of Patient Data
    config['trainingduration'] = 36  # How many files of data (set it to something above 200 for all)
    config['channel'] = 'All'        # Option of 10-10, 10-20, Parietal or All 
    config['trainFiles'] = findSession(config)  # Track files used for config 
    config['locsdir'] = r"C:\Users\uceerjp\Desktop\G-W Data\Understanding Non-starionarity in GW Dataset\Understanding-Non-stationarity-over-2-Years-with-ALS"
    config['classselect'] = 'Rest'   # Either 'Rest', 'Up', 'Down', 'Up and Down', 'All'

    if config['alias'] == 'LS':
        config['bpf'] = [55,85]
        config['dir'] = r"C:\Users\uceerjp\Desktop\G-W Data\eegLS"
    elif config['alias'] == 'GH':
        config['bpf'] = [1,5]
        config['dir'] = r"C:\Users\uceerjp\Desktop\G-W Data\eegGH"

    # Use the parallel processing version to process the data.
    [data, config] = PreProcess_parallel(config)

    # Check that channel information was successfully retrieved.
    if config.get('channels') is None:
        raise ValueError("No channel information was returned. Please verify that the files are valid and processed correctly.")

    # Save the processed data and config into a file
    with open('RSN.pkl', 'wb') as f:
        pickle.dump((data, config), f)

    # Import plotting helper from the original rsnhelpers (which does not use parallelism)
    from rsnhelpers import process_S_and_save_matrix_session

    process_S_and_save_matrix_session(data, output_dir="RSN", channel_labels=config['channels']['Channel'])


if __name__ == '__main__':
    main()
