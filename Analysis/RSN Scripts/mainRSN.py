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

from rsnhelpers import findSession, selectEEGChannels, PreProcess

# folder2 = r'C:\Users\uceerjp\Desktop\G-W Data\eegGH' # local directory 
# folder1 =  r'C:\Users\uceerjp\Desktop\G-W Data\eegLS' # local directory

folder2 = r'/home/uceerjp/GW/GH/' # server directory
folder1 = r'/home/uceerjp/GW/LS/' # server directory

# Get list of files (excluding directories)
filesInFolder1 = [f for f in glob.glob(os.path.join(folder1, '*')) if os.path.isfile(f)]
filesInFolder2 = [f for f in glob.glob(os.path.join(folder2, '*')) if os.path.isfile(f)]

# Extract the file names from the full paths
LS = [os.path.basename(f) for f in filesInFolder1]
GH = [os.path.basename(f) for f in filesInFolder2]

# Clear unnecessary variables if needed
del filesInFolder1, filesInFolder2, folder1, folder2


config = {}
config['fs'] =  500 ## Fs
config['resample'] = 200 # Downsample Fs
config['workingfs'] =  200
config['plen'] = 400 # Processing Epoch
config['class'] = 2 # Number of Classes 
config['alias'] = 'GH' # Patient Alias
config['patient'] = GH # List of Patient Data
config['trainingduration'] = 36 #How many files of data (set it to something above 200 for all)
config['channel'] = 'All' # Option of 10-10, 10-20, Patietal or All 
config['trainFiles'] = findSession(config) # Track files used for config 
# config['locsdir'] = r"C:\Users\uceerjp\Desktop\G-W Data\Understanding Non-starionarity in GW Dataset\Understanding-Non-stationarity-over-2-Years-with-ALS"
config['locsdir'] = r"/home/uceerjp/GW/"
config['classselect'] = 'Rest' # Either 'Rest', 'Up', 'Down', 'Up and Down', 'All'

if config['alias'] == 'LS':
    config['bpf'] = [55,85]
    #config['dir'] = r"C:\Users\uceerjp\Desktop\G-W Data\eegLS"
    config['dir'] = r'/home/uceerjp/GW/LS/' # server directory
elif config['alias'] == 'GH':
    config['bpf'] = [1,5]
    #config['dir'] = r"C:\Users\uceerjp\Desktop\G-W Data\eegGH"
    config['dir'] = r'/home/uceerjp/GW/GH/' # server directory
 
[data,config] = PreProcess(config) # Here lies the data for whatever channels you want to see :) 

# Save the processed data and config into a file
with open('RSN.pkl', 'wb') as f:
    pickle.dump((data, config), f)

#% Plot and save PLV for global structures
from rsnhelpers import process_S_and_save_matrix_session

process_S_and_save_matrix_session(data, output_dir="RSN", channel_labels=config['channels']['Channel'])
