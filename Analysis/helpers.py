# -*- coding: utf-8 -*-
"""
Functions for G-W Dataset Analysis
"""
import matplotlib as plt
import numpy as np 
import os
import glob
import scipy
from os.path import dirname, join as pjoin

def findSession(config):
    x = config['trainingduration']  # Number of sessions to consider
    a = config['patient']           # List containing filenames of patient data

    # Extract session numbers from the filenames in config['patient']
    b = [int(patient[4:6]) for patient in config['patient']]  # Extract session number from characters 5 and 6 and convert to int

    # Find the minimum session number
    c = min(b)

    idx = []
    missing_count = 0   # Counter for consecutive missing sessions
    max_missing = 2     # Maximum allowed consecutive missing sessions

    # Loop to find indices of the first 'x' sessions in sequence
    i = 0
    while i < x:
        value_to_search = c + i   # Calculate session number to search for

        if value_to_search in b:
            # Find indices where b equals value_to_search
            d = [j for j, val in enumerate(b) if val == value_to_search]

            # Append the found indices to idx
            idx.extend(d)

            # Reset the missing count as we found a session
            missing_count = 0
        else:
            # Handle case where value_to_search is not found in b
            print(f'Session {value_to_search} not found in array b')

            # Increase the missing count
            missing_count += 1

            # Check if consecutive missing sessions exceed the limit
            if missing_count >= max_missing:
                print('Reached maximum consecutive missing sessions. Stopping search.')
                break

        # Move to the next session
        i += 1

    # Select filenames based on the found indices
    f = [a[i] for i in idx]

    return f

def selectEEGChannels(data, configType, locsdir):
    # Load the .mat file
    filename = pjoin(locsdir, "EEGLabLocsMPICap.mat")
    loaded_data = scipy.io.loadmat(filename)
    
    # Extract channel information from the .mat file
    if 'Chanlocs' not in loaded_data:
        raise ValueError("The loaded .mat file does not contain 'Chanlocs'. Check the file format.")
    
    locs_struct = loaded_data['Chanlocs']

    # Extract labels and coordinates
    try:
        all_labels = [str(locs_struct['labels'][0, i][0]) for i in range(locs_struct['labels'].size)]
        all_coords = np.array([[locs_struct['X'][0, i][0, 0], 
                                locs_struct['Y'][0, i][0, 0], 
                                locs_struct['Z'][0, i][0, 0]] for i in range(locs_struct['X'].size)])
    except KeyError as e:
        raise ValueError(f"Key {e} not found in Chanlocs. Verify the structure of the .mat file.")

    # Define channel configurations
    channel_names_10_20 = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'P7/T5', 'P3', 'Pz', 'P4', 'P8/T6',
        'O1', 'O2'
    ]

    channel_names_10_10 = [
        'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
        'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
        'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
        'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
        'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
        'P7/T5', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8/T6',
        'PO7', 'PO3', 'POz', 'PO4', 'PO8',
        'O1', 'Oz', 'O2', 'Iz'
    ]

    parietal_channels = [
        'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7/T5', 'P8/T6', 'Pz',
        'PO1', 'PO2', 'PO3', 'PO4', 'POz', 'O1', 'Oz', 'O2'
    ]

    # Select the configuration
    if configType == '10-20':
        selected_channels = channel_names_10_20
    elif configType == '10-10':
        selected_channels = channel_names_10_10
    elif configType == 'Parietal':
        selected_channels = parietal_channels
    elif configType == 'All':
        selected_channels = all_labels  # Use all available channels
    else:
        raise ValueError("Invalid configuration type. Choose from '10-20', '10-10', 'Parietal', or 'All'.")

    # Match channels with indices and coordinates
    matched_channels = []
    selected_indices = []
    matched_coords = []

    for channel in selected_channels:
        if channel in all_labels:
            idx = all_labels.index(channel)
            selected_indices.append(idx)
            matched_channels.append(channel)
            matched_coords.append(all_coords[idx])
        else:
            print(f"Warning: Channel {channel} not found in the dataset.")

    if not matched_channels:
        raise ValueError("No matching channels found for the selected configuration.")

    # Select corresponding data
    selected_data = data[:, selected_indices]
    
    # Create structured output
    selected_channels_info = {
        'Channel': matched_channels,
        'X': [coord[0] for coord in matched_coords],
        'Y': [coord[1] for coord in matched_coords],
        'Z': [coord[2] for coord in matched_coords]
    }

    return selected_data, selected_channels_info

import scipy.io
import numpy as np
import warnings
from scipy.signal import resample, butter, filtfilt

def PreProcess(config):
    f = findSession(config)

    # Initialize a list to hold all the loaded and processed data
    allData = []

    # Iterate over each file in the list f
    for file in f:
        # Load the current file
        data_dir = config['dir']
        filename = pjoin(data_dir, file)
        data = scipy.io.loadmat(filename)

        # Check if the loaded data contains the expected fields
        if 'parameters' in data and 'signal' in data and 'states' in data:
            # Extract signal and states
            signal = data['signal']
            states = data['states']

            # Select EEG channels
            signal, channels = selectEEGChannels(signal, config['channel'], config['locsdir'])
            
            # Bandpass filter the signal
            signal = bandpass(signal, config['bpf'], config['fs'])
            #print(signal.shape)
            
            # Resample the signal (computational redundancy)
            signal = resample(signal, int(signal.shape[0] * config['workingfs'] / config['fs']), axis=0)
            
            #print(signal.shape)
            
            # Flatten the label_data to make sure it's 1D
            label_data = states['PresentationPhase'].flatten()
            label_data = label_data[0]
            
            # Ensure label_data is 1D
            #print("Shape of label_data:", label_data.shape)  # Expected output: (152560,)
            
            # Resample the flattened label_data
            label = resample(label_data, int(label_data.shape[0] * config['workingfs'] / config['fs']), axis=0)
            
            # Debug print to check the shape after resampling
            #print("Shape of resampled label:", label.shape)

            label = np.round(label)
            # label = label + 1  # Makes Re = 1, Up = 2, Down = 3

            # Store the processed data in allData
            allData.append({
                'parameters': data['parameters'],
                'signal': signal,
                'states': states,
                'label': label,
                'channels': channels
            })
        else:
            # Issue a warning if the file does not contain the expected fields
            warnings.warn(f'File {file} does not contain the expected fields.')

    #print(allData[0]['signal'].shape)
    S = []
    config['channels'] = allData[0]['channels']
    for data in allData:
        X = split_eeg_by_labels(data['signal'], data['label'], config['workingfs'])
        S.extend(X)

    # S = LRvRe(S)

    return S, config

def bandpass(data, bpf, fs):
    """
    Applies a bandpass filter to EEG data channel-wise.

    Parameters:
    - data: ndarray
        EEG data of shape (samples, channels).
    - bpf: list or tuple
        Bandpass frequency range [low_cutoff, high_cutoff].
    - fs: float
        Sampling frequency of the EEG data.

    Returns:
    - filtered_data: ndarray
        Bandpass-filtered EEG data with the same shape as the input.
    """
    # Design the bandpass filter
    b, a = butter(4, [bpf[0], bpf[1]], fs=fs, btype='band')
    
    # Initialize an array to store filtered data
    filtered_data = np.zeros_like(data)
    
    # Apply the filter to each channel (column) in the data
    for ch in range(data.shape[1]):
        filtered_data[:, ch] = filtfilt(b, a, data[:, ch])
    
    return filtered_data


import numpy as np
from scipy.signal import find_peaks

def split_eeg_by_labels(eeg, target,fs):
    """
    Splits EEG data into segments for each class based on the labels.

    Parameters:
        

    Returns:
        list: A list of dictionaries, where each dictionary contains EEG segments
              for classes 'L', 'R', and 'Re'.
    """
    x = target.flatten()
    
    # Create binary masks for each label
    L = (x == 2).astype(int)
    R = (x == 3).astype(int)
    Re = (x == 1).astype(int)

    # Flags to mark transitions
    flagL = np.diff(L, prepend=0, append=0) != 0
    flagR = np.diff(R, prepend=0, append=0) != 0
    flagRe = np.diff(Re, prepend=0, append=0) != 0
    
    # Find the locations of transitions
    locL, _ = find_peaks(flagL.astype(float))
    locR, _ = find_peaks(flagR.astype(float))
    locRe, _ = find_peaks(flagRe.astype(float))

    # Truncate to the minimum number of presentations
    a = [len(locL) // 2, len(locR) // 2, len(locRe) // 2]
    b = min(a)

    # Prepare the output list
    S = []
    for j in range(b):
        startL, stopL = locL[2 * j], locL[2 * j + 1]
        startR, stopR = locR[2 * j], locR[2 * j + 1]
        startRe, stopRe = locRe[2 * j], locRe[2 * j + 1]
        
        eeg2 = eeg[startL:stopL, :]
        eeg3 = eeg[startR:stopR, :]
        eeg1 = eeg[startRe:stopRe, :]
        
        S.append({'two': eeg2, 'three': eeg3, 'one': eeg1})
    
    return S



