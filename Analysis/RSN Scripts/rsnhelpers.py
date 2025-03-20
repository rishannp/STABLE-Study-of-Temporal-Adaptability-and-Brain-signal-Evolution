# -*- coding: utf-8 -*-
"""
Helper Functions for G-W Dataset Processing 
"""
import matplotlib as plt
import numpy as np 
import os
import glob
import scipy
from os.path import dirname, join as pjoin
import scipy.io
import warnings
from scipy.signal import resample, butter, filtfilt, hilbert

def findSession(config):
    """
    Returns only the files corresponding to calibration runs (odd run numbers)
    from config['patient'], but limits the selection to a specified number of sessions 
    as given by config['trainingduration'].
    
    For example, for a file "GHS001R01.mat":
      - Session is extracted as "001" (characters after the prefix "GHS").
      - Run number is extracted as "01" (the two digits after "R").
    Only files with an odd run number are included.
    """
    files = config['patient']
    sessions = {}
    for file in files:
        try:
            parts = file.split("R")
            # Extract session: remove the first three characters from the part before "R"
            session = parts[0][3:]
            # Extract the run number: first two characters after "R"
            run_str = parts[1][:2]
            run_num = int(run_str)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue
        if run_num % 2 == 1:  # Only include odd runs
            if session not in sessions:
                sessions[session] = []
            sessions[session].append(file)
    
    # Sort sessions numerically (as integers)
    sorted_sessions = sorted(sessions.keys(), key=lambda s: int(s))
    session_limit = config.get('trainingduration', len(sorted_sessions))
    selected_sessions = sorted_sessions[:session_limit]
    
    # Collect files from the selected sessions; sort the files within each session by run number.
    selected_files = []
    for session in selected_sessions:
        session_files = sorted(sessions[session], key=lambda f: int(f.split("R")[1][:2]))
        selected_files.extend(session_files)
    
    return selected_files


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

def bandpass(data, bpf, fs):
    """
    Applies a bandpass filter to EEG data channel-wise.
    """
    b, a = butter(4, [bpf[0], bpf[1]], fs=fs, btype='band')
    filtered_data = np.zeros_like(data)
    for ch in range(data.shape[1]):
        filtered_data[:, ch] = filtfilt(b, a, data[:, ch])
    return filtered_data

def plvfcn(eegData):
    """
    Computes the Phase Locking Value (PLV) matrix from EEG data.
    Uses all electrodes in the provided data.
    """
    numElectrodes = eegData.shape[1]
    numTimeSteps = eegData.shape[0]
    plvMatrix = np.zeros((numElectrodes, numElectrodes))
    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            phase1 = np.angle(hilbert(eegData[:, electrode1]))
            phase2 = np.angle(hilbert(eegData[:, electrode2]))
            phase_difference = phase2 - phase1
            plv = np.abs(np.sum(np.exp(1j * phase_difference)) / numTimeSteps)
            plvMatrix[electrode1, electrode2] = plv
            plvMatrix[electrode2, electrode1] = plv
    return plvMatrix

def PreProcess(config):
    """
    Loads each calibration file (only odd run numbers), processes the signal,
    computes the PLV matrix immediately, and returns a list S where each element is
    a dictionary with key 'rest' mapping to the PLV matrix.
    """
    f = findSession(config)
    S = []
    # Iterate over each file in the list f
    for file in f:
        print("Processing file:", file)  # Print the file name before processing
        data_dir = config['dir']
        filename = pjoin(data_dir, file)
        #print("file loaded")
        data = scipy.io.loadmat(filename)

        # Check if the loaded data contains the expected fields
        if 'parameters' in data and 'signal' in data and 'states' in data:
            signal = data['signal']
            states = data['states']

            # Select EEG channels
            signal, channels = selectEEGChannels(signal, config['channel'], config['locsdir'])
            
            # Bandpass filter the signal
            signal = bandpass(signal, config['bpf'], config['fs'])
            
            # Resample the signal
            signal = resample(signal, int(signal.shape[0] * config['workingfs'] / config['fs']), axis=0)
            
            # Compute the PLV matrix immediately to reduce storage burden
            #print("Computing PLV")
            plv_matrix = plvfcn(signal)
            
            # Append the PLV matrix under the key 'rest'
            S.append({'rest': plv_matrix})
        else:
            warnings.warn(f'File {file} does not contain the expected fields.')

    # Store channel information in config (from first file)
    config['channels'] = channels
    return S, config

#%% Data Science Helpers for PLV Computation
import matplotlib.pyplot as plt
import scipy.signal as sig

def save_plv_graph(plv_matrix, channel_labels, output_filename):
    """
    Save a PLV matrix as an image with electrode labels on both axes.
    """
    numElectrodes = plv_matrix.shape[0]
    # Use a fixed figure size (8 x 6 inches)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cax = ax.imshow(plv_matrix, cmap='hot', vmin=0, vmax=1)
    fig.colorbar(cax)
    ax.set_title("PLV Matrix", fontsize=14)
    
    # Set ticks for each electrode
    ax.set_xticks(np.arange(numElectrodes))
    ax.set_yticks(np.arange(numElectrodes))
    
    if len(channel_labels) == numElectrodes:
        ax.set_xticklabels(channel_labels, rotation=90, ha='right', fontsize=8)
        ax.set_yticklabels(channel_labels, fontsize=8)
    else:
        ax.set_xticklabels(np.arange(numElectrodes), fontsize=8)
        ax.set_yticklabels(np.arange(numElectrodes), fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

def process_S_and_save_plv(S, output_dir, channel_labels):
    """
    For each instance in S (a list of dictionaries where each dictionary has key 'rest'
    containing a PLV matrix), save the resulting PLV graph.
    
    Parameters:
        S (list): List of dictionaries with PLV matrices.
        output_dir (str): Directory to save PLV graphs.
        channel_labels (list): List of electrode labels for the axes.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i, instance in enumerate(S):
        output_filename = os.path.join(output_dir, f"plv_instance_{i}_rest.png")
        save_plv_graph(instance['rest'], channel_labels, output_filename)
        print(f"Saved PLV graph for instance {i}, class rest to {output_filename}")
