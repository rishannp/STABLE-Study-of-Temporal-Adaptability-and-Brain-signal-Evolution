# -*- coding: utf-8 -*-
"""
Helper Functions for G-W Dataset Processing 
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np 
import os
import glob
import scipy
from os.path import dirname, join as pjoin
import scipy.io
import warnings
from scipy.signal import resample, butter, filtfilt, hilbert, coherence, csd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    for electrode1 in tqdm(range(numElectrodes), desc="Computing PLV", leave=False):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            phase1 = np.angle(hilbert(eegData[:, electrode1]))
            phase2 = np.angle(hilbert(eegData[:, electrode2]))
            phase_difference = phase2 - phase1
            plv = np.abs(np.sum(np.exp(1j * phase_difference)) / numTimeSteps)
            plvMatrix[electrode1, electrode2] = plv
            plvMatrix[electrode2, electrode1] = plv
    return plvMatrix

def mscfcn(eegData, fs, nperseg=400):
    """
    Computes the Magnitude Squared Coherence (MSC) matrix from EEG data.
    """
    numElectrodes = eegData.shape[1]
    mscMatrix = np.zeros((numElectrodes, numElectrodes))
    for electrode1 in tqdm(range(numElectrodes), desc="Computing MSC", leave=False):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            f_vals, Cxy = coherence(eegData[:, electrode1], eegData[:, electrode2], fs=fs, nperseg=nperseg)
            avg_coh = np.mean(Cxy)
            mscMatrix[electrode1, electrode2] = avg_coh
            mscMatrix[electrode2, electrode1] = avg_coh
    return mscMatrix

def csdfcn(eegData, fs):
    """
    Computes the Cross-Spectral Density (CSD) matrix from EEG data.
    For each pair of electrodes, computes the CSD using Welch's method
    and averages the magnitude of the CSD across frequencies.
    """
    numElectrodes = eegData.shape[1]
    csdMatrix = np.zeros((numElectrodes, numElectrodes))
    nperseg = 400  # Adjust based on your needs
    for electrode1 in tqdm(range(numElectrodes), desc="Computing CSD", leave=False):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            f_vals, Pxy = csd(eegData[:, electrode1], eegData[:, electrode2], fs=fs, nperseg=nperseg)
            avg_csd = np.mean(np.abs(Pxy))
            csdMatrix[electrode1, electrode2] = avg_csd
            csdMatrix[electrode2, electrode1] = avg_csd
    return csdMatrix

def apply_laplacian_xyz(data, coords, radius=40):
    """
    Apply a Laplacian spatial filter using Euclidean distance between electrodes.
    Each electrode is re-referenced by subtracting the mean of nearby electrodes.

    Parameters:
        data (samples x channels)
        coords: dict with 'X', 'Y', 'Z' lists (must be same order as data)
        radius: proximity threshold in normalized space

    Returns:
        laplacian_data: filtered data of same shape
    """
    n_channels = data.shape[1]
    laplacian_data = np.zeros_like(data)
    all_coords = np.array([coords['X'], coords['Y'], coords['Z']]).T

    for i in range(n_channels):
        dists = np.linalg.norm(all_coords - all_coords[i], axis=1)
        neighbor_idxs = np.where((dists > 0) & (dists < radius))[0]

        if len(neighbor_idxs) > 0:
            neighbor_avg = np.mean(data[:, neighbor_idxs], axis=1)
            laplacian_data[:, i] = data[:, i] - neighbor_avg
        else:
            laplacian_data[:, i] = data[:, i]  # No neighbors found

    return laplacian_data


def process_single_file(file, config, bands):
    """
    Processes a single file: loads data, selects channels, resamples,
    applies bandpass filtering, and computes PLV, MSC, and CSD matrices
    for each frequency band.
    """
    # Print the current file being processed
    print(f"Processing file: {file}")
    
    data_dir = config['dir']
    filename = pjoin(data_dir, file)
    data_mat = scipy.io.loadmat(filename)
    if not ('parameters' in data_mat and 'signal' in data_mat and 'states' in data_mat):
        warnings.warn(f'File {file} does not contain the expected fields.')
        return None

    signal = data_mat['signal']
    # Process the signal: select channels
    signal, channels = selectEEGChannels(signal, config['channel'], config['locsdir'])

    # Apply Laplacian if enabled
    if config.get('laplacian', 0) == 1:
        signal = apply_laplacian_xyz(signal, channels)

    # Resample signal to working sampling frequency
    signal = resample(signal, int(signal.shape[0] * config['workingfs'] / config['fs']), axis=0)
    
    plv_dict = {}
    msc_dict = {}
    csd_dict = {}
    
    for band_name, freq_range in bands.items():
        filtered_signal = bandpass(signal, freq_range, config['workingfs'])
        plv_dict[band_name] = plvfcn(filtered_signal)
        msc_dict[band_name] = mscfcn(filtered_signal, fs=config['workingfs'], nperseg=256)
        #csd_dict[band_name] = csdfcn(filtered_signal, fs=config['workingfs'])
    
    return {'PLV': plv_dict, 'MSC': msc_dict, 'CSD': csd_dict}, channels


def PreProcess_parallel(config):
    """
    Loads each calibration file (only odd run numbers), processes the signal,
    computes multiple PLV, MSC and CSD matrices for different frequency bands
    (Delta, Theta, Alpha, Beta, Gamma), and returns a list S where each element is a dictionary
    with keys 'PLV', 'MSC', and 'CSD'. Uses parallel processing to speed up computation.
    """
    files = findSession(config)
    S = []
    channels = None  # To capture channel info from a successful file

    # Define frequency bands (Hz)
    bands = {
        'Delta': [1, 4],
        'Theta': [4, 7],
        'Alpha': [8, 13],
        'Beta': [13, 30],
        'Gamma': [30, 80]
    }
    
    # Use a ProcessPoolExecutor to process files in parallel.
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_file, file, config, bands): file for file in files}
        # Create a progress bar that tracks the completion of each file.
        for future in tqdm(as_completed(futures), total=len(futures), desc="Overall progress"):
            file = futures[future]
            try:
                result = future.result()
                if result is None:
                    continue
                processed_data, ch = result
                S.append(processed_data)
                # Capture channel info once (assuming it's consistent across files)
                if channels is None:
                    channels = ch
            except Exception as e:
                warnings.warn(f"Error processing file {file}: {e}")
    
    config['channels'] = channels
    return S, config


#%% Data Science Helpers for Plotting
def save_plv_graph_session(plv_matrix, channel_labels, output_filename, session_number, band_name):
    """
    Save a PLV matrix as an image with electrode labels on both axes,
    and include the session number and band name in the title.
    """
    numElectrodes = plv_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(plv_matrix, cmap='hot', vmin=0, vmax=1)
    fig.colorbar(cax)
    ax.set_title(f"Session {session_number} - {band_name} PLV", fontsize=14)
    ax.set_xticks(np.arange(numElectrodes))
    ax.set_yticks(np.arange(numElectrodes))
    if len(channel_labels) == numElectrodes:
        ax.set_xticklabels(channel_labels, rotation=90, ha='right', fontsize=2)
        ax.set_yticklabels(channel_labels, fontsize=2)
    else:
        ax.set_xticklabels(np.arange(numElectrodes), fontsize=2)
        ax.set_yticklabels(np.arange(numElectrodes), fontsize=2)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

def save_msc_graph_session(msc_matrix, channel_labels, output_filename, session_number, band_name):
    """
    Save an MSC matrix as an image with electrode labels on both axes,
    and include the session number and band name in the title.
    """
    numElectrodes = msc_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(msc_matrix, cmap='hot', vmin=0, vmax=1)
    fig.colorbar(cax)
    ax.set_title(f"Session {session_number} - {band_name} MSC", fontsize=14)
    ax.set_xticks(np.arange(numElectrodes))
    ax.set_yticks(np.arange(numElectrodes))
    if len(channel_labels) == numElectrodes:
        ax.set_xticklabels(channel_labels, rotation=90, ha='right', fontsize=2)
        ax.set_yticklabels(channel_labels, fontsize=2)
    else:
        ax.set_xticklabels(np.arange(numElectrodes), fontsize=2)
        ax.set_yticklabels(np.arange(numElectrodes), fontsize=2)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

def save_csd_graph_session(csd_matrix, channel_labels, output_filename, session_number, band_name):
    """
    Save a CSD matrix as an image with electrode labels on both axes,
    and include the session number and band name in the title.
    """
    numElectrodes = csd_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(csd_matrix, cmap='hot', vmin=0, vmax=1)
    fig.colorbar(cax)
    ax.set_title(f"Session {session_number} - {band_name} CSD", fontsize=14)
    ax.set_xticks(np.arange(numElectrodes))
    ax.set_yticks(np.arange(numElectrodes))
    if len(channel_labels) == numElectrodes:
        ax.set_xticklabels(channel_labels, rotation=90, ha='right', fontsize=2)
        ax.set_yticklabels(channel_labels, fontsize=2)
    else:
        ax.set_xticklabels(np.arange(numElectrodes), fontsize=2)
        ax.set_yticklabels(np.arange(numElectrodes), fontsize=2)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

def process_S_and_save_matrix_session(S, output_dir, channel_labels, config):
    """
    For each instance in S (a list of dictionaries where each dictionary has keys 'PLV', 'MSC',
    and 'CSD' containing dictionaries of matrices for each frequency band), plot the data for each band and
    measure with a title indicating the session number, measure type, and band, and save the resulting images 
    in separate subfolders for each feature type and band within the output_dir folder.
    
    Parameters:
        S (list): List of dictionaries with PLV, MSC, and CSD matrices.
        output_dir (str): Base directory to save the graphs.
        channel_labels (list): List of electrode labels for the axes.
    """
    
    if config.get('plots', 1) == 0:
        print("⚠️ Skipping plot generation (config['plots'] == 0)")
        return
    
    # Base subfolders for each feature type
    feature_subfolders = {
        "PLV": os.path.join(output_dir, "PLV"),
        "MSC": os.path.join(output_dir, "MSC"),
        "CSD": os.path.join(output_dir, "CSD")
    }
    
    # Create the base feature folders if they don't exist
    for folder in feature_subfolders.values():
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    for i, instance in enumerate(S):
        session_number = i + 1
        
        # Process PLV graphs for each band
        for band_name, plv_matrix in instance['PLV'].items():
            # Create a subfolder for this band inside the PLV folder
            band_folder = os.path.join(feature_subfolders["PLV"], band_name)
            if not os.path.exists(band_folder):
                os.makedirs(band_folder)
            output_filename = os.path.join(band_folder, f"plv_session_{session_number}_{band_name}.png")
            save_plv_graph_session(plv_matrix, channel_labels, output_filename, session_number, band_name)
            print(f"Saved PLV graph for Session {session_number}, Band {band_name} to {output_filename}")
        
        # Process MSC graphs for each band
        for band_name, msc_matrix in instance['MSC'].items():
            # Create a subfolder for this band inside the MSC folder
            band_folder = os.path.join(feature_subfolders["MSC"], band_name)
            if not os.path.exists(band_folder):
                os.makedirs(band_folder)
            output_filename = os.path.join(band_folder, f"msc_session_{session_number}_{band_name}.png")
            save_msc_graph_session(msc_matrix, channel_labels, output_filename, session_number, band_name)
            print(f"Saved MSC graph for Session {session_number}, Band {band_name} to {output_filename}")
        
        # Process CSD graphs for each band
        for band_name, csd_matrix in instance['CSD'].items():
            # Create a subfolder for this band inside the CSD folder
            band_folder = os.path.join(feature_subfolders["CSD"], band_name)
            if not os.path.exists(band_folder):
                os.makedirs(band_folder)
            output_filename = os.path.join(band_folder, f"csd_session_{session_number}_{band_name}.png")
            save_csd_graph_session(csd_matrix, channel_labels, output_filename, session_number, band_name)
            print(f"Saved CSD graph for Session {session_number}, Band {band_name} to {output_filename}")
