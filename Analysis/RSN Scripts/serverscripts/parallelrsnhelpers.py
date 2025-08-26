# -*- coding: utf-8 -*-
"""
Helper Functions for G-W Dataset Processing

parallelrsnhelpers.py
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
from scipy.signal import resample, butter, filtfilt, hilbert, coherence, csd, welch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d

# --------------------------
# File selection
# --------------------------
def findSession(config):
    """
    Returns only the files corresponding to calibration runs (odd run numbers)
    from config['patient'], limited by config['trainingduration'].
    Filenames like "GHS001R01.mat": session "001", run "01".
    """
    files = config['patient']
    sessions = {}
    for file in files:
        try:
            parts = file.split("R")
            session = parts[0][3:]
            run_str = parts[1][:2]
            run_num = int(run_str)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue
        if run_num % 2 == 1:
            if session not in sessions:
                sessions[session] = []
            sessions[session].append(file)
    sorted_sessions = sorted(sessions.keys(), key=lambda s: int(s))
    session_limit = config.get('trainingduration', len(sorted_sessions))
    selected_sessions = sorted_sessions[:session_limit]
    selected_files = []
    for session in selected_sessions:
        session_files = sorted(sessions[session], key=lambda f: int(f.split("R")[1][:2]))
        selected_files.extend(session_files)
    return selected_files

# --------------------------
# Channels + selection
# --------------------------
def selectEEGChannels(data, configType, locsdir):
    filename = pjoin(locsdir, "EEGLabLocsMPICap.mat")
    loaded_data = scipy.io.loadmat(filename)
    if 'Chanlocs' not in loaded_data:
        raise ValueError("The loaded .mat file does not contain 'Chanlocs'.")
    locs_struct = loaded_data['Chanlocs']

    try:
        all_labels = [str(locs_struct['labels'][0, i][0]) for i in range(locs_struct['labels'].size)]
        all_coords = np.array([[locs_struct['X'][0, i][0, 0],
                                locs_struct['Y'][0, i][0, 0],
                                locs_struct['Z'][0, i][0, 0]] for i in range(locs_struct['X'].size)])
    except KeyError as e:
        raise ValueError(f"Key {e} not found in Chanlocs.")

    channel_names_10_20 = [
        'Fp1','Fp2','F7','F3','Fz','F4','F8',
        'T7','C3','Cz','C4','T8',
        'P7/T5','P3','Pz','P4','P8/T6',
        'O1','O2'
    ]
    channel_names_10_10 = [
        'Fp1','Fpz','Fp2','AF7','AF3','AFz','AF4','AF8',
        'F7','F5','F3','F1','Fz','F2','F4','F6','F8',
        'FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8',
        'T7','C5','C3','C1','Cz','C2','C4','C6','T8',
        'TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8',
        'P7/T5','P5','P3','P1','Pz','P2','P4','P6','P8/T6',
        'PO7','PO3','POz','PO4','PO8',
        'O1','Oz','O2','Iz'
    ]
    parietal_channels = [
        'P1','P2','P3','P4','P5','P6','P7/T5','P8/T6','Pz',
        'PO1','PO2','PO3','PO4','POz','O1','Oz','O2'
    ]

    if configType == '10-20':
        selected_channels = channel_names_10_20
    elif configType == '10-10':
        selected_channels = channel_names_10_10
    elif configType == 'Parietal':
        selected_channels = parietal_channels
    elif configType == 'All':
        selected_channels = all_labels
    else:
        raise ValueError("Invalid configuration type.")

    matched_channels, selected_indices, matched_coords = [], [], []
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

    selected_data = data[:, selected_indices]
    selected_channels_info = {
        'Channel': matched_channels,
        'X': [c[0] for c in matched_coords],
        'Y': [c[1] for c in matched_coords],
        'Z': [c[2] for c in matched_coords]
    }
    return selected_data, selected_channels_info

# --------------------------
# Preprocessing helpers
# --------------------------
def bandpass(data, bpf, fs):
    b, a = butter(4, [bpf[0], bpf[1]], fs=fs, btype='band')
    return filtfilt(b, a, data, axis=0)

def _robust_channel_variance(x):
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) + 1e-12
    return np.median((x - med) ** 2, axis=0) / (mad ** 2)

def reject_bad_channels(x, zthr=5.0):
    rv = _robust_channel_variance(x)
    z = (rv - np.median(rv)) / (np.std(rv) + 1e-12)
    bad = np.where(np.abs(z) > zthr)[0].tolist()
    if bad:
        good = [i for i in range(x.shape[1]) if i not in bad]
        if good:
            x[:, bad] = np.mean(x[:, good], axis=1, keepdims=True)
    return x, bad

def apply_car(x):
    return x - np.mean(x, axis=1, keepdims=True)

def window_indices(n_samples, fs, length_sec, overlap):
    step = int(fs * length_sec * (1.0 - overlap))
    win = int(fs * length_sec)
    idx = []
    start = 0
    while start + win <= n_samples:
        idx.append((start, start + win))
        start += step if step > 0 else win
    return idx

# --------------------------
# Features: PLV (vectorized), iCoh, Band power, LZC, PLE
# --------------------------
def plvfcn_vectorized(eegData):
    """
    Fast PLV across all channel pairs using vectorized Hilbert phasors.
    eegData: (T, C)
    Returns: (C, C) PLV in [0,1] with ones on diag.
    """
    analytic = hilbert(eegData, axis=0)              # (T, C)
    U = analytic / (np.abs(analytic) + 1e-20)        # unit phasors
    T = U.shape[0]
    G = (U.conj().T @ U) / T                         # (C, C) complex
    PLV = np.abs(G)
    np.fill_diagonal(PLV, 1.0)
    return PLV

def icohfcn(eegData, fs, nperseg=256):
    """
    Imaginary part of coherency averaged over frequency.
    """
    from scipy.signal import welch, csd
    C = eegData.shape[1]
    f, Pxx = welch(eegData, fs=fs, nperseg=min(nperseg, eegData.shape[0]), axis=0)  # (F, C)
    denom = np.sqrt(np.maximum(Pxx, 1e-20))                                         # (F, C)

    iCoh = np.zeros((C, C), dtype=float)
    for j in range(C):
        for k in range(j+1, C):
            _, S_jk = csd(eegData[:, j], eegData[:, k], fs=fs, nperseg=min(nperseg, eegData.shape[0]))
            denom_jk = denom[:, j] * denom[:, k] + 1e-20
            coh = S_jk / denom_jk
            val = np.mean(np.abs(np.imag(coh)))
            iCoh[j, k] = val
            iCoh[k, j] = val
    return iCoh

def icohfcn_fft_band(xw, fs, band, freq_decim=1):
    """
    Vectorized imaginary coherency averaged over freq bins in 'band'.
    xw: (Tw, E) band-passed (or raw) window
    Returns: (E,E) matrix
    """
    Tw, E = xw.shape
    if Tw < 8:
        return np.zeros((E, E), dtype=float)

    # Hann taper to reduce leakage
    win = np.hanning(Tw)[:, None]
    X = np.fft.rfft(xw * win, axis=0)                  # (F, E)
    freqs = np.fft.rfftfreq(Tw, d=1.0/fs)

    fmask = (freqs >= band[0]) & (freqs <= band[1])
    fidx = np.where(fmask)[0]
    if fidx.size == 0:
        return np.zeros((E, E), dtype=float)
    if freq_decim > 1:
        fidx = fidx[::freq_decim]

    Xb = X[fidx, :]                                    # (Fb, E)
    Fb = Xb.shape[0]

    # Cross-spectral tensor S(f): (Fb, E, E)
    # S_f = conj(Xb_f)^T @ Xb_f (outer across channels) / Tw
    S = np.einsum('fe,fg->feg', np.conj(Xb), Xb) / float(Tw)    # (Fb, E, E)

    # PSD per freq & channel: P(f,e) = |X|^2 / Tw
    P = (np.abs(Xb)**2) / float(Tw)                              # (Fb, E)

    # Coherency per freq: C = S / sqrt(P_jj P_kk)
    denom = np.sqrt(P[:, :, None] * P[:, None, :]) + 1e-20       # (Fb, E, E)
    C = S / denom                                                # (Fb, E, E)

    # Imag coherency magnitude averaged over freq in band
    iCoh = np.mean(np.abs(np.imag(C)), axis=0)                   # (E, E)
    np.fill_diagonal(iCoh, 0.0)
    return iCoh


def bandpower_welch_per_channel(x, fs, band, nperseg=256):
    f_low, f_high = band
    f, Pxx = welch(x, fs=fs, nperseg=min(nperseg, x.shape[0]), axis=0)  # (F, C)
    band_mask = (f >= f_low) & (f <= f_high)
    if not np.any(band_mask):
        return np.zeros(Pxx.shape[1])
    bp = np.trapz(Pxx[band_mask, :], f[band_mask], axis=0)
    return bp

def lzc_per_channel(x):
    n, C = x.shape
    vals = np.zeros(C)
    for ch in range(C):
        env = np.abs(hilbert(x[:, ch]))
        thr = np.median(env) if np.isfinite(np.median(env)) else 0.0
        seq = (env > thr).astype(np.uint8)
        vals[ch] = _lempel_ziv_complexity_binary(seq)
    return vals

def _lempel_ziv_complexity_binary(seq):
    s = ''.join('1' if b else '0' for b in seq.tolist())
    n = len(s)
    if n <= 1: return 0.0
    i, c = 0, 1
    k = 1
    while True:
        if i + k > n:
            c += 1
            break
        substring = s[i:i+k]
        if s[:i].find(substring) != -1:
            k += 1
        else:
            c += 1
            i += k
            k = 1
        if i >= n:
            break
    norm = (n / np.log2(n)) if n > 1 else 1.0
    return c / norm

def ple_per_channel(x, fs, fmin=2, fmax=40, exclude_ranges=None, nperseg=512):
    if exclude_ranges is None:
        exclude_ranges = []
    f, Pxx = welch(x, fs=fs, nperseg=min(nperseg, x.shape[0]), axis=0)
    mask = (f >= fmin) & (f <= fmax)
    for lo, hi in exclude_ranges:
        mask &= ~((f >= lo) & (f <= hi))
    f_sel = f[mask]
    if f_sel.size < 5:
        return np.zeros(Pxx.shape[1])
    logf = np.log10(f_sel)
    betas = np.zeros(Pxx.shape[1])
    for ch in range(Pxx.shape[1]):
        psd = Pxx[mask, ch]
        psd = np.maximum(psd, 1e-20)
        logpsd = np.log10(psd)
        m, b = np.polyfit(logf, logpsd, 1)
        betas[ch] = -m
    return betas

# --------------------------
# Per-file processing (SPEC)
# --------------------------
def process_single_file(file, config, bands):
    """
    Preprocess: bad-chan -> BPF(1-97) -> CAR -> resample
    Window: 5s non-overlap
    Per band (pre-filtered once): PLV (vectorized), iCoh(FFT), BandPower, LZC
    PLE broadband per window.
    Shapes:
      FC: (E,E,W,B), BandPower: (E,B,W), LZC: (E,B,W), PLE: (E,W)
    """
    print(f"Processing file: {file}")
    data_dir = config['dir']
    filename = pjoin(data_dir, file)
    data_mat = scipy.io.loadmat(filename)
    if not ('parameters' in data_mat and 'signal' in data_mat and 'states' in data_mat):
        warnings.warn(f'File {file} does not contain the expected fields.')
        return None

    raw = data_mat['signal']                                  # (T_raw, C)
    raw, channels = selectEEGChannels(raw, config['channel'], config['locsdir'])
    pp = raw.astype(np.float64, copy=False)

    # Bad channel rejection
    if config['preproc'].get('bad_channel_reject', True):
        pp, bad = reject_bad_channels(pp, zthr=config['preproc'].get('bad_chan_zthr', 5.0))
        if bad:
            print(f"  Replaced bad channels (z>{config['preproc']['bad_chan_zthr']}): {bad}")

    # Full-range BPF (1-97) then CAR
    pp = bandpass(pp, config['preproc'].get('bandpass_full', [1, 97]), fs=config['fs'])
    if config['preproc'].get('apply_car', True):
        pp = apply_car(pp)

    # Resample to processing rate
    fs_proc = config.get('resample', config['fs'])
    if fs_proc != config['fs']:
        pp = resample(pp, int(pp.shape[0] * fs_proc / config['fs']), axis=0)
    fs = fs_proc
    T, E = pp.shape

    # Windowing
    wcfg = config.get('window', {})
    WIDX = window_indices(T, fs, wcfg.get('length_sec', 5.0), wcfg.get('overlap', 0.0))
    if len(WIDX) == 0:
        warnings.warn(f"{file}: too short after resample for a 5s window -- skipping.")
        return None

    W = len(WIDX)
    band_list = list(bands.items())
    B = len(band_list)

    # -------- Speedup 1: pre-filter whole session per band (reuse in all windows)
    session_band = {}
    phasor_band = {}  # for PLV
    for bname, (flo, fhi) in band_list:
        xf = bandpass(pp, [flo, fhi], fs=fs)                  # (T, E)
        session_band[bname] = xf
        analytic = hilbert(xf, axis=0)
        phasor_band[bname] = analytic / (np.abs(analytic) + 1e-20)  # unit phasors (T,E)

    # Allocate outputs
    fc_plv  = np.zeros((E, E, W, B), dtype=np.float32)
    fc_icoh = np.zeros((E, E, W, B), dtype=np.float32)
    bp_abs  = np.zeros((E, B, W),   dtype=np.float32)
    bp_rel  = np.zeros((E, B, W),   dtype=np.float32)
    lzc     = np.zeros((E, B, W),   dtype=np.float32)
    ple     = np.zeros((E, W),      dtype=np.float32)

    # helper
    def _bp(xwin, band):
        return bandpower_welch_per_channel(xwin, fs=fs, band=band, nperseg=min(256, xwin.shape[0]))

    icoh_method = (config.get('features', {}).get('icoh_method', 'fft') or 'fft').lower()
    freq_decim  = int(config.get('features', {}).get('icoh_freq_decim', 1))

    # nested tqdm: windows -> bands
    for w_i, (s0, s1) in enumerate(tqdm(WIDX, desc=f"{file} windows", leave=False)):
        xw_full = pp[s0:s1, :]                               # (Tw, E)

        # PLE broadband 2-40 Hz
        ple[:, w_i] = ple_per_channel(xw_full, fs=fs, fmin=2, fmax=40,
                                      exclude_ranges=[(48, 52), (98, 102)])

        # total power for relative normalization (1-97) for this window
        total_pw = _bp(xw_full, [1, 97])

        for b_i, (bname, (flo, fhi)) in enumerate(tqdm(band_list, desc=f"W{w_i+1} bands", leave=False)):
            # Slice pre-filtered arrays for this window
            xw  = session_band[bname][s0:s1, :]              # (Tw, E)
            Uw  = phasor_band[bname][s0:s1, :]               # (Tw, E)

            # PLV (vectorized via phasor Gram)
            if 'plv' in config['features'].get('fc', []):
                Tw = Uw.shape[0]
                G = (Uw.conj().T @ Uw) / float(Tw)
                fc_plv[:, :, w_i, b_i] = np.abs(G)

            # iCoh (fast FFT)
            if 'icoh' in config['features'].get('fc', []):
                if icoh_method == 'fft':
                    fc_icoh[:, :, w_i, b_i] = icohfcn_fft_band(xw, fs=fs, band=[flo, fhi], freq_decim=freq_decim)
                else:
                    # fallback: old slow Welch pairwise (not recommended)
                    fc_icoh[:, :, w_i, b_i] = icohfcn(xw, fs=fs, nperseg=min(256, xw.shape[0]))

            # Band power
            if config['features'].get('power', True):
                bp = _bp(xw, [flo, fhi])
                bp_abs[:, b_i, w_i] = bp
                bp_rel[:, b_i, w_i] = bp / np.maximum(total_pw, 1e-12)

            # LZC
            if config['features']['criticality'].get('lzc', True):
                lzc[:, b_i, w_i] = lzc_per_channel(xw)

    out = {
        'FC': {
            'PLV':  {'bands': [k for k,_ in band_list], 'data': fc_plv},
            'iCoh': {'bands': [k for k,_ in band_list], 'data': fc_icoh}
        },
        'BANDPOWER': {
            'bands': [k for k,_ in band_list],
            'absolute': bp_abs,
            'relative': bp_rel
        },
        'CRITICALITY': {
            'LZC': {'bands': [k for k,_ in band_list], 'data': lzc},
            'PLE': {'band': '2-40', 'data': ple}
        }
    }
    return out, channels


# --------------------------
# Parallel driver
# --------------------------
def PreProcess_parallel(config):

    max_workers = int(os.environ.get("GW_MAX_WORKERS", "0"))
    if max_workers <= 0:
        max_workers = os.cpu_count() or 1
        
    files = findSession(config)
    S = []
    channels = None

    # use global bands from config
    bands = config.get('bands', {
        '2-4':  [2,4],
        '4-7':  [4,7],
        '7-13': [7,13],
        '13-30':[13,30],
        '30-47':[30,47],
        '53-97':[53,97]
    })

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, file, config, bands): file for file in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Overall progress"):
            file = futures[future]
            try:
                result = future.result()
                if result is None:
                    continue
                processed_data, ch = result
                S.append(processed_data)
                if channels is None:
                    channels = ch
            except Exception as e:
                warnings.warn(f"Error processing file {file}: {e}")

    config['channels'] = channels
    return S, config

# --------------------------
# Plotters
# --------------------------

# Pastel palette
PASTEL = {
    'blue':  '#A3C4F3',
    'green': '#B9E3C6',
    'red':   '#F7B2AD',
    'purple':'#CDB4DB',
    'orange':'#FFD6A5',
    'grey':  '#CED4DA'
}

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_matrix_heatmap(mat, channel_labels, title, outpath, vmin=None, vmax=None, cmap='viridis'):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im)
    ax.set_title(title, fontsize=12)
    n = mat.shape[0]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    if len(channel_labels) == n:
        ax.set_xticklabels(channel_labels, rotation=90, ha='right', fontsize=4)
        ax.set_yticklabels(channel_labels, fontsize=4)
    else:
        ax.set_xticklabels(np.arange(n), fontsize=4)
        ax.set_yticklabels(np.arange(n), fontsize=4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def _smooth_series(x, y, factor=10, kind='cubic'):
    if len(x) < 3:
        return x, y
    f = interp1d(x, y, kind=kind)
    xs = np.linspace(x.min(), x.max(), int(len(x) * factor))
    ys = f(xs)
    return xs, ys

def plot_dataset_timeseries_mean_std(per_session_values, ylabel, title, outpath, color=PASTEL['blue']):
    """
    per_session_values: list of 1D arrays (values for that session),
                        we compute mean over windows and std over windows per session.
    """
    if len(per_session_values) == 0:
        return
    sess_idx = np.arange(1, len(per_session_values) + 1)
    means = np.array([np.mean(v) if v.size > 0 else np.nan for v in per_session_values])
    stds  = np.array([np.std(v)  if v.size > 0 else np.nan for v in per_session_values])

    # smooth (ignoring NaNs)
    mask = ~np.isnan(means)
    xs, ys = _smooth_series(sess_idx[mask], means[mask], factor=8, kind='cubic')

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, ys, lw=2, color=color)
    # simple linear interp for std on xs grid
    if np.any(mask):
        fstd = interp1d(sess_idx[mask], stds[mask], kind='linear', fill_value="extrapolate")
        std_interp = fstd(xs)
        ax.fill_between(xs, ys - std_interp, ys + std_interp, alpha=0.3, color=color)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Session (S1 .. S{})".format(len(per_session_values)))
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def process_S_and_save_matrix_session(S, output_dir, channel_labels, config):
    """
    Save all features:
      - FC matrices (PLV / iCoh): for each session & band
            * mean over windows heatmap
            * each-window heatmap
      - Time-series plots across sessions:
            * BandPower absolute (mean over channels & windows) ± std
            * BandPower relative (mean over channels & windows) ± std
            * LZC (mean over channels & windows) ± std
            * PLE (mean over channels & windows) ± std
      x-axis is session bins S1..Sn; curves smoothed; pastel colors; std shading.
    """
    if config.get('plots', 1) == 0:
        print("?? Skipping plot generation (config['plots'] == 0)")
        return

    # Base folders
    _ensure_dir(output_dir)
    fc_dir = os.path.join(output_dir, "FC")
    _ensure_dir(fc_dir)
    plv_dir = os.path.join(fc_dir, "PLV")
    icoh_dir = os.path.join(fc_dir, "iCoh")
    _ensure_dir(plv_dir); _ensure_dir(icoh_dir)

    bp_dir = os.path.join(output_dir, "BANDPOWER")
    _ensure_dir(bp_dir)
    bp_abs_dir = os.path.join(bp_dir, "Absolute")
    bp_rel_dir = os.path.join(bp_dir, "Relative")
    _ensure_dir(bp_abs_dir); _ensure_dir(bp_rel_dir)

    crit_dir = os.path.join(output_dir, "CRITICALITY")
    _ensure_dir(crit_dir)
    lzc_dir = os.path.join(crit_dir, "LZC")
    ple_dir = os.path.join(crit_dir, "PLE")
    _ensure_dir(lzc_dir); _ensure_dir(ple_dir)

    # ======================
    # FC: per-session saves
    # ======================
    for i, instance in enumerate(S):
        session_number = i + 1

        # ---- PLV ----
        if 'PLV' in instance['FC'] and instance['FC']['PLV']['data'].size > 0:
            bands = instance['FC']['PLV']['bands']
            data  = instance['FC']['PLV']['data']    # (E,E,W,B)
            E, _, W, B = data.shape
            for b_i, bname in enumerate(bands):
                # mean over windows
                mean_mat = data[:, :, :, b_i].mean(axis=2)  # (E,E)
                out_mean = os.path.join(plv_dir, f"session_{session_number}_band_{bname}_PLV_mean.png")
                save_matrix_heatmap(mean_mat, channel_labels,
                                    title=f"S{session_number} PLV mean over windows ({bname})",
                                    outpath=out_mean, cmap='viridis', vmin=0, vmax=1)

                # each-window heatmap
                perw_dir = os.path.join(plv_dir, f"session_{session_number}", bname)
                _ensure_dir(perw_dir)
                for w in range(W):
                    out_w = os.path.join(perw_dir, f"PLV_S{session_number}_{bname}_W{w+1}.png")
                    save_matrix_heatmap(data[:, :, w, b_i], channel_labels,
                                        title=f"S{session_number} PLV W{w+1} ({bname})",
                                        outpath=out_w, cmap='viridis', vmin=0, vmax=1)

        # ---- iCoh ----
        if 'iCoh' in instance['FC'] and instance['FC']['iCoh']['data'].size > 0:
            bands = instance['FC']['iCoh']['bands']
            data  = instance['FC']['iCoh']['data']    # (E,E,W,B)
            E, _, W, B = data.shape
            vmax = np.nanmax(data) if np.isfinite(np.nanmax(data)) else 1.0
            for b_i, bname in enumerate(bands):
                mean_mat = data[:, :, :, b_i].mean(axis=2)
                out_mean = os.path.join(icoh_dir, f"session_{session_number}_band_{bname}_iCoh_mean.png")
                save_matrix_heatmap(mean_mat, channel_labels,
                                    title=f"S{session_number} iCoh mean over windows ({bname})",
                                    outpath=out_mean, cmap='magma', vmin=0, vmax=vmax)

                perw_dir = os.path.join(icoh_dir, f"session_{session_number}", bname)
                _ensure_dir(perw_dir)
                for w in range(W):
                    out_w = os.path.join(perw_dir, f"iCoh_S{session_number}_{bname}_W{w+1}.png")
                    save_matrix_heatmap(data[:, :, w, b_i], channel_labels,
                                        title=f"S{session_number} iCoh W{w+1} ({bname})",
                                        outpath=out_w, cmap='magma', vmin=0, vmax=vmax)

    # =========================================
    # Dataset-level time-series (across sessions)
    # =========================================
    # We aggregate per session by averaging over windows (and channels where relevant).
    # Then we plot one curve with session bins S1..Sn and std shading.

    n_sessions = len(S)

    # ---- BandPower absolute & relative ----
    if n_sessions > 0 and 'BANDPOWER' in S[0]:
        bands = S[0]['BANDPOWER']['bands']
        B = len(bands)

        # For each band, compute per-session mean over (channels, windows)
        for mode, dkey in [('Absolute', 'absolute'), ('Relative', 'relative')]:
            per_band_values = {bname: [] for bname in bands}
            for i, inst in enumerate(S):
                data = inst['BANDPOWER'][dkey]  # (E,B,W)
                for b_i, bname in enumerate(bands):
                    # mean over channels, leave per-window values to compute std across windows
                    vals_w = data[:, b_i, :].mean(axis=0)  # (W,)
                    per_band_values[bname].append(vals_w)

            # Plot each band on its own chart
            for bname in bands:
                outpath = os.path.join(bp_abs_dir if mode=='Absolute' else bp_rel_dir,
                                       f"{mode}_timeseries_{bname}.png")
                plot_dataset_timeseries_mean_std(per_band_values[bname],
                                                 ylabel=f"{mode} Band Power ({bname})",
                                                 title=f"{mode} Band Power across Sessions ({bname})",
                                                 outpath=outpath,
                                                 color=PASTEL['green'] if mode=='Absolute' else PASTEL['orange'])

    # ---- LZC ----
    if n_sessions > 0 and 'CRITICALITY' in S[0] and 'LZC' in S[0]['CRITICALITY']:
        bands = S[0]['CRITICALITY']['LZC']['bands']
        per_band_values = {bname: [] for bname in bands}
        for i, inst in enumerate(S):
            data = inst['CRITICALITY']['LZC']['data']  # (E,B,W)
            for b_i, bname in enumerate(bands):
                vals_w = data[:, b_i, :].mean(axis=0)    # (W,)
                per_band_values[bname].append(vals_w)
        for bname in bands:
            outpath = os.path.join(lzc_dir, f"LZC_timeseries_{bname}.png")
            plot_dataset_timeseries_mean_std(per_band_values[bname],
                                             ylabel=f"LZC ({bname})",
                                             title=f"LZC across Sessions ({bname})",
                                             outpath=outpath,
                                             color=PASTEL['purple'])

    # ---- PLE (broadband 2-40) ----
    if n_sessions > 0 and 'CRITICALITY' in S[0] and 'PLE' in S[0]['CRITICALITY']:
        per_session_values = []
        for i, inst in enumerate(S):
            data = inst['CRITICALITY']['PLE']['data']  # (E,W)
            vals_w = data.mean(axis=0)                 # mean over channels -> (W,)
            per_session_values.append(vals_w)
        outpath = os.path.join(ple_dir, "PLE_timeseries.png")
        plot_dataset_timeseries_mean_std(per_session_values,
                                         ylabel="PLE (2-40 Hz)",
                                         title="PLE across Sessions",
                                         outpath=outpath,
                                         color=PASTEL['blue'])
