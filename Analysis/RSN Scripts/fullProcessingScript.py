# -*- coding: utf-8 -*-
"""
Full analysis for RSN_GH.pkl and RSN_LS.pkl
Just run this file – everything is self-contained.

This version:
- Handles unified PKL {'sessions','features','config','channels'} and also legacy tuple/list PKLs
- Includes CSD in all matrix-based analyses (PLV/MSC/CSD)
- Adds time-series for Bandpower (Abs/Rel), LZC (per band), PLE (broadband)
- REMOVED Louvain communities and Communicability everywhere
"""

# --------------------------------------------------------------------------- #
# 1. Imports
# --------------------------------------------------------------------------- #
import pathlib
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx

from tqdm import tqdm
from scipy.stats import entropy, linregress
# (Removed louvain import)

plt.rcParams["figure.dpi"] = 300   # nicer inline resolution (optional)

# --------------------------------------------------------------------------- #
# 2. File locations –- edit if your paths are different
# --------------------------------------------------------------------------- #
file_paths = {
    "GH": pathlib.Path(
        r"C:\Users\uceerjp\Desktop\G-W Data\Understanding Non-starionarity in GW Dataset"
        r"\Understanding-Non-stationarity-over-2-Years-with-ALS\Analysis\RSN Scripts\RSN_GH.pkl"
    ),
    "LS": pathlib.Path(
        r"C:\Users\uceerjp\Desktop\G-W Data\Understanding Non-starionarity in GW Dataset"
        r"\Understanding-Non-stationarity-over-2-Years-with-ALS\Analysis\RSN Scripts\RSN_LS.pkl"
    ),
}

# --------------------------------------------------------------------------- #
# 3. Fixed parameters
# --------------------------------------------------------------------------- #
# Connectivity features I will treat as adjacency matrices over time:
features_conn = ['PLV', 'MSC', 'CSD']
# Canonical bands (must match what was produced during preprocessing)
bands = ['Alpha', 'Beta', 'Delta', 'Gamma', 'Theta']
band_colors = {
    'Alpha': 'tab:blue',
    'Beta' : 'tab:orange',
    'Delta': 'tab:green',
    'Gamma': 'tab:red',
    'Theta': 'tab:purple'
}
# Entropy histogram bins for temporal-entropy (edge-wise)
TEMP_ENT_BINS = 10

# Container so both runs survive in RAM:
results = {}

# --------------------------------------------------------------------------- #
# 0. Small utilities (keep it tight + explicit)
# --------------------------------------------------------------------------- #
def _nanfix(x):
    """Replace nan/inf with zeros; return a copy to be safe."""
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0, copy=True)

def _is_unified(rsn):
    """Detect unified dict vs legacy tuple/list."""
    return isinstance(rsn, dict) and 'sessions' in rsn and 'channels' in rsn

def _get_sessions(rsn):
    """Return list of session dicts regardless of PKL layout."""
    if _is_unified(rsn):
        return rsn['sessions']
    # legacy: assume rsn[0] is the sessions list
    return rsn[0]

def _get_channels(rsn):
    """Return channel label list regardless of PKL layout."""
    if _is_unified(rsn):
        return rsn['channels']['Channel']
    # legacy: assume rsn[1] is config-like dict with 'channels'
    return rsn[1]['channels']['Channel']

def _stack_connectivity(rsn, feat, band):
    """
    Stack adjacency matrices across sessions into shape (C, C, S).
    feat ∈ {'PLV','MSC','CSD'}, band ∈ bands.
    """
    mats = []
    for sess in _get_sessions(rsn):
        mat = sess[feat][band]
        mat = _nanfix(mat)
        mats.append(mat[:, :, np.newaxis])
    return np.concatenate(mats, axis=2)  # (C,C,S)

def _edgewise_cv(stack):
    """Coefficient of variation per edge across time. stack: (C,C,S)."""
    mu = stack.mean(axis=2)
    sig = stack.std(axis=2, ddof=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = np.where(mu != 0, sig / mu, 0.0)
    return cv

def _matrix_entropy_over_time(stack):
    """
    Shannon entropy per session of the (upper-triangular) edge weight distribution.
    stack: (C,C,S) → returns (S,)
    """
    C, _, S = stack.shape
    ent = np.zeros(S)
    iu = np.triu_indices(C, k=1)
    for t in range(S):
        w = stack[:, :, t][iu]
        w = w[w > 0]
        if w.size == 0:
            ent[t] = 0.0
        else:
            p = w / w.sum()
            ent[t] = entropy(p)
    return ent

def _temporal_entropy_edgewise(stack, bins=10, rng=(0, 1)):
    """
    Temporal entropy per edge from its time-series histogram.
    stack: (C,C,S) → returns (C,C) entropy per edge.
    """
    C, _, _ = stack.shape
    out = np.zeros((C, C))
    for i in range(C):
        for j in range(i+1, C):
            series = stack[i, j, :]
            hist, _ = np.histogram(series, bins=bins, range=rng, density=True)
            hist = hist[hist > 0]
            p = hist / hist.sum() if hist.size else np.array([1.0])
            e = entropy(p)
            out[i, j] = out[j, i] = e
    return out

def _node_entropy_over_time(stack):
    """
    Node entropy over sessions: for each session, entropy of each node's
    edge weight distribution (row normalized).
    stack: (C,C,S) → returns (C,S)
    """
    C, _, S = stack.shape
    node_ent = np.zeros((C, S))
    for t in range(S):
        mat = stack[:, :, t].copy()
        np.fill_diagonal(mat, 0.0)
        row_sums = mat.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            P = np.where(row_sums > 0, mat / row_sums, 0.0)
        # entropy along rows
        for i in range(C):
            pi = P[i, :]
            pi = pi[pi > 0]
            node_ent[i, t] = entropy(pi) if pi.size else 0.0
    return node_ent

def _ensure_aggregates(rsn):
    """
    If rsn['features'] is missing (old PKLs), reconstruct minimal aggregates
    for Bandpower/LZC/PLE from 'sessions'. Otherwise return rsn['features'].
    Returns a dict with keys like in the new pipeline:
        features['BANDPOWER']['absolute'][band] -> (S,C)
        features['CRITICALITY']['LZC'][band]    -> (S,C)
        features['CRITICALITY']['PLE']          -> (S,C)
    """
    if _is_unified(rsn) and isinstance(rsn.get('features', None), dict):
        return rsn['features']

    warnings.warn("PKL has no 'features' key – reconstructing aggregates from 'sessions'.")
    sessions = _get_sessions(rsn)
    # Infer bands from first session PLV keys
    bands_here = list(sessions[0]['PLV'].keys())
    C = next(iter(sessions[0]['PLV'].values())).shape[0]
    S = len(sessions)

    bp_abs = {b: np.zeros((S, C)) for b in bands_here}
    bp_rel = {b: np.zeros((S, C)) for b in bands_here}
    lzc    = {b: np.zeros((S, C)) for b in bands_here}
    ple    = np.zeros((S, C))
    for s_idx, sess in enumerate(sessions):
        # Bandpower
        if 'BANDPOWER' in sess:
            for b in bands_here:
                if 'absolute' in sess['BANDPOWER'] and b in sess['BANDPOWER']['absolute']:
                    bp_abs[b][s_idx, :] = _nanfix(sess['BANDPOWER']['absolute'][b])
                if 'relative' in sess['BANDPOWER'] and b in sess['BANDPOWER']['relative']:
                    bp_rel[b][s_idx, :] = _nanfix(sess['BANDPOWER']['relative'][b])
        # LZC
        if 'CRITICALITY' in sess and 'LZC' in sess['CRITICALITY']:
            for b in bands_here:
                if b in sess['CRITICALITY']['LZC']:
                    lzc[b][s_idx, :] = _nanfix(sess['CRITICALITY']['LZC'][b])
        # PLE
        if 'CRITICALITY' in sess and 'PLE' in sess['CRITICALITY']:
            ple[s_idx, :] = _nanfix(sess['CRITICALITY']['PLE'])

    return {
        'BANDPOWER': {'absolute': bp_abs, 'relative': bp_rel},
        'CRITICALITY': {'LZC': lzc, 'PLE': ple}
    }

# --------------------------------------------------------------------------- #
# 4. Main loop –- runs twice (once per .pkl)
# --------------------------------------------------------------------------- #
for dataset_name, file_path in file_paths.items():
    print(f"\n=================  Processing dataset: {dataset_name}  =================")

    # ---- 4.1 Load unified RSN dict (or legacy) ----------------------------- #
    with open(file_path, "rb") as f:
        rsn = pickle.load(f)

    channels = _get_channels(rsn)
    C = len(channels)
    S = len(_get_sessions(rsn))

    # ---- 4.2 Stack connectivity matrices (PLV/MSC/CSD) --------------------- #
    stacked_data = {}
    for feat in features_conn:
        stacked_data[feat] = {}
        for band in bands:
            try:
                stack = _stack_connectivity(rsn, feat, band)  # (C,C,S)
            except KeyError:
                # If some band/feat missing for this dataset, skip gracefully
                warnings.warn(f"Missing {feat}-{band} in {dataset_name}; skipping.")
                continue
            stacked_data[feat][band] = stack

    # ---- 4.3 Coefficient of variation (CV) --------------------------------- #
    cv_data = {}
    for feat in stacked_data:
        cv_data[feat] = {}
        for band in stacked_data[feat]:
            cv_data[feat][band] = _edgewise_cv(stacked_data[feat][band])

    # ---- 4.4 Plot CV matrices ---------------------------------------------- #
    n_rows, n_cols = len(stacked_data), len(bands)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    fig.suptitle(f'{dataset_name}: Coefficient of Variation (CV) matrices', fontsize=16)

    feats_list = list(stacked_data.keys())
    for i, feat in enumerate(feats_list):
        for j, band in enumerate(bands):
            ax = axs[i, j] if n_rows > 1 else axs[j]
            if band not in stacked_data[feat]:
                ax.axis('off'); continue
            im = ax.imshow(cv_data[feat][band], cmap='hot')
            ax.set_title(f'{feat} — {band}', fontsize=10)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

    # ---- 4.5 Shannon entropy per adjacency (session-wise) ------------------ #
    matrix_entropy = {}
    for feat in stacked_data:
        matrix_entropy[feat] = {}
        for band in stacked_data[feat]:
            matrix_entropy[feat][band] = _matrix_entropy_over_time(stacked_data[feat][band])

    # ---- 4.6 Temporal entropy (edge-wise) ---------------------------------- #
    temporal_entropy = {}
    for feat in stacked_data:
        temporal_entropy[feat] = {}
        for band in stacked_data[feat]:
            temporal_entropy[feat][band] = _temporal_entropy_edgewise(
                stacked_data[feat][band], bins=TEMP_ENT_BINS, rng=(0, 1)
            )

    # ---- 4.7 Node entropy (distribution of edges per node) ----------------- #
    node_entropy = {}
    for feat in stacked_data:
        node_entropy[feat] = {}
        for band in stacked_data[feat]:
            node_entropy[feat][band] = _node_entropy_over_time(stacked_data[feat][band])  # (C,S)

    # ---- 4.8 Plot entropy time-series -------------------------------------- #
    fig, axs = plt.subplots(1, len(stacked_data), figsize=(18, 5), sharey=True)
    fig.suptitle(f'{dataset_name}: Shannon entropy over time', fontsize=16)
    if len(stacked_data) == 1:
        axs = [axs]

    for i, feat in enumerate(feats_list):
        ax = axs[i]
        for band in bands:
            if band in matrix_entropy[feat]:
                ax.plot(matrix_entropy[feat][band], label=band, color=band_colors[band])
        ax.set_title(feat)
        ax.set_xlabel('Session')
        if i == 0:
            ax.set_ylabel('Entropy')
        ax.grid(True)
        ax.legend(title="Band")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

    # ---- 4.9 Plot temporal entropy matrices -------------------------------- #
    fig2, axs2 = plt.subplots(len(stacked_data), len(bands), figsize=(20, 10))
    fig2.suptitle(f'{dataset_name}: temporal entropy (edge-wise)', fontsize=16)
    if len(stacked_data) == 1:
        axs2 = np.array([axs2])

    for i, feat in enumerate(feats_list):
        for j, band in enumerate(bands):
            ax = axs2[i, j]
            if band not in temporal_entropy[feat]:
                ax.axis('off'); continue
            im = ax.imshow(temporal_entropy[feat][band], cmap='hot')
            ax.set_title(f'{feat} — {band}', fontsize=10)
            ax.axis('off')
            fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

    # ---- 4.10 Plot node entropy time-series -------------------------------- #
    for feat in feats_list:
        fig3, axs3 = plt.subplots(1, len(bands), figsize=(25, 5), sharey=True)
        fig3.suptitle(f'{dataset_name}: node-wise entropy over time — {feat}', fontsize=16)

        for j, band in enumerate(bands):
            ax3 = axs3[j]
            if band not in node_entropy[feat]:
                ax3.axis('off'); continue
            ent_matrix = node_entropy[feat][band]  # (C,S)
            # spaghetti per node
            for row in ent_matrix:
                ax3.plot(row, alpha=0.35, lw=0.8)
            # mean + linear trend
            mean_ent = ent_matrix.mean(axis=0)
            sessions = np.arange(mean_ent.shape[0])
            if len(sessions) >= 2:
                slope, intercept, *_ = linregress(sessions, mean_ent)
                ax3.plot(sessions, slope*sessions + intercept, color='black', lw=2, label='Trend')
            ax3.plot(sessions, mean_ent, color='k', lw=1.2, alpha=0.9, label='Mean')
            ax3.set_title(band)
            ax3.set_xlabel('Session')
            if j == 0:
                ax3.set_ylabel('Node entropy')
            ax3.grid(True); ax3.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

    # ---- 4.11 Classical graph descriptors (NO louvain, NO communicability) - #
    graph_metrics       = {}
    node_clustering_map = {}

    for feat in feats_list:
        graph_metrics[feat]       = {}
        node_clustering_map[feat] = {}

        for band in bands:
            if band not in stacked_data[feat]:
                continue
            print(f"Graph metrics: {feat} — {band}")
            stack = stacked_data[feat][band]               # (C,C,S)
            n_nodes, _, n_sessions = stack.shape

            avg_cluster_series  = []
            assortativity_series = []
            node_cluster_matrix  = np.zeros((n_nodes, n_sessions))

            for t in tqdm(range(n_sessions), leave=False, desc=f"{feat}-{band}"):
                mat = stack[:, :, t].copy()
                np.fill_diagonal(mat, 0)
                G = nx.from_numpy_array(mat)

                # clustering
                try:
                    clust_dict = nx.clustering(G, weight='weight')
                    clust_vals = np.fromiter(clust_dict.values(), float)
                    avg_cluster_series.append(clust_vals.mean())
                    node_cluster_matrix[:, t] = clust_vals
                except Exception:
                    avg_cluster_series.append(np.nan)

                # assortativity
                try:
                    ar = nx.degree_pearson_correlation_coefficient(G, weight='weight')
                    assortativity_series.append(ar if np.isfinite(ar) else np.nan)
                except Exception:
                    assortativity_series.append(np.nan)

            graph_metrics[feat][band] = pd.DataFrame({
                'Session'      : np.arange(n_sessions),
                'AvgClustering': avg_cluster_series,
                'Assortativity': assortativity_series
            })
            node_clustering_map[feat][band] = node_cluster_matrix

    # ---- 4.12 Plot average clustering & assortativity ----------------------- #
    def _timeseries_panel(metric_name, y_label, data_dict):
        fig, axs = plt.subplots(1, len(feats_list), figsize=(18, 5), sharey=True)
        fig.suptitle(f'{dataset_name}: {metric_name} over time', fontsize=16)
        if len(feats_list) == 1:
            axs = [axs]
        for i, feat in enumerate(feats_list):
            ax = axs[i]
            for band in bands:
                if band in data_dict[feat]:
                    ax.plot(
                        data_dict[feat][band]['Session'],
                        data_dict[feat][band][metric_name],
                        label=band
                    )
            ax.set_title(feat)
            ax.set_xlabel('Session')
            if i == 0:
                ax.set_ylabel(y_label)
            ax.grid(True); ax.legend(title="Band")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

    _timeseries_panel('AvgClustering', 'Avg clustering', graph_metrics)
    _timeseries_panel('Assortativity', 'Assortativity', graph_metrics)

    # ---- 4.13 Plot node clustering heatmaps (channels × sessions) ----------- #
    for feat in feats_list:
        for band in bands:
            if band not in node_clustering_map[feat]:
                continue
            sns.heatmap(
                node_clustering_map[feat][band], cmap='viridis',
                cbar_kws={'label': 'Clustering coeff'}
            )
            plt.title(f'{dataset_name}: Node-wise clustering — {feat} | {band}')
            plt.xlabel('Session'); plt.ylabel('Node')
            plt.tight_layout(); plt.show()

    # ===================== NEW: Extra features time-series =================== #
    # ---- 4.14 Bandpower (Absolute & Relative) over time -------------------- #
    feats_agg = _ensure_aggregates(rsn)  # {'BANDPOWER':{...}, 'CRITICALITY':{...}}
    bp_abs = feats_agg['BANDPOWER']['absolute']  # dict band -> (S,C)
    bp_rel = feats_agg['BANDPOWER']['relative']  # dict band -> (S,C)

    # Quick sanity: shapes
    # print({b: bp_abs[b].shape for b in bp_abs}, {b: bp_rel[b].shape for b in bp_rel})

    # Per-band global summaries (mean across channels) – small panel
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(f'{dataset_name}: Bandpower over time (mean across channels)', fontsize=16)
    ax_abs, ax_rel = axs

    for band in bands:
        if band in bp_abs:
            ax_abs.plot(bp_abs[band].mean(axis=1), label=band, color=band_colors[band])
        if band in bp_rel:
            ax_rel.plot(bp_rel[band].mean(axis=1), label=band, color=band_colors[band])

    ax_abs.set_title('Absolute'); ax_abs.set_xlabel('Session'); ax_abs.set_ylabel('Power (a.u.)')
    ax_abs.grid(True);  ax_abs.legend(title='Band')

    ax_rel.set_title('Relative'); ax_rel.set_xlabel('Session'); ax_rel.set_ylabel('Relative Power')
    ax_rel.grid(True); ax_rel.legend(title='Band')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

    # NEW: Per-channel spaghetti (node-wise) across sessions for each band
    # I keep alpha low to avoid occlusion; this answers "plots for each node over sessions".
    for label, bp_dict in [('Absolute', bp_abs), ('Relative', bp_rel)]:
        for band in bands:
            if band not in bp_dict:
                continue
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            fig.suptitle(f'{dataset_name}: Bandpower ({label}) – node-wise over sessions — {band}', fontsize=14)
            # bp_dict[band] is (S, C) → plot each channel’s time series
            for ch in range(bp_dict[band].shape[1]):
                ax.plot(bp_dict[band][:, ch], alpha=0.25, lw=0.8)
            ax.plot(bp_dict[band].mean(axis=1), color='k', lw=2, label='Mean')
            ax.set_xlabel('Session'); ax.set_ylabel(f'{label} Power'); ax.grid(True); ax.legend()
            plt.tight_layout(); plt.show()

    # Heatmaps: channels × sessions (huge panels), per band, for both Abs and Rel
    for label, bp_dict, cmap in [
        ('Absolute', bp_abs, 'Blues'),
        ('Relative', bp_rel, 'Greens')
    ]:
        for band in bands:
            if band not in bp_dict: 
                continue
            sns.heatmap(
                bp_dict[band].T,  # (S,C) → (C,S) as rows=channels
                cmap=cmap, cbar_kws={'label': f'{label} Power'},
            )
            plt.title(f'{dataset_name}: Bandpower ({label}) — {band}')
            plt.xlabel('Session'); plt.ylabel('Channel')
            plt.tight_layout(); plt.show()

    # ---- 4.15 Lempel–Ziv Complexity (per band) over time ------------------- #
    lzc = feats_agg['CRITICALITY']['LZC']  # dict band -> (S,C)

    # Mean across channels as time-series
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle(f'{dataset_name}: LZC over time (mean across channels)', fontsize=16)
    for band in bands:
        if band in lzc:
            ax.plot(lzc[band].mean(axis=1), label=band, color=band_colors[band])
    ax.set_xlabel('Session'); ax.set_ylabel('LZC'); ax.grid(True); ax.legend(title='Band')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

    # Heatmaps per band (channels × sessions)
    for band in bands:
        if band not in lzc: continue
        sns.heatmap(
            lzc[band].T, cmap='Purples', cbar_kws={'label': 'LZC'}
        )
        plt.title(f'{dataset_name}: LZC — {band}')
        plt.xlabel('Session'); plt.ylabel('Channel')
        plt.tight_layout(); plt.show()

    # ---- 4.16 Power Law Exponent (broadband) over time --------------------- #
    ple = feats_agg['CRITICALITY']['PLE']  # (S,C)

    # Mean across channels as time-series
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle(f'{dataset_name}: PLE (2–40 Hz slope) over time', fontsize=16)
    ax.plot(ple.mean(axis=1), label='Mean across channels')
    ax.set_xlabel('Session'); ax.set_ylabel('PLE (β)'); ax.grid(True); ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

    # Heatmap channels × sessions
    sns.heatmap(
        ple.T, cmap='OrRd', cbar_kws={'label': 'PLE (β)'}
    )
    plt.title(f'{dataset_name}: PLE — channels × sessions')
    plt.xlabel('Session'); plt.ylabel('Channel')
    plt.tight_layout(); plt.show()

    # ---- 4.17 Store everything --------------------------------------------- #
    results[dataset_name] = dict(
        stacked_data      = stacked_data,     # dict[feat][band] -> (C,C,S)
        cv_data           = cv_data,          # dict[feat][band] -> (C,C)
        matrix_entropy    = matrix_entropy,   # dict[feat][band] -> (S,)
        temporal_entropy  = temporal_entropy, # dict[feat][band] -> (C,C)
        node_entropy      = node_entropy,     # dict[feat][band] -> (C,S)
        graph_metrics     = graph_metrics,    # dict[feat][band] -> DataFrame
        node_clustering   = node_clustering_map,    # dict[feat][band] -> (C,S)
        # REMOVED: communicability_metrics
        # REMOVED: louvain_communities_dict
        bandpower_abs     = bp_abs,           # dict[band] -> (S,C)
        bandpower_rel     = bp_rel,           # dict[band] -> (S,C)
        lzc               = lzc,              # dict[band] -> (S,C)
        ple               = ple,              # (S,C)
        channels          = channels
    )

    plt.close('all')   # free a bit of memory before next dataset

# --------------------------------------------------------------------------- #
print("\nAll datasets processed!  Use the `results` dict for further inspection.")
