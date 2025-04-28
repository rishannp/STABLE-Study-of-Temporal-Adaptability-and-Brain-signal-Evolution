# -*- coding: utf-8 -*-
"""
Full analysis for RSN_GH.pkl and RSN_LS.pkl
Just run this file – everything is self‑contained.
"""

# --------------------------------------------------------------------------- #
# 1. Imports
# --------------------------------------------------------------------------- #
import pathlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx

from tqdm import tqdm
from scipy.stats import entropy, linregress
from networkx.algorithms.community import louvain_communities

plt.rcParams["figure.dpi"] = 110   # nicer inline resolution (optional)

# --------------------------------------------------------------------------- #
# 2. File locations –‑ edit if your paths are different
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
features = ['PLV', 'MSC']
bands    = ['Alpha', 'Beta', 'Delta', 'Gamma', 'Theta']
band_colors = {
    'Alpha': 'tab:blue',
    'Beta' : 'tab:orange',
    'Delta': 'tab:green',
    'Gamma': 'tab:red',
    'Theta': 'tab:purple'
}

# Container so both runs survive in RAM:
results = {}

# --------------------------------------------------------------------------- #
# 4. Main loop –‑ everything inside runs twice (once per .pkl)
# --------------------------------------------------------------------------- #
for dataset_name, file_path in file_paths.items():
    print(f"\n=================  Processing dataset: {dataset_name}  =================")

    # ---- 4.1 Load data ------------------------------------------------------ #
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # ---- 4.2 Stack matrices ------------------------------------------------- #
    stacked_data = {}
    for feat in features:
        stacked_data[feat] = {}
        for band in bands:
            stacked = []
            for sample in data[0]:
                mat = sample[feat][band]
                mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
                stacked.append(mat[:, :, np.newaxis])
            stacked_data[feat][band] = np.concatenate(stacked, axis=2)

    # ---- 4.3 Coefficient of variation (CV) --------------------------------- #
    cv_data = {}
    for feat in features:
        cv_data[feat] = {}
        for band in bands:
            bandmatrix  = stacked_data[feat][band]          # 124×124×N
            n_channels  = bandmatrix.shape[0]
            cv_matrix   = np.zeros((n_channels, n_channels))

            for i in range(n_channels):
                for j in range(n_channels):
                    pair_t = bandmatrix[i, j, :]
                    mu, sig = np.mean(pair_t), np.std(pair_t)
                    cv_matrix[i, j] = sig / mu if mu != 0 else 0
            cv_data[feat][band] = cv_matrix

    # ---- 4.4 Plot CV matrices ---------------------------------------------- #
    n_rows, n_cols = len(features), len(bands)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    fig.suptitle(f'{dataset_name}: Coefficient of Variation (CV) matrices', fontsize=16)

    for i, feat in enumerate(features):
        for j, band in enumerate(bands):
            ax = axs[i, j]
            im = ax.imshow(cv_data[feat][band], cmap='hot')
            ax.set_title(f'{feat} — {band}', fontsize=10)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # ---- 4.5 Shannon entropy of each adjacency matrix ---------------------- #
    matrix_entropy = {}
    for feat in features:
        matrix_entropy[feat] = {}
        for band in bands:
            stack = stacked_data[feat][band]
            entropies = []
            for t in range(stack.shape[2]):
                mat = stack[:, :, t]
                w = mat[np.triu_indices_from(mat, k=1)]
                w = w[w > 0]
                probs = (w / np.sum(w)) if w.size else np.array([1.0])
                entropies.append(entropy(probs))
            matrix_entropy[feat][band] = np.array(entropies)

    # ---- 4.6 Temporal entropy (per edge) ----------------------------------- #
    temporal_entropy = {}
    for feat in features:
        temporal_entropy[feat] = {}
        for band in bands:
            stack = stacked_data[feat][band]               # 124×124×N
            n = stack.shape[0]
            ent_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    edge_series = stack[i, j, :]
                    hist, _ = np.histogram(edge_series, bins=10, range=(0, 1), density=True)
                    hist = hist[hist > 0]
                    probs = (hist / np.sum(hist)) if hist.size else np.array([1.0])
                    ent = entropy(probs)
                    ent_matrix[i, j] = ent_matrix[j, i] = ent
            temporal_entropy[feat][band] = ent_matrix

    # ---- 4.7 Node entropy (distribution of edges per node) ----------------- #
    node_entropy = {}
    for feat in features:
        node_entropy[feat] = {}
        for band in bands:
            stack = stacked_data[feat][band]               # 124×124×N
            n_nodes, _, n_sessions = stack.shape
            ent_matrix = np.zeros((n_nodes, n_sessions))
            for t in range(n_sessions):
                mat = stack[:, :, t]
                for i in range(n_nodes):
                    row = mat[i, :].copy()
                    row[i] = 0
                    if row.sum():
                        p = row / row.sum()
                        ent_matrix[i, t] = entropy(p[p > 0])
            node_entropy[feat][band] = ent_matrix

    # ---- 4.8 Plot entropy time‑series -------------------------------------- #
    fig, axs = plt.subplots(1, len(features), figsize=(18, 5), sharey=True)
    fig.suptitle(f'{dataset_name}: Shannon entropy over time', fontsize=16)

    for i, feat in enumerate(features):
        ax = axs[i]
        for band in bands:
            ax.plot(matrix_entropy[feat][band], label=band, color=band_colors[band])
        ax.set_title(feat)
        ax.set_xlabel('Session')
        if i == 0:
            ax.set_ylabel('Entropy')
        ax.grid(True)
        ax.legend(title="Band")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # ---- 4.9 Plot temporal entropy matrices -------------------------------- #
    fig2, axs2 = plt.subplots(len(features), len(bands), figsize=(20, 10))
    fig2.suptitle(f'{dataset_name}: temporal entropy (edge‑wise)', fontsize=16)

    for i, feat in enumerate(features):
        for j, band in enumerate(bands):
            ax = axs2[i, j]
            im = ax.imshow(temporal_entropy[feat][band], cmap='hot')
            ax.set_title(f'{feat} — {band}', fontsize=10)
            ax.axis('off')
            fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # ---- 4.10 Plot node entropy time‑series -------------------------------- #
    for feat in features:
        fig3, axs3 = plt.subplots(1, len(bands), figsize=(25, 5), sharey=True)
        fig3.suptitle(f'{dataset_name}: node‑wise entropy over time — {feat}', fontsize=16)

        for j, band in enumerate(bands):
            ax3 = axs3[j]
            ent_matrix = node_entropy[feat][band]
            for row in ent_matrix:
                ax3.plot(row, alpha=0.4, lw=0.8)
            mean_ent = ent_matrix.mean(axis=0)
            sessions = np.arange(mean_ent.shape[0])
            slope, intercept, *_ = linregress(sessions, mean_ent)
            ax3.plot(sessions, slope*sessions + intercept, color='black', lw=2, label='Trend')
            ax3.set_title(band)
            ax3.set_xlabel('Session')
            if j == 0:
                ax3.set_ylabel('Node entropy')
            ax3.grid(True)
            ax3.legend()
            ax3.set_ylim(4.0, 4.7)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # ---- 4.11 Classical graph descriptors ---------------------------------- #
    graph_metrics             = {}
    node_clustering           = {}
    communicability_metrics   = {}
    louvain_communities_dict  = {}

    for feat in features:
        graph_metrics[feat]            = {}
        node_clustering[feat]          = {}
        communicability_metrics[feat]  = {}
        louvain_communities_dict[feat] = {}

        for band in bands:
            print(f"Graph metrics: {feat} — {band}")
            stack = stacked_data[feat][band]               # 124×124×N
            n_nodes, _, n_sessions = stack.shape

            avg_cluster_series  = []
            assortativity_series = []
            node_cluster_matrix  = np.zeros((n_nodes, n_sessions))
            comm_matrix          = np.zeros((n_nodes, n_sessions))
            louvain_matrix       = np.zeros((n_nodes, n_sessions))

            for t in tqdm(range(n_sessions), leave=False, desc=f"{feat}-{band}"):
                mat = stack[:, :, t].copy()
                np.fill_diagonal(mat, 0)
                G = nx.from_numpy_array(mat)

                # clustering
                try:
                    clust_vals = np.fromiter(nx.clustering(G, weight='weight').values(), float)
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

                # communicability diagonal
                try:
                    comm = nx.communicability_exp(G)
                    comm_matrix[:, t] = np.array([comm[i][i] for i in range(n_nodes)])
                except Exception:
                    pass

                # louvain communities
                try:
                    comms = louvain_communities(G, weight='weight', resolution=1)
                    for idx, community in enumerate(comms):
                        for node in community:
                            louvain_matrix[node, t] = idx
                except Exception:
                    louvain_matrix[:, t] = -1

            graph_metrics[feat][band] = pd.DataFrame({
                'Session'      : np.arange(n_sessions),
                'AvgClustering': avg_cluster_series,
                'Assortativity': assortativity_series
            })
            node_clustering[feat][band]          = node_cluster_matrix
            communicability_metrics[feat][band]  = comm_matrix
            louvain_communities_dict[feat][band] = louvain_matrix

    # ---- 4.12 Plot average clustering & assortativity ----------------------- #
    def _timeseries_panel(metric_name, y_label, data_dict):
        fig, axs = plt.subplots(1, len(features), figsize=(18, 5), sharey=True)
        fig.suptitle(f'{dataset_name}: {metric_name} over time', fontsize=16)
        for i, feat in enumerate(features):
            ax = axs[i]
            for band in bands:
                ax.plot(
                    data_dict[feat][band]['Session'],
                    data_dict[feat][band][metric_name],
                    label=band
                )
            ax.set_title(feat)
            ax.set_xlabel('Session')
            if i == 0:
                ax.set_ylabel(y_label)
            ax.grid(True)
            ax.legend(title="Band")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    _timeseries_panel('AvgClustering', 'Avg clustering', graph_metrics)
    _timeseries_panel('Assortativity', 'Assortativity', graph_metrics)

    # ---- 4.13 Plot node clustering, communicability, Louvain heatmaps ------- #
    for feat in features:
        for band in bands:
            sns.heatmap(
                node_clustering[feat][band], cmap='viridis',
                cbar_kws={'label': 'Clustering coeff'}
            )
            plt.title(f'{dataset_name}: Node‑wise clustering — {feat} | {band}')
            plt.xlabel('Session')
            plt.ylabel('Node')
            plt.tight_layout(); plt.show()

            sns.heatmap(
                communicability_metrics[feat][band], cmap='magma',
                cbar_kws={'label': 'Communicability'}
            )
            plt.title(f'{dataset_name}: Communicability — {feat} | {band}')
            plt.xlabel('Session')
            plt.ylabel('Node')
            plt.tight_layout(); plt.show()

            sns.heatmap(
                louvain_communities_dict[feat][band], cmap='tab20',
                cbar_kws={'label': 'Community ID'}
            )
            plt.title(f'{dataset_name}: Louvain communities — {feat} | {band}')
            plt.xlabel('Session')
            plt.ylabel('Node')
            plt.tight_layout(); plt.show()

    # ---- 4.14 Number of communities per session ---------------------------- #
    fig, axs = plt.subplots(1, len(features), figsize=(18, 5), sharey=True)
    fig.suptitle(f'{dataset_name}: Number of Louvain communities per session', fontsize=16)

    for i, feat in enumerate(features):
        ax = axs[i]
        for band in bands:
            counts = [
                len(np.unique(louvain_communities_dict[feat][band][:, t][
                    louvain_communities_dict[feat][band][:, t] >= 0
                ]))
                for t in range(louvain_communities_dict[feat][band].shape[1])
            ]
            ax.plot(counts, label=band)
        ax.set_title(feat)
        ax.set_xlabel('Session')
        if i == 0:
            ax.set_ylabel('Community count')
        ax.grid(True)
        ax.legend(title="Band")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # ---- 4.15 Store everything --------------------------------------------- #
    results[dataset_name] = dict(
        stacked_data            = stacked_data,
        cv_data                 = cv_data,
        matrix_entropy          = matrix_entropy,
        temporal_entropy        = temporal_entropy,
        node_entropy            = node_entropy,
        graph_metrics           = graph_metrics,
        node_clustering         = node_clustering,
        communicability_metrics = communicability_metrics,
        louvain_communities_dict= louvain_communities_dict,
    )

    plt.close('all')   # free a bit of memory before next dataset

# --------------------------------------------------------------------------- #
print("\nAll datasets processed!  Use the `results` dict for further inspection.")
