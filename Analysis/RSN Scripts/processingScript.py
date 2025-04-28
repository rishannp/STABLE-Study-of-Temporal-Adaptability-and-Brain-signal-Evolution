# Processing Code

import pickle

# Load Data
file_path = "C:/Users/uceerjp/Desktop/G-W Data/Understanding Non-starionarity in GW Dataset/Understanding-Non-stationarity-over-2-Years-with-ALS/Analysis/RSN Scripts/RSN.pkl"

# Open and load the pkl file
with open(file_path, 'rb') as file:
    data = pickle.load(file)
    
import numpy as np

features = ['PLV', 'MSC', 'CSD']
bands = ['Alpha', 'Beta', 'Delta', 'Gamma', 'Theta']

# Create new variable to store stacked results
stacked_data = {}

###### STACK MATRICES ##########
for feat in features:
    stacked_data[feat] = {}
    for band in bands:
        stacked = []
        for sample in data[0]:
            mat = sample[feat][band]
            mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
            stacked.append(mat[:, :, np.newaxis])
        stacked_data[feat][band] = np.concatenate(stacked, axis=2)


###### CV ########

cv_data = {}

for feat in features:
    cv_data[feat] = {}
    matrix = stacked_data[feat]
    
    for band in bands:
        bandmatrix = matrix[band]  # shape: 124 x 124 x N
        n_channels = bandmatrix.shape[0]

        # Preallocate CV matrix
        cv_matrix = np.zeros((n_channels, n_channels))

        for i in range(n_channels):
            for j in range(n_channels):
                pair_t = bandmatrix[i, j, :]  # timeseries across trials
                mu = np.mean(pair_t)
                sig = np.std(pair_t)
                # Avoid division by zero
                cv = sig / mu if mu != 0 else 0
                cv_matrix[i, j] = cv

        cv_data[feat][band] = cv_matrix  # shape: 124 x 124

# Plots
import matplotlib.pyplot as plt

# Set up grid: 3 features x 5 bands
n_rows = len(features)
n_cols = len(bands)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
fig.suptitle('Coefficient of Variation (CV) Matrices per Feature and Band', fontsize=16)

# Plot each matrix
for i, feat in enumerate(features):
    for j, band in enumerate(bands):
        ax = axs[i, j]
        cv_matrix = cv_data[feat][band]

        im = ax.imshow(cv_matrix, cmap='hot')
        ax.set_title(f'{feat} - {band}', fontsize=10)
        ax.axis('off')

        # Optional: add colorbars per subplot
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # make room for suptitle
plt.show()

#%% Estimating Chaos in a Connectivity Matrix

# Shannon Entropy of each Adjacency Matrix
from scipy.stats import entropy

matrix_entropy = {}

for feat in features:
    matrix_entropy[feat] = {}
    for band in bands:
        stack = stacked_data[feat][band]  # shape: 124x124xN
        entropies = []
        for t in range(stack.shape[2]):
            mat = stack[:, :, t]
            # Extract upper triangle, flatten
            triu_indices = np.triu_indices_from(mat, k=1)
            weights = mat[triu_indices]
            weights = weights[weights > 0]  # remove zeros
            probs = weights / np.sum(weights)
            ent = entropy(probs)  # Shannon entropy
            entropies.append(ent)
        matrix_entropy[feat][band] = np.array(entropies)  # shape: (N_sessions,)



# Temporal Entropy
temporal_entropy = {}

for feat in features:
    temporal_entropy[feat] = {}
    for band in bands:
        stack = stacked_data[feat][band]  # shape: 124x124xN
        n = stack.shape[0]
        ent_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                edge_series = stack[i, j, :]  # shape: (N_sessions,)
                # Bin into histogram
                hist, _ = np.histogram(edge_series, bins=10, range=(0, 1), density=True)
                hist = hist[hist > 0]
                probs = hist / np.sum(hist)
                ent = entropy(probs)
                ent_matrix[i, j] = ent
                ent_matrix[j, i] = ent  # symmetric

        temporal_entropy[feat][band] = ent_matrix  # shape: 124x12
        
# Node Entropy
node_entropy = {}  # shape: [feature][band] = 124 x 106 matrix

for feat in features:
    node_entropy[feat] = {}
    for band in bands:
        stack = stacked_data[feat][band]  # shape: 124 x 124 x 106
        n_nodes, _, n_sessions = stack.shape
        ent_matrix = np.zeros((n_nodes, n_sessions))

        for t in range(n_sessions):
            mat = stack[:, :, t]
            for i in range(n_nodes):
                row = np.copy(mat[i, :])
                row[i] = 0  # remove self-connection
                if np.sum(row) == 0:
                    ent_matrix[i, t] = 0
                else:
                    p = row / np.sum(row)
                    p = p[p > 0]  # remove zeros
                    ent_matrix[i, t] = entropy(p)

        node_entropy[feat][band] = ent_matrix  # shape: 124 x 106


# Plots
import matplotlib.pyplot as plt

# ------- TIME SERIES: ONE PLOT PER FEATURE, BANDS COLORED --------
fig, axs = plt.subplots(1, len(features), figsize=(18, 5), sharey=True)
fig.suptitle('Shannon Entropy of Adjacency Matrices Over Time (per Feature)', fontsize=16)

colors = {
    'Alpha': 'tab:blue',
    'Beta': 'tab:orange',
    'Delta': 'tab:green',
    'Gamma': 'tab:red',
    'Theta': 'tab:purple'
}

for i, feat in enumerate(features):
    ax = axs[i]
    for band in bands:
        ts = matrix_entropy[feat][band]
        ax.plot(ts, label=band, color=colors[band])
    ax.set_title(f'{feat}', fontsize=12)
    ax.set_xlabel('Session')
    if i == 0:
        ax.set_ylabel('Entropy')
    ax.grid(True)
    ax.legend(title="Band")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()



# ------- PLOT 2: TEMPORAL ENTROPY MATRICES -------
fig2, axs2 = plt.subplots(len(features), len(bands), figsize=(20, 10))
fig2.suptitle('Temporal Entropy of Each Connection Over Time', fontsize=16)

for i, feat in enumerate(features):
    for j, band in enumerate(bands):
        ax = axs2[i, j]
        mat = temporal_entropy[feat][band]  # shape: 124x124
        im = ax.imshow(mat, cmap='hot')
        ax.set_title(f'{feat} - {band}', fontsize=10)
        ax.axis('off')
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# ------- PLOT 3: NODE TEMPORAL ENTROPY ---------
from scipy.stats import linregress

for feat in features:
    fig3, axs3 = plt.subplots(1, len(bands), figsize=(25, 5), sharey=True)
    fig3.suptitle(f'Node-Wise Entropy Over Time — {feat}', fontsize=16)

    for j, band in enumerate(bands):
        ax3 = axs3[j]
        ent_matrix = node_entropy[feat][band]  # shape: 124 x 106

        # Plot all individual node entropy time series
        for i in range(ent_matrix.shape[0]):
            ax3.plot(ent_matrix[i, :], alpha=0.4, linewidth=0.8)

        # Compute mean entropy per session
        mean_entropy = np.mean(ent_matrix, axis=0)
        sessions = np.arange(mean_entropy.shape[0])

        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(sessions, mean_entropy)
        trend_line = slope * sessions + intercept

        # Plot the trend line
        ax3.plot(sessions, trend_line, color='black', linewidth=2, label='Trend')
        ax3.set_title(f'{band}', fontsize=12)
        ax3.set_xlabel('Session')
        if j == 0:
            ax3.set_ylabel('Node Entropy')
        ax3.grid(True)
        ax3.legend()
        plt.ylim(4.0, 4.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
#%% Lets now look into traditional graph descriptors 
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from networkx.algorithms.community import louvain_communities

features = ['PLV', 'MSC', 'CSD']
bands = ['Alpha', 'Beta', 'Delta', 'Gamma', 'Theta']

graph_metrics = {}
node_clustering = {}
communicability_metrics = {}
louvain_communities_dict = {}

for feat in features:
    graph_metrics[feat] = {}
    node_clustering[feat] = {}
    communicability_metrics[feat] = {}
    louvain_communities_dict[feat] = {}
    
    for band in bands:
        print(f"Processing {feat} - {band}")
        stack = stacked_data[feat][band]
        n_nodes, _, n_sessions = stack.shape

        avg_cluster_series = []
        assortativity_series = []
        node_cluster_matrix = np.zeros((n_nodes, n_sessions))
        comm_matrix = np.zeros((n_nodes, n_sessions))
        louvain_matrix = np.zeros((n_nodes, n_sessions))

        for t in tqdm(range(n_sessions), desc=f"{feat}-{band}", leave=False):
            mat = np.copy(stack[:, :, t])
            np.fill_diagonal(mat, 0)
            G = nx.from_numpy_array(mat)

            try:
                clustering_dict = nx.clustering(G, weight='weight')
                clustering_values = np.array(list(clustering_dict.values()))
                avg_clustering = np.mean(clustering_values)
                avg_cluster_series.append(avg_clustering)
                node_cluster_matrix[:, t] = clustering_values
            except:
                avg_cluster_series.append(np.nan)
                node_cluster_matrix[:, t] = np.nan

            try:
                assortativity = nx.degree_pearson_correlation_coefficient(G, weight='weight')
                if not np.isfinite(assortativity):
                    assortativity = np.nan
                assortativity_series.append(assortativity)
            except:
                assortativity_series.append(np.nan)

            try:
                comm = nx.communicability_exp(G)
                comm_values = np.array([comm[i][i] for i in range(n_nodes)])
                comm_matrix[:, t] = comm_values
            except:
                comm_matrix[:, t] = np.nan

            try:
                communities = louvain_communities(G, weight='weight', resolution=1)
                for i, comm_group in enumerate(communities):
                    for node in comm_group:
                        louvain_matrix[node, t] = i
            except:
                louvain_matrix[:, t] = -1

        graph_metrics[feat][band] = pd.DataFrame({
            'Session': np.arange(n_sessions),
            'AvgClustering': avg_cluster_series,
            'Assortativity': assortativity_series
        })
        node_clustering[feat][band] = node_cluster_matrix
        communicability_metrics[feat][band] = comm_matrix
        louvain_communities_dict[feat][band] = louvain_matrix

# --- Plot average clustering coefficient ---
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle('Average Clustering Coefficient Over Time per Feature', fontsize=16)

for i, feat in enumerate(features):
    ax = axs[i]
    for band in bands:
        df = graph_metrics[feat][band]
        ax.plot(df['Session'], df['AvgClustering'], label=band)
    ax.set_title(feat)
    ax.set_xlabel("Session")
    if i == 0:
        ax.set_ylabel("Avg Clustering Coeff.")
    ax.grid(True)
    ax.legend(title="Band")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Plot assortativity over time ---
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle('Degree Assortativity Over Time per Feature', fontsize=16)

for i, feat in enumerate(features):
    ax = axs[i]
    for band in bands:
        df = graph_metrics[feat][band]
        ax.plot(df['Session'], df['Assortativity'], label=band)
    ax.set_title(feat)
    ax.set_xlabel("Session")
    if i == 0:
        ax.set_ylabel("Assortativity")
    ax.grid(True)
    ax.legend(title="Band")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Plot node-wise clustering time series (heatmaps) ---
for feat in features:
    for band in bands:
        matrix = node_clustering[feat][band]
        plt.figure(figsize=(14, 6))
        sns.heatmap(matrix, cmap='viridis', cbar_kws={'label': 'Clustering Coeff'})
        plt.title(f'Node-Wise Clustering Coefficient Over Time — {feat} | {band}')
        plt.xlabel('Session')
        plt.ylabel('Node')
        plt.tight_layout()
        plt.show()

# --- Plot communicability (diagonal) over time ---
for feat in features:
    for band in bands:
        matrix = communicability_metrics[feat][band]
        plt.figure(figsize=(14, 6))
        sns.heatmap(matrix, cmap='magma', cbar_kws={'label': 'Communicability'})
        plt.title(f'Node Communicability Over Time — {feat} | {band}')
        plt.xlabel('Session')
        plt.ylabel('Node')
        plt.tight_layout()
        plt.show()

# --- Plot Louvain community detection over time ---
for feat in features:
    for band in bands:
        matrix = louvain_communities_dict[feat][band]
        plt.figure(figsize=(14, 6))
        sns.heatmap(matrix, cmap='tab20', cbar_kws={'label': 'Community ID'})
        plt.title(f'Louvain Communities Over Time — {feat} | {band}')
        plt.xlabel('Session')
        plt.ylabel('Node')
        plt.tight_layout()
        plt.show()

# --- Number of Communities Over Time ---
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle('Number of Louvain Communities per Session', fontsize=16)

for i, feat in enumerate(features):
    ax = axs[i]
    for band in bands:
        matrix = louvain_communities_dict[feat][band]
        n_sessions = matrix.shape[1]
        community_counts = []

        for t in range(n_sessions):
            labels = matrix[:, t]
            valid_labels = labels[labels >= 0]
            num_comms = len(np.unique(valid_labels))
            community_counts.append(num_comms)

        ax.plot(range(n_sessions), community_counts, label=band)

    ax.set_title(feat)
    ax.set_xlabel('Session')
    if i == 0:
        ax.set_ylabel('Num. of Communities')
    ax.grid(True)
    ax.legend(title="Band")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
