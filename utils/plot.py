import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import pairwise_kernels

def compute_mmd(X, Y, kernel="rbf", gamma=None):
    """Compute Maximum Mean Discrepancy (MMD) between X and Y."""
    XX = pairwise_kernels(X, X, metric=kernel, gamma=gamma)
    YY = pairwise_kernels(Y, Y, metric=kernel, gamma=gamma)
    XY = pairwise_kernels(X, Y, metric=kernel, gamma=gamma)
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd

def compute_cka(X, Y):
    """Compute Centered Kernel Alignment (CKA) between X and Y."""
    def center_gram(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    K_X = X @ X.T
    K_Y = Y @ Y.T
    K_X_centered = center_gram(K_X)
    K_Y_centered = center_gram(K_Y)
    hsic = np.sum(K_X_centered * K_Y_centered)
    norm_X = np.sqrt(np.sum(K_X_centered ** 2))
    norm_Y = np.sqrt(np.sum(K_Y_centered ** 2))
    return hsic / (norm_X * norm_Y)


def plot_tsne_modility(
    emb_all,
    vis_2d,
    labels,
    vis_code_tree_c=None,
    epoch=0,
    level=0,
):
        
    print('emb_all.shape, vis_2d.shape, labels.shape', emb_all.shape, vis_2d.shape, labels.shape)

    label_to_name = {
        0: "Molecule 1D",
        1: "Molecule 2D",
        2: "Molecule 3D",
        3: "Gene",
        4: "Cell",
        5: "Gene Expression",
    }
    label_to_color = {
        0: "#1f77b4",  
        1: "#ff7f0e",  
        2: "#2ca02c",  
        3: "#F3EF0B",  
        4: "#9467bd",  
        5: "#F432bd", 
    }

    plt.figure(figsize=(10, 8), dpi=100)
    for label, name in label_to_name.items():
        idx = labels == label
        plt.scatter(
            vis_2d[idx, 0],
            vis_2d[idx, 1],
            color=label_to_color[label],
            label=name,
            alpha=0.7,
            s=1,
        )
    plt.scatter(
        vis_code_tree_c[:, 0],
        vis_code_tree_c[:, 1],
        color="#db0909",  
        label="Tree Code",
        alpha=0.9,
        s=20,
        edgecolors="black",
        )
    
    
    plt.tick_params(axis="both", which="major", labelsize=14)
    #plt.legend(fontsize=14, loc="upper right", markerscale=5)
    plt.title(f"t-SNE Visualization at Epoch {epoch}", fontsize=16)

    if os.path.exists("vis_tnse") is False:
        os.makedirs("vis_tnse")

    path = f"vis_tsne/Ours_m_tsne_epoch_{epoch}_level{level}.png"
    plt.savefig(path)
    plt.close()
    return path

def plot_tsne_cluster(
    emb_all: np.ndarray,       
    vis_2d: np.ndarray,        
    labels: np.ndarray = None,  
    vis_code_tree_c: np.ndarray = None, 
    epoch: int = 0,
    level: int = 0,
) -> str:

    # import pdb; pdb.set_trace()  # DEBUG: Check input shapes
    # assert vis_2d.shape[1] == emb_all.shape[1], "Mismatch between 2D projection and original embedding."

    dists = cdist(vis_2d, vis_code_tree_c)  # (N, K)
    labels = np.argmin(dists, axis=1)  


    label_name_list = np.unique(labels)

    cmap_list = ['tab20', 'Set1', 'Set2', 'Set3', 'Accent', 'Paired', 'Dark2']
    color_palette = []

    for cmap_name in cmap_list:
        cmap = plt.cm.get_cmap(cmap_name)
        colors = [cmap(i) for i in range(cmap.N)]
        color_palette.extend(colors)

    color_palette = list({tuple(c): c for c in color_palette}.values())

    while len(color_palette) < 64:
        color_palette.extend(color_palette)

    color_palette = color_palette[:64]

    label_to_color = {label: color_palette[i % len(color_palette)] for i, label in enumerate(label_name_list)}

    plt.figure(figsize=(10, 8), dpi=100)
    for label in label_name_list:
        idx = labels == label
        plt.scatter(
            vis_2d[idx, 0],
            vis_2d[idx, 1],
            color=label_to_color[label],
            label=str(label),
            alpha=0.7,
            s=1,
        )

    if vis_code_tree_c is not None:
        plt.scatter(
            vis_code_tree_c[:, 0],
            vis_code_tree_c[:, 1],
            color="#db0909",
            label="Center",
            alpha=0.9,
            s=40,
            edgecolors="black",
        )

    plt.tick_params(axis="both", which="major", labelsize=14)
    # plt.legend(fontsize=12, loc="upper right", markerscale=5)
    plt.title(f"t-SNE Visualization at Epoch {epoch}, Level {level}", fontsize=16)

    os.makedirs("vis_tsne", exist_ok=True)
    path = f"vis_tsne/Ours_h_tsne_epoch_{epoch}_level{level}.png"
    plt.savefig(path)
    plt.close()
    return path


def label_to_cluster_center(data, labels):

    unique_labels = np.unique(labels)
    centers = []
    
    for label in unique_labels:
        idx = labels == label
        center = np.mean(data[idx], axis=0)
        centers.append(center)
    
    return np.array(centers)




def plot_tsne(emb_all, labels, epoch=0, m0_tree_rout_dict=None):

    
    cluster_center_list_level_high = []
    index_center = []
    for level in range(m0_tree_rout_dict.shape[1]):
        cluster_center = label_to_cluster_center(
            emb_all[:m0_tree_rout_dict.shape[0]],
            m0_tree_rout_dict[:, level].detach().cpu().numpy(),
        )
        cluster_center_list_level_high.append(cluster_center)
        index_center.append((
            np.concatenate(cluster_center_list_level_high).shape[0] - cluster_center.shape[0],
            np.concatenate(cluster_center_list_level_high).shape[0]
            ))


    emb_all = np.unique(emb_all, axis=0)

    num_points_aim = 5000
    if emb_all.shape[0] > num_points_aim:
        indices = np.random.choice(emb_all.shape[0], num_points_aim, replace=False)
        emb_all = emb_all[indices]
        labels = labels[indices]

    labels = np.array(labels, dtype=int)
    
    drer = TSNE(n_components=2, random_state=42, perplexity=20)
    data_center = np.concatenate(cluster_center_list_level_high, axis=0)
    num_data = emb_all.shape[0]
    
    data_all = np.concatenate([emb_all, data_center], axis=0)
    
    vis_all = drer.fit_transform(data_all)
    vis_emb = vis_all[:num_data, :]
    vis_center = vis_all[num_data:, :]

        

    
    path_dict = {}
    for level in range(6):
        
        start = index_center[level][0] if level > 0 else 0
        end = index_center[level][1] if level < len(index_center) else vis_center.shape[0]
        
        vis_code_tree_c = vis_emb[start:end]
        path = plot_tsne_modility(
            emb_all,
            vis_emb,
            labels,
            vis_code_tree_c=vis_code_tree_c,
            epoch=epoch,
            level=level
        )
        path_dict['modility_' + str(level)] = path

        path = plot_tsne_cluster(
            emb_all,
            vis_emb,
            labels,
            vis_code_tree_c=vis_code_tree_c,
            epoch=epoch,
            level=level
        )
        path_dict['cluster_' + str(level)] = path

        
    return path_dict
