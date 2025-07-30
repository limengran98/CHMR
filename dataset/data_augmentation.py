import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from tqdm import tqdm
import os
import hnswlib

def build_similarity_matrix(mol_features, top_k, cache_path):
    """
    Use HNSWlib approximate nearest neighbor search for top-K neighbors.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached similarity matrix: {cache_path}")
        loader = np.load(cache_path)
        W = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
        return W

    print("Computing similarity matrix with HNSWlib (approximate)...")
    N = len(mol_features)

    # Convert bitvectors to numpy dense matrix (uint8)
    fp_array = np.zeros((N, 1024), dtype=np.float32)
    for i, fp in enumerate(mol_features):
        arr = np.zeros((1,), dtype=np.uint8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        fp_array[i] = arr

    # Build HNSW index
    dim = 1024  # Vector dimension
    index = hnswlib.Index(space='l2', dim=dim)  # Use L2 distance for similarity
    index.init_index(max_elements=N, ef_construction=200, M=16)

    # Add data to the index
    index.add_items(fp_array)

    # Search top-K neighbors
    labels, distances = index.knn_query(fp_array, k=top_k+1)  # includes self in neighbors

    rows, cols, data = [], [], []
    for i in range(N):
        for j, dist in zip(labels[i][1:], distances[i][1:]):  # skip self (index 0)
            rows.append(i)
            cols.append(j)
            data.append(1 - dist)  # Convert distance to similarity (1 - distance)

    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    W = normalize(A, norm='l1', axis=1)

    # Cache
    np.savez_compressed(
        cache_path,
        data=W.data, indices=W.indices, indptr=W.indptr, shape=W.shape
    )
    print(f"Saved similarity matrix: {cache_path}")
    return W

def propagate_features(X, W, known_mask, propagate_steps=3, report_missing=True, label="", fallback="nearest"):
    """
    Propagate features via sparse matrix multiplication.
    """
    N, feat_dim = X.shape

    if report_missing:
        # ✅ Step 0: Report initial missing rate (relative to total samples)
        missing_mask = (X == 0).all(axis=1) & (~known_mask)
        initial_missing = missing_mask.sum()
        initial_missing_rate = initial_missing / N
        print(
            f"{label} - Propagation step 0/{propagate_steps}: "
            f"Initial missing rate = {initial_missing_rate*100:.2f}%, "
            f"Missing samples = {initial_missing}/{N}"
        )

    for step in range(propagate_steps):
        X = W.dot(X)
        # Restore known values
        X[known_mask] = X[known_mask]
        if report_missing:
            # Check how many rows are still fully zero
            missing_mask = (X == 0).all(axis=1) & (~known_mask)
            missing = missing_mask.sum()
            missing_rate = missing / N
            print(
                f"{label} - Propagation step {step+1}/{propagate_steps}: "
                f"Remaining missing rate = {missing_rate*100:.2f}%, "
                f"Missing samples = {missing}/{N}"
            )

    # Fallback for nodes still missing
    missing_mask = (X == 0).all(axis=1) & (~known_mask)
    final_missing = missing_mask.sum()
    final_missing_rate = final_missing / N
    print(
        f"{label} - Final missing rate after propagation: "
        f"{final_missing_rate*100:.2f}%, "
        f"Missing samples = {final_missing}/{N}"
    )

    if final_missing_rate > 0:
        print(f"⚠️  {label}: {missing_mask.sum()} nodes still missing after propagation.")

        if fallback == "mean":
            print(f"{label}: Filling missing nodes with mean value.")
            mean_val = np.nanmean(X[known_mask], axis=0)
            X[missing_mask] = mean_val

        elif fallback == "nearest":
            print(f"{label}: Filling missing nodes with nearest neighbor.")
            for i in np.where(missing_mask)[0]:
                # Use W row to find neighbors
                neighbors = W[i].nonzero()[1]
                if len(neighbors) > 0:
                    X[i] = X[neighbors[0]]  # Take first neighbor
                else:
                    X[i] = np.nanmean(X[known_mask], axis=0)  # fallback to mean
        else:
            print(f"{label}: No fallback, missing nodes remain zero.")

    return X


def build_feature_table_graph_matrix(
    mol_ids, mol_features, target_ids, features, fill_type, label,
    W=None, propagate_steps=3
):
    """
    Aligns and fills missing features using precomputed similarity matrix W.
    """
    tqdm.write(f"Building feature table for {label} (fill='{fill_type}', steps={propagate_steps})...")
    feat_dim = features.shape[1]
    target_id_to_index = {k: i for i, k in enumerate(target_ids)}
    N = len(mol_ids)

    # Initialize feature matrix
    X = np.full((N, feat_dim), np.nan)
    known_mask = np.zeros(N, dtype=bool)

    # Fill known values
    for i, mid in enumerate(mol_ids):
        if mid in target_id_to_index:
            X[i] = features[target_id_to_index[mid]]
            known_mask[i] = True

    if fill_type == "zero":
        X[~known_mask] = 0
        return X

    elif fill_type == "mean":
        mean_val = np.nanmean(features, axis=0)
        X[~known_mask] = mean_val
        return X

    elif fill_type == "graph":
        if W is None:
            raise ValueError("Similarity matrix W must be provided for 'graph' fill_type")
        # Initialize missing values to 0
        X[np.isnan(X)] = 0
        # Propagate
        X = propagate_features(X, W, known_mask, propagate_steps=propagate_steps, label=label)
        return X

    else:
        raise ValueError(f"Unknown fill_type: {fill_type}")

def augment_modalities(folder, fill_method='graph', top_k=10, propagate_steps=5):
    """
    Aligns modalities by mol_id and fills missing data using specified strategy.
    Results are cached in an .npz file to avoid recomputation.

    fill_method: 'mean', 'zero', 'graph'
    """
    cache_path = os.path.join(folder, f"aligned_modalities_{fill_method}2.npz")

    if os.path.exists(cache_path):
        print(f"Loading cached aligned modalities: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    # ==== Step 1: Load molecule data and calculate fingerprint ====
    mol_df = pd.read_csv(f"{folder}/structure.csv.gz").drop_duplicates(subset="mol_id")
    # mol_df = mol_df.sample(frac=0.1, random_state=42)
    mol_ids = mol_df["mol_id"].values
    smiles_list = mol_df["smiles"].tolist()

    print("Generating molecular fingerprints...")
    mol_features = []
    for smi in tqdm(smiles_list, desc="SMILES to fingerprint"):
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        mol_features.append(fp)  # Keep as list of ExplicitBitVect

    # ==== Step 2: Load and handle modalities separately ====
    tqdm.write("Loading modality data...")
    cp_bray_df = pd.read_csv(f"{folder}/CP-Bray.csv.gz")
    cp_bray_feat = np.load(f"{folder}/CP-Bray_feature.npz")["data"]

    cp_jump_df = pd.read_csv(f"{folder}/CP-JUMP.csv.gz")
    cp_jump_feat = np.load(f"{folder}/CP-JUMP_feature.npz")["data"]

    crispr_df = pd.read_csv(f"{folder}/G-CRISPR.csv.gz")
    crispr_feat = np.load(f"{folder}/G-CRISPR_feature.npz")["data"]

    orf_df = pd.read_csv(f"{folder}/G-ORF.csv.gz")
    orf_feat = np.load(f"{folder}/G-ORF_feature.npz")["data"]

    ge_df = pd.read_csv(f"{folder}/GE.csv.gz")
    ge_feat = np.load(f"{folder}/GE_feature.npz")["data"]
    express_ids = ge_df["mol_id"].values
    express_feat = ge_feat

    # ==== Step 3: Precompute similarity matrix if needed ====
    if fill_method == "graph":
        similarity_cache = os.path.join(folder, f"similarity_matrix_top{top_k}.npz")
        W = build_similarity_matrix(mol_features, top_k, cache_path=similarity_cache)
    else:
        W = None

    # ==== Step 4: Align and fill features for each modality ====
    aligned_cp_bray_feat = build_feature_table_graph_matrix(
        mol_ids, mol_features, cp_bray_df["mol_id"].values, cp_bray_feat,
        fill_method, "CP-Bray", W=W, propagate_steps=propagate_steps
    )
    aligned_cp_jump_feat = build_feature_table_graph_matrix(
        mol_ids, mol_features, cp_jump_df["mol_id"].values, cp_jump_feat,
        fill_method, "CP-JUMP", W=W, propagate_steps=propagate_steps
    )
    aligned_crispr_feat = build_feature_table_graph_matrix(
        mol_ids, mol_features, crispr_df["mol_id"].values, crispr_feat,
        fill_method, "G-CRISPR", W=W, propagate_steps=propagate_steps
    )
    aligned_orf_feat = build_feature_table_graph_matrix(
        mol_ids, mol_features, orf_df["mol_id"].values, orf_feat,
        fill_method, "G-ORF", W=W, propagate_steps=propagate_steps
    )
    aligned_express_feat = build_feature_table_graph_matrix(
        mol_ids, mol_features, express_ids, express_feat,
        fill_method, "Expression", W=W, propagate_steps=propagate_steps
    )


    # # ==== Step 4: Save cache ====    
    # print("Saving to cache:", cache_path)
    # np.savez_compressed(cache_path,
    #     mol_id=mol_ids,
    #     mol_feat=mol_features,
    #     crispr_feat=aligned_crispr_feat,
    #     orf_feat=aligned_orf_feat,
    #     cp_bray_feat=aligned_cp_bray_feat,
    #     cp_jump_feat=aligned_cp_jump_feat,
    #     express_feat=aligned_express_feat
    # )

    return {
        "mol_id": mol_ids,
        "mol_feat": mol_features,
        "crispr_feat": aligned_crispr_feat,
        "orf_feat": aligned_orf_feat,
        "cp_bray_feat": aligned_cp_bray_feat,
        "cp_jump_feat": aligned_cp_jump_feat,
        "express_feat": aligned_express_feat
    }


