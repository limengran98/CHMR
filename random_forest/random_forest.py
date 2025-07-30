import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from ogb.graphproppred import GraphPropPredDataset

def main(args):
    all_probs = {}
    train_label_props = {}

    dataset_name = args.dataset_name
    dataset_key = dataset_name.replace("-", "_")

    # === Dataset-specific hyperparameters ===
    if dataset_name == "ogbg-molbace":
        n_estimators = 500
        rep = 7
        class_weight = {0: 1, 1: 1.43}
        feat = np.concatenate([
            np.load(f"./raw_data/{dataset_key}/mgf_feat.npy"),
            np.load(f"./raw_data/{dataset_key}/maccs_feat.npy"),
            np.load(f"./raw_data/{dataset_key}/rdk_feat.npy")
        ], axis=1)

    elif dataset_name == "ogbg-molclintox":
        n_estimators = 75
        rep = 5
        class_weight = {0: 1, 1: 10}
        feat = np.concatenate([
            np.load(f"./raw_data/{dataset_key}/mgf_feat.npy"),
            np.load(f"./raw_data/{dataset_key}/maccs_feat.npy"),
            np.load(f"./raw_data/{dataset_key}/unimol_feat.npy")
        ], axis=1)

    elif dataset_name == "ogbg-molhiv":
        n_estimators = 1000
        rep = 0
        class_weight = {0: 1, 1: 10}
        feat = np.concatenate([
            np.load(f"./raw_data/{dataset_key}/mgf_feat.npy"),
            np.load(f"./raw_data/{dataset_key}/maccs_feat.npy")
        ], axis=1)

    elif dataset_name == "ogbg-molsider":
        n_estimators = 500
        rep = 4
        class_weight = {0: 1, 1: 3.4}
        feat = np.concatenate([
            np.load(f"./raw_data/{dataset_key}/mgf_feat.npy"),
            np.load(f"./raw_data/{dataset_key}/maccs_feat.npy"),
            np.load(f"./raw_data/{dataset_key}/rdk_feat.npy")
        ], axis=1)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    print("features size:", feat.shape[1])

    # === Load data
    dataset = GraphPropPredDataset(name=dataset_name, root="./raw_data/")
    df_smi = pd.read_csv(f"./raw_data/{dataset_key}/mapping/mol.csv.gz")
    smiles = df_smi["smiles"]
    outcomes = df_smi.set_index("smiles").drop(["mol_id"], axis=1)

    X = pd.DataFrame(feat, index=smiles, columns=[f"f{i}" for i in range(feat.shape[1])])

    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]

    # eval_scores, test_scores = [], []
    # eval_ap_scores, test_ap_scores = [], []
    all_prob_matrix = []

    for target in tqdm(outcomes.columns):
        val_key = dataset_name, target, rep, "val"
        test_key = dataset_name, target, rep, "test"

        if val_key in all_probs:
            print("Skipping", val_key[:-1])
            continue

        Y = outcomes[target]
        y_train, y_val, y_test = Y.loc[X_train.index], Y.loc[X_val.index], Y.loc[X_test.index]

        if y_train.sum() == 0:
            continue

        y_val, y_test = y_val.dropna(), y_test.dropna()
        X_v, X_t = X_val.loc[y_val.index], X_test.loc[y_test.index]
        y_tr = y_train.dropna()
        train_label_props[dataset_name, target, rep] = y_tr.mean()

        print(f"Fitting model for {target}...")
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            criterion='entropy',
            class_weight=class_weight,
            random_state=rep
        )
        rf.fit(X_train.loc[y_tr.index], y_tr)

        all_probs[val_key] = pd.Series(rf.predict_proba(X_v)[:, 1], index=X_v.index)
        all_probs[test_key] = pd.Series(rf.predict_proba(X_t)[:, 1], index=X_t.index)
        all_prob_matrix.append(rf.predict_proba(X)[:, 1])


    # === Save prediction matrix
    all_prob_matrix = np.stack(all_prob_matrix, axis=1)
    save_dir = f"./raw_data/{dataset_key}/raw"
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "rf_pred.npy"), all_prob_matrix)
    print("Done!")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ogbg-molhiv")
    args = parser.parse_args()
    main(args)
