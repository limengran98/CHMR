import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import DataStructs
from ogb.graphproppred import GraphPropPredDataset
from rdkit.Chem import MACCSkeys 
import numpy as np
from unimol_tools import UniMolRepr
# from unimol_tools import UniMolRepr

def getmorganfingerprint(mol):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2))

def getmaccsfingerprint(mol):
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return [int(b) for b in fp.ToBitString()]

def getrdkfingerprint(mol, rdk_max_path=7, rdk_fp_size=1024):
    fp = Chem.RDKFingerprint(mol, maxPath=rdk_max_path, fpSize=rdk_fp_size)
    arr = np.zeros((rdk_fp_size,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    rdk_fps = arr.tolist()
    
    return rdk_fps

def main(dataset_name):
    #dataset = GraphPropPredDataset(name=dataset_name)


    if dataset_name in ['chembl2k', "broad6k", "biogenadme"]:
        df_smi = pd.read_csv(f"./raw_data/{dataset_name}/raw/assays.csv.gz")
        smiles = df_smi["smiles"]
    else:
        df_smi = pd.read_csv(f"./raw_data/{dataset_name}/mapping/mol.csv.gz".replace("-", "_"))
        smiles = df_smi["smiles"]


    mgf_feat_list = []
    maccs_feat_list = []
    rdk_feat_list = []
    for ii in tqdm(range(len(smiles))):
        rdkit_mol = AllChem.MolFromSmiles(smiles.iloc[ii])

        mgf = getmorganfingerprint(rdkit_mol)
        mgf_feat_list.append(mgf)

        maccs = getmaccsfingerprint(rdkit_mol)
        maccs_feat_list.append(maccs)

        rdk = getrdkfingerprint(rdkit_mol)
        rdk_feat_list.append(rdk)


    mgf_feat = np.array(mgf_feat_list, dtype="int64")
    maccs_feat = np.array(maccs_feat_list, dtype="int64")
    rdk_feat = np.array(rdk_feat_list, dtype="int64")

    clf = UniMolRepr(data_type='molecule', remove_hs=False)
    smiles_list = df_smi['smiles'].tolist()
    unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=True)
    unimol_feat = np.array(unimol_repr['cls_repr'])

    

    print("morgan feature shape: ", mgf_feat.shape)
    print("maccs feature shape: ", maccs_feat.shape)
    print("rdk feature shape: ", rdk_feat.shape)
    print("unimol shape: ", unimol_feat.shape)

    save_path = f"./raw_data/{dataset_name}".replace("-", "_")
    print("saving feature in %s" % save_path)
    np.save(os.path.join(save_path, "mgf_feat.npy"), mgf_feat)
    np.save(os.path.join(save_path, "maccs_feat.npy"), maccs_feat)
    np.save(os.path.join(save_path, "rdk_feat.npy"), rdk_feat)
    np.save(os.path.join(save_path, "unimol_feat.npy"), unimol_feat)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='gnn')
    parser.add_argument("--dataset_name", type=str, default="ogbg-molhiv")
    args = parser.parse_args()

    main(args.dataset_name)