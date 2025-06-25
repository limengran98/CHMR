import os
import torch
import logging
import pandas as pd
import numpy as np
import os.path as osp
from torch_geometric.data import InMemoryDataset
from sklearn.model_selection import train_test_split

from .data_utils import read_graph_list
from rdkit import Chem
from rdkit.Chem import AllChem

import random
def get_maccs_fingerprint(mol):
    """
    生成 MACCS Keys 指纹，返回长度 167 的 0/1 列表
    """
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return [int(b) for b in fp.ToBitString()]

logger = logging.getLogger(__name__)

class PretrainMoleculeDataset(InMemoryDataset):
    def __init__(self, name='pretrain', root ='raw_data', transform=None, pre_transform = None):
        '''
            - name (str): name of the pretraining dataset: pretrain_all
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects
        ''' 
        self.name = name
        self.dir_name = '_'.join(name.split('-'))
        self.original_root = root
        self.root = osp.join(root, self.dir_name)
        self.processed_root = osp.join(osp.abspath(self.root))

        self.num_tasks = 1
        self.eval_metric = 'customize'
        self.task_type = 'pretrain'
        self.__num_classes__ = '-1'
        self.binary = 'False'

        super(PretrainMoleculeDataset, self).__init__(self.processed_root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.total_data_len = self.__len__()

    
    def get_idx_split(self):
        full_idx = list(range(self.total_data_len))
        num_test = max(1, int(0.01 * self.total_data_len))  
        test_idx = random.sample(full_idx, num_test)  
        train_idx = list(set(full_idx))  
        valid_idx = []  

        return {
            'train': torch.tensor(train_idx, dtype=torch.long),
            'valid': torch.tensor(valid_idx, dtype=torch.long),
            'test': torch.tensor(test_idx, dtype=torch.long)
        }

    @property
    def processed_file_names(self):
        return ['mol_data_processed.pt']

    def process(self):

        mol_data_path = osp.join(self.root, 'raw', 'structure.csv.gz')
        print('Processing molecule data at folder: ' , mol_data_path)

        mol_df = pd.read_csv(mol_data_path, compression='gzip')
        mol_df = mol_df.drop_duplicates(subset="mol_id")
        
        mol_3d_path = osp.join(self.root, 'raw', 'cls_repr_3d_unimol.pt')
        cls_tensor = torch.load(mol_3d_path)

        mol_1d_path = osp.join(self.root, 'raw', 'structure.csv.gz')
        structure_df = pd.read_csv(mol_1d_path)
        mol_df = structure_df.drop_duplicates(subset="mol_id")
        smiles_list = mol_df["smiles"].tolist()
        molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        structure_feature = np.array(
            [AllChem.GetMorganFingerprintAsBitVect(m, 4, nBits=1024) for m in molecules]
        )

        maccs_fps = np.array([get_maccs_fingerprint(mol) for mol in molecules],
                        dtype=np.int8)
        tensor_fp = torch.tensor(np.concatenate((structure_feature, maccs_fps), axis=1))

        data_list = read_graph_list(mol_df, cls_tensor, tensor_fp, keep_id=True)

        self.total_data_len = len(data_list)

        print('Pretrain molecule data loading finished with length ', self.total_data_len)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])




# main file
if __name__ == "__main__":
    pass