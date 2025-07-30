import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_pyg import read_graph_pyg

import torch
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Lipinski import *
from rdkit.Chem import AllChem


def get_maccs_fingerprint(mol):
    """
    Generate MACCS Keys fingerprint, return 167-bit 0/1 list.
    """
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return [int(b) for b in fp.ToBitString()]

def smiles_to_fingerprint(df, 
                         smiles_col="smiles",
                         morgan_radius=1,
                         morgan_n_bits=1024,
                         rdk_max_path=5,
                         rdk_fp_size=1024,
                         include_rdk=False,
                         include_maccs=True):

    smiles_list = df[smiles_col].tolist()
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]

    morgan_fps = []
    for mol in mols:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, morgan_radius, nBits=morgan_n_bits)
        arr = np.zeros((morgan_n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        morgan_fps.append(arr)
    morgan_fps = np.array(morgan_fps, dtype=np.int8)

    rdk_fps = []
    if include_rdk:
        for mol in mols:
            fp = Chem.RDKFingerprint(mol, maxPath=rdk_max_path, fpSize=rdk_fp_size)
            arr = np.zeros((rdk_fp_size,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            rdk_fps.append(arr)
        rdk_fps = np.array(rdk_fps, dtype=np.int8)

    maccs_fps = []
    if include_maccs:
        maccs_fps = np.array([get_maccs_fingerprint(mol) for mol in mols],
                             dtype=np.int8)

    combined = [morgan_fps]
    if include_rdk:
        combined.append(rdk_fps)
    if include_maccs:
        combined.append(maccs_fps)

    combined = np.concatenate(combined, axis=1)
    return torch.tensor(combined)

import numpy as np
from unimol_tools import UniMolRepr

class PygGraphPropPredDataset(InMemoryDataset):
    def __init__(self, name, root = 'dataset', transform=None, pre_transform = None, meta_dict = None):
        self.name = name
        
        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-')) 
            if osp.exists(osp.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'

            self.original_root = root
            self.root = osp.join(root, self.dir_name)
            
            master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col=0, keep_default_na=False)
            if not self.name in master:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]
            
        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict

        if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.root)

        self.download_name = self.meta_info['download_name']
        self.num_tasks = int(self.meta_info['num tasks'])
        self.eval_metric = self.meta_info['eval metric']
        self.task_type = self.meta_info['task type']
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.binary = self.meta_info['binary'] == 'True'

        super(PygGraphPropPredDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_idx_split(self, split_type = None):
        if split_type is None:
            split_type = self.meta_info['split']
            
        path = osp.join(self.root, 'split', split_type)

        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]
        valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]
        test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header = None).values.T[0]

        return {'train': torch.tensor(train_idx, dtype = torch.long), 'valid': torch.tensor(valid_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        if self.binary:
            return ['data.npz']
        else:
            file_names = ['edge']
            if self.meta_info['has_node_attr'] == 'True':
                file_names.append('node-feat')
            if self.meta_info['has_edge_attr'] == 'True':
                file_names.append('edge-feat')
            return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        url = self.meta_info['url']
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)
        else:
            print('Stop downloading.')
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        if self.meta_info['additional node files'] == 'None':
            additional_node_files = []
        else:
            additional_node_files = self.meta_info['additional node files'].split(',')

        if self.meta_info['additional edge files'] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info['additional edge files'].split(',')

        data_list = read_graph_pyg(
            self.raw_dir,
            add_inverse_edge=add_inverse_edge,
            additional_node_files=additional_node_files,
            additional_edge_files=additional_edge_files,
            binary=self.binary
        )

        mol_csv_path = os.path.join(self.raw_dir, 'assays.csv.gz')
        if not os.path.exists(mol_csv_path):
            raise FileNotFoundError(f"{mol_csv_path} not found")

        mol_df = pd.read_csv(mol_csv_path, compression='gzip')
        mol_data = smiles_to_fingerprint(mol_df)

        if len(mol_data) != len(data_list):
            raise ValueError(f"mol_data rows ({len(mol_data)}) and graphs ({len(data_list)}) mismatch")

        clf = UniMolRepr(data_type='molecule', remove_hs=False)
        unimol_repr = clf.get_repr(mol_df["smiles"].tolist(), return_atomic_reprs=True)
        unimol_feat = torch.tensor(unimol_repr['cls_repr'])

        # Only load rf_pred for specific datasets
        use_rf_pred = self.name.split("-")[-1] in ["molbace", "molclintox", "molhiv", "molsider"]
        if use_rf_pred:
            mol_rf_path = os.path.join(self.raw_dir, 'rf_pred.npy')
            if not os.path.exists(mol_rf_path):
                raise FileNotFoundError(f"{mol_rf_path} not found for dataset {self.name}")
            rf_pred = np.load(mol_rf_path)
            rf_pred = torch.tensor(rf_pred)

        for i, data in enumerate(data_list):
            data.mol_features = mol_data[i].unsqueeze(0)
            data.unimol_features = unimol_feat[i].unsqueeze(0)
            if use_rf_pred:
                data.rf_pred = rf_pred[i].unsqueeze(0)


        if self.task_type == 'subtoken prediction':
            graph_label_notparsed = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values
            graph_label = [str(graph_label_notparsed[i][0]).split(' ') for i in range(len(graph_label_notparsed))]

            for i, g in enumerate(data_list):
                g.y = graph_label[i]

        else:
            if self.binary:
                graph_label = np.load(osp.join(self.raw_dir, 'graph-label.npz'))['graph_label']
            else:
                graph_label = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values

            has_nan = np.isnan(graph_label).any()

            for i, g in enumerate(data_list):
                if 'classification' in self.task_type:
                    if has_nan:
                        g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.float32)
                    else:
                        g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.long)
                else:
                    g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.float32)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    pyg_dataset = PygGraphPropPredDataset(name='ogbg-code2')
    print(pyg_dataset.num_classes)
    split_index = pyg_dataset.get_idx_split()
    print(pyg_dataset[0])
    print([pyg_dataset[i].x[1] for i in range(100)])
    print(pyg_dataset[split_index['train']])
    print(pyg_dataset[split_index['valid']])
    print(pyg_dataset[split_index['test']])
