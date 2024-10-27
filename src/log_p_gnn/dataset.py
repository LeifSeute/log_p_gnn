"""
Defines a dataset class for dgl graphs along with their corresponding mol_name and mol_tag.
"""

from torch.utils.data import Dataset
import dgl
import torch
from torch.utils.data.dataset import ConcatDataset
from .data_utils import get_hierarchical_graph, load_nx_dataset, is_single_bead
from typing import Dict, List
from pytorch_lightning import LightningDataModule
import random
from torch.utils.data import DataLoader
import copy
import logging

class DGLDataset(Dataset):
    """
    A dataset storing graphs, mol_names and mol_tags.
    The graphs have the following node types:
    - global: a single node storing global features such as logp values (keys: OCO, HD, CLF)
    - aa: nodes representing atoms in the atomistic graph
    - cg: nodes representing coarse-grained particles in the coarse-grained graph

    The graphs have the following edge types:
    - aa_to_aa: edges between atoms in the atomistic graph
    - cg_to_cg: edges between particles in the coarse-grained graph
    - aa_to_cg: edges between atoms and particles in the hierarchical graph
    - cg_to_aa: edges between particles and atoms in the hierarchical graph

    The dataset is constructed from a list of atomistic and coarse-grained graphs.
    """
    def __init__(self, graphs:dgl.DGLGraph, mol_names:List[str], mol_tags:List[str]):
        """
        Constructs a dataset of DGL graphs.
        """
        self.graphs = graphs
        self.mol_names = mol_names
        self.mol_tags = mol_tags

    
    @classmethod
    def load(cls, dspath:str=None):
        """
        Loads the dataset from a path to a .csv file containing mol_name, mol_tag cgsmiles and logP values. 
        """
        aa_graphs, cg_graphs, mol_names, mol_tags, log_ps = load_nx_dataset(dspath)
        
        graphs = cls._construct_dgl_graphs(aa_graphs=aa_graphs, cg_graphs=cg_graphs, targets=log_ps)
        
        mol_names = mol_names
        mol_tags = mol_tags

        return cls(graphs, mol_names, mol_tags)


    @staticmethod
    def _construct_dgl_graphs(aa_graphs, cg_graphs, targets:Dict[str,List]={}):
        """
        Turn a list of atomistic and coarse-grained graphs into a list of DGL graphs.
        """
        graphs = []
        for i, (aa_graph, cg_graph) in enumerate(zip(aa_graphs, cg_graphs)):
            g = get_hierarchical_graph(aa_graph=aa_graph, cg_graph=cg_graph, featurize=True)
            for k, v in targets.items():
                # Add target values to the global node
                # Add a new axis because dim0 of a global feature tensor must be 1
                g.nodes['global'].data[k] = torch.tensor([v[i]]).unsqueeze(dim=0)

            graphs.append(g)
        return graphs
        
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]
    
    def collate_fn(self, batch):
        return dgl.batch(batch)
    
    def to(self, device):
        for g in self.graphs:
            g.to(device)
        return self

    def split(self, train_tags, val_tags, test_tags):
        """
        Splits the dataset into train, validation and test sets.
        """
        train_idx = [i for i, mol_tag in enumerate(self.mol_tags) if mol_tag in train_tags]
        val_idx = [i for i, mol_tag in enumerate(self.mol_tags) if mol_tag in val_tags]
        test_idx = [i for i, mol_tag in enumerate(self.mol_tags) if mol_tag in test_tags]
        
        train_graphs = [self.graphs[i] for i in train_idx]
        val_graphs = [self.graphs[i] for i in val_idx]
        test_graphs = [self.graphs[i] for i in test_idx]

        train_mol_names = [self.mol_names[i] for i in train_idx]
        val_mol_names = [self.mol_names[i] for i in val_idx]
        test_mol_names = [self.mol_names[i] for i in test_idx]

        train_mol_tags = [self.mol_tags[i] for i in train_idx]
        val_mol_tags = [self.mol_tags[i] for i in val_idx]
        test_mol_tags = [self.mol_tags[i] for i in test_idx]

        train_dataset = DGLDataset(train_graphs, train_mol_names, train_mol_tags)
        val_dataset = DGLDataset(val_graphs, val_mol_names, val_mol_tags)
        test_dataset = DGLDataset(test_graphs, test_mol_names, test_mol_tags)

        return train_dataset, val_dataset, test_dataset
    

    def split_single_bead_train(self, ratio=[0.8,0.1,0.1], seed=0):
        """
        Splits the dataset into train, validation and test sets such that all single-bead molecules are in the training set.
        """

        single_bead_moltags = [self.mol_tags[i] for i, g in enumerate(self.graphs) if is_single_bead(g)]
        multi_bead_moltags = [self.mol_tags[i] for i, g in enumerate(self.graphs) if not is_single_bead(g)]

        # now distribute the remaining multi-bead molecules:
        n = len(multi_bead_moltags) + len(single_bead_moltags)
        n_train = int(n*ratio[0]) - len(single_bead_moltags)
        n_train = max(n_train, 0)
        n_val = int(n*ratio[1])
        n_test = n - n_train - n_val

        assert n_train >= 0
        assert n_val >= 0
        assert n_test >= 0

        # shuffle the multi-bead molecules
        random.seed(seed)
        multi_bead_moltags = copy.deepcopy(multi_bead_moltags)
        random.shuffle(multi_bead_moltags)

        train_moltags = single_bead_moltags + multi_bead_moltags[:n_train]
        val_moltags = multi_bead_moltags[n_train:n_train+n_val]
        test_moltags = multi_bead_moltags[n_train+n_val:]

        return self.split(train_moltags, val_moltags, test_moltags)

    def random_split(self, ratio=[0.8,0.1,0.1], seed=0):
        """
        Splits the dataset into train, validation and test sets randomly.
        """
        n = len(self.graphs)
        n_train = int(n*ratio[0])
        n_val = int(n*ratio[1])
        n_test = n - n_train - n_val

        assert n_train >= 0
        assert n_val >= 0
        assert n_test >= 0

        idx = list(range(n))
        random.seed(seed)
        random.shuffle(idx)

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train+n_val]
        test_idx = idx[n_train+n_val:]

        train_mol_tags = [self.mol_tags[i] for i in train_idx]
        val_mol_tags = [self.mol_tags[i] for i in val_idx]
        test_mol_tags = [self.mol_tags[i] for i in test_idx]

        return self.split(train_mol_tags, val_mol_tags, test_mol_tags)
    
    def sklearn_split(self):
        from sklearn.model_selection import train_test_split

        tags_train, tags_test, y_train, y_test = train_test_split(self.mol_tags, [0]*len(self.mol_tags), test_size=0.10, random_state=42)

        return self.split(train_tags=tags_train, val_tags=tags_test, test_tags=tags_test)
    
    def __add__(self, other: Dataset)->Dataset:
        return DGLDataset(self.graphs + other.graphs, self.mol_names + other.mol_names, self.mol_tags + other.mol_tags)

    def filter(self, min_logp=None, max_logp=None):
        """
        Sets the target to nan if it is outside the specified range.
        Then filters out molecules that have only nan targets.
        """
        if min_logp is None and max_logp is None:
            return self
        
        new_graphs = []
        new_mol_names = []
        new_mol_tags = []
        for g, mol_name, mol_tag in zip(self.graphs, self.mol_names, self.mol_tags):
            for target_key in g.nodes['global'].data.keys():
                if (min_logp is not None and g.nodes['global'].data[target_key][0] < min_logp) or (max_logp is not None and g.nodes['global'].data[target_key][0] > max_logp):
                    g.nodes['global'].data[target_key][0] = torch.tensor(float('nan'), dtype=torch.float32, device=g.nodes['global'].data[target_key].device)
            if not all(torch.isnan(g.nodes['global'].data[target_key]) for target_key in g.nodes['global'].data.keys()):
                new_graphs.append(g)
                new_mol_names.append(mol_name)
                new_mol_tags.append(mol_tag)
        return DGLDataset(new_graphs, new_mol_names, new_mol_tags)


class DataModule(LightningDataModule):
    """
    A LightningDataModule for the DGLDataset.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.experiment.training.batch_size
        self.extend_train_epoch = cfg.data.extend_train_epoch

    def setup(self, stage=None):
        self.dataset = DGLDataset.load(self.cfg.data.dataset_path)
        if 'min_logp' in self.cfg.data and 'max_logp' in self.cfg.data:
            self.dataset = self.dataset.filter(min_logp=self.cfg.data.min_logp, max_logp=self.cfg.data.max_logp) 
        if 'sklearn_split' in self.cfg.data and self.cfg.data.sklearn_split:
            logging.info('Using sklearn split with seed 42 and val=test')
            self.train_dataset, self.val_dataset, self.test_dataset = self.dataset.sklearn_split()
        else:
            logging.info('Using random split with seed 0')
            self.train_dataset, self.val_dataset, self.test_dataset = self.dataset.split_single_bead_train(seed=self.cfg.data.seed, ratio=self.cfg.data.split_ratio)

        print(f"Train: {len(self.train_dataset)}")
        print(f"Val: {len(self.val_dataset)}")
        print(f"Test: {len(self.test_dataset)}")

        if not self.cfg.data.extra_dataset_path is None:
            self.extra_dataset = DGLDataset.load(self.cfg.data.extra_dataset_path)

            # split the dataset into train and rest. rest is both val and trian because its val part is only used for logging, not for early stopping...
            self.extra_dataset_tr, extra_dataset_val, extra_dataset_test = self.extra_dataset.random_split(ratio=[self.cfg.data.extra_train_mols, 0., 1-self.cfg.data.extra_train_mols], seed=self.cfg.data.seed)
            self.extra_dataset = extra_dataset_test + extra_dataset_val
        else:
            self.extra_dataset = None
            self.extra_dataset_tr = None

        print(f"Train: {len(self.train_dataset)}")	
        print(f"Val: {len(self.val_dataset)}")
        print(f"Test: {len(self.test_dataset)}")

        print(f"Extra Train: {len(self.extra_dataset_tr)}") if not self.extra_dataset_tr is None else None
        print(f"Extra Val: {len(self.extra_dataset)}") if not self.extra_dataset is None else None

            

    def train_dataloader(self):
        
        extended_train_set = self.train_dataset + self.extra_dataset_tr if not self.extra_dataset_tr is None else self.train_dataset
        if self.extend_train_epoch > 0:
            for _ in range(self.extend_train_epoch):
                extended_train_set += self.train_dataset

        return DataLoader(extended_train_set, batch_size=self.batch_size, shuffle=True, collate_fn=extended_train_set.collate_fn, drop_last=True, num_workers=2)

    def val_dataloader(self):
        if self.extra_dataset is None:
            return [DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.val_dataset.collate_fn, num_workers=1)]
        else:
            return [DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.val_dataset.collate_fn, num_workers=1),
                    DataLoader(self.extra_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.extra_dataset.collate_fn, num_workers=1)]

    def test_dataloader(self):
        if self.extra_dataset is None:
            return [DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.val_dataset.collate_fn, num_workers=1)]
        else:
            return [DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.val_dataset.collate_fn, num_workers=1),
                    DataLoader(self.extra_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.extra_dataset.collate_fn, num_workers=1)]