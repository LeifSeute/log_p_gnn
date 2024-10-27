MAX_BOND_ORDER = 4
MAX_NUM_H = 4

# first 86 elements:
ELEMENTS = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']

import torch
import dgl
import networkx as nx
from typing import List
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
import cgsmiles


def get_data_path():
    return (Path(__file__).parent.parent.parent / 'data').resolve().absolute()
    

def encode_beads_onehot(fragname:str)->np.ndarray:
    '''
    Returns a one-hot encoding of the fragment name as an array of shape (21,) where the first 3 entries one-hot encode the size, the next 5 entries one-hot encode the polarity, the next 7 entries one-hot encode the level of the polarity, the next 3 entries one-hot encode the first degree label, and the last 3 entries one-hot encode the second degree label according to the encoding in encode_beads_int.
    '''
    int_encoding = encode_beads_int(fragname)

    onehot_encoding = [np.zeros(3), np.zeros(5), np.zeros(7), np.zeros(3), np.zeros(3)]

    for idx, enc in enumerate(int_encoding):
        onehot_encoding[idx][enc] = 1

    return np.concatenate(onehot_encoding, axis=0).flatten()


def encode_beads_int(fragname:str)->List[int]:
    """
    Decodes beadtypes into numbers.

    - size is econded as 0, 1, 2 correspoding
      to regular, small, and tiny beads

    - polarity is econded as 0, 1, 2, 3, 4 corresponding
      to charged (Q), polar (P), neutral (N), hydrophobic (C),
      and halo compounds (X).

    - each polarity has a level going 0 to 5 with exception of Q
      and X beads that have 5, and 4 levels respectively

    - first degree labels can be a or d; no label is 0

    - second degree labels can be r or h; no label is 0


    encoding is a list of length 5, where the entries are integers running from [0-2, 0-4, 0-6, 0-2, 0-2]
    """
    encoding = [0,0,0,0,0]
    polarity_encoding = {'Q': 0,
                         'P': 1,
                         'N': 2,
                         'C': 3,
                         'X': 4}

    for idx, token in enumerate(str(fragname)):
        # does the sizes
        if idx == 0 and token == 'S':
            encoding[0] = 1
        elif idx == 0 and token == 'T':
            encoding[0] = 2
        elif idx == 0:
            encoding[0] = 0

        if token in 'Q P N C X'.split():
            encoding[1] = polarity_encoding[token]

        if token.isdigit():
            encoding[2] = int(token)
            assert encoding[2] <= 6, f"Level of polarity must be between 0 and 6, but got {encoding[2]}"

    if 'a' in fragname:
        encoding[3] = 1
    elif 'd' in fragname:
        encoding[3] = 2

    if 'r' in fragname:
        encoding[4] = 1
    if 'h' in fragname:
        encoding[4] = 2

    return encoding



def one_hot_encode_element(element: str) -> torch.Tensor:
    t = torch.tensor([1 if element == e else 0 for e in ELEMENTS])
    if sum(t) != 1:
        raise ValueError(f"Element {element} not found in the list of elements.")
    return t


def rename_node_type(graph, old_name, new_name):
    # Create new graph data dictionaries
    new_node_data = {}
    new_edge_data = {}
    new_edges = {}

    # Copy node features to new node type name
    for ntype in graph.ntypes:
        if ntype == old_name:
            new_node_data[new_name] = graph.nodes[ntype].data
        else:
            new_node_data[ntype] = graph.nodes[ntype].data

    # Adjust edge data and connectivity
    for etype in graph.canonical_etypes:
        src_type, edge_type, dst_type = etype
        # Rename src and dst node types if they match old_name
        new_src_type = new_name if src_type == old_name else src_type
        new_dst_type = new_name if dst_type == old_name else dst_type
        new_etype = (new_src_type, edge_type, new_dst_type)

        # Copy edge features and connectivity under the new etype
        new_edge_data[new_etype] = graph.edges[etype].data
        new_edges[new_etype] = (graph.edges(etype)[0], graph.edges(etype)[1])

    # Create a new graph with the modified schema
    new_graph = dgl.heterograph(new_edges, num_nodes_dict={nt: graph.number_of_nodes(nt) for nt in graph.ntypes})

    # Assign node and edge data to the new graph
    for ntype in new_graph.ntypes:
        new_graph.nodes[ntype].data.update(new_node_data[ntype])
    for etype in new_graph.canonical_etypes:
        new_graph.edges[etype].data.update(new_edge_data[etype])

    return new_graph


def one_hot_encode_aromaticity(aromaticity: List[bool])->torch.Tensor:
    """
    One-hot encodes the aromaticity of a list of atoms.
    Returns a tensor of shape (n_atoms, 2) where the first column is 1 if the atom is aromatic and the second column is 1 if the atom is not aromatic.
    """
    return torch.tensor([[1, 0] if aro else [0, 1] for aro in aromaticity]).float()

def one_hot_encode_hcount(hcount: List[int])->torch.Tensor:
    """
    One-hot encodes the hydrogen count of a list of atoms.
    Returns a tensor of shape (n_atoms, 5) where the first column is 1 if the atom has 0 hydrogens, the second column is 1 if the atom has 1 hydrogen, etc. 
    """
    return torch.tensor([[1 if h == i else 0 for i in range(MAX_NUM_H+1)] for h in hcount]).float()

def encode_formal_charge(charge: List[int])->torch.Tensor:
    """
    Encodes the formal charge of a list of atoms.
    Returns a tensor of shape (n_atoms, 1) with the formal charge of each atom. Here we choose the identity for simplicity now and clamp the charge to [-3, 3].
    """
    return torch.tensor(charge).float().clamp(-3, 3).unsqueeze(-1)

def get_aa_features(node_features: dict)->dict:
    """
    Transforms the node features of the all-atom graph as stored by CGSmiles into a dictionary of torch tensors.
    """

    aa_features = {}
    idxs = node_features['idx']
    assert (np.array(idxs) == np.arange(len(idxs))).all(), f'Node indices should be in order but found {idxs} Otherwise, implement reshuffling.'

    for key, value in node_features.items():
        if key == 'element':
            encoding = [one_hot_encode_element(v) for v in value]
            aa_features[key] = torch.stack(encoding, dim=0).float()
        elif key == 'aromatic':
            aa_features[key] = one_hot_encode_aromaticity(value)
        elif key == 'hcount':
            aa_features[key] = one_hot_encode_hcount(value)
        elif key == 'charge':
            aa_features[key] = encode_formal_charge(value)
        # elif key == 'idx':
        #     aa_features[key] = torch.tensor([node_idx]).float()


    return aa_features

def get_cg_features(node_features: dict)->dict:
    """
    Transforms the node features of the coarse-grained graph as stored by CGSmiles into a dictionary of torch tensors.
    """
    cg_features = {}
    idxs = node_features['idx']
    assert (np.array(idxs) == np.arange(len(idxs))).all(), 'Node indices should be in order. Otherwise, implement reshuffling.'

    for key, value in node_features.items():
        if key == 'fragname':
            cg_features[key] = torch.tensor(np.array([encode_beads_onehot(v) for v in value])).float()
        # elif key == 'idx':
        #     cg_features[key] = torch.tensor(value).float()

    return cg_features

def get_edge_features_aa(edge_features: dict)->dict:
    """
    Currently not used.
    """
    edge_features_out = {}
    # for key, value in edge_features.items():
        # if key == 'idx1' or key == 'idx2':
        #     edge_features_out[key] = torch.tensor(value).float()

    return edge_features_out

def get_edge_features_cg(edge_features: dict)->dict:
    """
    Currently not used.
    """
    edge_features_out = {}
    # for key, value in edge_features.items():
        # if key == 'idx1' or key == 'idx2':
        #     edge_features_out[key] = torch.tensor(value).float()

    return edge_features_out

def featurize_graph(g:dgl.DGLGraph, node_features_aa: dict, node_features_cg: dict, edge_features_aa: dict=None, edge_features_cg: dict=None)->dgl.DGLGraph:
    """
    Featurizes a dgl heterograph with node and edge features.
    """
    for key, value in node_features_aa.items():
        g.nodes['aa'].data[key] = value

    for key, value in node_features_cg.items():
        g.nodes['cg'].data[key] = value

    if edge_features_aa is None:
        for key, value in edge_features_aa.items():
            g.edges['aa_to_aa'].data[key] = value

    if edge_features_cg is None:
        for key, value in edge_features_cg.items():
            g.edges['cg_to_cg'].data[key] = value

    return g



def get_hierarchical_graph(aa_graph: nx.Graph, cg_graph: nx.Graph, featurize:bool=True)->dgl.DGLGraph:
    """
    Constructs a dgl heterograph from the all-atom and coarse-grained networkx graphs.
    For this, the fragid attribute of the nodes in the all-atom graph is used to determine to which hypernodes in the coarse-grained graph the nodes belong.
    Edges between the all-atom and coarse-grained nodes are added if the all-atom node belongs to the respective hypernode.

    Args:
    - aa_graph: nx.Graph
        The all-atom graph
    - cg_graph: nx.Graph
        The coarse-grained graph

    Returns:
    - hetero_graph: dgl.DGLGraph
        The constructed heterograph with node types 'aa' and 'cg' and edge types 'aa_to_aa', 'cg_to_cg', 'aa_to_cg', 'cg_to_aa'
    """

    aa_data = get_node_data(aa_graph)
    cg_data = get_node_data(cg_graph)

    aa_edge_data = get_edge_data(aa_graph)
    cg_edge_data = get_edge_data(cg_graph)

    aa_edge_idxs = (aa_edge_data['idx1'], aa_edge_data['idx2'])
    cg_edge_idxs = (cg_edge_data['idx1'], cg_edge_data['idx2'])

    # define edges between aa-nodes and cg-nodes:
    # the node aa_i is connected to the node cg_j if the node aa_i contains j as fragid
    inter_level_edge_idx_1 = []
    inter_level_edge_idx_2 = []
    for node_idx, fragids in zip(aa_data['idx'], aa_data['fragid']):
        for fragid in fragids:
            # cg_node_idx = cg_data['idx'][fragid]
            cg_node_idx = fragid
            inter_level_edge_idx_1.append(node_idx)
            inter_level_edge_idx_2.append(cg_node_idx)


    # now make all edges undirected:
    inter_level_edges_reversed_1 = inter_level_edge_idx_2
    inter_level_edges_reversed_2 = inter_level_edge_idx_1

    aa_edge_idxs = (aa_edge_idxs[0] + aa_edge_idxs[1], aa_edge_idxs[1] + aa_edge_idxs[0])

    cg_edge_idxs = (cg_edge_idxs[0] + cg_edge_idxs[1], cg_edge_idxs[1] + cg_edge_idxs[0])

    # construct a dgl heterograph:
    data_dict = {
        ('aa', 'aa_to_aa', 'aa'): aa_edge_idxs,
        ('cg', 'cg_to_cg', 'cg'): cg_edge_idxs,
        ('aa', 'aa_to_cg', 'cg'): (inter_level_edge_idx_1, inter_level_edge_idx_2),
        ('cg', 'cg_to_aa', 'aa'): (inter_level_edges_reversed_1, inter_level_edges_reversed_2),
        ('global', 'global_to_global', 'global'): ([0], [0])
    }

    hetero_graph = dgl.heterograph(data_dict)

    if featurize:
        node_features_aa = get_aa_features(aa_data)
        node_features_cg = get_cg_features(cg_data)
        edge_features_aa = get_edge_features_aa(aa_edge_data)
        edge_features_cg = get_edge_features_cg(cg_edge_data)

        hetero_graph = featurize_graph(g=hetero_graph, node_features_aa=node_features_aa, node_features_cg=node_features_cg, edge_features_aa=edge_features_aa, edge_features_cg=edge_features_cg)

    return hetero_graph



def get_node_data(subgraph: nx.Graph):
    """
    Returns a dictionary of lists where each key is a feature and each value is a list of that feature for each node in the subgraph.
    The dictionary also contains a key 'idx' which is a list of the node indices defining the order-to-node mapping.
    """
    node_idxs, node_data = zip(*[(idx, feat_dict) for idx, feat_dict in subgraph.nodes(data=True)])

    # node indices should be in order (otherwise, reshuffle is needed):
    # assert (np.array(node_idxs) == np.arange(len(node_idxs))).all()

    # reshape the list of dicts to a dict of lists:
    all_feat_keys = set([k for d in node_data for k in d.keys()])

    node_data = {k: [d.get(k, None) for d in node_data] for k in all_feat_keys}
    node_data['idx'] = list(node_idxs)
    # now sort node data dict lists such that idx is in order:
    idx_order = np.argsort(node_data['idx'])
    for k, v in node_data.items():
        node_data[k] = [v[i] for i in idx_order]            
    return node_data
#%%

def get_edge_data(subgraph: nx.Graph):
    """
    Returns a dictionary of lists where each key is a feature and each value is a list of that feature for each edge in the subgraph.
    The dictionary also contains keys 'idx1' and 'idx2' which are lists of the node indices defining the order-to-node mapping.
    """
    edge_idxs1, edge_idxs2, edge_data = zip(*[(idx1, idx2, feat_dict) for idx1, idx2, feat_dict in subgraph.edges(data=True)]) if len(subgraph.edges) > 0 else ([], [], [])

    # reshape the list of dicts to a dict of lists:
    all_feat_keys = set([k for d in edge_data for k in d.keys()])

    edge_data = {k: [d.get(k, None) for d in edge_data] for k in all_feat_keys}
    edge_data['idx1'] = list(edge_idxs1)
    edge_data['idx2'] = list(edge_idxs2)
    return edge_data


def dgl_from_cgsmiles(cgsmiles_str:str)->dgl.DGLGraph:
    """
    Constructs a featurized dgl graph from a CGSmiles string.
    """
    assert isinstance(cgsmiles_str, str), f"Expected cgsmiles_str to be a string, but got {type(cgsmiles_str)}"
    cg_graph, aa_graph = cgsmiles.resolve.MoleculeResolver.from_string(cgsmiles_str, legacy=True).resolve_all()
    g = get_hierarchical_graph(aa_graph=aa_graph, cg_graph=cg_graph, featurize=True)
    return g


def load_nx_dataset(p:Path)->Tuple[List[nx.Graph], List[nx.Graph], List[str], List[str], Dict[str, List[float]]]:
    """
    Loads a dataset of networkx graphs and log p values from a csv file.
    Returns a list of all-atom graphs, a list of coarse-grained graphs, a list of molnames, a list of moltags, and a dictionary of log p values.
    """
    df = pd.read_csv(str(p)) if str(p).endswith('.csv') else pd.read_csv(str(p), delim_whitespace=True)

    aa_mols, cg_mols, molnames, moltags = [], [], [], []
    log_ps = {'OCO':[], 'HD':[], 'CLF':[]}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        
        mol_name, mol_tag, cgsmiles_str = row['mol_name'], row['mol_tag'], row['cgsmiles_str']

        cg_mol, aa_mol = cgsmiles.resolve.MoleculeResolver.from_string(cgsmiles_str, legacy=True).resolve_all()

        log_p_oco = row['OCO']
        log_p_hd = row['HD']
        log_p_clf = row['CLF']

        log_ps['OCO'].append(log_p_oco)
        log_ps['HD'].append(log_p_hd)
        log_ps['CLF'].append(log_p_clf)

        aa_mols.append(aa_mol)
        cg_mols.append(cg_mol)
        molnames.append(mol_name)
        moltags.append(mol_tag)


    return aa_mols, cg_mols, molnames, moltags, log_ps


def is_single_bead(graph: dgl.DGLGraph)->bool:
    """
    Returns True if the coarse-grained graph is a single bead, i.e., if it has only one node.
    """
    return graph.number_of_nodes('cg') == 1


def get_in_feats(g, in_feat_names:List[str], lvl='aa'):
    return torch.cat([g.nodes[lvl].data[in_feat_name] for in_feat_name in in_feat_names], dim=-1)

def get_in_feat_size(in_feat_names, lvl='aa'):

    assert lvl in ['aa', 'cg'], f'received lvl {lvl}'
    size = 0

    if lvl == 'aa':
        for in_feat_name in in_feat_names:
            if in_feat_name == 'element':
                size += len(ELEMENTS)
            elif in_feat_name == 'aromatic':
                size += 2
            elif in_feat_name == 'hcount':
                size += MAX_NUM_H + 1
            elif in_feat_name == 'charge':
                size += 1

    elif lvl == 'cg':
        for in_feat_name in in_feat_names:
            if in_feat_name == 'fragname':
                size += 21

    return size