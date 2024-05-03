MAX_BOND_ORDER = 4

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
from torch.utils.data import Dataset


class DGLDataset(Dataset):
    def __init__(self, dspath:str=None, graphs:list=None):
        assert not (dspath is None and graphs is None), 'Either dspath or graphs must be provided'
        assert not (dspath is not None and graphs is not None), 'Only one of dspath or graphs must be provided'

        if graphs is not None:
            self.graphs = graphs
        else:    
            self.graphs, _ = dgl.load_graphs(str(dspath))
        
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]
    
    def collate_fn(self, batch):
        return dgl.batch(batch)
    


def homgraph_to_hetgraph(g, global_feats:List[str]=["logp"]):
    """
    Creates a heterograph with node type atom and edge type bond from the homograph g.
    creates a node type global that represens a single feature, which is simply the zeroth entry of the global_feats feature types (for which we assert that they are all the same).
    """

    # Initialize node and edge connectivity for the atom type
    src_nodes, dst_nodes = g.edges()
    data_dict = {
        ('atom', 'bond', 'atom'): (src_nodes, dst_nodes),
        ('global', 'global_self', 'global'): (torch.tensor([0]), torch.tensor([0]))
    }
    
    # Create the heterograph
    hg = dgl.heterograph(data_dict)
    
    # Transfer node features from homograph to heterograph for 'atom' type
    for key, value in g.ndata.items():
        hg.nodes['atom'].data[key] = value
    
    # Transfer edge features from homograph to heterograph for 'bond' type
    for key, value in g.edata.items():
        hg.edges['bond'].data[key] = value
    
    # Assert that the global features are consistent
    if len(set([g.ndata[feat][0] for feat in global_feats])) != 1:
        raise ValueError("All global features must be the same across all nodes.")
    
    # Initialize global node feature
    global_feature = g.ndata[global_feats[0]][0]  # We take the first element as all are asserted to be the same
    for feat in global_feats:
        hg.nodes['global'].data[feat] = torch.tensor([global_feature])

    hg.nodes['atom'].data['degree'] = hg.in_degrees(etype='bond')
    
    return hg

def one_hot_encode_element(element):
    t = torch.tensor([1 if element == e else 0 for e in ELEMENTS])
    if sum(t) != 1:
        raise ValueError(f"Element {element} not found in the list of elements.")
    return t

def networkx_to_dgl(aa_mol: nx.Graph):
    dgl_graph = dgl.from_networkx(aa_mol)

    # edge features (dgl has every edge 2 times since it treats them as directed edges, so we need to set the bond order for both directions)
    ############################
    edge_orders = nx.get_edge_attributes(aa_mol, 'order')
    # For each edge in DGL (which will now correctly handle undirected edges as two directed edges)
    dgl_graph.edata['bond_order'] = torch.zeros(dgl_graph.num_edges(), MAX_BOND_ORDER)
    for (u, v), order in edge_orders.items():
        # Since DGL treats these edges as directed, we need to find the edge in both directions
        edge_u_v = dgl_graph.edge_ids(u, v)
        edge_v_u = dgl_graph.edge_ids(v, u)
        bond_order_u_v = torch.tensor([1 if i == order else 0 for i in range(MAX_BOND_ORDER)])
        bond_order_v_u = torch.tensor([1 if i == order else 0 for i in range(MAX_BOND_ORDER)])
        dgl_graph.edata['bond_order'][edge_u_v] = bond_order_u_v
        dgl_graph.edata['bond_order'][edge_v_u] = bond_order_v_u
    ############################

    # global features: (do not use formal charge as node feature as it breaks symmetry for certain molecules, e.g. carboxylate)
    log_p = aa_mol.nodes(data='logp')[0]
    total_charge = sum([float(c) for _, c in aa_mol.nodes(data='charge')])

    # node features:
    # one-hot encode the element:
    atomic_number = [one_hot_encode_element(data['element']) for _, data in aa_mol.nodes(data=True)]
    is_aromatic = [1 if data['aromatic'] else 0 for _, data in aa_mol.nodes(data=True)]
    # NOTE: do a good encoding of the fragment id later on

    dgl_graph.ndata['logp'] = torch.ones(dgl_graph.num_nodes()) * log_p
    dgl_graph.ndata['total_charge'] = torch.ones(dgl_graph.num_nodes()) * total_charge

    dgl_graph.ndata['atomic_number'] = torch.stack(atomic_number).float()
    dgl_graph.ndata['is_aromatic'] = torch.tensor(is_aromatic).float()

    return dgl_graph


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
