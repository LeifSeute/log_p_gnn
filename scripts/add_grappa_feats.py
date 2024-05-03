#%%
import torch
torch.set_float32_matmul_precision('medium')
import dgl
import grappa
from copy import deepcopy

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
        new_edges[new_etype] = (graph.edges(etype=etype)[0], graph.edges(etype=etype)[1])

    # Create a new graph with the modified schema
    new_graph = dgl.heterograph(new_edges, num_nodes_dict={nt: graph.number_of_nodes(nt) for nt in graph.ntypes if nt != old_name}.update({new_name: graph.number_of_nodes(old_name)}))

    # Assign node and edge data to the new graph
    for ntype in new_graph.ntypes:
        new_graph.nodes[ntype].data.update(new_node_data[ntype])
    for etype in new_graph.canonical_etypes:
        new_graph.edges[etype].data.update(new_edge_data[etype])

    return new_graph


#%%
grappa_model = grappa.utils.loading_utils.model_from_tag('grappa-1.2')

dspath = '../data/dgl_dataset.bin'
dspath = '../data/all_atom_dgl_dataset.bin'

graphs, _ = dgl.load_graphs(dspath)
#%%
graphs[0].nodes['atom'].data['atomic_number']
# %%
# only take first sub module of grappa:
grappa_gnn = [c for c in grappa_model.children()][0].to('cuda')
# %%
for i in range(len(graphs)):
    try:
        # raise Exception
        print(i, end='\r')
        graph = deepcopy(graphs[i])
        # rename the node type to 'n1':
        graph = rename_node_type(graph, old_name='atom', new_name='n1')
        atomic_numbers = graph.nodes['n1'].data['atomic_number']
        MAX_GRAPPA_ELEMENT = 53
        atomic_numbers = atomic_numbers[:, :MAX_GRAPPA_ELEMENT]
        
        MAX_DEGREE = 7
        degrees = graph.in_degrees(etype='bond')
        #one hot encode:
        degrees_ = torch.zeros((degrees.shape[0], MAX_DEGREE)).float()
        for d in degrees:
            degrees_[d] = 1
            

        graph.nodes['n1'].data['degree'] = degrees_

        graph.nodes['n1'].data['atomic_number'] = atomic_numbers
        graph.nodes['n1'].data['ring_encoding'] = torch.zeros((atomic_numbers.shape[0], 7)).float()

        graph.nodes['n1'].data['charge_model'] = torch.zeros_like(atomic_numbers[:,0]).float()
        graph.nodes['n1'].data['partial_charge'] = torch.zeros((atomic_numbers.shape[0])).float()

        graph = graph.to('cuda')

        graph = grappa_gnn(graph).to('cpu')
    except:
        print(i, end='\r')
        graph = deepcopy(graphs[i])
        graph = graph.to('cpu')
        grappa_gnn = grappa_gnn.to('cpu')
        graph = deepcopy(graphs[i])
        # rename the node type to 'n1':
        graph = rename_node_type(graph, old_name='atom', new_name='n1')
        atomic_numbers = graph.nodes['n1'].data['atomic_number']
        MAX_GRAPPA_ELEMENT = 53
        atomic_numbers = atomic_numbers[:, :MAX_GRAPPA_ELEMENT]
        
        MAX_DEGREE = 7
        degrees = graph.in_degrees(etype='bond')
        #one hot encode:
        degrees_ = torch.zeros((degrees.shape[0], MAX_DEGREE)).float()
        for d in degrees:
            degrees_[d] = 1
            

        graph.nodes['n1'].data['degree'] = degrees_

        graph.nodes['n1'].data['atomic_number'] = atomic_numbers
        graph.nodes['n1'].data['ring_encoding'] = torch.zeros((atomic_numbers.shape[0], 7)).float()

        graph.nodes['n1'].data['charge_model'] = torch.zeros_like(atomic_numbers[:,0]).float()
        graph.nodes['n1'].data['partial_charge'] = torch.zeros((atomic_numbers.shape[0])).float()
        with torch.no_grad():
            graph = grappa_gnn(graph)

        grappa_gnn.to('cuda')
    graphs[i].nodes['atom'].data['grappa_features'] = graph.nodes['n1'].data['h']
# %%
# save the updated graphs:
dgl.save_graphs('../data/dgl_dataset_all_atom_with_grappa.bin', graphs)
# %%
print(graphs[0].nodes['atom'].data['grappa_features'].shape)
# %%
