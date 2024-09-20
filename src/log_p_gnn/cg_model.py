"""
Defines a dgl gcn with message passing between aa and cg nodes.
"""

#%%

import torch
import dgl
from log_p_gnn.data_utils import get_in_feats, get_in_feat_size
from log_p_gnn.scaling_layer import LearnableSizeScaling
import dgl.nn as dglnn
import torch.nn as nn
from typing import List, Tuple
from dgl.utils import expand_as_pair
#%%

class GraphIdentity(nn.Module):
    def __init__(self):
        super(GraphIdentity, self).__init__()

    def forward(self, graph, feat):
        feat_src, feat_dst = expand_as_pair(feat, graph)
        return feat_dst


class HeteroGNN(nn.Module):
    def __init__(self, rel_names, msg_module=dglnn.GraphConv(in_feats=32, out_feats=32, activation=torch.nn.ELU()), edge_names=None):
        super(HeteroGNN, self).__init__()

        if edge_names is None:
            edge_names = rel_names

        # Define convolution for each relation type
        edge_modules = {
            rel: msg_module
            for rel in rel_names
        }

        # add identity for all other relations:
        for rel in edge_names:
            if rel not in edge_modules.keys():
                edge_modules[rel] = GraphIdentity()

        self.conv = dglnn.HeteroGraphConv(edge_modules, aggregate='sum')

        self.in_nodes = []
        self.out_nodes = []
        for relname in rel_names:
            in_name, out_name = relname.split('_to_')
            self.in_nodes.append(in_name)
            self.out_nodes.append(out_name)

    def forward(self, g):
        assert all([n in g.ntypes for n in self.in_nodes])
        inputs = {k: g.nodes[k].data['h'] for k in g.ntypes if not 'global' in k}
        outputs = self.conv(g, inputs)
        
        # set outputs as node data:
        for k in self.out_nodes:
            g.nodes[k].data['h'] = outputs[k]

        return g

class NodeWiseLayer(nn.Module):
    def __init__(self, update_module=torch.nn.Linear(in_features=32, out_features=32), node_type='aa'):
        super(NodeWiseLayer, self).__init__()
        self.update_module = update_module
        self.node_type = node_type

    def forward(self, g):
        g.nodes[self.node_type].data['h'] = self.update_module(g.nodes[self.node_type].data['h'])
        return g


class GraphConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats=None, activation=torch.nn.ELU(), rel_names=['cg_to_cg', 'aa_to_aa'], edge_names=None):
        super(GraphConvLayer, self).__init__()

        if out_feats is None:
            out_feats = in_feats

        # msg_module = dglnn.SAGEConv(in_feats=in_feats, out_feats=out_feats, activation=activation, aggregator_type='mean')
        msg_module = dglnn.GraphConv(in_feats=in_feats, out_feats=out_feats, activation=activation)

        self.hetero_gnn = HeteroGNN(rel_names=rel_names, msg_module=msg_module, edge_names=edge_names)

    def forward(self, g):
        g = self.hetero_gnn(g)
        return g



class MLPLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation=torch.nn.ELU(), layer_norm:bool=True, p_dropout:float=0.0, skip_connection:bool=False):
        super(MLPLayer, self).__init__()
        self.linear = torch.nn.Linear(in_feats, out_feats)
        self.ln = torch.nn.LayerNorm(out_feats) if layer_norm else torch.nn.Identity()
        self.drop = torch.nn.Dropout(p_dropout)
        self.activation = activation if activation is not None else torch.nn.Identity()

        if skip_connection:
            assert in_feats == out_feats, 'Skip connection requires in_feats == out_feats'
        self.skip_connection = skip_connection

    def forward(self, x):
        h = self.linear(x)
        h = self.drop(h)
        h = self.activation(h)

        if self.skip_connection:
            h = h + x

        h = self.ln(h)

        return h


class Pooling(nn.Module):
    def __init__(self, lvl='aa', pred_scaling:bool=False, min_power=1., max_power=1., in_feats=32, out_feats=1):
        super().__init__()

        self.lvl = lvl

        self.pred_scaling = pred_scaling

        if self.pred_scaling:
            self.min_power = min_power
            self.max_power = max_power
            self.scale_network = torch.nn.Sequential(torch.nn.ELU(), torch.nn.Linear(in_feats-out_feats, out_feats))
            self.in_feats = in_feats
            self.out_feats = out_feats

    def forward(self, g:dgl.DGLGraph)->Tuple[torch.tensor, torch.tensor]:
        """
        Returns an output tensor of shape (num_batch, out_feats) and a tensor of shape (num_batch) containing the number of nodes in the graph.
        """
        g.nodes[self.lvl].data['ones_for_num_nodes'] = torch.ones((g.number_of_nodes(self.lvl),1), device=g.device)


        # num_nodes_aa = num_nodes if 'aa' == self.lvl else dgl.sum_nodes(g, 'ones_for_num_nodes', ntype='aa')
        if not self.pred_scaling:
            num_nodes = dgl.sum_nodes(g, 'ones_for_num_nodes', ntype=self.lvl)
            output = dgl.sum_nodes(g, 'h', ntype=self.lvl)
            output = output / num_nodes
            g.ndata.pop('ones_for_num_nodes')
        else:
            # unbatch the graphs:
            graphs = dgl.unbatch(g)
            output_list = []
            num_nodes_list = []
            for graph in graphs:
                output = graph.nodes[self.lvl].data['h'] # shape (num_nodes, in_feats)
                
                output_scale_pred = output[:,self.out_feats:] # shape (num_nodes, in_feats-out_feats)
                scaling_score = self.scale_network(output_scale_pred) # shape (num_nodes, out_feats)
                powers = self.min_power + (self.max_power - self.min_power) * torch.sigmoid(scaling_score) # shape (num_nodes, out_feats)

                output_value = output[:,:self.out_feats] # shape (num_nodes, out_feats)

                num_node = graph.number_of_nodes(self.lvl)
                num_node_tensor = torch.ones((num_node, self.out_feats), device=output.device) * num_node # shape (num_nodes, out_feats)

                # now we apply: output = 1/num_nodes * sum_i output_i * num_nodes_i**power_i = sum_i output_i * num_nodes ** (power_i-1)

                output_value = output_value * num_node_tensor ** (powers - 1.)
                output_value = output_value.sum(dim=0, keepdim=True) # sum over nodes

                output_list.append(output_value)
                num_nodes_list.append(num_node)

            output = torch.cat(output_list, dim=0) # shape (num_batch, out_feats)
            num_nodes = torch.tensor(num_nodes_list, device=output.device) # shape (num_batch)

        return output, num_nodes



class HGCN(torch.nn.Module):
    def __init__(self,
                 in_feat_names={'aa':['element', 'charge'],'cg':['fragname']},
                 out_feats=1,
                 hidden_feats=32,
                 activation=torch.nn.ELU(),
                 layer_norm:bool=True,
                 p_dropout:float=0.0,
                 skip_connection:bool=True,
                 layers:List[str]=['cg_to_cg', 'cg', 'cg_to_aa', 'aa', 'aa_to_cg'],
                 pooling_lvl='aa',
                 scaling:bool=True,
                 scaling_mode:str='global', # 'global' or 'local'
                 scaling_power_min=1.,
                 scaling_power_max=1.,
    ):
        """
        Implements message passing between high-resolution all-atom (aa) nodes and coarse-grained (cg) nodes in a heterogeneous graph, where one aa-node can belong to several cg nodes.

        Model flow:
        - Layers comprising nodewise MLP updates ('cg'/'aa') and message passing between given nodes ('cg_to_aa' etc.) all with width hidden_feats.
        - The last layer must be a cg MLP. It maps to feats of dim readout_layers[0] or out_feats respectively.
        - Then, cg node features are pooled given by the pooling operation.
        - Afterwards, a readout network is applied, whose in_feat dims of each layer are given by readout_layers (pass empty list for no readout network).
        - If scaling is true, the output is scaled by learnable mean and std dev.

        Args:
            in_feat_names (dict): A dictionary specifying the input features for 'aa' and 'cg' node types. Example: {'aa': ['element', 'charge'], 'cg': ['fragname']}
            out_feats (int): Number of output features for the final layer. Default is 1.
            hidden_feats (int): Number of hidden features in the MLP and GCN layers. Default is 32.
            activation (callable): Activation function used in MLP and GCN layers. Default is ELU.
            layer_norm (bool): If True, apply layer normalization in MLP layers. Default is True.
            p_dropout (float): Dropout probability for MLP layers. Default is 0.0.
            scaling (bool): If True, apply a learnable scaling layer to the output. Default is False.
            skip_connection (bool): If True, adds skip connections in MLP layers. Requires in_feats == out_feats. Default is True.
            layers (list of str): Defines the sequence of layers to be applied in the model. The list must start with ['cg', 'aa'] and end with either 'cg' or 'aa'. Example: ['cg', 'aa', 'cg_to_cg', 'cg']
            pooling (str): Aggregation method for pooling ('sum' is the only supported value). Default is 'sum'.
            readout_layers (list of int): Specifies the number of features in each readout layer. Default is [4].

        Returns:
            torch.Tensor: A tensor of shape (num_batch, out_feats) representing the model output after message passing and readout.
        """      
        super().__init__()

        EDGE_NAMES = ['cg_to_cg', 'aa_to_aa', 'cg_to_aa', 'aa_to_cg']

        self.in_feat_names = in_feat_names

        self.in_feats = {lvl: get_in_feat_size(in_feat_names[lvl], lvl=lvl) for lvl in ['aa', 'cg']}

        self.layers = torch.nn.ModuleList()


        for node_type in ['cg', 'aa']:
            module = MLPLayer(self.in_feats[node_type], hidden_feats, activation=activation, layer_norm=layer_norm, p_dropout=p_dropout, skip_connection=False)

            self.layers.append(
                NodeWiseLayer(
                    module,
                    node_type=node_type
                )
            )

        for key in layers:
            if key in ['cg', 'aa']:
                module = MLPLayer(in_feats=hidden_feats, out_feats=hidden_feats, activation=activation, layer_norm=layer_norm, p_dropout=p_dropout, skip_connection=skip_connection)
                self.layers.append(
                    NodeWiseLayer(
                        module,
                        node_type=node_type
                    )
                )

            elif key in ['cg_to_cg', 'aa_to_aa', 'cg_to_aa', 'aa_to_cg']:
                module = GraphConvLayer(in_feats=hidden_feats, out_feats=hidden_feats, activation=activation, rel_names=[key], edge_names=EDGE_NAMES) # NOTE: this can be optimized by putting more keys into rel_names

                self.layers.append(
                    module
                )

        last_lvl_width = out_feats if scaling and scaling_mode=='global' else hidden_feats

        module = MLPLayer(in_feats=hidden_feats, out_feats=last_lvl_width, activation=None, layer_norm=layer_norm, p_dropout=p_dropout, skip_connection=False)
        self.layers.append(
            NodeWiseLayer(
                module,
                node_type=pooling_lvl
            )
        )

        self.pooling = Pooling(lvl=pooling_lvl, pred_scaling=scaling and scaling_mode=='local', min_power=scaling_power_min, max_power=scaling_power_max, out_feats=out_feats, in_feats=hidden_feats if scaling_mode=='local' else out_feats)

        self.scaling = LearnableSizeScaling(out_feats, min_power=scaling_power_min, max_power=scaling_power_max) if scaling and scaling_mode=='global' else None


    def forward(self, g):
        """
        Returns a tensor of shape (num_batch, out_feats) containing the output of the model.
        """
        
        # write features to graph:
        feats_aa = get_in_feats(g, self.in_feat_names['aa'], lvl='aa')
        feats_cg = get_in_feats(g, self.in_feat_names['cg'], lvl='cg')

        assert torch.all(torch.isfinite(feats_aa)), 'feats_aa contains NaNs or Infs'
        assert torch.all(torch.isfinite(feats_cg)), 'feats_cg contains NaNs or Infs'

        g.nodes['aa'].data['h'] = feats_aa
        g.nodes['cg'].data['h'] = feats_cg

        # apply neural network:
        for layer in self.layers:
            g = layer(g)

        # pool:
        output, num_nodes = self.pooling(g)

        if not self.scaling is None:
            output = self.scaling(output, num_nodes)

        return output


#%%

if __name__ == '__main__':
    from log_p_gnn.dataset import DataModule
    from omegaconf import OmegaConf


    conf = OmegaConf.load('../../configs/train_test.yaml')

    conf.data.dataset_path = '../../'+conf.data.dataset_path
    conf.data.extra_dataset_path = None
    module = DataModule(conf)

    module.setup()
    val_dataloader = module.val_dataloader()[0]
    # %%
    # model = HGCN(layers=['cg', 'aa', 'cg'])
    model = HGCN(layers=['cg', 'aa', 'aa_to_aa', 'cg'])
    for batch in val_dataloader:
        g = model(batch)
