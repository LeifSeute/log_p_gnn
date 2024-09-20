"""
Defines a dgl gcn over the aa graph.
"""

import torch
import dgl
from .data_utils import get_in_feats, get_in_feat_size
from .scaling_layer import LearnableScaling

class GraphLayerWrapper(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, g):
        feats = g.ndata['h']
        g.ndata['h'] = self.layer(g, feats)
        return g

class NodeLayerWrapper(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, g):
        feats = g.ndata['h']
        g.ndata['h'] = self.layer(feats)
        return g

class AllAtomGCN(torch.nn.Module):
    def __init__(self, in_feat_names, out_feats, hidden_feats=32, num_layers=2, activation=torch.nn.ELU(), layer_norm:bool=True, p_dropout:float=0.0, scaling:bool=False):
        
        
        super().__init__()

        self.in_feat_names = in_feat_names

        self.in_feats = get_in_feat_size(in_feat_names)

        self.layers = torch.nn.ModuleList()
        self.layers.append(
            NodeLayerWrapper(
                torch.nn.Sequential(
                    torch.nn.Linear(self.in_feats, hidden_feats),
                    torch.nn.LayerNorm(hidden_feats) if layer_norm else torch.nn.Identity(),
                    torch.nn.Dropout(p_dropout),
                    activation,
                )
            )
        )

        for _ in range(num_layers):
            self.layers.append(GraphLayerWrapper(dgl.nn.GraphConv(hidden_feats, hidden_feats, activation=activation)))
            self.layers.append(NodeLayerWrapper(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_feats, hidden_feats),
                    torch.nn.LayerNorm(hidden_feats) if layer_norm else torch.nn.Identity(),
                    torch.nn.Dropout(p_dropout),
                    activation,
                )
            ))

        self.layers.append(
            NodeLayerWrapper(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_feats, out_feats),
                    activation,
                )
            )
        )

        self.scaling = LearnableScaling(out_feats) if scaling else torch.nn.Identity()

    def forward(self, g):
        """
        Returns a tensor of shape (num_batch, out_feats) containing the output of the model.
        """
        g_ = dgl.node_type_subgraph(g, ['aa'])
        feats = get_in_feats(g_, self.in_feat_names)
        g_.ndata['h'] = feats
        for layer in self.layers:
            g_ = layer(g_)

        # pool:
        g.nodes['aa'].data['h'] = g_.nodes['aa'].data['h']
        output = dgl.sum_nodes(g, 'h', ntype='aa')

        output = self.scaling(output)

        return output
