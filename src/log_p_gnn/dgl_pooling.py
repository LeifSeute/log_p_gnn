import torch
import torch.nn as nn

try:
    import dgl
    dgl_available = True
except ImportError:
    dgl_available = False

if dgl_available:

        
    class DGLPooling(nn.Module):
        '''
        Given a batched DGL graph, this module computes pooled quantities like max, mean,
        mean squared, etc., for each graph in the batch.
        '''
        def __init__(self, max=True, mean=True, mean_squared=False, sum=False, var=False):
            super().__init__()
            self.max = max
            self.mean = mean
            self.sum = sum

        def forward(self, g):
            '''
            bg: a batched DGL heterogeneous graph (produced by dgl.batch([graph1, graph2, ...]))
            '''

            outputs = []

            if self.max:
                # DGL built-in function to compute max pooling over nodes in each graph
                max_pool = dgl.max_nodes(g, 'h', ntype='atom')
                outputs.append(max_pool)

            if self.mean:
                # DGL built-in function to compute mean pooling over nodes in each graph
                mean_pool = dgl.mean_nodes(g, 'h', ntype='atom')
                outputs.append(mean_pool)

            if self.sum:
                # DGL built-in function to compute sum over nodes in each graph
                sum_pool = dgl.sum_nodes(g, 'h', ntype='atom')
                outputs.append(sum_pool)


            return torch.cat(outputs, dim=1)