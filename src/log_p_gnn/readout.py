import torch
import torch.nn as nn

class Readout(torch.nn.Module):
    '''
    Given a pooled graph global feature, this module computes an output featue vector using an mlp with skip connections, layer normalization and dropout.
    Essentially, this is just an mlp implementation.
    '''
    def __init__(self, in_features, out_features, hidden_features=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.Sequential(*[FeedForwardLayer(in_feats=in_features if i==0 else hidden_features, hidden_feats=hidden_features, out_feats=hidden_features if i<num_layers-1 else out_features, dropout=dropout if i<num_layers-1 else 0.0, skip=False if i in [0,num_layers-1] else True, layer_norm=True) for i in range(num_layers)])

    def forward(self, x):
        return self.layers(x)
    
class Denormalize(torch.nn.Module):
    '''
    Given a mean and a std, this module denormalizes the input tensor by multiplying it by the std and adding the mean. These values can be made learnable.
    By default, the mean and std are set to 0 and 1, respectively, and are not learnable, i.e. by default this layer has no effect.
    '''
    def __init__(self, mean=0, std=1, learnable=False):
        super().__init__()
        if isinstance(mean, torch.Tensor):
            mean = mean.item()
        if isinstance(std, torch.Tensor):
            std = std.item()
        self.mean = torch.nn.Parameter(torch.tensor(mean, dtype=torch.float32), requires_grad=learnable)
        self.std = torch.nn.Parameter(torch.tensor(std, dtype=torch.float32), requires_grad=learnable)

    def forward(self, x):
        return x*self.std + self.mean


class FeedForwardLayer(nn.Module):
    """
    From Grappa:
    Simple MLP with one hidden layer and, optionally, a skip connection, dropout at the end and layer normalization.
    
    ----------
    Parameters:
    ----------
        in_feats (int): The number of features in the input and output.
        hidden_feats (int, optional): The number of features in the hidden layer of the feed-forward network.
                                     Defaults to `in_feats`, following the original transformer paper.
        activation (torch.nn.Module): The activation function to be used in the feed-forward network. Defaults to nn.ELU().
        dropout (float): Dropout rate applied in the feed-forward network. Defaults to 0.
        skip (bool): If True, adds a skip connection between the input and output of the feed-forward network. Defaults to False.
        layer_norm (bool): If True, applies layer normalization after the input and before the output of the feed-forward network. Defaults to True.
    """
    def __init__(self, in_feats:int, hidden_feats:int=None, out_feats:int=None, activation:torch.nn.Module=nn.ELU(), dropout:float=0.3, skip:bool=True, layer_norm:bool=True):
        super().__init__()
        if hidden_feats is None:
            hidden_feats = in_feats
        if out_feats is None:
            out_feats = in_feats

        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats

        self.linear1 = nn.Linear(in_feats, hidden_feats)
        self.linear2 = nn.Linear(hidden_feats, out_feats)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.skip = skip
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm1 = nn.LayerNorm(in_feats)

        skip_is_possible = (out_feats/in_feats).is_integer()
        assert skip_is_possible or not skip, f"Skip connection is not possible with {in_feats} input features and {out_feats} output features."


    def forward(self, x):
        if self.layer_norm:
            x = self.norm1(x)
        x_out = self.linear1(x)
        x_out = self.activation(x_out)
        x_out = self.linear2(x_out)
        x_out = self.dropout(x_out)
        if self.skip:
            x = torch.repeat_interleave(x, int(self.out_feats/self.in_feats), dim=-1)
            x_out = x_out + x
        return x_out