import torch
import torch.nn as nn

class LearnableSizeScaling(torch.nn.Module):
    def __init__(self, num_features, min_power, max_power=None, set_id=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(num_features))
        if max_power is None:
            max_power = min_power
        if min_power is None:
            min_power = max_power
        assert not (min_power is None or max_power is None)
        self.power = RangeConstrainedParameter(min_val=min_power, max_val=max_power)
        self.set_id = set_id

    def forward(self, x, num_nodes):
        if not self.set_id:
            return self.scale * x * num_nodes ** self.power()
        else:
            return x

class LearnableScaling(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(num_features))
        self.bias = torch.nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return self.scale * x + self.bias


class RangeConstrainedParameter(nn.Module):
    def __init__(self, min_val, max_val):
        """
        A learnable parameter constrained to a given range [min_val, max_val].
        """
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        
        # Initialize unconstrained parameter
        self.raw_param = nn.Parameter(torch.randn(1)) 

    def forward(self):
        # Apply sigmoid and rescale to [min_val, max_val]
        constrained_param = self.min_val + (self.max_val - self.min_val) * torch.sigmoid(self.raw_param)
        return constrained_param

    # # Usage example
    # module = RangeConstrainedParameter(min_val=0.1, max_val=0.9)
    # constrained_param = module()
    # print(constrained_param)
