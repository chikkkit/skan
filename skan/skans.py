import torch.nn as nn
import torch
from .basis import lshifted_softplus

class SKANLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, basis_function=lshifted_softplus, device='cpu'):
        super(SKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.basis_function = basis_function
        self.device = device
        # add bias
        if bias:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features+1).to(device))
        else:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
    
    def forward(self, x):
        x = x.view(-1, 1, self.in_features)
        # add bias
        if self.use_bias:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=2)

        y = self.basis_function(x, self.weight)
        
        y = torch.sum(y, dim=2)
        return y
    
    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
    
class SKANNetwork(nn.Module):
    def __init__(self, layer_sizes, basis_function=lshifted_softplus, bias=True, device='cpu'):
        super(SKANNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            self.layers.append(SKANLinear(layer_sizes[i], layer_sizes[i+1], bias=bias, 
                                             basis_function=basis_function, device=device))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x