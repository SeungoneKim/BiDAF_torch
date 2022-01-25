import torch
import torch.nn as nn
import torch.nn.functional as F


class HighwayNetwork(nn.Module):
    def __init__(self, emb_dim, num_layers=2):
        super(HighwayNetwork, self).__init__()

        self.num_layers = num_layers
        self.H = nn.ModuleList([nn.Linear(emb_dim, emb_dim)
                               for _ in range(num_layers)])
        self.T = nn.ModuleList([nn.Linear(emb_dim, emb_dim)
                               for _ in range(num_layers)])
        self.G = nn.ModuleList([nn.Linear(emb_dim, emb_dim)
                               for _ in range(num_layers)])

    def forward(self, x):
        # y = f(Q(x))*T(x)+(1-T(x))*G(x)
        # where Q and G is affine transformation
        # and f in non-linear transformation
        # (0 < T < 1)

        for layer in range(self.num_layers):
            t = F.sigmoid(self.T[layer](x))
            h = F.relu(self.H[layer](x))
            g = self.G[layer](x)

            x = t * h + (1-t) * g

        return x

    
    
