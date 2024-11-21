'''
Created on Jun 24 2024
@author: Chamara
Contains several util classes for the diffusion models

** Unless otherwise noted, '_minmax' keyword added to all the processes and variables related
to min-max related data. '_val' keyword added to all the processes and variables related to the
chunk value data. **
'''
import math
import torch
import torch.nn as nn

# sinusoidal position embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # arange a torch tensor and send it to device
        emb = x[:, None] * emb[None, :]  # add another dimension using None. similar to the function reshape in tensor
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # last dimension. e.g., vertically concat if the num dim is 2
        return emb