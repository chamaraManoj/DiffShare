'''
Created on Jun 24 2024
@author: Chamara
Contains the denoiser models used for the diffusion

** Unless otherwise noted, '_minmax' keyword added to all the processes and variables related
to min-max related data. '_val' keyword added to all the processes and variables related to the
chunk value data. **
'''

import torch.nn as nn

import util_classes as utils
import u_net_minmax

class Denoiser(nn.Module):
    def __init__(self,
                 diffusion_time_embedding_dim_minmax,  # time embedding dimensions used for minmax network.
                 ):
        super(Denoiser, self).__init__()

        # ++++++++++++++ common layers +++++++++++++++ #
        self.time_embedding = utils.SinusoidalPosEmb(diffusion_time_embedding_dim_minmax[0])

        # ++++++ definition for the model used for the values tranasformation +++#
        self.unet = u_net_minmax.UNet(
            image_channels=1,
            n_channels=64,
            ch_mults=(1, 2, 3),
            is_attn=(False, False, False),
            n_blocks=3
        )

    def forward(self,
                perturbed_x_minmax,
                diffusion_timestep):
        x = self.unet(perturbed_x_minmax, diffusion_timestep)

        return x
