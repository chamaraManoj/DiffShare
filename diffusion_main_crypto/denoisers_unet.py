'''
Created on Jun 24 2024
@author: Chamara
Contains the denoiser models used for the diffusion

** Unless otherwise noted, '_minmax' keyword added to all the processes and variables related
to min-max related data. '_val' keyword added to all the processes and variables related to the
chunk value data. **
'''

import torch
import torch.nn as nn

import util_classes as utils
import u_net


class Denoiser(nn.Module):
    '''
    Denoiser class which is the entry point for the DM model training
    '''
    def __init__(self,
                 dim_val,
                 dim_minmax,
                 hidden_dims_val,  # dimensions of the Conv2 network transaction data
                 diffusion_time_embedding_dim_val,  # dimension of time that should convert the time for Conv2
                 hidden_dims_minmax,  # dimensions of the NN network used for minmax data
                 diffusion_time_embedding_dim_minmax,  # time embedding dimensions used for minmax network.
                 # time embedding dimension is changing according to the nn network hidden layer e.g., 256, 512, 256
                 n_times=1000):
        super(Denoiser, self).__init__()

        # ++++++++++++++ common layers +++++++++++++++ #
        self.time_embedding = utils.SinusoidalPosEmb(diffusion_time_embedding_dim_minmax[0])

        # ++++++ definition for the model used for the values tranasformation +++#
        _, _, img_C = dim_val

        # initiate U_Net model for a denoising block
        self.unet = u_net.UNet(
            x_val_channels=dim_val[2],  # gives the number of channels
            x_minmax_channels=dim_minmax[1],  # gives the number of channels
            n_channels=64,
            ch_mults=(1, 2),
            is_attn=(True, True),
            n_blocks=2
        )

    # start training the model
    def forward(self,
                perturbed_x_val,
                x_minmax,
                x_relative_pos,
                x_val_prev,
                diffusion_timestep):
        x = self.unet(perturbed_x_val,
                      x_minmax,
                      x_val_prev,
                      x_relative_pos,
                      diffusion_timestep)

        return x
