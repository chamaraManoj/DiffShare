'''
Created on Jun 24 2024
@author: Chamara
Diffusion model which combines two basic diffuson process together
1. Generation of the chunk data
2. Generation of min-max values of the columns.
3. Combine the above two processes together

** Unless otherwise noted, '_minmax' keyword added to all the processes and variables related
to min-max related data. '_val' keyword added to all the processes and variables related to the
chunk value data. **
'''
import torch
from torch import nn
import numpy as np

import config

# running diffusion
class Diffusion(nn.Module):
    def __init__(self,
                 model,
                 data_dim_minmax=[1, 256],
                 data_dim_val=None,
                 n_times=1000,
                 beta_range=[1e-4, 2e-2],
                 device='cuda'):

        super(Diffusion, self).__init__()

        self.n_times = n_times

        # get dimensions of two data types
        self.latent_R_minmax, self.latent_C_minmax = data_dim_minmax
        self.img_H_val, self.img_W_val, self.img_C_val = data_dim_val

        # get separate models
        self.model = model

        # define linear variance schedule(betas)
        beta_1, beta_T = beta_range
        betas = torch.linspace(start=beta_1, end=beta_T, steps=n_times).to(device)  # follows DDPM paper
        self.sqrt_betas = torch.sqrt(betas)

        # define alpha for forward diffusion kernel
        self.alphas = 1 - betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)

        self.device = device

    def extract(self, a, t, x_shape):
        """
            from lucidrains' implementation
                https://github.com/lucidrains/denoising-diffusion-pytorch/blob/beb2f2d8dd9b4f2bd5be4719f37082fe061ee450/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L376
        """
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    # value range conversion helper functions
    def scale_to_minus_one_to_one(self, x):
        # according to the DDPMs paper, normalization seems to be crucial to train reverse process network
        return x * 2 - 1

    # scle betweeen minus one to plus one only  for selected columns
    def scale_to_minus_one_to_one_sel_cols(self, x, col_id):

        x[:, :, :, col_id] = x[:, :, :, col_id] * 2 - 1

        return x

    def reverse_scale_to_zero_to_one(self, x):
        return (x + 1) * 0.5

    # scle betweeen minus one to plus one only  for selected columns
    def reverse_scale_to_zero_to_onesel_cols(self, x, col_id):

        x[:, :, :, col_id] = (x[:, :, :, col_id] +1 ) * 0.5

        return x

    # add noise gradually to the input sample until time t
    def make_noisy(self, x_zeros, t):
        # perturb x_0 into x_t (i.e., take x_0 samples into forward diffusion kernels)
        epsilon = torch.randn_like(x_zeros).to(self.device)

        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)

        # Let's make noisy sample!: i.e., Forward process with fixed variance schedule
        #      i.e., sqrt(alpha_bar_t) * x_zero + sqrt(1-alpha_bar_t) * epsilon
        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar

        # ========== detach() ?? ========== #
        return noisy_sample.detach(), epsilon

    # train the model adding noise and denoising the U-Net
    def forward(self,
                x_val,
                x_minmax,
                x_relative_pos,
                x_prev_val,
                ):

        # ++++++ scale data to -1 to 1 ++++++ #
        # value processing
        # select the columns to be scaled in the matrix
        rem_cols = config.col_id_exclude_from_scale
        all_cols = list(np.arange(x_val.shape[3]))
        sel_cols = [i for i in all_cols if i not in rem_cols]

        x_val = self.scale_to_minus_one_to_one_sel_cols(x_val, col_id=sel_cols)
        B_val, _, _, _ = x_val.shape

        # # previous value processing
        x_prev_val = self.scale_to_minus_one_to_one_sel_cols(x_prev_val, col_id=sel_cols)
        B_prev_val, _, _, _ = x_prev_val.shape

        # minmax processing
        x_minmax = self.scale_to_minus_one_to_one(x_minmax)
        B_minmax, _ = x_minmax.shape

        # (1) randomly choose diffusion time-step.
        # get the t values for according to the batch size
        t = torch.randint(low=0, high=self.n_times, size=(B_val,)).long().to(self.device)

        # (2) forward diffusion process: perturb both data with fixed variance schedule
        # Make the original data noisy.
        perturbed_img_val, epsilon_val = self.make_noisy(x_val, t)
        # perturbed_latents_minmax, epsilon_minmax = self.make_noisy(x_minmax, t)

        # (3) predict epsilon(noise) given perturbed data at diffusion-timestep t.
        pred_epsilon_val = self.model(perturbed_img_val,
                                      x_minmax,
                                      x_relative_pos,
                                      x_prev_val,
                                      t)

        return perturbed_img_val, epsilon_val, pred_epsilon_val

    # denoise the values at time t
    def denoise_at_t(self,
                     x_t_val,
                     x_minmax,
                     relative_pos,
                     x_val_prev,
                     timestep,
                     t):

        B_val, _, _, _ = x_t_val.shape
        if t > 1:
            z_val = torch.randn_like(x_t_val).to(self.device)
        else:
            z_val = torch.zeros_like(x_t_val).to(self.device)

        # at inference, we use predicted noise(epsilon) to restore perturbed data sample.
        epsilon_pred_val = self.model(x_t_val,
                                      x_minmax,
                                      relative_pos,
                                      x_val_prev,
                                      timestep
                                      )

        alpha_val = self.extract(self.alphas, timestep, x_t_val.shape)
        sqrt_alpha_val = self.extract(self.sqrt_alphas, timestep, x_t_val.shape)
        sqrt_one_minus_alpha_bar_val = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t_val.shape)
        sqrt_beta_val = self.extract(self.sqrt_betas, timestep, x_t_val.shape)

        x_t_minus_1_val = 1 / sqrt_alpha_val * (x_t_val - (
                1 - alpha_val) / sqrt_one_minus_alpha_bar_val * epsilon_pred_val) + sqrt_beta_val * z_val

        return x_t_minus_1_val.clamp(-1., 1)

    # sample the data
    def sample(self,
               N,
               relative_pos,
               x_val_prev,
               x_minmax
               ):
        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        x_t_val = torch.randn((N, self.img_C_val, self.img_H_val, self.img_W_val)).to(self.device)

        # autoregressively denoise from x_T to x_0
        #     i.e., generate image from noise, x_T
        for t in range(self.n_times - 1, -1, -1):
            timestep = torch.tensor([t]).repeat_interleave(N, dim=0).long().to(self.device)
            x_t_val = self.denoise_at_t(x_t_val,
                                        x_minmax,
                                        relative_pos,
                                        x_val_prev,
                                        timestep,
                                        t)

        rem_cols = config.col_id_exclude_from_scale
        all_cols = list(np.arange(x_t_val.shape[3]))
        sel_cols = [i for i in all_cols if i not in rem_cols]

        x_0_val = self.reverse_scale_to_zero_to_onesel_cols(x_t_val, sel_cols)
        return x_0_val
