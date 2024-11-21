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
import numpy as np
import torch
import tqdm
from torch import nn


# running diffusion vae_model
class Diffusion(nn.Module):
    def __init__(self,
                 model,
                 n_times=1000,
                 beta_range=[1e-4, 2e-2],
                 device='cuda'):

        super(Diffusion, self).__init__()

        self.n_times = n_times
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

    def scale_to_minus_one_to_one(self, x):
        # according to the DDPMs paper, normalization seems to be crucial to train reverse process network
        return x * 2 - 1

    def clamp_to_minus_one_to_one(self, x):

        return x.clamp(-1, 1)

    def scale_between_give_region(self, m, c, x):

        return m * x + c

    def reverse_scale_to_zero_to_one(self, x):
        return (x + 1) * 0.5

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

    def forward(self,
                x_minmax,
                ):

        # minmax processing
        x_minmax = self.scale_to_minus_one_to_one(x_minmax)
        B_minmax, _, _ = x_minmax.shape

        # (1) randomly choose diffusion time-step.
        # get the t values for according to the batch size
        t = torch.randint(low=0, high=self.n_times, size=(B_minmax,)).long().to(self.device)

        # (2) forward diffusion process: perturb both data with fixed variance schedule
        # Make the original data noisy.
        perturbed_latents_minmax, epsilon_minmax = self.make_noisy(x_minmax, t)

        # (3) predict epsilon(noise) given perturbed data at diffusion-timestep t.
        pred_epsilon_minmax = self.model(
            perturbed_latents_minmax,
            t)

        return perturbed_latents_minmax, epsilon_minmax, pred_epsilon_minmax

    def denoise_at_t(self,
                     x_t_minmax,
                     timestep,
                     t):

        B_minmax, _, _ = x_t_minmax.shape
        if t > 1:
            z_minmax = torch.randn_like(x_t_minmax).to(self.device)
        else:
            z_minmax = torch.zeros_like(x_t_minmax).to(self.device)

        # at inference, we use predicted noise(epsilon) to restore perturbed data sample.
        epsilon_pred_minmax = self.model(
            x_t_minmax,
            timestep)

        alpha_minmax = self.extract(self.alphas, timestep, x_t_minmax.shape)
        sqrt_alpha_minmax = self.extract(self.sqrt_alphas, timestep, x_t_minmax.shape)
        sqrt_one_minus_alpha_bar_minmax = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t_minmax.shape)
        sqrt_beta_minmax = self.extract(self.sqrt_betas, timestep, x_t_minmax.shape)

        # denoise at time t, utilizing predicted noise
        x_t_minus_1_minmax = 1 / sqrt_alpha_minmax * (x_t_minmax - (
                1 - alpha_minmax) / sqrt_one_minus_alpha_bar_minmax * epsilon_pred_minmax) + sqrt_beta_minmax * z_minmax

        return x_t_minus_1_minmax.clamp(-1., 1)

    def sample(self, N):
        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        x_t_minmax = torch.randn((N, 908)).to(self.device)
        x_t_minmax = x_t_minmax[:, None, :]

        for t in tqdm.tqdm(range(self.n_times - 1, -1, -1)):
            timestep = torch.tensor([t]).repeat_interleave(N, dim=0).long().to(self.device)
            x_t_minmax = self.denoise_at_t(
                x_t_minmax,
                timestep,
                t)

        x_0_minmax = self.reverse_scale_to_zero_to_one(x_t_minmax)

        return x_0_minmax
