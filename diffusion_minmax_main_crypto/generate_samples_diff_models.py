'''
Created on Jun 24 2024
@author: Chamara
Generate the samples for the minmax model

** Unless otherwise noted, '_minmax' keyword added to all the processes and variables related
to min-max related data. '_val' keyword added to all the processes and variables related to the
chunk value data. **

 - Generating the min-max samples with 1228
'''

import logging
import os
import time
import pandas as pd
import numpy as np
import torch
import argparse

import denoisers_unet_minmax
import diffusion_minmax

parser = argparse.ArgumentParser()
# Add arguments
parser.add_argument("data_path", type=str, help="Add the path to your data")  # Positional argument
parser.add_argument("class_type", type=str, help="Mention the data type required, Normal or Normal_rug")
# Parse the arguments
args = parser.parse_args()

cuda = True
DEVICE = torch.device("cuda:0" if cuda else "cpu")

# ++++++++ Parameters for training +++++++++ #
num_of_row_minmax = 1
num_of_col_minmax = 256  # latent vector size

# dimensions of the chunk transactions and minmax values
dim_minmax = (num_of_row_minmax, num_of_col_minmax)

# hidden layer dimensions of the two models
hidden_dim = 256  # for val model
n_layers = 8  # for val model
hidden_dims_val = [hidden_dim for _ in range(n_layers)]
# hidden dims for the minmax DM
hidden_dims_minmax = [256, 512, 512, 512, 256]
timestep_embedding_dim_minmax = [256, 512, 512, 512]

# common parameters used by both diffusion models
n_timesteps = 1000
beta_range = [1e-4, 2e-2]
lr = 5e-5
train_batch_size = 64
alpha = 0.7

model_save_steps = 25
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)


def ToTensorMinMaxData(sample):
    """Convert samples into a ndarrya and then into Tensors.
    used for min-max value data"""

    sample = sample.values
    sample = sample.astype(np.float32)
    return torch.from_numpy(sample)


# save the chunks for a given transaction
def save_samples(chunk_val,
                 token_name,
                 path_out):
    path_out = path_out

    if not os.path.exists(path_out):
        os.makedirs(path_out)

    chunk_val = chunk_val.detach().cpu().numpy().squeeze()

    # save chunk data
    chunk_val = pd.DataFrame(data=chunk_val)

    chunk_val.to_csv(path_out + '/' + token_name + '.csv', index=False, header=False)

    return

cols = ['gap_block',
        'amount_token',
        'amount_verified_token',
        'token_price',
        'verified_token_price',
        'amount_usd',
        'gap_timestamp']

feats = {
    'gap_block': 16,
    'amount_token': 128,
    'amount_verified_token': 72,
    'token_price': 128,
    'verified_token_price': 22,
    'amount_usd': 72,
    'gap_timestamp': 16
}

parser = argparse.ArgumentParser()
# Add arguments
parser.add_argument("data_path", type=str, help="Add the path to your data")  # Positional argument
parser.add_argument("class_type", type=str, help="Mention the data type required, Normal or Normal_rug")
# Parse the arguments
args = parser.parse_args()

models = ['model_15000']
data_type = 'Normal'
total_bits_read = 0
for feat in feats:
    print(feat)
    bits = feats[feat] * 2
    bit_range = np.arange(total_bits_read, total_bits_read + bits)
    total_bits_read += bits


    num_of_row_minmax = 1
    num_of_col_minmax = bits  # latent vector size

    # dimensions of the chunk transactions and minmax values
    dim_minmax = (num_of_row_minmax, num_of_col_minmax)

    # path_model = (r'/opt/home/e126410/data_synthesis/blockchain/codes/crypto/df_model_minmax/version_5_gpu/'
    #               r'trained_models_') + data_type + '_' + str(raws) + '_' + feat + '/' + model + '.pt'
    path_model = os.getcwd() + '/trained_models/' + args.class_type + '_' + feat + '/' + 'model_15000.pt'
    path_synth = os.getcwd() + '/synth_data/' + args.class_type + '_' + feat

    df_model_minmax = denoisers_unet_minmax.Denoiser(
        dim_minmax=dim_minmax,
        hidden_dims_val=hidden_dims_val,
        hidden_dims_minmax=hidden_dims_minmax,
        diffusion_time_embedding_dim_minmax=timestep_embedding_dim_minmax,
        n_times=n_timesteps).to(DEVICE)
    # diffusion model. the main entry point of the diffusion process
    diffusion_model = diffusion_minmax.Diffusion(
        model=df_model_minmax,
        data_dim_minmax=dim_minmax,
        n_times=n_timesteps,
        beta_range=beta_range,
        device=DEVICE).to(DEVICE)

    diffusion_model.eval()
    diffusion_model.load_state_dict(torch.load(path_model,
                                               weights_only=True))
    diffusion_model.to(DEVICE)

    all_synth_data = []
    with torch.no_grad():
        for i in range(10):
            print(i)
            x_0_minmax = diffusion_model.sample(N=100)
            all_synth_data.append(x_0_minmax)

            save_samples(chunk_val=x_0_minmax,
                         token_name='token_' + str(i),
                         path_out=path_synth)
