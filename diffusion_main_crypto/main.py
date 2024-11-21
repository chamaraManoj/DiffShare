'''
Created on Jun 24 2024
@author: Chamara
Main fucntion which initiates the diffusion model

** Unless otherwise noted, '_minmax' keyword added to all the processes and variables related
to min-max related data. '_val' keyword added to all the processes and variables related to the
chunk value data. **

 - Chunk values and min-max values are generated while having a link between them.
 - Add conditional input
    - Previous chunk
    - chunk id
    - is the last id
 - first, we use the transformer with U-net architecture for the module
'''

import logging
import os
import time
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm

# custom files including classes
import denoisers_unet
import diffusion
import pre_process as pre

cuda = True
DEVICE = torch.device("cuda:0" if cuda else "cpu")

# ++++++++ Parameters for training +++++++++ #
DATA_VERSION = 'version_3'
SET_VERSION = 's1234'
seqs = [25]
prev_chunks = [25]
num_of_row_vals = [26]
num_of_col_vals = [22]
batch_sizes = [64]
epochs = 20000

# taking argpase data for the data location
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("data_path", type=str, help="Add the path to your data")  # Positional argument
# Parse the arguments
args = parser.parse_args()


for s_i, seq in enumerate(seqs):

    num_prev_chunk = prev_chunks[s_i]
    num_of_row_val = num_of_row_vals[s_i]  # maximum  number of rows assigned for chunk
    num_of_col_val = num_of_col_vals[s_i]  # number of cols read from a chunk

    num_of_row_minmax = 1
    num_of_col_minmax = 908  # latent vector size

    # dimensions of the chunk transactions and minmax values
    dim_minmax = (num_of_row_minmax, num_of_col_minmax)
    dim_val = (num_of_row_val, num_of_col_val, 1)

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
    train_batch_size = batch_sizes[s_i]
    alpha = 0.7
    subset_samples = 400
    model_save_steps = 100
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)

    # defining the paths
    path_curr = os.getcwd()
    path_data = args.data_path
    path_model = path_curr + '/trained_models'+'_seq_'+str(seq)
    if not os.path.exists(path_model):
        os.makedirs(path_model)
    print(path_curr)
    print(path_data)
    print(path_model)

    # ++++++++++ Loading data +++++++++++++++++ #
    # function for loading data
    transformed_dataset = pre.TransactionDataset(
        csv_file=path_data + '/' + 'Normal.csv',
        path_trans_data=path_data + '/' + 'chunk_values',
        path_min_max_data=path_data + '/' + 'chunk_minmax_values',
        # providing minmax data at global level
        transform_chunk_data=transforms.Compose([
            pre.ToTensorTransData()
        ]),
        transform_prev_chunk_data=transforms.Compose([
            pre.ToTensorTransData()
        ]),
        transform_min_max_data=transforms.Compose([pre.ToTensorMinMaxData()]),
        num_of_rows=num_of_row_val,
        num_of_cols=num_of_col_val,
        num_of_rows_from_prev_chunk=num_prev_chunk
    )

    # Denoiser vae_model which uses simple MLP architecture
    df_model = denoisers_unet.Denoiser(dim_val=dim_val,
                                       dim_minmax=dim_minmax,
                                       hidden_dims_val=hidden_dims_val,
                                       diffusion_time_embedding_dim_val=256,
                                       hidden_dims_minmax=hidden_dims_minmax,
                                       diffusion_time_embedding_dim_minmax=timestep_embedding_dim_minmax,
                                       n_times=n_timesteps).to(DEVICE)

    # diffusion model. the main entry point of the diffusion process
    diffusion_model = diffusion.Diffusion(
        model=df_model,
        data_dim_minmax=dim_minmax,
        data_dim_val=dim_val,
        n_times=n_timesteps,
        beta_range=beta_range,
        device=DEVICE).to(DEVICE)

    optimizer = Adam(diffusion_model.parameters(), lr=lr)
    denoising_loss = nn.MSELoss()

    loss_data = []

    # train through the epochs
    for epoch in range(epochs):

        train_loader = DataLoader(dataset=transformed_dataset,
                                  batch_size=train_batch_size,
                                  sampler=SubsetRandomSampler(
                                      torch.randint(high=len(transformed_dataset), size=(subset_samples,)))
                                  )

        # different noise values and list to keep those values
        noise_prediction_loss_val = 0
        noise_prediction_loss_minmax = 0
        noise_prediction_loss_combined = 0
        loss_list_val = []
        loss_list_minmax = []
        loss_list_combined = []

        # read the batch data and train batch by batch
        for batch_idx, (x_val, x_min_max, x_relative_pos, x_prev_val, total_idx) in tqdm(enumerate(train_loader),
                                                                                         total=len(train_loader)):
            optimizer.zero_grad()

            # push the data to CUDA
            x_val = x_val.to(DEVICE)
            x_min_max = x_min_max.to(DEVICE).squeeze()
            x_relative_pos = x_relative_pos.type(torch.FloatTensor).to(DEVICE)
            x_prev_val = x_prev_val.to(DEVICE)

            noisy_input_val, epsilon_val, pred_epsilon_val = diffusion_model(
                x_val,
                x_min_max,
                x_relative_pos,
                x_prev_val,
            )

            loss_val = denoising_loss(pred_epsilon_val, epsilon_val)

            noise_prediction_loss_val += loss_val.item()

            loss_list_val.append(noise_prediction_loss_val)

            loss_val.backward()
            optimizer.step()

        if epoch % model_save_steps == 0:
            logging.info("Saving model at epoch: " + str(epoch))
            torch.save(diffusion_model.state_dict(), path_model + '/model_' + str(epoch) + '.pt')

        denoise_loss_val = noise_prediction_loss_val / batch_idx
        denoise_loss_minmax = noise_prediction_loss_minmax / batch_idx
        denoise_loss_combined = noise_prediction_loss_combined / batch_idx

        logging.info(
            "Epoch: " + str(epoch + 1) + " Loss val: " + str(
                denoise_loss_val) + ' Loss min-max: ' + str(
                denoise_loss_minmax) + ' Loss combined: ' + str(
                denoise_loss_combined))

        print("\tEpoch", epoch + 1, "complete!", "\tDenoising Loss val: ", denoise_loss_val)
        print("\tEpoch", epoch + 1, "complete!", "\tDenoising Loss min-max: ", denoise_loss_minmax)
        print("\tEpoch", epoch + 1, "complete!", "\tDenoising Loss combined: ", denoise_loss_combined)

    print('Training done')
