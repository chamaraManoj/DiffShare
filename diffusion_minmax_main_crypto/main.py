'''
Created on Jun 24 2024
@author: Chamara
Main fucntion which initiates the diffusion model for generating the minmax data.

** Unless otherwise noted, '_minmax' keyword added to all the processes and variables related
to min-max related data. '_val' keyword added to all the processes and variables related to the
chunk value data. **

This script mainly contains the process for crypto related minmax generation
'''

import logging
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm
import argparse

import denoisers_unet_minmax
import diffusion_minmax
import pre_process as pre


cuda = True
DEVICE = torch.device("cuda:0" if cuda else "cpu")



parser = argparse.ArgumentParser()
# Add arguments
parser.add_argument("data_path", type=str, help="Add the path to your data")  # Positional argument
parser.add_argument("class_type", type=str, help="Mention the data type required, Normal or Normal_rug")
# Parse the arguments
args = parser.parse_args()

feats = {
    'gap_block': 16,
    'amount_token': 128,
    'amount_verified_token': 72,
    'token_price': 128,
    'verified_token_price': 22,
    'amount_usd': 72,
    'gap_timestamp': 16
}

# ++++++++ Parameters for training +++++++++ #
rows_per_sample = 1
epochs = 15001
down_sample_factor = 2
total_bits_read = 0
for feat in feats:

    # ++++++++++ add feature specific pararmeters +++++++++++ #
    # number of bits used represent each bits in the dataset
    bits = feats[feat] * 2
    bit_range = np.arange(total_bits_read, total_bits_read + bits)
    total_bits_read += bits

    num_of_row_minmax = 1
    num_of_col_minmax = bits  # latent vector size

    # dimensions of the chunk transactions and minmax values
    dim_minmax = (num_of_row_minmax, num_of_col_minmax)

    # ++++++++++ parameters used for each feature ++++++++++ #
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
    train_batch_size = 32
    alpha = 0.7

    model_save_steps = 1000
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ++++++++ define path variables ++++++++ #
    path_curr = os.getcwd()
    # path_data = r'/opt/home/e126410/data_synthesis/blockchain/data/version_3_global_s1234_seq_25'
    path_data = args.data_path
    path_model = path_curr + '/trained_model_' + args.class_type + '_' + feat
    if not os.path.exists(path_model):
        os.makedirs(path_model)

    # timestr = time.strftime("%Y%m%d-%H%M%S")
    # logging.basicConfig(
    #     filename=path_curr + '/logs/' + timestr + '.log',
    #     filemode='w',
    #     level=logging.INFO,
    #     format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    #     datefmt='%Y-%m-%d %H:%M:%S')
    # logging.info('Logging start')
    #
    # # start logging data
    # logging.info('n_timesteps: ' + str(n_timesteps))
    # logging.info('lr: ' + str(lr))
    # logging.info('train batch size: ' + str(train_batch_size))
    # logging.info('alpha: ' + str(alpha))

    # ++++++++++ Loading data +++++++++++++++++ #
    # function for loading data
    transformed_dataset = pre.TransactionDataset(
        csv_file=path_data + '/' + args.class_type + '.csv',
        path_min_max_data=path_data + '/' + 'chunk_minmax_values',
        transform_min_max_data=transforms.Compose([pre.ToTensorMinMaxData()]),
        bit_range=bit_range
    )

    # Denoiser vae_model which uses simple MLP architecture
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
        # data_dim_val=dim_val,
        n_times=n_timesteps,
        beta_range=beta_range,
        device=DEVICE).to(DEVICE)

    optimizer = Adam(diffusion_model.parameters(), lr=lr)
    denoising_loss = nn.MSELoss()

    loss_data = []

    # start training the model
    for epoch in range(epochs):

        train_loader = DataLoader(dataset=transformed_dataset,
                                  batch_size=train_batch_size,
                                  # shuffle=True,
                                  sampler=SubsetRandomSampler(
                                      torch.randint(high=len(transformed_dataset), size=(100,)))
                                  )

        # different noise values and list to keep those values
        noise_prediction_loss_val = 0
        noise_prediction_loss_minmax = 0
        noise_prediction_loss_combined = 0
        loss_list_val = []
        loss_list_minmax = []
        loss_list_combined = []

        # read the batch data and train batch by batch
        for batch_idx, (x_min_max) in tqdm(enumerate(train_loader),
                                           total=len(train_loader)):
            optimizer.zero_grad()

            print(x_min_max.shape)
            x_min_max = x_min_max.to(DEVICE)
            noisy_input_minmax, epsilon_minmax, pred_epsilon_minmax = diffusion_model(
                x_min_max,
            )

            loss_minmax = denoising_loss(pred_epsilon_minmax, epsilon_minmax)

            noise_prediction_loss_minmax += loss_minmax.item()

            loss_list_minmax.append(loss_minmax)

            loss_minmax.backward()
            optimizer.step()

        if epoch % model_save_steps == 0:
            logging.info("Saving model at epoch: " + str(epoch))
            torch.save(diffusion_model.state_dict(), path_model + '/model_' + str(epoch) + '.pt')

        denoise_loss_minmax = noise_prediction_loss_minmax / batch_idx

        logging.info(
            "Epoch: " + str(epoch + 1) + ' Loss min-max: ' + str(
                denoise_loss_minmax))

        print("\tEpoch", epoch + 1, "complete!", "\tDenoising Loss min-max: ", denoise_loss_minmax)