'''
Function which generate samples from the trained ML models.

while we have the generated model, we get the trained model from GPU.

In this step,
1. load the model
2. generate samples while giving the real data as the feedback. e.g.,
when generating sequence of 5 chunks, all the conditions for previous 4 chunks are selected from a suitable
5 chunks transaction from the data. Diffusion model, in this code, does not take own generated samples as
the input
'''

import os.path
import numpy as np
import torch
import pandas as pd
import argparse

# user defined dependancies
import denoisers_unet
import diffusion
import generate_samples_minmax as gsm

VERSION = 'version_3'
SET = 's1234'

cuda = True
DEVICE = torch.device("cuda:0" if cuda else "cpu")

num_of_row_val = 26  # maximum  number of rows assigned for chunk
num_of_col_val = 22  # number of cols read from a chunk

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
hidden_dims_minmax = [256, 512, 256, 256]
timestep_embedding_dim_minmax = [256, 512, 256]
beta_range = [1e-4, 2e-2]
n_timesteps = 1000

# taking argpase data for the data location
parser = argparse.ArgumentParser()
# Add arguments
parser.add_argument("path_synth", type=str, help="Add the path to your data")  # Positional argument
# Parse the arguments
args = parser.parse_args()


# ++++++++++++ Transformer classes for the data ++++++++++++
class NormalizeCols(object):
    '''
    Normalize the columns of data using min-max normalization
    '''

    def __call__(self, sample):
        sample = sample.values
        for c in range(sample.shape[1]):
            sample[:, c] = (sample[:, c] - sample[:, c].min()) / (sample[:, c].max() - sample[:, c].min() + 1)
        return sample


class ToTensorMinMaxData(object):
    """Convert samples into a ndarrya and then into Tensors.
    used for min-max value data"""

    def __call__(self, sample):
        sample = sample.values
        sample = sample.astype(np.float32)
        return torch.from_numpy(sample)


class ToTensorTransData(object):
    """Convert samples into a ndarrya and then into Tensors."""

    def __call__(self, sample):
        sample = sample.values
        sample = sample.astype(np.float32)
        sample = sample[np.newaxis, ...]

        return torch.from_numpy(sample)


# take the distribution of real transaction
def get_chunk_number_dist_real_transaction(path_in,
                                           dataset_name):
    print(path_in + '/' + dataset_name + '.csv')
    chunk_data = pd.read_csv(path_in + '/' + dataset_name + '.csv')

    chunk_start = chunk_data[chunk_data['chunk_type'] == 1]['chunk_id'].values

    return chunk_start


def process_curr_chunk_to_prev_chunk(x_prev_chunk):
    x_prev_chunk = torch.flip(x_prev_chunk, dims=[1])

    # plt.imshow(x_prev_chunk.detach().cpu().numpy()[0,0])
    # plt.show()

    zeros_rows = torch.zeros_like(x_prev_chunk[:, :, 0, :])[:, :, None, :]
    data_prev_chunk_val = torch.concat((x_prev_chunk, zeros_rows), dim=2)

    zeros_cols = torch.zeros_like(data_prev_chunk_val[:, :, :, 0])[:, :, :, None]
    data_prev_chunk_val = torch.concat((data_prev_chunk_val, zeros_cols), dim=3)

    # plt.imshow(data_prev_chunk_val.detach().cpu().numpy()[0,0])
    # plt.show()

    return data_prev_chunk_val


# randomly select a transcation with the given number of transactions
def select_token_transaction(num_of_chunks):
    path_in = r'C:\Research\Projects\Data-synthesis\Data\real-chunked-tokens' + '/' + VERSION + '_global_' + SET
    chunk_data = pd.read_csv(path_in + '/chunk_data_short.csv')

    # select tokens with given number of chunks
    token_IDs = chunk_data[(chunk_data['chunk_id'] == num_of_chunks) & (chunk_data['chunk_type'] == 1)][
        'token_ID'].values

    # randomly select the tokens with the given id
    token_ID = np.random.choice(token_IDs, 1)[0]

    # select the chunk data distribution
    sel_chunk_data = chunk_data[chunk_data['token_ID'] == token_ID]

    return sel_chunk_data

# This function will read a minmax distribution from a already created distribution.
# one sample will be read
def get_global_minmax_data_from_csv_file(path_in):
    arr = pd.read_csv(path_in,
                      header=None).values
    np.random.shuffle(arr)

    return arr[0]


# save the chunks for a given transaction
def save_samples(chunk_val,
                 token_name,
                 chunk_id,
                 model_name,
                 path_out):
    path_out = path_out + '/' + model_name
    path_token = path_out + '/' + token_name
    if not os.path.exists(path_token):
        os.makedirs(path_token)

    chunk_val = chunk_val.detach().cpu().numpy().squeeze()
    # save chunk data
    chunk_val = pd.DataFrame(data=chunk_val)
    chunk_val.to_csv(path_token + '/' + chunk_id + '_val.csv', index=False, header=False)

    return


'''TO DOOOO'''


def get_global_minmax_data(path_model_minmax, path_out, csv_name):
    minmax_trace = gsm.generate_sampples(path_model_minmax=path_model_minmax,
                                         path_out=path_out,
                                         csv_name=csv_name)

    minmax_trace[minmax_trace < 0.5] = 0
    minmax_trace[minmax_trace >= 0.5] = 1

    return minmax_trace


# in Diffshare we have defined this value to be 25. User can defined this value based on their requirement
seqs = [25]

for seq in seqs:

    # denoiser model used
    df_model = denoisers_unet.Denoiser(dim_val=dim_val,
                                       dim_minmax=dim_minmax,
                                       hidden_dims_val=hidden_dims_val,
                                       diffusion_time_embedding_dim_val=256,
                                       hidden_dims_minmax=hidden_dims_minmax,
                                       diffusion_time_embedding_dim_minmax=timestep_embedding_dim_minmax,
                                       n_times=n_timesteps).to(DEVICE)

    # overall diffusion model
    diffusion_model = diffusion.Diffusion(
        model=df_model,
        data_dim_minmax=dim_minmax,
        data_dim_val=dim_val,
        n_times=n_timesteps,
        beta_range=beta_range,
        device=DEVICE).to(DEVICE)

    diffusion_model.eval()
    diffusion_model.to(DEVICE)

    synth_version = 'v7-all-conditons-in-v5-and-adding-minmax-conditions'
    synth_minmax_version = 'v2-global-minmax-normal-rugpull'

    # dataset version. For crypto we have two data types either ""normal" or "rugpull"
    dataset = 'normal'
    # trained model name for the chunk value generation
    model_val_name = 'model_' + dataset
    # trained model name used for the minmax generation
    model_minmax_name = 'model_' + dataset

    # define the models
    # path which contains the synthetic data for the main chunk data

    # path_synth_analysis = r'/opt/home/e126410/data_synthesis/blockchain/synth/crypto/' + synth_version + '/' + dataset + '/seq_' + str(
    #     seq)
    path_synth_analysis = args.path_synth_analysis

    # path which contains the minmax information
    # path_synth_minmax_analysis = r'/opt/home/e126410/data_synthesis/blockchain/synth/crypto/minmax/' + synth_minmax_version
    path_synth_minmax_analysis = args.path_synth_minmax_analysis

    # path data to extract the metadata of the tokens
    path_data = os.getcwd()

    diffusion_model.load_state_dict(torch.load(os.getcwd() + '/models/main_chunk/' + model_val_name + '.pt'))

    # get the real tranasction chunk sequence
    if dataset == 'normal':
        chunk_dist = get_chunk_number_dist_real_transaction(path_data,
                                                            dataset_name='Normal')
    else:
        chunk_dist = get_chunk_number_dist_real_transaction(path_data,
                                                            dataset_name='Normal_rug')

    # generate N number of tranasctions from the model
    N = 1000
    count = 0
    chunk_limit = 60

    with torch.no_grad():
        while count < N:
            print('token ' + str(count))
            # sample a chunk value from the distribution
            num_of_chunks = np.random.choice(chunk_dist, 1)[0]
            print('   Number of chunks ' + str(num_of_chunks))

            if num_of_chunks > chunk_limit or num_of_chunks == 1:
                continue
            count += 1

            x_val_prevs = torch.zeros(1, 1, num_of_row_val, num_of_col_val)

            # get the minmax values from the csv file. this csv file contains the synthesized minmax data from the minma
            # ax generators for each features
            path_minmax = os.getcwd() + '/synth_minmax/' + model_minmax_name + '.csv'
            x_minmax = get_global_minmax_data_from_csv_file(path_in=path_minmax)
            x_minmax = torch.from_numpy(x_minmax).to(DEVICE)[None, :]
            x_minmax = x_minmax.to(torch.float32)

            # iteratively generate the chunks for a given sequence
            for c in range(num_of_chunks):
                print('                         chunk ' + str(c))
                if c == 0:
                    x_val_prev = x_val_prevs[c].to(DEVICE)
                    x_val_prev = x_val_prev[None, :, :, :]

                relative_pos = (c + 1) * 100 / num_of_chunks
                relative_pos = torch.Tensor([relative_pos]).type(torch.FloatTensor).to(DEVICE)
                # runt the sampling method in diffusion moodel
                x_0_val = diffusion_model.sample(N=1,
                                                 relative_pos=relative_pos,
                                                 x_val_prev=x_val_prev,
                                                 x_minmax=x_minmax
                                                 )
                # save the generated samples
                save_samples(chunk_val=x_0_val,
                             token_name='token_' + str(count),
                             chunk_id='chunk_' + str(c),
                             model_name=model_val_name,
                             path_out=path_synth_analysis)

                # take the previously generated sample as the input to the next chunk
                x_val_prev = x_0_val[:, :, :-1, :-1]
                x_val_prev = process_curr_chunk_to_prev_chunk(x_val_prev)
