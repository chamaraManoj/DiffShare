'''
Created on Jun 24 2024
@author: Chamara
Description: Pre-processing functions for processing data for diffusion models
1. Read the data using PyTorch DataLoader
2. Transform the data to Tensors
'''

import numpy as np
import pandas as pd
import os

import torch
from torch.utils.data import Dataset, DataLoader

cuda = True
DEVICE = torch.device("cuda:0" if cuda else "cpu")


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


def re_arrange_cols(df):
    cols = ['gap_block',
            'amount_token',
            'amount_verified_token',
            'token_price',
            'verified_token_price',
            'amount_usd',
            'gap_timestamp',
            'sender_address_bit_0',
            'sender_address_bit_1',
            'sender_address_bit_2',
            'sender_address_bit_3',
            'sender_address_bit_4',
            'sender_address_bit_5',
            'sender_address_bit_6',
            'sender_address_bit_7',
            'sender_address_bit_8',
            'sender_address_bit_9',
            'sender_address_bit_10',
            'sender_address_bit_11',
            'category_bit_0',
            'category_bit_1'
            ]

    df = df[cols]
    return df


# ++++++++++++++ Dataset class for reading the samples +++++++++
# NOTE: This function should be changed after the deicision of feeding the data chunk by chunk
class TransactionDataset(Dataset):
    '''
    - Read samples from both the chunk value and the min-max normalization dataset
    - call the related data transformation functions
    - Define input and output tensors: (ToDO)
    '''

    def __init__(self, csv_file,
                 path_trans_data,
                 path_min_max_data,
                 transform_chunk_data,
                 transform_prev_chunk_data,
                 transform_min_max_data,
                 num_of_rows,
                 num_of_cols,
                 num_of_rows_from_prev_chunk):
        # chunk paths contains the token name and the number of chunks in it
        # this is process at the DF_pre_processing.create_chunks.py
        self.chunk_data = pd.read_csv(csv_file)
        self.transform_chunk_val_data = transform_chunk_data
        self.transform_prev_chunk_val_data = transform_prev_chunk_data
        self.transform_chunk_min_max_data = transform_min_max_data
        self.path_trans_data = path_trans_data
        self.path_minmax_data = path_min_max_data

        self.column_order = []

        self.row_chunk_val = num_of_rows
        self.col_chunk_val = num_of_cols
        self.rows_from_pre_chunk = num_of_rows_from_prev_chunk

        self.total = 0

    def __len__(self):
        return len(self.chunk_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # select the chunk value to be read
        chunk_data = self.chunk_data.iloc[idx].values

        token_name = chunk_data[1]
        chunk_id = chunk_data[2]
        chunk_prev_id = chunk_data[4]
        chunk_reltive_pos = chunk_data[5]

        # ++++++ read chunk value related data ++++++ #
        # read the chunk data
        path_chunk_val_data = self.path_trans_data + '/' + token_name + '/chunk_' + str(chunk_id) + '.csv'
        data_chunk_val = pd.read_csv(path_chunk_val_data)
        data_chunk_val = re_arrange_cols(data_chunk_val)

        # add the 0s columns and rows to the chunk
        if self.transform_chunk_val_data:
            data_chunk_val = self.transform_chunk_val_data(data_chunk_val)

            if data_chunk_val.shape[1] % 2 == 1:
                zeros_rows = torch.zeros_like(data_chunk_val[:, 0, :])[:, None, :]
                data_chunk_val = torch.concat((data_chunk_val, zeros_rows), dim=1)

            zeros_cols = torch.zeros_like(data_chunk_val[:, :, 0])[:, :, None]
            data_chunk_val = torch.concat((data_chunk_val, zeros_cols), dim=2)

        ## read the previous chunk
        if chunk_prev_id == -1:
            data_prev_chunk_val = torch.zeros(
                (data_chunk_val.shape[0], data_chunk_val.shape[1], data_chunk_val.shape[2]))
        else:
            path_chunk_val_data = self.path_trans_data + '/' + token_name + '/chunk_' + str(chunk_prev_id) + '.csv'
            data_prev_chunk_val = pd.read_csv(path_chunk_val_data)
            data_prev_chunk_val = re_arrange_cols(data_prev_chunk_val)
            data_prev_chunk_val = data_prev_chunk_val.iloc[data_prev_chunk_val.shape[0] - self.rows_from_pre_chunk:,
                                  :]  # get the last 10 tranasactions
            if self.transform_prev_chunk_val_data:
                data_prev_chunk_val = self.transform_chunk_val_data(data_prev_chunk_val)
                data_prev_chunk_val = torch.flip(data_prev_chunk_val, dims=[1])

                if data_prev_chunk_val.shape[1] % 2 == 1:
                    zeros_rows = torch.zeros_like(data_prev_chunk_val[:, 0, :])[:, None, :]
                    data_prev_chunk_val = torch.concat((data_prev_chunk_val, zeros_rows), dim=1)

                zeros_cols = torch.zeros_like(data_prev_chunk_val[:, :, 0])[:, :, None]
                data_prev_chunk_val = torch.concat((data_prev_chunk_val, zeros_cols), dim=2)

        # ++++++ read min-max related data ++++++ #
        # read the min-max data
        path_chunk_minmax_data = self.path_minmax_data + '/' + token_name + '.csv'
        data_chunk_minmax = pd.read_csv(path_chunk_minmax_data)
        if self.transform_chunk_min_max_data:
            data_chunk_minmax = self.transform_chunk_min_max_data(data_chunk_minmax)

        return (data_chunk_val,
                data_chunk_minmax,
                chunk_reltive_pos,
                data_prev_chunk_val,
                idx)
