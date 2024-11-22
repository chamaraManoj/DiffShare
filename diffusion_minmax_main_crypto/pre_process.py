'''
Created on Jun 24 2024
@author: Chamara
Description: Pre-processing functions for processing data for diffusion models
1. Read the data using PyTorch DataLoader
2. Transform the data to Tensors
'''

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

# Processing
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
        sample = sample.astype(np.float32)
        sample = sample[np.newaxis, ...]

        return torch.from_numpy(sample)


# ++++++++++++++ Dataset class for reading the samples +++++++++
# NOTE: This function should be changed after the deicision of feeding the data chunk by chunk
class TransactionDataset(Dataset):
    '''
    - Read samples from both the chunk value and the min-max normalization dataset
    - call the related data transformation functions
    - Define input and output tensors: (ToDO)
    '''

    def __init__(self,
                 csv_file,
                 path_min_max_data,
                 transform_min_max_data,
                 bit_range
                 ):
        # chunk paths contains the token name and the number of chunks in it
        # this is process at the DF_pre_processing.create_chunks.py
        self.chunk_data = pd.read_csv(csv_file)
        self.transform_chunk_min_max_data = transform_min_max_data
        self.path_minmax_data = path_min_max_data
        self.bit_range = bit_range

    def __len__(self):
        return len(self.chunk_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # select the chunk value to be read
        chunk_data = self.chunk_data.iloc[idx].values

        token_name = chunk_data[1]

        # ++++++ read min-max related data ++++++ #
        # read the min-max data
        path_chunk_minmax_data = self.path_minmax_data + '/' + token_name + '.csv'
        data_chunk_minmax = pd.read_csv(path_chunk_minmax_data,
                                        usecols=self.bit_range)
        if self.transform_chunk_min_max_data:
            data_chunk_minmax = self.transform_chunk_min_max_data(data_chunk_minmax)

        return data_chunk_minmax
