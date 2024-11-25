'''
Combine chunk together and re-construct the original values
'''

import numpy as np
import pandas as pd
import os
import argparse

import config


def measure_gap_between_chunks(data, n=10, th=0.2):
    for c in range(len(data) - 1):
        last_row_prev_chunk = data[c, -1, 1:]
        first_row_curr_chunk = data[c + 1, 0, 1:]

        diff = first_row_curr_chunk - last_row_prev_chunk

        for feat in range(1, data.shape[2]):

            diff_val = diff[feat - 1]
            if np.abs(diff_val) > th:
                vals_from_prev_chunk = data[c, -n:, feat]
                vals_from_next_chunk = data[c + 1, 1:n + 1, feat]

                w_prev = 1 - (1 / np.arange(1, n + 1))
                w_next = 1 / np.arange(1, n + 1)
                weighted_sum_vals_from_prev_chunk = np.sum(vals_from_prev_chunk * w_prev)
                weighted_sum_vals_from_next_chunk = np.sum(vals_from_next_chunk * w_next)

                val_avg = (weighted_sum_vals_from_next_chunk + weighted_sum_vals_from_prev_chunk) / 2
                data[c + 1, 0, feat] = val_avg

    data = np.concatenate(data)

    return data


# read chunk data
def read_chunk_data(path_token):
    chunks = os.listdir(path_token)
    list_chunks_num = []
    list_chunks_cat = []
    for c in range(len(chunks)):
        df = pd.read_csv(path_token + '/chunk_' + str(c) + '_val.csv', header=None)
        arr_num = df.values[:-1, :7]
        arr_cat = df.values[:-1, 7:-1]
        arr_cat[arr_cat < 0.5] = 0
        arr_cat[arr_cat >= 0.5] = 1
        list_chunks_num.append(arr_num)
        list_chunks_cat.append(arr_cat)

    token_num = measure_gap_between_chunks(np.asarray(list_chunks_num), n=10, th=0.2)
    token_cat = np.concatenate(list_chunks_cat, axis=0)
    final_token = np.concatenate([token_num, token_cat], axis=1)

    return final_token


# read min max data
def read_minmax(path_token):
    chunk_minmax = pd.read_csv(path_token, header=None).values.flatten()
    chunk_minmax[chunk_minmax < 0.5] = 0
    chunk_minmax[chunk_minmax >= 0.5] = 1

    return chunk_minmax


def convert_bit_sequence_int(bit_seq):
    str_seq = ''
    for bit in bit_seq:
        str_seq += str(int(bit))
    # return BitArray(bin=str_seq).int
    return int(str_seq, 2)


def reconstruct_minmax(chunk_minmax):
    cols = list(config.bit_dict.keys())

    bit_dict = config.bit_dict
    decimal_multiply = config.decimal_multiply
    bit_pointer = 0
    list_minmax = []
    for c_i, col in enumerate(cols):
        bits = bit_dict[col] * 2
        bits_selected = chunk_minmax[bit_pointer:bit_pointer + bits]
        bit_pointer += bits

        min_bit = bits_selected[:int(bits / 2)]
        max_bit = bits_selected[int(bits / 2):]
        min_val = convert_bit_sequence_int(min_bit)
        max_val = convert_bit_sequence_int(max_bit)

        if col in decimal_multiply.keys():
            multi_power = decimal_multiply[col]
            divider = np.power(10, multi_power)
            min_val = min_val / divider
            max_val = max_val / divider

        list_minmax.append([min_val, max_val])

    return list_minmax


def re_scale(list_minmax, chunk_data):
    return_chunk_data = chunk_data.copy()
    list_cols = []
    for c in range(len(list_minmax)):
        val_data = chunk_data[:, c]
        # plot_data(val_data)
        sign_arr = np.zeros(val_data.shape)
        sign_arr[val_data < 0] = -1
        sign_arr[val_data >= 0] = 1

        val_data = np.abs(val_data)
        # plot_data(val_data)
        min_val = list_minmax[c][0]
        max_val = list_minmax[c][1]
        val_data = val_data * (max_val - min_val) + min_val
        val_data = val_data * sign_arr
        return_chunk_data[:, c] = val_data
        # list_cols.append(val_data.reshape([-1, 1]))

    # data = np.concatenate(list_cols, axis=1)

    return return_chunk_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument("data_path", type=str, help="Add the path to your data")  # Positional argument
    parser.add_argument('is_normalized', type=bool, help="If you want to normalize the data")
    parser.add_argument('dataset', type=str, help="select between rugpull or normal")
    # Parse the arguments
    args = parser.parse_args()
    path_data = args.data_path
    dataset = args.dataset
    is_normalized = args.is_normalized

    # dataset = 'rugpull'
    # path_in = r'C:\Research\Projects\Data-synthesis\Synth_analaysis\crypto\chunk_val/' + dataset + '/synth_data/images/' + model_val
    # path_in_minmax = r'C:\Research\Projects\Data-synthesis\Synth_analaysis\crypto\minmax\v4-batch-minmax-ind-feat/' + dataset + '\synth_data\minmax.csv'

    path_in = path_data + '/' + dataset
    path_in_minmax = path_data + '/' + 'minmax_' + dataset + '.csv'

    if is_normalized:
        path_out = path_data + '/' + dataset + '_out/normalized'
    else:
        path_out = path_data + '/' + dataset + '_out/non-normalized'
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    if not is_normalized:

        # read minmax data and randomly pick one of the chunk size
        minmax_data = pd.read_csv(path_in_minmax).values

        list_tokens = os.listdir(path_in)
        for t, token in enumerate(list_tokens):
            print('token ' + str(t))
            # read chunk data
            chunk_data = read_chunk_data(path_token=path_in + '/' + '/token_' + str(t + 1))

            # extract the corresponding minmax value
            chunk_minmax = minmax_data[t, :]

            # reconstruct min max values
            list_minmax = reconstruct_minmax(chunk_minmax)

            # re-scale data
            scaled_output = re_scale(list_minmax, chunk_data)

            df = pd.DataFrame(columns=config.cols,
                              data=scaled_output)

            df.to_csv(path_out + '/token_' + str(t + 1) + '.csv',
                      index=False)


    else:
        list_tokens = os.listdir(path_in)
        for t, token in enumerate(list_tokens):
            print('token ' + str(t))
            # read chunk data
            chunk_data = read_chunk_data(path_token=path_in + '/' + token)

            df = pd.DataFrame(columns=config.cols,
                              data=chunk_data)

            df.to_csv(path_out + '/' + token + '.csv',
                      index=False)
