'''
This is a configuration file which holds important parameters of the process
'''

chunk_size = 25

minmax_vector = 908

columns_to_read = [
    'block_number',
    'timestamp',
    'pool_balance',
    'category',
    'sender_address',
    'amount_token',
    'amount_verified_token',
    'token_price',
    'verified_token_price',
    'amount_usd',
    'created_at'
]

cat_var_cols = ['sender_address',
                      'category']
# cat_var_bits = [12, 2]
cat_var_bits = {
    'sender_address': 12,
    'category': 2
}
# time variables
time_vars = ['timestamp']
# columns to drop
drop_cols = ['sender_address', 'category', 'created_at']

# dict for actions
action_dict = {'deposit': 0,
               'buy': 1,
               'sell': 2,
               'withdraw': 3}

# order of the columns in a dataframe
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

# columns ids to be exclude from normalization.
# these ids are inherently are normalized between -1 to 1
col_id_exclude_from_scale = [cols.index('amount_token'),
                             cols.index('amount_verified_token'),
                             cols.index('amount_usd')]


