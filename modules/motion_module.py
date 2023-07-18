import gdown
import os
import torch

__state_dict_file = 'models/mm_sd_v15.ckpt'


def load_state_dict():
    print('Checking motion module...')
    if not os.path.isfile(__state_dict_file):
        print('Motion module not found, downloading...')
        gdown.download(id='1ql0g_Ys4UCz2RnokYlBjyOYPbttbIpbu', output=__state_dict_file, quiet=False)
        if not os.path.isfile(__state_dict_file):
            exit('Failed to download motion module!')

    print('Loading motion module state dictionary...')
    return torch.load('models/mm_sd_v15.ckpt', map_location='cpu')
