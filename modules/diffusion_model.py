import os
import urllib.request

from safetensors import safe_open

from modules.utils import get_model_path

__model_file = get_model_path('toonyou_beta3.safetensors')


def load_model():
    print('Checking diffusion model...')
    if not os.path.isfile(__model_file):
        print('Diffusion model not found, downloading...')
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'MyApp/1.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve('http://civitai.com/api/download/models/78775', __model_file)

    print('Loading diffusion model state dictionary...')

    state_dict = {}
    with safe_open(__model_file, framework="pt", device="cpu") as model:
        for key in model.keys():
            state_dict[key] = model.get_tensor(key)

    return state_dict
