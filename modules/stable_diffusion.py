from diffusers import AutoencoderKL
from huggingface_hub import snapshot_download
from transformers import CLIPTextModel, CLIPTokenizer

__snapshot_dir = None


def init_snapshot(path: str):
    global __snapshot_dir
    __snapshot_dir = path
    snapshot_download(repo_id='runwayml/stable-diffusion-v1-5', local_dir=__snapshot_dir)


def load_tokenizer():
    return CLIPTokenizer.from_pretrained(__snapshot_dir, subfolder='tokenizer')


def load_text_encoder():
    return CLIPTextModel.from_pretrained(__snapshot_dir, subfolder='text_encoder')


def load_vae():
    return AutoencoderKL.from_pretrained(__snapshot_dir, subfolder='vae')
