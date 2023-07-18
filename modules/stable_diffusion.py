from diffusers import AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import snapshot_download
from transformers import CLIPTextModel, CLIPTokenizer

from deps.AnimateDiff.animatediff.models.unet import UNet3DConditionModel

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


def load_unet():
    unet = UNet3DConditionModel.from_pretrained_2d(__snapshot_dir, subfolder="unet", unet_additional_kwargs={
        'unet_use_cross_frame_attention': False,
        'unet_use_temporal_attention': False,
        'use_motion_module': True,
        'motion_module_resolutions': [1, 2, 4, 8],
        'motion_module_mid_block': False,
        'motion_module_decoder_only': False,
        'motion_module_type': "Vanilla",
        'motion_module_kwargs': {
            'num_attention_heads': 8,
            'num_transformer_block': 1,
            'attention_block_types': ["Temporal_Self", "Temporal_Self" ],
            'temporal_position_encoding': True,
            'temporal_position_encoding_max_len': 24,
            'temporal_attention_dim_div': 1
        }
    })
    if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
    else: print('WARNING: XFormers is not installed. XFormers memory-efficient attention is disabled.')
    return unet
