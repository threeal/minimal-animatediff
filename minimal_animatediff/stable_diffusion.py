import warnings

from diffusers import AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import snapshot_download
from transformers import CLIPTextModel, CLIPTokenizer

from deps.AnimateDiff.animatediff.models.unet import UNet3DConditionModel


class StableDiffusionSnapshot:
    def __init__(self):
        path = "snapshots/stable_diffusion"
        snapshot_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            local_dir=path,
            allow_patterns=[
                "text_encoder/*.json",
                "text_encoder/*model.bin",
                "tokenizer/*",
                "unet/*.json",
                "unet/*model.bin",
                "vae/*.json",
                "vae/*model.bin",
            ],
        )

        self.text_encoder = CLIPTextModel.from_pretrained(path, subfolder="text_encoder")
        self.tokenizer = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer")
        self.vae = AutoencoderKL.from_pretrained(path, subfolder="vae")

        self.unet = UNet3DConditionModel.from_pretrained_2d(
            path,
            subfolder="unet",
            unet_additional_kwargs={
                "unet_use_cross_frame_attention": False,
                "unet_use_temporal_attention": False,
                "use_motion_module": True,
                "motion_module_resolutions": [1, 2, 4, 8],
                "motion_module_mid_block": False,
                "motion_module_decoder_only": False,
                "motion_module_type": "Vanilla",
                "motion_module_kwargs": {
                    "num_attention_heads": 8,
                    "num_transformer_block": 1,
                    "attention_block_types": ["Temporal_Self", "Temporal_Self"],
                    "temporal_position_encoding": True,
                    "temporal_position_encoding_max_len": 24,
                    "temporal_attention_dim_div": 1,
                },
            },
        )

        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()
        else:
            warnings.warn("XFormers is not installed. memory-efficient is disabled", RuntimeWarning)
