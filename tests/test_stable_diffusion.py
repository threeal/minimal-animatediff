from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from minimal_animatediff.stable_diffusion import StableDiffusion

from deps.AnimateDiff.animatediff.models.unet import UNet3DConditionModel


def test_init_stable_diffusion():
    sd = StableDiffusion()
    assert isinstance(sd.text_encoder, CLIPTextModel)
    assert isinstance(sd.tokenizer, CLIPTokenizer)
    assert isinstance(sd.vae, AutoencoderKL)
    assert isinstance(sd.unet, UNet3DConditionModel)
