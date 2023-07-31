from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from minimal_animatediff.stable_diffusion import StableDiffusionSnapshot

from deps.AnimateDiff.animatediff.models.unet import UNet3DConditionModel


def test_init_stable_diffusion_snapshot():
    snapshot = StableDiffusionSnapshot()
    assert isinstance(snapshot.text_encoder, CLIPTextModel)
    assert isinstance(snapshot.tokenizer, CLIPTokenizer)
    assert isinstance(snapshot.vae, AutoencoderKL)
    assert isinstance(snapshot.unet, UNet3DConditionModel)
