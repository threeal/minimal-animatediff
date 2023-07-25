import pytest
import torch

from deps.AnimateDiff.animatediff.utils.util import save_videos_grid
from modules.animation_pipeline import create_animation_pipeline


@pytest.mark.dependency()
def test_create_animation_pipeline():
    global pipeline
    pipeline = create_animation_pipeline()


@pytest.mark.dependency(depends=['test_create_animation_pipeline'])
def test_run_animation_pipeline():
    global pipeline, sample
    torch.manual_seed(16372571278361863751)
    sample = pipeline(
        'best quality, masterpiece, 1girl, cloudy sky, dandelion, alternate hairstyle,',
        negative_prompt     = '',
        num_inference_steps = 25,
        guidance_scale      = 7.5,
        width               = 512,
        height              = 512,
        video_length        = 16,
    ).videos


@pytest.mark.dependency(depends=['test_run_animation_pipeline'])
def test_save_animation_pipeline():
    global sample
    save_videos_grid(sample, 'samples/sample.gif')
