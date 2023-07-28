from filehash import FileHash
import pytest
import torch
import os

from deps.AnimateDiff.animatediff.utils.util import save_videos_grid
from minimal_animatediff.animation_pipeline import create_animation_pipeline

hasher = FileHash("md5")

dir_path = os.path.dirname(os.path.realpath(__file__))
PIPELINE = None
SAMPLE = None


@pytest.mark.dependency()
def test_create_animation_pipeline():
    global PIPELINE
    PIPELINE = create_animation_pipeline()


@pytest.mark.dependency(depends=["test_create_animation_pipeline"])
def test_run_animation_pipeline():
    global SAMPLE
    assert PIPELINE is not None

    torch.manual_seed(16372571278361863751)
    SAMPLE = PIPELINE(
        "best quality, masterpiece, 1girl, cloudy sky, dandelion, alternate hairstyle,",
        negative_prompt="",
        num_inference_steps=10,
        guidance_scale=7.5,
        width=512,
        height=512,
        video_length=16,
    ).videos


@pytest.mark.dependency(depends=["test_run_animation_pipeline"])
def test_save_animation_gif():
    assert SAMPLE is not None
    gif_path = os.path.join(dir_path, "samples/sample.gif")
    save_videos_grid(SAMPLE, gif_path)
    assert hasher.hash_file(gif_path) == "30cc5fb2a6446f0849889b7a84ec1c42"


@pytest.mark.dependency(depends=["test_run_animation_pipeline"])
def test_save_animation_mp4():
    assert SAMPLE is not None
    mp4_path = os.path.join(dir_path, "samples/sample.mp4")
    save_videos_grid(SAMPLE, mp4_path)
    assert hasher.hash_file(mp4_path) == "76494406baff329aa68d9c22ad0436cc"
