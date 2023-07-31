import sys

from diffusers import DDIMScheduler

from deps.AnimateDiff.animatediff.pipelines.pipeline_animation import AnimationPipeline
import deps.AnimateDiff.animatediff.utils.convert_from_ckpt as cvt

from .dream_booth import DreamBoothModel
from .motion_module import MotionModule
from .stable_diffusion import StableDiffusion


def create_animation_pipeline():
    sd = StableDiffusion()

    pipeline = AnimationPipeline(
        text_encoder=sd.text_encoder,
        tokenizer=sd.tokenizer,
        vae=sd.vae,
        unet=sd.unet,
        scheduler=DDIMScheduler(
            **{
                "num_train_timesteps": 1000,
                "beta_start": 0.00085,
                "beta_end": 0.012,
                "beta_schedule": "linear",
                "steps_offset": 1,
                "clip_sample": False,
            }
        ),
    )

    pipeline.to("cuda")

    print("Loading motion module to the animation pipeline...")
    mm = MotionModule("mm_sd_v15.ckpt")
    _, unexpected = pipeline.unet.load_state_dict(mm.states, strict=False)
    if len(unexpected) > 0:
        sys.exit("Failed to load motion module to the animation pipeline!")

    print("Loading diffusion model to the animation pipeline...")
    db_model = DreamBoothModel("toonyou_beta3.safetensors")

    converted_vae_checkpoint = cvt.convert_ldm_vae_checkpoint(db_model.states, pipeline.vae.config)
    pipeline.vae.load_state_dict(converted_vae_checkpoint)

    converted_unet_checkpoint = cvt.convert_ldm_unet_checkpoint(db_model.states, pipeline.unet.config)
    pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)

    pipeline.text_encoder = cvt.convert_ldm_clip_checkpoint(db_model.states)

    pipeline.to("cuda")

    return pipeline
