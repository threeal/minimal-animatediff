from diffusers import DDIMScheduler

from deps.AnimateDiff.animatediff.pipelines.pipeline_animation import AnimationPipeline
import modules.motion_module as mm
import modules.stable_diffusion as sd


def create_animation_pipeline():
    print('Creating animation pipeline...')
    pipeline = AnimationPipeline(
        text_encoder=sd.load_text_encoder(),
        tokenizer=sd.load_tokenizer(),
        vae=sd.load_vae(),
        unet=sd.load_unet(),
        scheduler=DDIMScheduler(**{
            'num_train_timesteps': 1000,
            'beta_start': 0.00085,
            'beta_end': 0.012,
            'beta_schedule': 'linear',
            'steps_offset': 1,
            'clip_sample': False
        }),
    )

    pipeline.to("cuda")

    print('Loading motion module to the animation pipeline...')
    _, unexpected = pipeline.unet.load_state_dict(mm.load_state_dict(), strict=False)
    if len(unexpected) > 0:
        exit('Failed to load motion module to the animation pipeline!')

    return pipeline
