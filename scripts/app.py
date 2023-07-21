import gradio
from modules.animation_pipeline import create_animation_pipeline
import torch

from deps.AnimateDiff.animatediff.utils.util import save_videos_grid


def generate(prompt):
    pipeline = create_animation_pipeline()

    torch.manual_seed(16372571278361863751)

    sample = pipeline(
        prompt,
        negative_prompt     = '',
        num_inference_steps = 25,
        guidance_scale      = 7.5,
        width               = 512,
        height              = 512,
        video_length        = 16,
    ).videos

    save_videos_grid(sample, 'samples/sample.gif')

    return 'samples/sample.gif'


if __name__ == '__main__':
    app = gradio.Interface(
        fn=generate,
        inputs=gradio.Textbox(label='Prompt'),
        outputs=gradio.Image(label='Sample')
    )

    app.launch(share=True,enable_queue=True)
