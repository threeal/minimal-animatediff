import gradio
from modules.animation_pipeline import create_animation_pipeline
import torch

from deps.AnimateDiff.animatediff.utils.util import save_videos_grid


if __name__ == '__main__':
    pipeline = create_animation_pipeline()

    def generate(prompt, progress=gradio.Progress()):
        torch.manual_seed(16372571278361863751)

        def update(i, t, latents):
            progress(i / float(26), desc='Sampling')

        progress(0, desc='Sampling')
        sample = pipeline(
            prompt,
            negative_prompt= '',
            num_inference_steps = 25,
            guidance_scale = 7.5,
            width = 512,
            height = 512,
            video_length = 16,
            callback = update
        ).videos

        progress(25 / float(26), desc='Converting')
        save_videos_grid(sample, 'samples/sample.gif')

        progress(1)
        return 'samples/sample.gif'

    app = gradio.Interface(
        fn=generate,
        inputs=gradio.Textbox(label='Prompt'),
        outputs=gradio.Image(label='Sample')
    )

    app.launch(share=True,enable_queue=True)
