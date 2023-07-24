import gradio
import torch

from deps.AnimateDiff.animatediff.utils.util import save_videos_grid
from modules.animation_pipeline import create_animation_pipeline


if __name__ == '__main__':
    pipeline = create_animation_pipeline()

    def generate(
        prompt,
        neg_prompt,
        steps,
        width,
        height,
        frames,
        progress=gradio.Progress()
    ):
        torch.manual_seed(16372571278361863751)

        def update(i, _, __):
            progress(i / float(steps + 1), desc='Sampling')

        progress(0, desc='Sampling')
        sample = pipeline(
            prompt,
            negative_prompt= neg_prompt,
            num_inference_steps = int(steps),
            guidance_scale = 7.5,
            width = int(width),
            height = int(height),
            video_length = int(frames),
            callback = update
        ).videos

        progress(steps / float(steps + 1), desc='Converting')
        save_videos_grid(sample, 'samples/sample.gif')

        progress(1)
        return 'samples/sample.gif'

    app = gradio.Interface(
        fn=generate,
        inputs=[
            gradio.Textbox(label='Prompt'),
            gradio.Textbox(label='Negative Prompt'),
            gradio.Number(label='Steps', value=25, minimum=1),
            gradio.Number(label='Width', value=512, minimum=1),
            gradio.Number(label='Height', value=512, minimum=1),
            gradio.Number(label='Frames', value=16, minimum=1),
        ],
        outputs=gradio.Image(label='Sample')
    )

    app.launch(share=True,enable_queue=True)
