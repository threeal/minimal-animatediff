from huggingface_hub import snapshot_download

def init_stable_diffusion_dir(path: str):
    print('Initializign Stable Diffusion directory...')
    snapshot_download(repo_id='runwayml/stable-diffusion-v1-5', local_dir=path)
