from huggingface_hub import snapshot_download

__snapshot_dir = ""

def init_snapshot(path: str):
    __snapshot_dir = path
    snapshot_download(repo_id='runwayml/stable-diffusion-v1-5', local_dir=__snapshot_dir)
