import subprocess
import os

def install_git_lfs():
    subprocess.run(["git", "lfs", "install"], check=True)


def clone_stable_diffusion(path: str):
    install_git_lfs()

    url = "https://huggingface.co/runwayml/stable-diffusion-v1-5"
    subprocess.run(["git", "clone", "--depth", "1", url, path], check=True)


def check_stable_diffusion(path: str):
    print('Checking for Stable Diffusion model...')
    git_dir = os.path.join(path, ".git")
    if not os.path.isdir(git_dir):
        print(' Stable Diffusion model not found, cloning...')
        clone_stable_diffusion(path)
