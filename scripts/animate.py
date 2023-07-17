import argparse
from modules.stable_diffusion import init_stable_diffusion_dir

def main(args):
    init_stable_diffusion_dir('models/stable_diffusion')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main(args)
