import argparse
from modules.stable_diffusion import check_stable_diffusion

def main(args):
    check_stable_diffusion('models/stable_diffusion')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main(args)
