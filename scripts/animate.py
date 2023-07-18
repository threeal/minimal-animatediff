import argparse
import modules.stable_diffusion as sd

def main(args):
    print('Initializing Stable Diffusion snapshot...')
    sd.init_snapshot('models/stable_diffusion')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main(args)
