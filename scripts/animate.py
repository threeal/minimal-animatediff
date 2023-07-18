import argparse
import modules.stable_diffusion as sd


def main(args):
    print('Initializing Stable Diffusion snapshot...')
    sd.init_snapshot('models/stable_diffusion')

    print('Loading tokenizer...')
    tokenizer = sd.load_tokenizer()

    print('Loading text encoder...')
    tokenizer = sd.load_text_encoder()

    print('Loading variable auto encoder...')
    tokenizer = sd.load_vae()

    print('Loading U-Net...')
    unet = sd.load_unet()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main(args)
