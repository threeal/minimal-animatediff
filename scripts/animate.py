import argparse
import modules.stable_diffusion as sd


def main(args):
    tokenizer = sd.load_tokenizer()
    tokenizer = sd.load_text_encoder()
    tokenizer = sd.load_vae()
    unet = sd.load_unet()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main(args)
