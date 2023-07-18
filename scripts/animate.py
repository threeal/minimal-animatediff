import argparse
from modules.pipeline import create_animation_pipeline


def main(args):
    pipeline = create_animation_pipeline()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main(args)
