import argparse

import yaml

from preprocessor import aihub_mmv, iemocap, selvas, man2, girl2, one


def main(config):
    if "AIHub-MMV" in config["dataset"]:
        aihub_mmv.prepare_align(config)
    if "IEMOCAP" in config["dataset"]:
        iemocap.prepare_align(config)
    if "man2" in config['dataset']:
        man2.prepare_align(config)
    if "selvas" in config['dataset']:
        selvas.prepare_align(config)
    if "girl2" in config['dataset']:
        girl2.prepare_align(config)
    if "one" in config['dataset']:
        one.prepare_align(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)
