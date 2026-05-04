#!/usr/bin/env python3
"""Self-supervised training entrypoint using jsonargparse config parsing."""

from timm import utils
from timme.engine import TrainConfig
from timme.apps.train_ssl import train_ssl


def parse_args() -> TrainConfig:
    """Parse command line arguments using jsonargparse."""
    try:
        from jsonargparse import ArgumentParser
    except ImportError:
        raise ImportError(
            "jsonargparse is required for train_ssl_jsonargparse. Install with: pip install jsonargparse[signatures]"
        )

    parser = ArgumentParser(
        description='Self-Supervised Learning Training with timme.engine',
        default_config_files=['~/.config/timme/train_ssl.yaml'],
        print_config='--dump-config',
    )
    parser.add_argument(
        '--config',
        action='config',
        help='Path to YAML/JSON config file. Can be specified multiple times.',
    )
    parser.add_class_arguments(TrainConfig, nested_key=None, fail_untyped=False)
    return parser.instantiate_classes(parser.parse_args())


def main():
    """CLI entrypoint for SSL training with jsonargparse."""
    utils.setup_default_logging()
    cfg = parse_args()
    train_ssl(cfg)


if __name__ == '__main__':
    main()
