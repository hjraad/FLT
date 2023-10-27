#----------------------------------------------------------------------------
# Created By  : Mohammad Abdizadeh & Hadi Jamali-Rad
# Created Date: 23-Nov-2020
# 
# Refactored By: Sayak Mukherjee
# Last Update: 13-Oct-2023
# ---------------------------------------------------------------------------
# File contains the main function which is the starting point.
# ---------------------------------------------------------------------------

import torch
import logging
import warnings
import random
import numpy as np

from pathlib import Path
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf

from optim import get_method

def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")

    return parser


def main(args: Namespace):

    config_path = Path(args.config)
    config = OmegaConf.load(config_path)

    log_path = Path(config.project.path + '/' + config.project.experiment_name)
    if not Path.exists(log_path):
        Path.mkdir(log_path, exist_ok=True, parents=True)

    logging.basicConfig(level = logging.INFO,
                        filemode = 'w',
                        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename = log_path.joinpath('log.txt'))
    logger = logging.getLogger()

    if args.log_level == "ERROR" or True:
        warnings.filterwarnings("ignore")

    # Set seed
    if config.project.seed != -1:

        # if -1 then keep randomised
        random.seed(config.project.seed)
        np.random.seed(config.project.seed)
        torch.manual_seed(config.project.seed)
        torch.cuda.manual_seed(config.project.seed)
        torch.cuda.manual_seed_all(config.project.seed)

        logger.info('Set seed to %d.' % config.project.seed)

    # set device
    assert config.trainer.accelerator in ["cuda", "cpu", "auto"]

    if config.trainer.accelerator == "cuda":
        if not torch.cuda.is_available():
            raise ValueError('accelerator is set to cuda but cuda is not found.')
        device = "cuda"
    elif config.trainer.accelerator == "cpu":
        device = "cpu"
    elif config.trainer.accelerator == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    logger.info(f'Running on {device}.')

    get_method(config.federated.method)(config, device)

if __name__ == '__main__':

    args = get_parser().parse_args()
    main(args)