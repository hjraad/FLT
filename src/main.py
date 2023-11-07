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
    parser.add_argument("--scenario", type=int, required=False, default=-1, help="Scenario to execute")
    parser.add_argument("--config", type=str, required=False, default=None, help="Path to a model config file")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")

    return parser


def run_experiment(config):

    log_path = Path(config.project.path + '/scenario' + str(config.federated.scenario) + '/' + config.project.experiment_name)
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

    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()

def main(args: Namespace):

    if args.scenario == -1 and args.config is None:
        raise ValueError('Either scenario or config path needs to be mentioned.')
    elif args.scenario != -1 and args.config is not None:
        raise ValueError('Both scenario or config path cannot be provided.')

    if args.config is not None:
        config_path = Path(args.config)
        config = OmegaConf.load(config_path)

        run_experiment(config)

    else:
        for scenario in range(3):
            # root_path = Path(f'../configs/scenario_{args.scenario}')
            root_path = Path(f'../configs/scenario_{scenario + 1}')
            all_configs = [x for x in root_path.glob('**/*') if x.is_file()]

            for config_path in all_configs:
                print(f'Running {config_path.stem}')
                torch.cuda.empty_cache()
                config = OmegaConf.load(config_path)
                run_experiment(config)

if __name__ == '__main__':

    args = get_parser().parse_args()
    main(args)