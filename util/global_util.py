import configparser
import logging
import os
import shutil
import time
import torch
import yaml
from typing import Dict
import random
import gym
import numpy as np
from time import strftime, localtime

import pfrl.utils.random_seed


def get_project_root() -> str:
    """
    Get the absolute path of the project root

    Returns path string of the project root
    -------

    """
    return os.path.join(os.path.dirname(__file__), "..")


def load_yaml(filename) -> Dict:
    with open(filename) as f:
        return yaml.safe_load(f)


def load_cfg_config(config_dir: str, filename: str) -> configparser.RawConfigParser:
    config = configparser.RawConfigParser()
    config.read(os.path.join(config_dir, filename))
    return config


def setup_logger(filename="train.log", use_console_log=True, use_file_log=False):
    """Sets up the logger"""
    # remove the old log file
    if os.path.isfile(filename):
        os.remove(filename)

    log_level = logging.INFO
    # formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] -- %(message)s")

    logger = logging.getLogger()
    logger.setLevel(log_level)

    if use_console_log:
        # add a console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if use_file_log:
        # add a file handler
        handler = logging.FileHandler(filename)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def set_random_seeds(random_seed: int):
    """
    Setup all possible random seeds so results can be reproduced
    """
    pfrl.utils.random_seed.set_random_seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(random_seed)
    # tf.set_random_seed(random_seed) # if you use tensorflow
    random.seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.cuda.manual_seed(random_seed)
    if hasattr(gym.spaces, "prng"):
        gym.spaces.prng.seed(random_seed)


def check_create_parent_dir_for_file(file_path: str):
    """
    Check if the parent folder of the file exists. If not, create the parent folder

    Parameters
    ----------
    file_path

    Returns
    -------

    """
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_standard_current_time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def get_current_time_for_filename():
    return strftime("%Y-%m-%d_%H-%M-%S", localtime())


def copy_config_files_to_output(config_dir: str, output_dir: str):
    shutil.copy(config_dir, output_dir)


def get_device(gpu: int):
    if gpu is not None and gpu >= 0:
        logging.info("Use device: CUDA:{}".format(gpu))
        return torch.device("cuda:{}".format(gpu))
    else:
        logging.info("Use device: CPU")
        return torch.device("cpu")
