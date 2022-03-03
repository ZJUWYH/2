import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from collections import Counter
import torch.nn as nn
import gc
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import random
from DNN_model import *
from config import *


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results.

    Args:
        seed (int): Number of the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import torch.nn as nn


# seed_everything(CFG.seed)

class Preprocessing:

    def drop_sensors(df, sensor_index):
        df0 = df.copy()
        df0.drop(df0.columns[sensor_index], axis=1, inplace=True)
        return df0

    def drop_units(df, unit_index):
        df0 = df.copy()
        df0.drop(df0[df0[df0.columns[0]].isin(unit_index)].index, axis=0, inplace=True)
        return df0.reset_index(drop=True)

    def add_timeseries(df):
        df0 = df.copy()
        df0["Time"] = df0.groupby(["Unit"]).cumcount() + 1
        return df0


def save_model_weights(model, filename, verbose=1, cp_folder=""):
    """
    Saves the weights of a PyTorch model.

    Args:
        model (torch model): Model to save the weights of.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        cp_folder (str, optional): Folder to save to. Defaults to "".
    """
    if verbose:
        print(f"\n -> Saving weights to {os.path.join(cp_folder, filename)}\n")
    torch.save(model.state_dict(), os.path.join(cp_folder, filename))


def worker_init_fn(worker_id):
    """
    Handles PyTorch x Numpy seeding issues.

    Args:
        worker_id (int): Id of the worker.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def fusion_tG(tG, dev_tG, dev2_tG):
    input_features = []
    for unit in range(len(tG)):
        feature_unit = np.vstack([tG[unit],
                                  dev_tG[unit],
                                  dev2_tG[unit]]).reshape(-1)
        input_features.append(feature_unit)
    return np.array(input_features)
