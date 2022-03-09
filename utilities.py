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
from config import *
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
from DNN_model import GRUAutoEncoder, CustomModel


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


#####ae

def collate_fn(data):  # pad 数据
    data.sort(key=lambda x: len(x), reverse=True)
    data = pack_sequence(data)
    return data


def encode_feature_extraction(instance):
    """
    instance is a 类的实例，含有原有的时间序列数据
    return the extracted feature as a list n*num_feature
    """
    aemodel_encode = GRUAutoEncoder()
    aemodel_encode.load_state_dict(torch.load("./model_checkpoints_ae/ae/model_ae.pt", map_location="cpu"))
    all_features_list = []
    for idx in range(len(instance)):
        with torch.no_grad():
            ae_result = aemodel_encode.encoder_GRU(instance[idx]["input"].unsqueeze(0),
                                                   torch.zeros(1, 1, CFG.ae_hidden_layer))[0].squeeze(0)
        all_features_list.append(ae_result)
    return all_features_list


def feature_preprocess(feature_list):
    """
    :param feature_list: 前一个函数出来的feature list
    :return: 首先按照时间步加权求和，再做z_score
    """
    feature_list_expend_condense = []
    for idx in range(len(feature_list)):
        c_l = []
        time = len(feature_list[idx])
        for i in range(time):
            c_l.append(0.001 + i * (2 - 2 * 0.001 * time) / ((time - 1) * time))
        feature_list_expend_condense.append(torch.mm(torch.FloatTensor(c_l).unsqueeze(0),
                                                     feature_list[idx]).squeeze())
    feature_list_expend_condensed = torch.vstack(feature_list_expend_condense)
    for col in range(feature_list_expend_condensed.shape[1]):  # z-score
        testlist = feature_list_expend_condensed[:, col]
        feature_list_expend_condensed[:, col] = (testlist - torch.mean(testlist)) / torch.std(testlist)
    return feature_list_expend_condensed


def get_input(instance):
    """
    :param instance: 特征提取后一个类的实例，以ae方法为例
    :return: 获得的n个样本*num——feature的array
    """
    input = []
    lenth = len(instance)
    for idx in range(lenth):
        input.append(instance[idx]["input"].numpy())
    return np.array(input)
