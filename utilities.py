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
# from DNN_model import GRUAutoEncoder, CustomModel, LSTMAutoEncoder
import json
import copy
import math


def clones(module, N):
    "生成n个相同的层"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def save_dict(pet, filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(pet))


def load_dict(filename):
    with open(filename) as f:
        pet = json.loads(f.read())
    return pet


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

def set_seed(seed):
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        print("Set seed failed,details are ",e)
        pass
    import numpy as np
    np.random.seed(seed)
    import random as python_random
    python_random.seed(seed)


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


def my_collate(batch):  # pad 数据包含字典的哦
    # batch contains a list of tuples of structure (sequence, target)
    data = [item["input"] for item in batch]
    data = pack_sequence(data, enforce_sorted=False)
    targets = [item["mode"] for item in batch]
    res = {
        "input": data,
        "mode": torch.vstack(targets)

    }

    return res


def encode_feature_extraction(instance):
    """
    instance is a 类的实例，含有原有的时间序列数据
    return the extracted feature as a list n*num_feature
    """
    aemodel_encode = LSTMAutoEncoder()
    aemodel_encode.load_state_dict(torch.load("./model_checkpoints_ae/ae/model_ae.pt", map_location="cpu"))
    all_features_list = []
    for idx in range(len(instance)):
        with torch.no_grad():
            ae_result = aemodel_encode.encoder1(instance[idx]["input"].unsqueeze(0))[0]  # .squeeze(0)
            ae_result = aemodel_encode.encoder2(ae_result)[1][0].squeeze()
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

##tsfresh
def get_dataframe_ofcuttedsequence(instance,idx):#instance is from AircraftDataset_expend
    """
    :param instance:
    :param idx:
    :return: a dataframe of ctuued sequence
    """
    testarray=np.array(instance[idx]["input"],dtype=np.float64)
    testdataframe=pd.DataFrame(testarray,columns=['T24','T30','T50','P30','Ps30','phi'])
    testdataframe["Unit"]=idx+1
    testdataframe=Preprocessing.add_timeseries(testdataframe)
    return testdataframe

def get_Accuracy(hat_RUL, RUL_true, lifetime_list):
    """

    :return: RUL升序排序后前20个，前40个，前**个的误差
    """
    # hat_RUL = get_RUL(mu_Gamma, C_Gamma, sigma_2, w, data, data_train)
    # RUL_frame = pd.read_csv(path, header=None)
    # RUL = RUL_frame.values[:, 0]
    Accuracy = np.zeros((100, 2))
    Accuracy[:, 0] = RUL_true
    Accuracy[:, 1] = abs(hat_RUL - RUL_true) / (lifetime_list + RUL_true)
    num_A = np.argsort(Accuracy[:, 0])
    iAccuracy = np.zeros((100, 2))
    for ia in range(0, 100):
        iAccuracy[ia, 0] = Accuracy[int(num_A[ia]), 0]
        iAccuracy[ia, 1] = Accuracy[int(num_A[ia]), 1]
    rul0 = [25, 50, 75, 100, 125, 300]
    Accuracy_RUL = np.mat(np.zeros((6, 3)))
    for ir in range(0, 6):
        Accuracy_RUL[ir, 0] = int(rul0[ir])
        num_rul = np.argwhere((iAccuracy[:, 0] <= rul0[ir]))
        #(cycletime_sim_raw > 50) & (cycletime_sim_raw < 350)
        Accuracy_RUL[ir, 1] = np.mean(iAccuracy[num_rul[:, 0], 1])
        s_error = np.std(iAccuracy[num_rul[:, 0], 1])/math.sqrt(len(num_rul))
        Accuracy_RUL[ir, 2] = s_error
    Accuracy_RUL[5, 0] = 150
    return Accuracy_RUL