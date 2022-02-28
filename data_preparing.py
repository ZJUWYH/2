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
from utilities import *
from config import *


class AircraftDataset(Dataset):
    def __init__(self, df):
        self.df = df.groupby("Unit").agg(list).reset_index()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        data = {}
        sensor = ['T24','T30','T50','P30','Nf','Nc','Ps30',
                      'phi','NRf','NRc','BPR','htBleed','W31','W32']
        multi_sensor = []
        for sensor_name in sensor:
            multi_sensor.append(np.array(self.df[sensor_name].values.tolist()[idx]))
            single_sensor = np.array(self.df[sensor_name].values.tolist()[idx])[:, None]
            data[sensor_name] = torch.tensor(single_sensor, dtype=torch.float)
        multi_sensor = np.vstack(multi_sensor).transpose(1, 0)
        data["input"] = torch.tensor(multi_sensor, dtype=torch.float)
        data["lifetime"] = torch.tensor(len(multi_sensor), dtype=torch.int64)
        data["timeseries"] = torch.tensor(np.array(self.df["Time"].values.tolist()[idx])[:, None], dtype=torch.int64)

        return data

class TrainingFeature(Dataset):
    def __init__(self, G, dev_G, dev2_G, Tal0):
        self.G = G
        self.dev_G = dev_G
        self.dev2_G = dev2_G
        self.Tal0 = Tal0
        self.prepare_data()

    def prepare_data(self):
        input_features = []
        for unit in range(len(self.G)):
            for time in range(int(self.Tal0[unit]) - 1):
                feature_unit_time = np.vstack([self.G[unit][time],
                                               self.dev_G[unit][time],
                                               self.dev2_G[unit][time]]).reshape(-1)
                input_features.append(feature_unit_time)
        self.input = np.array(input_features)

        total_RUL = []
        for unit in range(len(self.G)):
            lifetime_unit = int(self.Tal0[unit])
            RUL_unit = np.array([lifetime_unit - 1 - i for i in range(lifetime_unit - 1)],
                                dtype=np.int64)
            total_RUL.append(RUL_unit)
        self.total_RUL = np.hstack(total_RUL)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        data = {
            "input": torch.tensor(self.input[idx], dtype=torch.float),
            "RUL": torch.tensor(self.total_RUL[idx], dtype=torch.int64)
        }

        return data


class TestingFeature(Dataset):
    def __init__(self, tG, dev_tG, dev2_tG, RUL):
        self.tG = tG
        self.dev_tG = dev_tG
        self.dev2_tG = dev2_tG
        self.RUL = RUL
        self.prepare_data()

    def prepare_data(self):
        input_features = []
        for unit in range(len(self.tG)):
            feature_unit = np.vstack([self.tG[unit],
                                      self.dev_tG[unit],
                                      self.dev2_tG[unit]]).reshape(-1)
            input_features.append(feature_unit)

        self.input = np.array(input_features)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        data = {
            "input": torch.tensor(self.input[idx], dtype=torch.float),
            "RUL": torch.tensor(self.RUL[idx], dtype=torch.int64)
        }

        return data

