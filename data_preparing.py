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
# from utilities import fusion_tG
from config import *
from sklearn.cluster import KMeans


class AircraftDataset(Dataset):
    def __init__(self, df):
        self.df = df.groupby("Unit").agg(list).reset_index()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        data = {}
        sensor = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30',
                  'phi', 'NRf', 'NRc', 'BPR', 'htBleed', 'W31', 'W32']
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
    def __init__(self, tG, dev_tG, dev2_tG, RUL, classifier):  # ,labels):
        self.tG = tG
        self.dev_tG = dev_tG
        self.dev2_tG = dev2_tG
        self.RUL = RUL
        self.classifier = classifier
        self.prepare_data()

    def prepare_data(self):
        input_features = []
        for unit in range(len(self.tG)):
            feature_unit = np.vstack([self.tG[unit],
                                      self.dev_tG[unit],
                                      self.dev2_tG[unit]]).reshape(-1)
            input_features.append(feature_unit)
        self.input = np.array(input_features)
        self.labels = self.classifier.predict(self.input)

        # self.labels=labels

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        data = {
            "input": torch.tensor(self.input[idx], dtype=torch.float),
            "RUL": torch.tensor(self.RUL[idx], dtype=torch.int64),
            "label": torch.tensor(self.labels[idx], dtype=torch.int64)
        }

        return data


class Classified_mean_test_features(TestingFeature):
    def __init__(self, tG, dev_tG, dev2_tG, RUL, classifier_in):  # classifier_in为训练好的一个分类器
        super().__init__(tG, dev_tG, dev2_tG, RUL, classifier_in)

    def __len__(self):
        return CFG.num_in_feature_classes

    def __getitem__(self, idx):
        data = {
            "input": torch.tensor(np.mean(self.input[np.where(self.labels == idx)], axis=0), dtype=torch.float),
            "label": torch.tensor(idx, dtype=torch.int64)
        }
        return data


class Classified_mean_train_features(TrainingFeature):
    def __init__(self, G, dev_G, dev2_G, Tal0, classifier):
        super().__init__(G, dev_G, dev2_G, Tal0)
        self.classifier = classifier
        self.labels = self.classifier.predict(self.input)

    def __len__(self):
        return CFG.num_in_feature_classes

    def __getitem__(self, idx):
        data = {
            "input": torch.tensor(np.mean(self.input[np.where(self.labels == idx)], axis=0), dtype=torch.float),
            "label": torch.tensor(idx, dtype=torch.int64)
        }
        return data
