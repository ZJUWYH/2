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


# class AircraftDataset(Dataset):
#     def __init__(self, df):
#         self.df = df.groupby("Unit").agg(list).reset_index()
#
#     def __len__(self):
#         return self.df.shape[0]
#
#     def __getitem__(self, idx):
#         data = {}
#         sensor = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30',
#                   'phi', 'NRf', 'NRc', 'BPR', 'htBleed', 'W31', 'W32']
#         multi_sensor = []
#         for sensor_name in sensor:
#             multi_sensor.append(np.array(self.df[sensor_name].values.tolist()[idx]))
#             single_sensor = np.array(self.df[sensor_name].values.tolist()[idx])[:, None]
#             data[sensor_name] = torch.tensor(single_sensor, dtype=torch.float)
#         multi_sensor = np.vstack(multi_sensor).transpose(1, 0)
#         data["input"] = torch.tensor(multi_sensor, dtype=torch.float)
#         data["lifetime"] = torch.tensor(len(multi_sensor), dtype=torch.int64)
#         data["timeseries"] = torch.tensor(np.array(self.df["Time"].values.tolist()[idx])[:, None], dtype=torch.int64)
#
#         return data
class AircraftDataset(Dataset):
    def __init__(self, df, labels):# df is a dataframe and label is an array indicate the true failure mode
        self.df = df.groupby("Unit").agg(list).reset_index()
        self.labels=labels

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        data = {}
#         sensor = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30',
#                   'phi', 'NRf', 'NRc', 'BPR', 'htBleed', 'W31', 'W32']
        sensor=['T24','T30','T50','P30','Ps30','phi']
        multi_sensor = []
        for sensor_name in sensor:
            multi_sensor.append(np.array(self.df[sensor_name].values.tolist()[idx]))
            single_sensor = np.array(self.df[sensor_name].values.tolist()[idx])[:, None]
            #data[sensor_name] = torch.tensor(single_sensor, dtype=torch.float)
        multi_sensor = np.vstack(multi_sensor).transpose(1, 0)
        data["input"] = torch.tensor(multi_sensor, dtype=torch.float)
        data["lifetime"] = torch.tensor(len(multi_sensor), dtype=torch.int64)
        #data["timeseries"] = torch.tensor(np.array(self.df["Time"].values.tolist()[idx])[:, None], dtype=torch.int64)
        if self.labels[idx].item()==-1:
            data["mode"]=torch.tensor([1,0],dtype=torch.float)
        elif self.labels[idx].item()==1:
            data["mode"]=torch.tensor([0,1],dtype=torch.float)
        return data


# class AircraftDataset_expend(AircraftDataset):  # 截断原有的数据集，获得海量的数据
#     def __init__(self, df, add_zero):
#         super().__init__(df)
#         self.add_zero = add_zero
#         self.cut_data()
#
#     def cut_data(self):
#         lenth = super().__len__()
#         input_signal = []
#         RUL = []
#         for unit in range(lenth):
#             unit_input = super().__getitem__(unit)["input"]
#             unit_life = super().__getitem__(unit)["lifetime"]
#             if self.add_zero:
#                 for time in range(3, unit_life):
#                     input_tensor = torch.zeros(525, 14, dtype=torch.float)
#                     input_tensor[0:time] = unit_input[0:time]
#                     unit_RUL = unit_life - time
#                     input_signal.append(input_tensor)
#                     RUL.append(unit_RUL)
#             else:
#                 for time in range(3, unit_life):
#                     input_tensor = unit_input[0:time]
#                     unit_RUL = unit_life - time
#                     input_signal.append(input_tensor)
#                     RUL.append(unit_RUL)
#
#         self.RUL = np.array(RUL)
#         self.input_signal = input_signal
#
#     def __len__(self):
#         return len(self.RUL)
#
#     def __getitem__(self, idx):
#         data = {
#             "input": self.input_signal[idx],
#             "RUL": torch.tensor(self.RUL[idx], dtype=torch.int64)
#         }
#
#         return data
class AircraftDataset_expend(AircraftDataset):  # 截断原有的数据集，获得海量的数据
    def __init__(self, df,labels,add_zero):
        super().__init__(df,labels)
        self.add_zero = add_zero
        self.feature = CFG.ae_input_layer
        self.cut_data()


    def cut_data(self):
        lenth = super().__len__()
        input_signal = []
        RUL = []
        label=[]
        for unit in range(lenth):
            unit_input = super().__getitem__(unit)["input"]
            unit_label=super().__getitem__(unit)["mode"]
            unit_life = len(unit_input)
            if self.add_zero:
                for time in range(7, unit_life):
                    input_tensor = torch.zeros(525, self.feature, dtype=torch.float)
                    input_tensor[0:time] = unit_input[0:time]
                    unit_RUL = unit_life - time
                    input_signal.append(input_tensor)
                    RUL.append(unit_RUL)
                    label.append(unit_label)
            else:
                for time in range(7, unit_life):
                    input_tensor = unit_input[0:time]
                    unit_RUL = unit_life - time
                    input_signal.append(input_tensor)
                    RUL.append(unit_RUL)
                    label.append(unit_label)

        self.RUL = np.array(RUL)
        self.input_signal = input_signal
        self.all_labels = label

    def __len__(self):
        return len(self.RUL)

    def __getitem__(self, idx):
        data = {
            "input": self.input_signal[idx],
            "lifetime": len(self.input_signal[idx]),
            "RUL": torch.tensor(self.RUL[idx], dtype=torch.int64),
            "mode":self.all_labels[idx]
        }

        return data


# class AircraftDataset_expend_norul(AircraftDataset_expend): #不包含RUL，方便后续的pad操作
#     def __init__(self, df):
#         super().__init__(df, add_zero=False)
#
#     def __getitem__(self, idx):
#         return self.input_signal[idx]


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


class Classified_mean_test_features(TestingFeature):  # 分类testingdata，得到每类的均值
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


#####ae model
class AircraftDataset_expend_feature_extraction(AircraftDataset_expend):
    """
    输入为feature list
    """

    def __init__(self, df, labels, all_feature_list,add_zero):
        super().__init__(df, labels,add_zero)
        self.all_feature_list = all_feature_list

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data["input"] = self.all_feature_list[idx]
        return data


class AircraftDataset_no_expend_feature_extraction(AircraftDataset):
    """
    输入为feature list,针对的是没有扩展的数据集
    """

    def __init__(self, df, labels,all_feature_list):
        super().__init__(df,labels)
        self.all_feature_list = all_feature_list

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data["input"] = self.all_feature_list[idx]
        return data
