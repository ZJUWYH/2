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
import torch.nn.functional as F
from utilities import *
from config import *
from data_preparing import *

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        #         self.cfg=cfg
        #         self.input_dim=cfg.input_dim
        #         self.hidden_size=cfg.hidden_size
        #         self.num_classes=cfg.num_classes

        self.mlp = nn.Sequential(
            nn.Linear(CFG.input_feature, CFG.hidden_size),
            nn.ReLU(),
        )
        # self.lstm1=nn.LSTM(CFG.hidden_size, CFG.hidden_size//2, dropout=0.1, batch_first=True, bidirectional=True)
        self.logits = nn.Sequential(
            nn.Linear(CFG.hidden_size, CFG.num_classes),
        )

    def forward(self, x):
        features = self.mlp(x)
        # features, _ = self.lstm1(features)
        pred = self.logits(features)
        return pred

class CustomLoss(nn.Module):
    def __init__(self):
        self.kernel=CFG.kernel
        super().__init__()

    def get_kernel(self,train_data,test_data):
        """

        :param train_data: 类的一个实例，字典取["input"], batch_size(1024)*input_feature，
        :param test_data:  1*input_feature
        :return: a diag matrix representing weights
        """
        #n_batch=len(train_data["input"],dtye=torch.float)
        n_batch=CFG.batch_size
        diag=torch.zeros(n_batch,dtype=float).to(CFG.device)
        for idx in range(n_batch):
            diag[idx]=torch.exp(
                -F.pairwise_distance(test_data[0],train_data[idx],p=2)
            )
        diag[torch.where(diag < torch.median(diag))] = 0
        return torch.diag(diag/torch.sum(diag))

    def get_kernel_loss(self, pred_batch, target_batch, train_data, test_data):
        """
        :param pred: batch_size*1,
        :param target: batch_size*1, 实例["RUL"]
        :param train_data:
        :param test_data:
        :return: the loss
        """
        n_batch = CFG.batch_size
        loss = torch.zeros(1, dtype=torch.float).to(CFG.device)
        kernel=self.get_kernel(train_data,test_data).float()
        if self.kernel:
            residual = (target_batch - pred_batch).float()
            loss = torch.inner(torch.mm(kernel,residual).squeeze(-1),residual.squeeze(-1))
            return loss
        else:
            residual = (target_batch - pred_batch).squeeze(-1).float()
            loss = torch.inner(residual, residual)/n_batch
            return loss

    def forward(self,pred_batch, target_batch, train_data, test_data):
        return self.get_kernel_loss(pred_batch, target_batch, train_data, test_data)

class myscheduler():
    def __init__(self,optimizer,decay):
        self.optimizer = optimizer
        self.decay = decay      #学习率衰减值
    def step(self,loss,last_loss):
        if loss.item()>last_loss.item():
            lr=self.decay*last_loss.item()*0.9
        else:
            lr=self.decay*loss.item()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


