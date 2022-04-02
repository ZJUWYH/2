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
from utilities import clones
from utilities import *
from config import *
from data_preparing import *


class CustomModel(nn.Module):
    def __init__(self, num_feature=CFG.input_feature, num_hidden=CFG.hidden_size):
        super(CustomModel, self).__init__()
        #         self.cfg=cfg
        #         self.input_dim=cfg.input_dim
        #         self.hidden_size=cfg.hidden_size
        #         self.num_classes=cfg.num_classes
        self.num_feature = num_feature
        self.num_hidden = num_hidden

        self.mlp = nn.Sequential(
            nn.Linear(self.num_feature, self.num_hidden),
            nn.ReLU(),
        )
        # self.lstm1=nn.LSTM(CFG.hidden_size, CFG.hidden_size//2, dropout=0.1, batch_first=True, bidirectional=True)
        self.logits = nn.Sequential(
            nn.Linear(self.num_hidden, CFG.num_classes),
        )

    def forward(self, x):
        features = self.mlp(x)
        # features, _ = self.lstm1(features)
        pred = self.logits(features)
        return pred


class CustomModel2(nn.Module):
    def __init__(self, num_feature, num_hidden):
        super(CustomModel2, self).__init__()
        #         self.cfg=cfg
        #         self.input_dim=cfg.input_dim
        #         self.hidden_size=cfg.hidden_size
        #         self.num_classes=cfg.num_classes
        self.num_feature = num_feature
        self.num_hidden = num_hidden

        self.mlp = nn.Sequential(
            nn.Linear(self.num_feature, self.num_hidden),
            nn.ReLU(),
            nn.Linear(self.num_hidden, self.num_hidden // 2),
            nn.ReLU(),
        )
        # self.lstm1=nn.LSTM(CFG.hidden_size, CFG.hidden_size//2, dropout=0.1, batch_first=True, bidirectional=True)
        self.logits = nn.Sequential(
            nn.Linear(self.num_hidden // 2, CFG.num_classes),
        )

    def forward(self, x):
        features = self.mlp(x)
        # features, _ = self.lstm1(features)
        pred = self.logits(features)
        return pred


class CustomModel3(nn.Module):
    def __init__(self, num_feature, num_hidden):
        super(CustomModel3, self).__init__()
        #         self.cfg=cfg
        #         self.input_dim=cfg.input_dim
        #         self.hidden_size=cfg.hidden_size
        #         self.num_classes=cfg.num_classes
        self.num_feature = num_feature
        self.num_hidden = num_hidden

        self.mlp = nn.Sequential(
            nn.Linear(self.num_feature, self.num_hidden),
            nn.ReLU(),
            nn.Linear(self.num_hidden, self.num_hidden // 2),
            nn.ReLU(),
            nn.Linear(self.num_hidden // 2, self.num_hidden // 4),
            nn.ReLU(),
        )
        # self.lstm1=nn.LSTM(CFG.hidden_size, CFG.hidden_size//2, dropout=0.1, batch_first=True, bidirectional=True)
        self.logits = nn.Sequential(
            nn.Linear(self.num_hidden // 4, CFG.num_classes),
        )

    def forward(self, x):
        features = self.mlp(x)
        # features, _ = self.lstm1(features)
        pred = self.logits(features)
        return pred


class CustomLoss(nn.Module):
    def __init__(self):
        self.kernel = CFG.kernel
        super().__init__()

    def get_kernel(self, train_data, test_data):
        """

        :param train_data: 类的一个实例，字典取["input"], batch_size(1024)*input_feature，
        :param test_data:  1*input_feature
        :return: a diag matrix representing weights
        """
        # n_batch=len(train_data["input"],dtye=torch.float)
        n_batch = CFG.batch_size
        diag = torch.zeros(n_batch, dtype=float).to(CFG.device)
        for idx in range(n_batch):
            diag[idx] = torch.exp(
                -F.pairwise_distance(test_data[0], train_data[idx], p=2)
            )
        # diag[torch.where(diag < torch.quantile(diag, 0.75, dim=0,
        #                                        keepdim=False,
        #                                        interpolation='nearest'))] = 0
        diag[torch.where(diag < torch.median(diag))] = 0
        return torch.diag(diag / torch.sum(diag))

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
        kernel = self.get_kernel(train_data, test_data).float()
        if self.kernel:
            residual = (target_batch - pred_batch).float()
            loss = torch.inner(torch.mm(kernel, residual).squeeze(-1), residual.squeeze(-1))
            return loss
        else:
            residual = (target_batch - pred_batch).squeeze(-1).float()
            loss = torch.inner(residual, residual) / n_batch
            return loss

    def forward(self, pred_batch, target_batch, train_data, test_data):
        return self.get_kernel_loss(pred_batch, target_batch, train_data, test_data)


class myscheduler():
    def __init__(self, optimizer, decay):
        self.optimizer = optimizer
        self.decay = decay  # 学习率衰减值

    def step(self, loss, last_loss):
        if loss.item() > last_loss.item():
            lr = self.decay * last_loss.item() * 0.9
        else:
            lr = self.decay * loss.item()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


######aemodel


class GRUAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = CFG.ae_input_layer
        self.hidden_layer = CFG.ae_hidden_layer
        self.encoder_GRU = nn.GRU(self.input_layer, self.hidden_layer, batch_first=True)
        self.decoder_GRU = nn.GRU(self.hidden_layer, self.input_layer, batch_first=True)

    def forward(self, x):
        x, h0 = self.encoder_GRU(x,
                                 torch.zeros(1, CFG.ae_batch_size, CFG.ae_hidden_layer).to(CFG.device))
        decoded_output, hidden = self.decoder_GRU(x,
                                                  h0)
        # torch.zeros(1, CFG.ae_batch_size, CFG.ae_input_layer).to(CFG.device))

        return decoded_output, h0


class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_layer=CFG.ae_input_layer,
                 hidden_layer=CFG.ae_hidden_layer,
                 ae_num_class=2,
                 seq_len=CFG.max_len):
        super().__init__()
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.num_class = ae_num_class
        self.seq_len = seq_len
        self.encoder1 = nn.LSTM(self.input_layer, self.hidden_layer, batch_first=True)
        self.encoder2 = nn.LSTM(self.hidden_layer, self.hidden_layer * 2, batch_first=True)
        self.decoder1 = nn.LSTM(self.hidden_layer * 2, self.hidden_layer, batch_first=True)
        self.decoder2 = nn.LSTM(self.hidden_layer, self.input_layer, batch_first=True)
        self.logits = nn.Sequential(
            nn.Linear(self.hidden_layer * 2, self.hidden_layer),
            nn.ReLU(),
            nn.Linear(self.hidden_layer, self.num_class),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        x, (_, _) = self.encoder1(x)
        x, (h0, _) = self.encoder2(x)  # h0 1*batch_size*hidden_dim
        # input=h0.squeeze().unsqueeze(1).repeat(1,self.seq_len,1) #input: batch_size,seq_len*hidden_dim
        pad = pad_packed_sequence(x)
        for idx in range(len(pad[1])):
            length = pad[1][idx]
            pad[0][:, idx][0:length] = h0.squeeze()[idx]
        input = pack_padded_sequence(pad[0], pad[1], enforce_sorted=False)
        input, (_, _) = self.decoder1(input)
        output, (_, _) = self.decoder2(input)

        label = self.logits(h0)

        return output, label


# attention based model

class AML_model(nn.Module):

    def __init__(self, num_feature, num_hidden, num_attention, num_hidden_2, num_classes=2, num_head=5,
                 shared=True, attention=True,expand=True):

        super().__init__()
        self.num_feature = num_feature
        self.num_hidden = num_hidden
        self.num_attention = num_attention
        self.num_classes = num_classes
        self.num_head = num_head
        self.num_hidden_2 = num_hidden_2
        self.shared=shared
        self.attention=attention
        self.expand=expand

        self.shared_layer = nn.Sequential(
            nn.Linear(self.num_feature, self.num_hidden),
            nn.ReLU(),
        )
        for layer in self.shared_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))

        #         self.attention_W = nn.Parameter(torch.Tensor(
        #             self.num_hidden, self.num_attention))
        #         self.projection_h = nn.Parameter(torch.Tensor(
        #             self.num_attention, self.num_hidden))
        #         for tensor in [self.attention_W, self.projection_h]:
        #             nn.init.xavier_normal_(tensor, gain=nn.init.calculate_gain('relu'))

        #         self.attention_b = nn.Parameter(torch.Tensor(self.num_attention))
        #         for tensor in [self.attention_b]:
        #             nn.init.zeros_(tensor, )

        self.attention_net = nn.Sequential(
            nn.Linear(self.num_hidden, self.num_attention),
            nn.ReLU(),
            nn.Softmax(dim=1))
        assert self.num_hidden % self.num_head == 0 and self.num_attention % self.num_head==0
        self.multi_attention_net = clones(nn.Sequential(
            nn.Linear(self.num_hidden // self.num_head, self.num_attention // self.num_head),
            nn.ReLU(),
            nn.Linear(self.num_attention // self.num_head, self.num_hidden // self.num_head),
            nn.Softmax(dim=1)), self.num_head)
        for layer in self.attention_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
        for idx in range(self.num_head):
            for layer in self.multi_attention_net[idx]:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))

        # self.logic = nn.Sequential(
        #     nn.Linear(self.num_hidden, self.num_hidden // 2),
        #     nn.ReLU(),
        #     nn.Linear(self.num_hidden // 2, 1),
        #     # nn.Linear(self.num_hidden, 1)
        # )
        if self.expand:
            self.into_logic = self.num_hidden * 2
            self.num_hidden_2 = self.num_hidden_2 * 2
        else:
            self.into_logic = self.num_hidden

        self.logic = nn.Sequential(
             nn.Linear(self.into_logic, self.num_hidden_2),
             nn.ReLU(),
             nn.Linear(self.num_hidden_2, self.num_hidden_2//2),
             nn.ReLU(),
             nn.Linear(self.num_hidden_2 // 2, 1),
             #nn.Linear(self.num_hidden, 1)
        )
        for layer in self.logic:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):

        # x=self.shared(x)
        #         attention_temp = F.relu(torch.tensordot(
        #             x, self.attention_W, dims=([-1], [0])) + self.attention_b)
        #         attention_temp =torch.tensordot(
        #             x, self.attention_W, dims=([-1], [0])) + self.attention_b
        #         normalized_att_score = F.softmax(torch.tensordot(
        #             attention_temp, self.projection_h, dims=([-1], [0])), dim=1)
        #         normalized_att_score = F.softmax(
        #             attention_temp,dim=1)
        #         normalized_att_score = self.attention_net(x)
        if self.shared:
            x = self.shared_layer(x)

        if self.attention:
            list1 = [layer(x.view(x.shape[0], self.num_head, self.num_hidden // self.num_head)[:, idx, :]) \
                     for idx, layer in enumerate(self.multi_attention_net)]
            normalized_att_score = torch.concat(list1, dim=1)
            selected = x * normalized_att_score
            if self.expand:
                x = torch.concat([x, selected], dim=1)
            elif not self.expand:
                x = selected

        x = self.logic(x)
        if self.attention:
            return x, torch.mean(normalized_att_score, dim=0)
        else:
            return x


class MHA_layer(nn.Module):
    def __init__(self, num_hidden, num_attention, num_classes=2, num_head=20,expand=True):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_attention = num_attention
        self.num_classes = num_classes
        self.num_head = num_head
        self.expand=expand
        self.attention_net = nn.Sequential(
            nn.Linear(self.num_hidden, self.num_attention),
            nn.ReLU(),
            nn.Softmax(dim=1))
        self.multi_attention_net = clones(nn.Sequential(
            nn.Linear(self.num_hidden // self.num_head, self.num_attention // self.num_head),
            nn.ReLU(),
            nn.Softmax(dim=1)), self.num_head)
        for layer in self.attention_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
        for idx in range(self.num_head):
            for layer in self.multi_attention_net[idx]:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        list1 = [layer(x.view(x.shape[0], self.num_head, self.num_hidden // self.num_head)[:, idx, :]) \
                 for idx, layer in enumerate(self.multi_attention_net)]
        normalized_att_score = torch.concat(list1, dim=1)
        selected = x * normalized_att_score
        if self.expand:
            x = torch.concat([x, selected], dim=1)
        elif not self.expand:
            x = selected
        return x, torch.mean(normalized_att_score, dim=0)


class DNN_layer(nn.Module):

    def __init__(self, num_hidden, num_hidden_2, num_classes=2):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.num_hidden_2=num_hidden_2
        self.logic = nn.Sequential(
            nn.Linear(self.num_hidden, self.num_hidden_2),
            nn.ReLU(),
            nn.Linear(self.num_hidden_2, self.num_hidden_2//2),
            nn.ReLU(),
            nn.Linear(self.num_hidden_2 // 2, self.num_classes),
            # nn.Linear(self.num_hidden, 1)
        )
        for layer in self.logic:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = self.logic(x)
        return x


class AML_model2(nn.Module):

    def __init__(self, num_feature, num_hidden, num_attention, num_hidden_2, num_classes=2, num_head=20,
                 shared=False, attention=True,expand=False,fix_mode_part=False):

        super().__init__()
        self.num_feature = num_feature
        self.num_hidden = num_hidden
        self.num_attention = num_attention
        self.num_classes = num_classes
        self.num_head = num_head
        self.num_hidden_2 = num_hidden_2
        self.shared=shared
        self.attention=attention
        self.expand=expand
        self.fix_mode_part=fix_mode_part

        self.shared_layer = nn.Sequential(
            nn.Linear(self.num_feature, self.num_hidden),
            nn.ReLU(),
        )
        for layer in self.shared_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))

        if self.expand:
            self.into_logic = self.num_hidden * 2
            self.num_hidden_2 = self.num_hidden_2 * 2
        else:
            self.into_logic = self.num_hidden

        self.A_d = MHA_layer(self.num_hidden, self.num_attention, num_classes=2, num_head=self.num_head,
                             expand=self.expand)

        self.Y_d = DNN_layer(self.into_logic, self.num_hidden_2, self.num_classes)
        if self.fix_mode_part:
            for p in self.parameters():
                p.requires_grad = False

        self.A_u = MHA_layer(self.num_hidden, self.num_attention, num_classes=2, num_head=self.num_head,
                             expand=self.expand)
        self.A_s = MHA_layer(self.num_hidden, self.num_attention, num_classes=2, num_head=self.num_head,
                             expand=self.expand)

        self.Y_u = DNN_layer(self.into_logic, self.num_hidden_2, 1)
        self.Y_s = DNN_layer(self.into_logic, self.num_hidden_2, 1)

    def forward(self, x):
        if self.shared:
            x = self.shared_layer(x)

        if self.attention:

            D, w_d = self.A_d(x)
            D = self.Y_d(D)
            log_mode = F.log_softmax(D,dim=1)
            mode = torch.exp(log_mode)
            R_u, w_u = self.A_u(x)
            R_u = self.Y_u(R_u)
            R_s, w_s = self.A_s(x)
            R_s = self.Y_s(R_s)
            y_pred = torch.sum(mode * torch.concat([R_u, R_s], dim=1), dim=1)

            return y_pred, log_mode, [w_d, w_u, w_s]

        else:
            D = self.Y_d(x)
            log_mode = F.log_softmax(D, dim=1)
            mode = torch.exp(log_mode)
            R_u = self.Y_u(x)
            R_s = self.Y_s(x)
            y_pred = torch.sum(mode * torch.concat([R_u, R_s], dim=1), dim=1)

            return y_pred, log_mode


