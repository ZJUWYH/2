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

class CFG:
    seed=42
    input_dim=3
    input_feature=18
    num_workers=2
    hidden_size=64
    batch_size=2048
    num_classes=1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer="Adam"
    scheduler="ExponentialLR"
    lr= 1e-1
    epoches=300
    kernel=True
    print_training_process=True
    sc_Gamma=0.999#指数型学习率衰减曲线
    decay = 1e-5
    num_in_feature_classes = 3
    warmup_prop=0.1
    jump_out_value=64

    ## autoencoder
    ae_batch_size=512
    ae_input_layer=6
    ae_hidden_layer=8
    ae_lr=0.001
    max_len=525

    ## ts
    ts_input_dim=30
    ts_hidden_dim=64



#seed_everything(CFG.seed)