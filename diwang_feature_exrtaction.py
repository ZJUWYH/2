import pandas
from matplotlib import pyplot
import numpy as np
from numpy import *
import math
import random
import datetime

DATA_PATH = "./Data_FD003/preprocessed data/"

# Read normalized data after log transformation
Dataframe = pandas.read_csv(DATA_PATH + 'TD_data.csv', header=None)  # training data
Data = Dataframe.values
# colnames = ['T24','T30','T50','P30','Nf','Nc','Ps30','phi','NRf','NRc', \
# 'BPR','htBleed','W31','W32']
selected = np.array([0, 1, 2, 3, 4, 7, 8])  # ,11,13,14])
Data = Data[:, selected]


# colnames = ['T24','T30','T50','P30','Ps30','phi']#, 'BPR','W31','W32']


# Feature extraction for training data
def psi(t):
    # Tal0 = 500
    t1 = t  # /Tal0
    return (np.mat([1, t1, t1 * t1]))


def PSI(tal, p):
    Psi = mat(ones((tal, p)))
    for it in range(0, tal):
        Psi[it, :] = psi(it)
    return (Psi)


def dev_psi(t):
    # Tal0 = 500
    t1 = t  # /Tal0
    return (np.mat([0, 1, 2 * t1]))


def dev_PSI(tal, p):
    Psi = mat(ones((tal, p)))
    for it in range(0, tal):
        Psi[it, :] = dev_psi(it)
    return (Psi)


def dev2_psi(t):
    # Tal0 = 500
    t1 = t  # /Tal0
    return (np.mat([0, 0, 2]))


def dev2_PSI(tal, p):
    Psi = mat(ones((tal, p)))
    for it in range(0, tal):
        Psi[it, :] = dev2_psi(it)
    return (Psi)


# Original Training units
Nu = 100;  # number of units
PHI = mat(zeros((Data.shape[0], 4)))
dev_PHI = mat(zeros((Data.shape[0], 4)))
dev2_PHI = mat(zeros((Data.shape[0], 4)))
for iu in range(0, Nu):
    num_unit = np.argwhere(Data[:, 0] == iu + 1)
    tal = num_unit.shape[0]  # tal: failure time
    PHI[num_unit[:, 0], 0] = iu + 1
    PHI[num_unit[:, 0], 1:] = PSI(tal, 3)  # degradation basis for all 100 training units
    dev_PHI[num_unit[:, 0], 0] = iu + 1
    dev_PHI[num_unit[:, 0], 1:] = dev_PSI(tal, 3)
    dev2_PHI[num_unit[:, 0], 0] = iu + 1
    dev2_PHI[num_unit[:, 0], 1:] = dev2_PSI(tal, 3)

# 小数定标规范法
# Normalization
alpha_G = 0.1
alpha_dev_G = 10
alpha_dev2_G = 1000

Num = int(max(Data[:, 0]))  # number of units
Tal0 = np.zeros((Num, 1))  # failure time of units
for iu in range(0, Num):
    num_unit = np.argwhere(Data[:, 0] == iu + 1)
    Data_unit = Data[num_unit[:, 0], 1:]
    Tal0[iu] = Data_unit.shape[0]

Ntw0 = int(max(Tal0))  # width of time window

# G
G = np.zeros((Nu, Ntw0, Data.shape[1] - 1))
dev_G = np.zeros((Nu, Ntw0, Data.shape[1] - 1))
dev2_G = np.zeros((Nu, Ntw0, Data.shape[1] - 1))
Gamma = np.zeros((Nu, Data.shape[1] - 1, 3))
for iu in range(0, Nu):
    num_unit = np.argwhere(Data[:, 0] == iu + 1)
    Data_unit = Data[num_unit[:, 0], 1:]
    phi = PHI[num_unit[:, 0], 1:]
    dev_phi = dev_PHI[num_unit[:, 0], 1:]
    dev2_phi = dev2_PHI[num_unit[:, 0], 1:]
    tal = int(Tal0[iu])
    # pyplot.figure()
    for i_sensor in range(0, Data.shape[1] - 1):
        sensor_unit = np.mat(Data_unit[:, i_sensor]).T
        Gamma_unit = (phi.T * phi).I * phi.T * sensor_unit
        g_unit = np.array(phi * Gamma_unit)
        dev_g_unit = np.array(dev_phi * Gamma_unit)
        dev2_g_unit = np.array(dev2_phi * Gamma_unit)
        G[iu, :tal, i_sensor] = g_unit[:, 0] * alpha_G
        dev_G[iu, :tal, i_sensor] = dev_g_unit[:, 0] * alpha_dev_G
        dev2_G[iu, :tal, i_sensor] = dev2_g_unit[:, 0] * alpha_dev2_G
        Gamma[iu, i_sensor, :] = reshape(Gamma_unit, 3)
        # pyplot.subplot(2,5,i_sensor+1)
        # pyplot.plot(sensor_unit,'.')
        # pyplot.plot(g_unit)
        # pyplot.plot(dev_g_unit)
        # pyplot.plot(dev2_g_unit)
        # pyplot.title(colnames[i_sensor], loc='right')
    # pyplot.show()

# Construct new training units that fail at some time
TestDataframe = pandas.read_csv(DATA_PATH + 'Test_data.csv', header=None)  # training data
TestData = TestDataframe.values
TestData = TestData[:, selected]

K = TestData.shape[1] - 1  # number of sensor types
Tal = np.zeros((Nu, 1))  # failure time of units
for iu in range(0, Nu):
    num_unit = np.argwhere(TestData[:, 0] == iu + 1)
    TestData_unit = TestData[num_unit[:, 0], 1:]
    Tal[iu] = TestData_unit.shape[0]

Ntw = int(max(Tal)) + 1  # width of time window 475

total_testdata = np.zeros((Nu, Ntw, K))
for iu in range(0, Nu):
    num_unit = np.argwhere(TestData[:, 0] == iu + 1)
    TestData_unit = TestData[num_unit[:, 0], 1:]
    total_testdata[iu, -int(Tal[iu]):, :] = TestData_unit

Num = total_testdata.shape[0]  # number of units
# rG
rG = np.zeros((total_testdata.shape[0], total_testdata.shape[2]))
dev_rG = np.zeros((total_testdata.shape[0], total_testdata.shape[2]))
dev2_rG = np.zeros((total_testdata.shape[0], total_testdata.shape[2]))
rGamma = np.zeros((Num, TestData.shape[1] - 1, 3))
sigma2 = np.zeros((Num, TestData.shape[1] - 1))  # sample variance
for iu in range(0, Num):
    iu
    tal = Ntw - np.max(np.argwhere(total_testdata[iu, :, 0] == 0)) - 1
    phi = PSI(tal, 3)
    dev_phi = dev_PSI(tal, 3)
    dev2_phi = dev2_PSI(tal, 3)
    Data_unit = total_testdata[iu, -tal:, :]
    for i_sensor in range(0, TestData.shape[1] - 1):
        sensor_unit = np.mat(Data_unit[:, i_sensor]).T
        Gamma_unit = (phi.T * phi).I * phi.T * sensor_unit
        g_unit = phi * Gamma_unit
        dev_g_unit = dev_phi * Gamma_unit
        dev2_g_unit = dev2_phi * Gamma_unit
        rGamma[iu, i_sensor, :] = reshape(Gamma_unit, 3)
        rG[iu, i_sensor] = g_unit[-1, 0] * alpha_G
        dev_rG[iu, i_sensor] = dev_g_unit[-1, 0] * alpha_dev_G
        dev2_rG[iu, i_sensor] = dev2_g_unit[-1, 0] * alpha_dev2_G
        sigma2[iu, i_sensor] = (sensor_unit - g_unit).T * (sensor_unit - g_unit) / (tal - 1)
        # pyplot.subplot(2,5,i_sensor+1)
        # pyplot.plot(sensor_unit,'.')
        # pyplot.plot(g_unit)
        # pyplot.plot(dev_g_unit)
        # pyplot.plot(dev2_g_unit)
        # pyplot.title(colnames[i_sensor], loc='right')
    # pyplot.show()

# rG update
Weight = np.zeros((Num, TestData.shape[1] - 1, Nu))
for iu in range(0, Num):
    iu
    tal = int(Tal[iu])
    phi = PSI(tal, 3)
    dev_phi = dev_PSI(tal, 3)
    dev2_phi = dev2_PSI(tal, 3)
    Data_unit = total_testdata[iu, -tal:, :]
    for i_sensor in range(0, TestData.shape[1] - 1):
        sensor_unit = np.mat(Data_unit[:, i_sensor]).T
        for io in range(0, Nu):
            Gamma_unit = np.mat(Gamma[io, i_sensor, :]).T
            g_unit = phi * Gamma_unit
            Weight[iu, i_sensor, io] = exp(
                -(sensor_unit - g_unit).T * (sensor_unit - g_unit) / (2 * sigma2[iu, i_sensor]))

W = Weight / np.tile(reshape(sum(Weight, 2), (Num, TestData.shape[1] - 1, 1)), (1, 1, Nu))

uG0 = np.zeros((total_testdata.shape[0], total_testdata.shape[2], Nu))
dev_uG0 = np.zeros((total_testdata.shape[0], total_testdata.shape[2], Nu))
dev2_uG0 = np.zeros((total_testdata.shape[0], total_testdata.shape[2], Nu))
for iu in range(0, Num):
    tal = int(Tal[iu])
    phi = PSI(tal, 3)
    dev_phi = dev_PSI(tal, 3)
    dev2_phi = dev2_PSI(tal, 3)
    Data_unit = total_testdata[iu, -tal:, :]
    for i_sensor in range(0, TestData.shape[1] - 1):
        sensor_unit = np.mat(Data_unit[:, i_sensor]).T
        for io in range(0, Nu):
            Gamma_unit = np.mat(Gamma[io, i_sensor, :]).T
            g_unit = phi * Gamma_unit
            dev_g_unit = dev_phi * Gamma_unit
            dev2_g_unit = dev2_phi * Gamma_unit
            weight = W[iu, i_sensor, io]
            uG0[iu, i_sensor, io] = g_unit[-1, 0] * weight
            dev_uG0[iu, i_sensor, io] = dev_g_unit[-1, 0] * weight
            dev2_uG0[iu, i_sensor, io] = dev2_g_unit[-1, 0] * weight

uG = sum(uG0, 2) * alpha_G
dev_uG = sum(dev_uG0, 2) * alpha_dev_G
dev2_uG = sum(dev2_uG0, 2) * alpha_dev2_G

Lambda = 0
tG = (Lambda * rG + (1 - Lambda) * uG)
dev_tG = (Lambda * dev_rG + (1 - Lambda) * dev_uG)
dev2_tG = (Lambda * dev2_rG + (1 - Lambda) * dev2_uG)

#RUL
path = DATA_PATH+'RUL.csv'
RUL_frame = pandas.read_csv(path, header=None)
RUL = RUL_frame.values[:, 0]

delete_index = [79, 95]
Tal_delete = np.delete(Tal, delete_index, axis=0).reshape(-1)
tG_delete = np.delete(tG, delete_index, axis=0)
dev_tG_delete = np.delete(dev_tG, delete_index, axis=0)
dev2_tG_delete = np.delete(dev2_tG, delete_index, axis=0)
RUL_delete = np.delete(RUL, delete_index)
Tal0=Tal0.reshape(-1)

