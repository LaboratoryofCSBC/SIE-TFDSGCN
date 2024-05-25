import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import scipy.io as scio
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from scipy.signal import welch, csd
from scipy.signal import hilbert

# 计算 PLV
def compute_plv(data):
    num_channels, num_samples = data.shape
    plv_matrix = np.zeros((num_channels, num_channels))

    for ch1 in range(num_channels):
        # for ch2 in range(ch1 + 1, num_channels):
        for ch2 in range(ch1, num_channels):
            phase_diff = np.angle(np.exp(1j * (data[ch1, :] - data[ch2, :])))
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_matrix[ch1, ch2] = plv
            plv_matrix[ch2, ch1] = plv  # 对称性

    return plv_matrix

def plv_mean(out):
    plv = compute_plv(out[0])
    for i in range(1, out.shape[0]):  # 576个样本
        plv += compute_plv(out[i])
    plv = plv / out.shape[0]
    return plv
# 定义函数来计算MSC
def compute_msc(channels):
    num_channels, num_samples = channels.shape
    msc_matrix = np.zeros((num_channels, num_channels))  # 22x22 的MSC矩阵

    for i in range(num_channels):
        for j in range(i, num_channels):
            if i == j:
                msc_matrix[i, j] = 1.0  # 对角线元素为1，因为信号与自身的MSC为1
            else:
                f, pxx_i = welch(channels[i, :], nperseg=256)   # 自动将时域信号转为频域
                f, pxx_j = welch(channels[j, :], nperseg=256)   # 自动将时域信号转为频域
                f, cxy = csd(channels[i, :], channels[j, :], nperseg=256)  # 自动将时域信号转为频域
                # print('pxx_i', pxx_i)
                # print('pxx_j', pxx_j)
                # print('cxy', abs(cxy)**2)
                msc = abs(cxy)**2 / (pxx_i * pxx_j)
                # print('msc', msc)
                msc_matrix[i, j] = np.mean(msc)
                msc_matrix[j, i] = np.mean(msc)  # 对称性质

    return msc_matrix


def msc_mean(out):
    msc = compute_msc(out[0])
    for i in range(1, out.shape[0]):  # 576个样本
        msc += compute_msc(out[i])
    msc = msc / out.shape[0]
    return msc
# pearson计算
def pearson_mean(out):
    pearson = np.corrcoef(out[0])
    for i in range(1, out.shape[0]):  # 576个样本
        pearson += np.corrcoef(out[i])
    pearson = pearson / out.shape[0]
    return abs(pearson)
    # return pearson

class EEG_Graph():
    def __init__(self, data, value_arr, adj):
        '''
        :param data: data.shape：[channel, batch_size*feature]
        '''
        self.number_of_nodes = data.shape[1]
        self.features = data.astype(np.float32)
        def get_edge(self):
            adj3 = np.zeros((self.number_of_nodes, self.number_of_nodes))
            for i in range(self.number_of_nodes):
                for j in range(self.number_of_nodes):
                    if(adj[i,j] == 1):
                        adj3[i, j] = value_arr[i][j]
            return adj, adj3

        self.adj, self.adj3 = get_edge(self)
        self.adj = self.adj + np.eye(self.adj.shape[0])
        degree = np.array(self.adj3.sum(1))  # 为每个结点计算度
        degree = np.diag(np.power(degree, -1))
        self.adj3 = np.eye(self.adj.shape[0]) - np.dot(degree, self.adj3)
        ''''''
        '''
        当没有邻接节点时
        '''
        for i in range(22):
            for j in range(22):
                if math.isnan(self.adj3[i][j]):
                    for k in range(22):
                        self.adj3[i][k] = 0
                    self.adj3[i][i] = 1

# subject = 1
for i in range(48):
    num_nodes = 10+i*10
    print('num_nodes', num_nodes)
    for subject in range(1,10):
        path_folder = "/home/tang/work/WTZ/BCI/Data/2a/gdf_4S 0.5-40未归一化"
        path_i = path_folder + "/A0" + str(subject) + ".mat"
        matdata_i = scio.loadmat(path_i)
        features = matdata_i["data"].reshape(576, -1)
        features = scale(features, axis=0).reshape(576, 22, -1)


        path_plv_msc_pea = "/home/tang/work/WTZ/BCI/Data/2a/adj and weight/PLV_MSC_PCC_weight/0.5-40_576_value_" + str(subject) + ".mat"
        value_arr = scio.loadmat(path_plv_msc_pea)["data"]
        path_adj = "/home/tang/work/WTZ/BCI/Data/2a/adj and weight/adj_OR/0.5-40_576_OR_top_indices" + str(num_nodes) + "_adj_" + str(subject) + ".mat"
        adj_arr = scio.loadmat(path_adj)["data"]

        G_i = EEG_Graph(features, value_arr, adj_arr)
        # adj = torch.as_tensor(torch.from_numpy(G_i.adj), dtype=torch.float32)
        adj3 = torch.as_tensor(torch.from_numpy(G_i.adj3), dtype=torch.float32)
        features = torch.as_tensor(torch.from_numpy(features), dtype=torch.float32)

        x = torch.einsum("jk,ikl->ijl", (adj3, features))  # [样本数，通道数，特征数]

        '''
        相关性矩阵                                
        '''
        eeg_data = np.angle(hilbert(x))
        plv_matrix = plv_mean(eeg_data)  # PLV
        # msc_matrix = msc_mean(x)         # MSC
        # pcc_matrix = pearson_mean(x)     # PCC
        # plv_msc_pcc_matrix = (plv_matrix+msc_matrix+pcc_matrix)/3  # PLV_MSC_PCC
        # print('plv_msc_pcc_matrix', plv_msc_pcc_matrix)

        filename = "/home/tang/work/WTZ/BCI/GCN-eeg/source_Graph/OR_Weight/SIE_plv/" + str(num_nodes) + "SIE_plv_0.5-40_576_value_" + str(subject) + ".mat"
        scio.savemat(filename,  {'data': plv_matrix})


