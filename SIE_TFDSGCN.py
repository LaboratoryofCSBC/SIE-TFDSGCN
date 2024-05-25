import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import scipy.io as scio
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import scale



SIE = True
channel = 22
Dropout_rate = 0.5
intput_size = (10, channel, 1000)

k1, k2, k3 = (1, 40), (1, 70), (1, 86)
k1_padding, k2_padding, k3_padding = (0, 0), (0, 15), (0, 23)
cls = 4

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")

path_folder = "./data/gdf_4S 0.5-40"


def select_person(i):
    path_plv_msc_pea = "./data/gdf_adj and weight/PLV_MSC_PCC_weight/0.5-40_576_value_" + str(i) + ".mat"
    value_arr = scio.loadmat(path_plv_msc_pea)["data"]
    path_adj = "./data/gdf_adj and weight/adj_AND/0.5-40_576_AND_top_indices160_adj_" + str(i) + ".mat"
    adj_arr = scio.loadmat(path_adj)["data"]

    path_plv_msc_pea = "./data/SIE_AND/AND_weight/SIE_plv_msc_pcc/160SIE_plv_msc_pcc_0.5-40_576_value_" + str(i) + ".mat"
    graph_value_arr = scio.loadmat(path_plv_msc_pea)["data"]

    path_i = path_folder + "/A0" + str(i) + ".mat"
    matdata_i = scio.loadmat(path_i)
    features = matdata_i["data"].reshape(576, -1)
    features = scale(features, axis=0).reshape(576, 22, -1)

    G_i = EEG_Graph(features, value_arr, adj_arr)
    graph_adj = np.ones((22, 22))
    np.fill_diagonal(graph_adj, 0)
    G_i_SIE = EEG_Graph_SIE(features, graph_value_arr, graph_adj)

    label = matdata_i["label"]
    features = torch.as_tensor(torch.from_numpy(features), dtype=torch.float32)
    train_feature, test_feature = features[0:288], features[288:]
    label = torch.as_tensor(torch.from_numpy(label), dtype=torch.float32).argmax(dim=1)
    train_label, test_label = label[0:288], label[288:]
    train_dataset = DataLoader(TensorDataset(train_feature, train_label), shuffle=True, batch_size=64)
    test_dataset = DataLoader(TensorDataset(test_feature, test_label), shuffle=True, batch_size=288)

    adj_L = torch.as_tensor(torch.from_numpy(G_i.adj3), dtype=torch.float32)
    adj_G = torch.as_tensor(torch.from_numpy(G_i_SIE.adj3), dtype=torch.float32)

    return train_dataset, test_dataset, G_i, adj_L, adj_G

class EEG_Graph():
    def __init__(self, data, value_arr, adj):
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
        degree = np.array(self.adj3.sum(1))
        degree = np.diag(np.power(degree, -1))
        self.adj3 = np.eye(self.adj.shape[0]) - np.dot(degree, self.adj3)

        for i in range(22):
            for j in range(22):
                if math.isnan(self.adj3[i][j]):
                    for k in range(22):
                        self.adj3[i][k] = 0
                    self.adj3[i][i] = 1

class EEG_Graph_SIE():
    def __init__(self, data, value_arr, adj):

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
        degree = np.array(self.adj3.sum(1))
        degree = np.diag(np.power(degree, -1))
        self.adj3 = np.eye(self.adj.shape[0]) - np.dot(degree, self.adj3)

        for i in range(22):
            for j in range(22):
                if math.isnan(self.adj3[i][j]):
                    for k in range(22):
                        self.adj3[i][k] = 0
                    self.adj3[i][i] = 1

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))  # 参数矩阵：
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)-self.bias  # 矩阵相乘：input * weight
        output = F.relu(torch.matmul(adj, support))

        return output

class SIE_TFDSGCN(nn.Module):

    def __init__(self, adj_L, adj_G):
        super(SIE_TFDSGCN, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=k1, stride=(1, 2), padding=k1_padding)
        self.conv1_2 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=k2, stride=(1, 2), padding=k2_padding)
        self.conv1_3 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=k3, stride=(1, 2), padding=k3_padding)
        self.batchNorm = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=60, kernel_size=(1, 15), stride=(1, 1))
        self.batchNorm2 = nn.BatchNorm2d(60)
        self.pooling = nn.MaxPool2d((1, 10))

        self.GCN = GraphConvolution(2040, 32)
        self.bn_ = nn.BatchNorm1d(22)

        self.liner1 = nn.Linear(704, cls)
        self.dropout = nn.Dropout(Dropout_rate)

        self.adj_L = adj_L.to(DEVICE)
        self.adj_G = nn.Parameter(adj_G, requires_grad=True)

    def forward(self, x):
        ''''''
        adj = self.adj_L
        if SIE == True:
            x = torch.einsum("jk,ikl->ijl", (adj, x))
        x = x.unsqueeze(1)

        x1 = F.elu(self.batchNorm(self.conv1_1(x)))
        x2 = F.elu(self.batchNorm(self.conv1_2(x)))
        x3 = F.elu(self.batchNorm(self.conv1_3(x)))
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.pooling(x)
        x = self.dropout(x)

        x = F.elu(self.batchNorm2(self.conv2(x)))
        x = self.dropout(x)

        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (x.size(0), x.size(1), -1))

        adj = self.adj_G
        x = self.GCN(x, adj)
        x = self.bn_(x)

        x = x.view(x.size()[0], -1)
        x = self.liner1(x)

        return x

if __name__ == '__main__':
    SIE = True
    x = torch.randn(intput_size).cuda()  # This is just an example, not an actual sample input.

    person = 1  # subject
    train_dataset, test_dataset, G_i, adj_L, adj_G = select_person(person)

    m = SIE_TFDSGCN(adj_L=adj_L, adj_G=adj_G).to(DEVICE)
    print(m(x))

    from torchsummary import summary
    summary(m, (channel, 1000))