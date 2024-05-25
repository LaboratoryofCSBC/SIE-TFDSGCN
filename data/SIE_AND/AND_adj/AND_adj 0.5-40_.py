import scipy.io as scio
import numpy as np

adj = np.zeros((22,22))
num = 0
# R = 0.8
# 创建一个(22, 22)的全零数组
# plv_result_array = np.zeros((22, 22))
# msc_result_array = np.zeros((22, 22))
# pearson_result_array = np.zeros((22, 22))
# top_indices = 480
for i in range(48):
    top_indices = 10+i*10
    print('top_indices', top_indices)
    for subject in range(1,10):
        path_plv = "/home/tang/work/WTZ/BCI/GCN-eeg/source_Graph/Weight/SIE_plv/" + str(top_indices) + "SIE_plv_0.5-40_576_value_" + str(subject) + ".mat"
        plv_arr = scio.loadmat(path_plv)["data"]
        path_msc = "/home/tang/work/WTZ/BCI/GCN-eeg/source_Graph/Weight/SIE_msc/" + str(top_indices) + "SIE_msc_0.5-40_576_value_" + str(subject) + ".mat"
        msc_arr = scio.loadmat(path_msc)["data"]
        path_pearson = "/home/tang/work/WTZ/BCI/GCN-eeg/source_Graph/Weight/SIE_pcc/" + str(top_indices) + "SIE_pcc_0.5-40_576_value_" + str(subject) + ".mat"
        pearson_arr = scio.loadmat(path_pearson)["data"]
        # print('plv_arr', plv_arr.shape)
        '''
        去除1
        '''
        for i in range(22):
            plv_arr[i][i] = 0
            msc_arr[i][i] = 0
            pearson_arr[i][i] = 0
        # print('plv_arr', plv_arr)
        # 找到最大的前100个元素的索引
        plv_top_indices = np.unravel_index(np.argpartition(plv_arr, -top_indices, axis=None)[-top_indices:], plv_arr.shape)
        msc_top_indices = np.unravel_index(np.argpartition(msc_arr, -top_indices, axis=None)[-top_indices:], msc_arr.shape)
        pearson_top_indices = np.unravel_index(np.argpartition(pearson_arr, -top_indices, axis=None)[-top_indices:], pearson_arr.shape)
        # print(plv_top_indices)
        # 每次都要清空！！！否则会叠加
        plv_result_array = np.zeros((22, 22))
        msc_result_array = np.zeros((22, 22))
        pearson_result_array = np.zeros((22, 22))
        # 将这100个元素对应的位置设置为1
        plv_result_array[plv_top_indices] = 1
        msc_result_array[msc_top_indices] = 1
        pearson_result_array[pearson_top_indices] = 1

        for i in range(22):
            for j in range(22):
                if((plv_result_array[i][j]==1 and msc_result_array[i][j]==1 and pearson_result_array[i][j]==1) and i != j):
                # if ((plv_result_array[i][j] == 1 ) and i != j):
                    adj[i][j] = 1
                    num += 1
                else:
                    adj[i][j] = 0
                if i == j:
                    adj[i][j] = 0
        # print('adj', adj)
        print('num', num)
        num = 0


        # # 保存

        # filename = str(top_indices) + "SIE_0.5-40_576_AND_top_indices" + str(top_indices) + "_adj_" + str(subject) + ".mat"
        # scio.savemat(filename,  {'data': adj})