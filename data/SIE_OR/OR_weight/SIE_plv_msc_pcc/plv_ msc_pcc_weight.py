import scipy.io as scio
import numpy as np
correlation_sum = 0
for i in range(48):
    num_nodes = 10+i*10
    print('num_nodes', num_nodes)
    for subject in range(1,10):
        # path_plv = "/home/tang/work/WTZ/BCI/GCN-eeg/plv/0.5-40_576_value_" + str(subject) + ".mat"
        path_plv = "/home/tang/work/WTZ/BCI/GCN-eeg/source_Graph/OR_Weight/SIE_plv/" + str(num_nodes) + "SIE_plv_0.5-40_576_value_" + str(subject) + ".mat"
        plv_arr = scio.loadmat(path_plv)["data"]
        # path_msc = "/home/tang/work/WTZ/BCI/GCN-eeg/MSC/0.5-40_576_value_"+ str(subject) +".mat"
        path_msc = "/home/tang/work/WTZ/BCI/GCN-eeg/source_Graph/OR_Weight/SIE_msc/" + str(num_nodes) + "SIE_msc_0.5-40_576_value_" + str(subject) + ".mat"
        msc_arr = scio.loadmat(path_msc)["data"]
        # path_pearson = "/home/tang/work/WTZ/BCI/GCN-eeg/pearson/0.5-40_576_value_" + str(subject) + ".mat"
        path_pearson = "/home/tang/work/WTZ/BCI/GCN-eeg/source_Graph/OR_Weight/SIE_pcc/" + str(num_nodes) + "SIE_pcc_0.5-40_576_value_" + str(subject) + ".mat"
        pearson_arr = scio.loadmat(path_pearson)["data"]
        # print('plv_arr', plv_arr)
        # print('msc_arr', msc_arr)
        plv_msc_pea_arr = plv_arr + msc_arr + pearson_arr
        # print('plv_msc_arr', plv_msc_pea_arr)
        plv_msc_pea_arr = plv_msc_pea_arr/3
        # print('2:plv_msc_arr', plv_msc_pea_arr)
        # print('sum', sum(sum(plv_msc_pea_arr)))
        correlation_sum += sum(sum(plv_msc_pea_arr))
        # correlation_sum += sum(sum(pearson_arr))

        # # 保存

        filename = "/home/tang/work/WTZ/BCI/GCN-eeg/source_Graph/OR_Weight/SIE_plv_msc_pcc/" + str(num_nodes) + "SIE_plv_msc_pcc_0.5-40_576_value_" + str(subject) + ".mat"
        scio.savemat(filename,  {'data': plv_msc_pea_arr})
    print('correlation_sum', correlation_sum)
    correlation_sum = 0