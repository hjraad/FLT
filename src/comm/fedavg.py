#----------------------------------------------------------------------------
# Created By  : Mohammad Abdizadeh & Hadi Jamali-Rad
# Created Date: 23-Nov-2020
# 
# Refactored By: Sayak Mukherjee
# Last Update: 27-Oct-2023
# ---------------------------------------------------------------------------
# File contains the code for federated averaging.
# ---------------------------------------------------------------------------

import torch
import tqdm

from collections import OrderedDict

def FedAvg(net_local_list, clustering_matrix, dict_users):
            
    for idx in tqdm.tqdm(range(len(net_local_list))):
        one_w_avg = OrderedDict()
        w_local = net_local_list[idx].state_dict()
        for k in w_local.keys():
            one_w_avg[k]=torch.zeros_like(w_local[k])
        
        for k in w_local.keys():
            # w_avg[idx][k] = 0#*w_avg[idx][k]

            sum_len = 0
            for i in range(0, len(net_local_list)):
                if clustering_matrix[idx][i] == 1:
                    ith = net_local_list[i]
                    one_w_avg[k] += ith.state_dict()[k]*len(dict_users[i])
                    # net_local_list[i] = ith
                    sum_len += len(dict_users[i])

            one_w_avg[k] = torch.div(one_w_avg[k], sum_len)
        net_local_list[idx].load_state_dict(one_w_avg)
    return net_local_list