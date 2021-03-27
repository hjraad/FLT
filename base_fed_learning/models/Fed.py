#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
from collections import OrderedDict
import tqdm

def FedAvg(net_local_list, clustering_matrix, dict_users):
    # w_avg = [] # copy.deepcopy(w)
    # for idx in range(len(net_local_list)):
    #     w_avg.append(OrderedDict())
    #     for k in net_local_list[idx].state_dict().keys():
    #         w_avg[idx][k]=0
            
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
