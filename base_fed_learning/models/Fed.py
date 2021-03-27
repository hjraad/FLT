#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
from collections import OrderedDict


def FedAvg(w, clustering_matrix, dict_users):
    w_avg = [] # copy.deepcopy(w)
    for idx in range(len(w)):
        w_avg.append(OrderedDict())
        for k in w[idx].keys():
            w_avg[idx][k]=0
            
    for idx in range(len(w)):
        for k in w_avg[idx].keys():
            # w_avg[idx][k] = 0#*w_avg[idx][k]

            sum_len = 0
            for i in range(0, len(w)):
                if clustering_matrix[idx][i] == 1:
                    w_avg[idx][k] += w[i][k].to('cpu')*len(dict_users[i])
                    sum_len += len(dict_users[i])

            w_avg[idx][k] = torch.div(w_avg[idx][k], sum_len)
    
    return w_avg
