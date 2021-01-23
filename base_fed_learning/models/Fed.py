#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn

def FedAvg(w, clustering_matrix, dict_users):
    w_avg = copy.deepcopy(w)
    for idx in range(len(w)):
        for k in w_avg[idx].keys():
            w_avg[idx][k] = 0*w_avg[idx][k]

            weight_sum = 0
            for i in range(0, len(w)):
                # weighted averaging allowing soft thresholding
                w_avg[idx][k] += w[i][k]*len(dict_users[i])*clustering_matrix[idx][i]
                # w_avg[idx][k] += w[i][k]*len(dict_users[i])
                weight_sum += len(dict_users[i])*clustering_matrix[idx][i]

            w_avg[idx][k] = torch.div(w_avg[idx][k], weight_sum)
    
    return w_avg
