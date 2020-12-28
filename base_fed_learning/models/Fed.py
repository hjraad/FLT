#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn

def FedAvg(w, clustering_matrix):

    w_avg = copy.deepcopy(w)
    for idx in range(len(w)):
        for k in w_avg[idx].keys():
            w_avg[idx][k] = 0*w_avg[idx][k]
            counter = 0
            for i in range(0, len(w)):
                if clustering_matrix[idx][i] == 1:
                    w_avg[idx][k] += w[i][k]
                    counter = counter + 1

            w_avg[idx][k] = torch.div(w_avg[idx][k], counter)
    
    return w_avg
