#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    #indices = [idxs_labels[i] for i, x in enumerate(idxs_labels) if x == 0]
    indices_array = np.zeros((10,10000), dtype='int64') - 1
    for i in range(len(idxs_labels.T)):
        #k=indices_array[ idxs_labels[1][i] ].index(-1)
        k = np.where(indices_array[ idxs_labels[1][i] ] == -1)[0][0]
        indices_array[ idxs_labels[1][i] ][k] = idxs_labels[0][i]
        
    # indices0 =[]
    # indices1 =[]
    # indices2 =[]
    # indices3 =[]
    # indices4 =[]
    # indices5 =[]
    # indices6 =[]
    # indices7 =[]
    # indices8 =[]
    # indices9 =[]
    # for i in range(len(idxs_labels.T)):
    #     if idxs_labels[1][i] == 0:
    #         indices0 = index0 + idxs_labels[0][i]
    #     if idxs_labels[1][i] == 1:
    #         indices1 = index1 + idxs_labels[0][i]
    #     if idxs_labels[1][i] == 2:
    #         indices2 = index2 + idxs_labels[0][i]
    #     if idxs_labels[1][i] == 3:
    #         indices3 = index3 + idxs_labels[0][i]
    #     if idxs_labels[1][i] == 4:
    #         indices4 = index4 + idxs_labels[0][i]
    #     if idxs_labels[1][i] == 5:
    #         indices5 = index5 + idxs_labels[0][i]
    #     if idxs_labels[1][i] == 6:
    #         indices6 = index6 + idxs_labels[0][i]
    #     if idxs_labels[1][i] == 7:
    #         indices7 = index7 + idxs_labels[0][i]
    #     if idxs_labels[1][i] == 8:
    #         indices8 = index8 + idxs_labels[0][i]
    #     if idxs_labels[1][i] == 9:
    #         indices9 = index9 + idxs_labels[0][i]
    cluster_num = 5
    cluster_length = num_users // cluster_num
    cluster = np.zeros((cluster_num,2), dtype='int64')
    for i in range(cluster_num):
        cluster[i] = np.random.choice(10, 2, replace=False)
            
    for i in range(num_users):
        cluster_index = (i//cluster_length)
        k = np.where(indices_array[ cluster[cluster_index][0] ] == -1)[0][0]
        rand_set = set(np.random.choice(k-1, num_imgs, replace=False))
        dict_users[i] = np.concatenate((dict_users[i], indices_array[ cluster[cluster_index][0] ][list(rand_set)]), axis=0)
                                       
        k = np.where(indices_array[ cluster[cluster_index][1] ] == -1)[0][0]
        rand_set = set(np.random.choice(k-1, num_imgs, replace=False))
        dict_users[i] = np.concatenate((dict_users[i], indices_array[ cluster[cluster_index][1] ][list(rand_set)]), axis=0)
    
    # divide and assign
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, 2, replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def mnist_noniid_cluster(dataset, num_users, cluster, cluster_num):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    #indices = [idxs_labels[i] for i, x in enumerate(idxs_labels) if x == 0]
    indices_array = np.zeros((10,10000), dtype='int64') - 1
    for i in range(len(idxs_labels.T)):
        #k=indices_array[ idxs_labels[1][i] ].index(-1)
        k = np.where(indices_array[ idxs_labels[1][i] ] == -1)[0][0]
        indices_array[ idxs_labels[1][i] ][k] = idxs_labels[0][i]
        
    # indices0 =[]
    # indices1 =[]
    # indices2 =[]
    # indices3 =[]
    # indices4 =[]
    # indices5 =[]
    # indices6 =[]
    # indices7 =[]
    # indices8 =[]
    # indices9 =[]
    # for i in range(len(idxs_labels.T)):
    #     if idxs_labels[1][i] == 0:
    #         indices0 = index0 + idxs_labels[0][i]
    #     if idxs_labels[1][i] == 1:
    #         indices1 = index1 + idxs_labels[0][i]
    #     if idxs_labels[1][i] == 2:
    #         indices2 = index2 + idxs_labels[0][i]
    #     if idxs_labels[1][i] == 3:
    #         indices3 = index3 + idxs_labels[0][i]
    #     if idxs_labels[1][i] == 4:
    #         indices4 = index4 + idxs_labels[0][i]
    #     if idxs_labels[1][i] == 5:
    #         indices5 = index5 + idxs_labels[0][i]
    #     if idxs_labels[1][i] == 6:
    #         indices6 = index6 + idxs_labels[0][i]
    #     if idxs_labels[1][i] == 7:
    #         indices7 = index7 + idxs_labels[0][i]
    #     if idxs_labels[1][i] == 8:
    #         indices8 = index8 + idxs_labels[0][i]
    #     if idxs_labels[1][i] == 9:
    #         indices9 = index9 + idxs_labels[0][i]

    cluster_length = num_users // cluster_num

    for i in range(num_users):
        cluster_index = (i//cluster_length)
        k = np.where(indices_array[ cluster[cluster_index][0] ] == -1)[0][0]
        rand_set = set(np.random.choice(k-1, num_imgs, replace=False))
        dict_users[i] = np.concatenate((dict_users[i], indices_array[ cluster[cluster_index][0] ][list(rand_set)]), axis=0)
                                       
        k = np.where(indices_array[ cluster[cluster_index][1] ] == -1)[0][0]
        rand_set = set(np.random.choice(k-1, num_imgs, replace=False))
        dict_users[i] = np.concatenate((dict_users[i], indices_array[ cluster[cluster_index][1] ][list(rand_set)]), axis=0)
    
    # divide and assign
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, 2, replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
