#----------------------------------------------------------------------------
# Created By  : Mohammad Abdizadeh & Hadi Jamali-Rad
# Created Date: 23-Nov-2020
# 
# Refactored By: Sayak Mukherjee
# Last Update: 30-Nov-2023
# ---------------------------------------------------------------------------
# File contains the dataset samplers.
# ---------------------------------------------------------------------------

import numpy as np
import scipy.optimize

def func(x, y, C, K, F):
    return np.sum(F + np.exp(C*x*(y**2)), axis=0) - K

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
    num_shards, num_imgs = num_users, int(len(dataset)/num_users)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()[idxs]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def mnist_noniid_cluster(dataset, num_users, cluster):
    """
    Author: Mohammad Abdizadeh
    Sample clustered non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    cluster_size = cluster.shape[0]
    num_shards, num_imgs = num_users, int(len(dataset)/num_users)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()[idxs]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    
    indices_array = np.zeros((10,10000), dtype='int64') - 1
    for i in range(len(idxs_labels.T)):
        k = np.where(indices_array[ idxs_labels[1][i] ] == -1)[0][0]
        indices_array[ idxs_labels[1][i] ][k] = idxs_labels[0][i]

    nr_in_clusters = num_users // cluster_size

    users = np.arange(0,num_users,1)
    # np.random.shuffle(users)
    # print(users)
    # for i in range(cluster_size):
        # print(sorted(users[i*nr_in_clusters:(i+1)*nr_in_clusters]))
    for i, user in enumerate(users):
        cluster_index = min((i//nr_in_clusters), cluster_size-1)
        class_index_range = np.where(cluster[cluster_index] != -1)[0]
        for j in class_index_range:
            k = np.where(indices_array[ cluster[cluster_index][j] ] == -1)[0][0]
            # TODO: pick one of these methods: random vs shard based
            #rand_set = np.random.choice(k-1, int(num_imgs/len(class_index_range)), replace=False)
            index = (i % nr_in_clusters) * int(num_imgs/len(class_index_range))
            rand_set = np.arange(index, index + int(num_imgs/len(class_index_range)))
            dict_users[user] = np.concatenate((dict_users[user], indices_array[ cluster[cluster_index][j] ][rand_set]), axis=0)
    
    return dict_users

def emnist_noniid_cluster(dataset, num_users, cluster, 
                          random_shuffle=False, 
                          sequential_sampling=False):
    '''
    Author: Hadi Jamali-Rad 
    Sampling method for non-IID federated-type power-law based EMNIST dataset. 
    Parameters: 
        dataset: dataset (torchvision)
        num_users: number of clients/users
        cluster: np.arrary with cluster label ID's
        random_shuffle: randomly shuflles images in one cluster
        sequential_sampling: reorders images in one cluster sequentially based on label ID
    returns:
        dict_users: dictionary containing the indicies of samples per user/client
    '''
    nr_clusters = cluster.shape[0]
    nr_classes = len(np.unique(cluster.flatten()))
    nr_samples = len(dataset)
    nr_clients = num_users
    
    dict_users = {i: np.array([], dtype='int64') for i in range(nr_clients)}
    idxs = np.arange(nr_samples)
    labels = dataset.train_labels.numpy()
    
    # reshape the data indices according to lables in cluster
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    reshaped_idx_labels = np.empty((2,0))
    for class_id in cluster.flatten()[0:47]:
        reshaped_idx_labels = np.concatenate((reshaped_idx_labels, idxs_labels[:,np.where(idxs_labels[1,:] == class_id)[0]]), axis=1)
    reshaped_idx_labels = reshaped_idx_labels.astype('int')
    
    # 124-127: Assign 
    if nr_classes % nr_clusters > 0: # uneven clusters
        (nr_classes_a, nr_classes_b) = nr_classes // (nr_clusters- 1), nr_classes % (nr_clusters - 1)
    else:
        (nr_classes_a, nr_classes_b) = nr_classes // (nr_clusters), nr_classes % (nr_clusters)
    
    (len_a, len_b) = np.floor(nr_samples/nr_classes * nr_classes_a), np.floor(nr_samples/nr_classes*nr_classes_b)
    
    border_indices = [0]
    if nr_classes % nr_clusters > 0:
        for ind_cluster in range(nr_clusters - 1):
            border_indices.append(int(len_a*(ind_cluster + 1)))  
    border_indices.append(nr_samples) 
    
    if random_shuffle == True: # users from that seq. randomly pick from different labels
        reshaped_idx_labels = np.transpose(reshaped_idx_labels)
        for ind in range(len(border_indices) - 1):
            np.random.shuffle(reshaped_idx_labels[border_indices[ind]:border_indices[ind + 1], :])
        reshaped_idx_labels = np.transpose(reshaped_idx_labels)
    
    CC = 1
    nr_clients_per_cluster = nr_clients // nr_clusters
    
    KK = int(len_a)
    FF = np.floor(len_a/2/nr_clients_per_cluster) # starting point for the first user, it doesn't overflow
    yy = np.arange(1, nr_clients_per_cluster + 1)
    x = scipy.optimize.fsolve(func, x0=0, args=(yy[:, None], CC, KK, FF))
    li_a = [int(np.floor(FF + np.exp(CC*x*(i**2)))) for i in yy]
    arr_a = np.array(li_a)
    arr_a[0] += KK - np.sum(arr_a)
    if nr_classes % nr_clusters > 0:
        arr_a = np.tile(arr_a, nr_clusters- 1)
    else: 
        arr_a = np.tile(arr_a, nr_clusters)
    
    if nr_classes % nr_clusters > 0: 
        KK = int(len_b)
        FF = np.floor(len_b/2/nr_clients_per_cluster)
        yy = np.arange(1, nr_clients_per_cluster + 1)
        x = scipy.optimize.fsolve(func, x0=0, args=(yy[:, None], CC, KK, FF))
        li_b = [int(np.floor(FF + np.exp(CC*x*(i**2)))) for i in yy]
        arr_b = np.array(li_b)
        arr_b[0] += KK - np.sum(arr_b)
    else:
        arr_b = np.empty(0)
    
    final_arr = np.concatenate((arr_a, arr_b))
    
    prev_ind = 0
    for ii in range(nr_clients): # creating the output dict of users: user -> [idx]
        dict_users[ii] = reshaped_idx_labels[0, prev_ind: prev_ind + final_arr[ii]]
        prev_ind += final_arr[ii]
        
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

def cifar_noniid_cluster(dataset, num_users, cluster):
    """
    Author: Hadi Jamali-Rad and Mohammad Abdizadeh
    Sample clustered non-I.I.D client data from CIFAR10 dataset
    Parameters:
        dataset
        num_users
        cluster: cluster formation np.array
    
    :param num_users:
    Returns:
        dic_users: dictionary of user data sample indices
    """
    cluster_size = cluster.shape[0]
    num_shards, num_imgs = num_users, int(len(dataset)/num_users)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)[idxs]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    
    indices_array = np.zeros((10,10000), dtype='int64') - 1
    for i in range(len(idxs_labels.T)):
        k = np.where(indices_array[ idxs_labels[1][i] ] == -1)[0][0]
        indices_array[ idxs_labels[1][i] ][k] = idxs_labels[0][i]

    nr_in_clusters = num_users // cluster_size

    for i in range(num_users):
        cluster_index = (i//nr_in_clusters)
        class_index_range = np.where(cluster[cluster_index] != -1)[0]
        for j in class_index_range:
            # TODO: pick one of these methods: random vs shard based
            #k = np.where(indices_array[ cluster[cluster_index][j] ] == -1)[0][0]
            #rand_set = np.random.choice(k-1, int(num_imgs/len(class_index_range)), replace=False)
            index = (i % nr_in_clusters) * int(num_imgs/len(class_index_range))
            rand_set = np.arange(index, index + int(num_imgs/len(class_index_range)))
            dict_users[i] = np.concatenate((dict_users[i], indices_array[ cluster[cluster_index][j] ][rand_set]), axis=0)
    
    return dict_users

def cluster_testdata_dict(dataset, dataset_type, num_users, cluster):
    """
    By: Mohammad Abdizadeh
    Sample clustered non-I.I.D client data from MNIST dataset
    Parameters:
        dataset: target dataset
        dataset_type 
        num_users
        cluster: cluster 2D array
    Returns:
        dict_users: user data sample index dictionary
    """
    cluster_size = cluster.shape[0]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    if dataset_type in ['cifar', 'CIFAR10', 'cinic10', 'CINIC10']:
        labels = np.array(dataset.targets)
    else:
        labels = dataset.train_labels.numpy()

    nr_in_clusters = num_users // cluster_size

    for i in range(num_users):
        cluster_index = (i//nr_in_clusters)
        for k in range(len(labels)):
            if labels[k] in cluster[cluster_index]:
                dict_users[i] = np.concatenate((dict_users[i], np.array([k])), axis=0)
    
    return dict_users
