#----------------------------------------------------------------------------
# Created By  : Mohammad Abdizadeh & Hadi Jamali-Rad
# Created Date: 23-Nov-2020
# 
# Refactored By: Sayak Mukherjee
# Last Update: 30-Nov-2023
# ---------------------------------------------------------------------------
# File contains the code for clustering clients.
# ---------------------------------------------------------------------------

import os
import pickle
import umap
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

import time

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from sympy.utilities.iterables import multiset_permutations

from models import get_model
from datasets.load_dataset import load_dataset
from datasets.utils import DatasetSplit
from optim.flt_pretrain import FLTPretrain

logger = logging.getLogger(__name__)

def min_matching_distance(center_0, center_1):
    if len(center_0) < len(center_1):
        center_small = center_0
        center_big = center_1
    else:
        center_small = center_1
        center_big = center_0

    distance = np.inf
    if len(center_small) > 0:
        s = set(range(len(center_big)))
        print(s)
        for p in tqdm(multiset_permutations(s)):
            summation = 0

            for i in range(len(center_small)):
                summation = summation + (np.linalg.norm(center_small[i] - center_big[p][i])**2)

            dist = np.sqrt(summation)/len(center_small)
            if dist < distance:
                distance = dist
    
    return distance

def get_extractor(config, device):

    pretrained_dataset_name = config.dataset.pre_trained_dataset

    if pretrained_dataset_name in ['CIFAR10', 'CIFAR100', 'CIFAR20']:
        model_args = {
                        'num_hiddens': config.model.num_hiddens,
                        'num_residual_layers': config.model.num_residual_layers, 
                        'num_residual_hiddens': config.model.num_residual_hiddens,
                        'latent_size': config.model.latent_dim,
                        }
        
    else:
        model_args = {
            'latent_size': config.model.latent_dim,
            }

    model_name = config.model.extractor_backbone

    logger.info(f'Using extractor {model_name}')

    extractor = get_model(model_name)(model_args)

    import_path = Path(config.project.path).joinpath('flt_artifacts').joinpath(model_name + '_' + pretrained_dataset_name + '.tar')
    if not Path.exists(import_path):
        logger.info(f'Extractor not found! Pre-training extractor.')
        trainer = FLTPretrain(config, extractor, model_name, pretrained_dataset_name, device)
        extractor = trainer.train()

    else:
        logger.info(f'Loading pre-trained extractor.')
        extractor_dict = extractor.state_dict()
        loaded_dict = torch.load(import_path)
        extractor_dict.update(loaded_dict)
        extractor.load_state_dict(extractor_dict)

    if config.trainer.finetune_epochs > 0:
        trainer = FLTPretrain(config, extractor, model_name, config.dataset.name, device)
        extractor = trainer.finetune()

    return extractor

def manifold_approximation_umap(config, use_AE, device):

    # check if manifold approximation is needed
    if use_AE and config.model.manifold_dim == config.model.latent_dim:
        raise AssertionError("We don't need manifold learning, AE dim = 2 !")
    
    pretrained_dataset_name = config.pre_trained_dataset.name.upper()
    
    dataset, _, _= load_dataset(pretrained_dataset_name, 
                                config.dataset.path, 
                                dataset_split=config.dataset.dataset_split)
    
    import_path = Path(config.project.path).joinpath('flt_artifacts') # .joinpath('umap_reducer_' + pretrained_dataset_name + '.p')
    if use_AE: 
        ae_model = get_extractor(config, device)
        ae_model = ae_model.to(device)

        trainloader = DataLoader(dataset['train'], batch_size = config.dataset.train_batch_size)
        embeddings = []
        for batch_idx, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            _, x_comp = ae_model(images)
            embeddings.append(x_comp.cpu().detach().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        
        logger.info('Using AE for E2E encoding ...')
        umap_data = embeddings
        umap_model_address = import_path.joinpath('umap_reducer_' + pretrained_dataset_name + '_' + config.model.extractor_backbone + '.p')
    else:
        data_list = [data[0] for data in dataset['train']]
        data_tensor = torch.cat(data_list, dim=0)
        data_2D_np = torch.reshape(data_tensor, (data_tensor.shape[0], -1)).numpy()

        logger.info('AE not used in this scenario ...')
        umap_data = data_2D_np
        umap_model_address = import_path.joinpath('umap_reducer_' + pretrained_dataset_name + '.p')
        

    logger.info('Training UMAP on AE embedding ...')
    umap_reducer = umap.UMAP(n_components=config.model.manifold_dim, random_state=config.project.seed)
    _ = umap_reducer.fit_transform(umap_data)
    pickle.dump(umap_reducer, open(umap_model_address, 'wb'))

    return umap_reducer

def clustering_single(num_users):
    clustering_matrix = np.ones((num_users, num_users))
                
    return clustering_matrix

def clustering_seperate(num_users):
    clustering_matrix = np.eye(num_users)
                
    return clustering_matrix

def clustering_perfect(config, dict_users, dataset_train, cluster):

    idxs_users = np.arange(config.federated.num_users)
    ar_label = np.zeros((config.federated.num_users, config.dataset.num_classes))-1

    for idx in idxs_users:
        local_train = DataLoader(DatasetSplit(dataset_train, dict_users[idx]), 
                                 batch_size=config.dataset.train_batch_size, 
                                 shuffle=True)
        label_matrix = np.empty(0, dtype=int)
        for _, (_, labels) in enumerate(local_train):
            label_matrix = np.concatenate((label_matrix, labels.numpy()), axis=0)
        label_matrix = np.unique(label_matrix)
        ar_label[idx][0:len(label_matrix)] = label_matrix
    
    clustering_matrix = np.zeros((config.federated.num_users, config.federated.num_users))
    for idx in idxs_users:
        for idx0 in idxs_users:
            if np.all(ar_label[idx0][0:len(cluster.T)] == ar_label[idx][0:len(cluster.T)]):
                clustering_matrix[idx][idx0] = 1
                
    return clustering_matrix

def clustering_umap(config, dict_users, dataset_train, device):

    use_AE = False #TODO: Remove hard-coding

    pretrained_dataset_name = config.dataset.pre_trained_dataset
    import_path = Path(config.project.path).joinpath('flt_artifacts')
    
    if use_AE:
        umap_model_address = import_path.joinpath('umap_reducer_' + pretrained_dataset_name + '_' + config.model.extractor_backbone + '.p')
    else:
        umap_model_address = import_path.joinpath('umap_reducer_' + pretrained_dataset_name + '.p')
    
    if not Path.exists(umap_model_address):
        logger.info(f'Reducer not found! Creating reducer.')
        reducer = manifold_approximation_umap(config, use_AE, device)

    else:
        logger.info(f'Loading pre-trained extractor.')
        with open(umap_model_address, 'rb') as fp:
            reducer = pickle.load(fp)

    idxs_users = np.arange(config.federated.num_users)
    
    input_dim = dataset_train[0][0].shape[-1]
    channel_dim = dataset_train[0][0].shape[0]
    
    centers = np.zeros((config.federated.num_users, 2, 2))
    for idx in tqdm(idxs_users, desc='Clustering progress'):
        images_matrix = np.empty((0, channel_dim*input_dim*input_dim))
        local_train = DataLoader(DatasetSplit(dataset_train, dict_users[idx]), 
                                 batch_size=config.dataset.train_batch_size, 
                                 shuffle=True)
        #TODO: concatenate the matrices
        for batch_idx, (images, labels) in enumerate(local_train):
            # if batch_idx == 3:# TODO: abalation test
            #     break
            ne = images.numpy().flatten().T.reshape((len(labels), channel_dim*input_dim*input_dim))
            images_matrix = np.vstack((images_matrix, ne))
        embedding1 = reducer.transform(images_matrix)
        X = list(embedding1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(X))
        centers[idx,:,:] = kmeans.cluster_centers_
    
    clustering_matrix_soft = np.zeros((config.federated.num_users, config.federated.num_users))
    clustering_matrix = np.zeros((config.federated.num_users, config.federated.num_users))

    for idx0 in tqdm(idxs_users, desc='Clustering matrix generation'):
        for idx1 in idxs_users:
            c0 = centers[idx0]
            c1 = centers[idx1]

            distance = min_matching_distance(c0, c1)

            clustering_matrix_soft[idx0][idx1] = distance
        
            if distance < 1:
                clustering_matrix[idx0][idx1] = 1
            else:
                clustering_matrix[idx0][idx1] = 0

    return clustering_matrix, clustering_matrix_soft, centers

def clustering_encoder(config, dict_users, dataset_train, ae_model, device):

    idxs_users = np.arange(config.federated.num_users.num_users)

    centers = np.zeros((config.federated.num_users.num_users, 2, 2))
    embedding_matrix = np.zeros((len(dict_users[0])*config.federated.num_users.num_users, 2))
    for user_id in tqdm(idxs_users, desc='Custering in progress ...'):
        local_train = DataLoader(DatasetSplit(dataset_train, dict_users[user_id]), 
                                 batch_size=config.dataset.train_batch_size, 
                                 shuffle=True)

        embeddings = []
        for batch_idx, (images, labels) in enumerate(local_train):
            images = images.to(device)
            _, x_comp = ae_model(images)
            embeddings.append(x_comp.cpu().detach().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        
        # ----------------------------------
        # use Kmeans to cluster the data into 2 clusters
        X = list(embeddings)
        embedding_matrix[user_id*len(dict_users[0]): len(dict_users[0])*(user_id + 1),:] = embeddings
        kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(X))
        centers[user_id,:,:] = kmeans.cluster_centers_
    
    clustering_matrix_soft = np.zeros((config.federated.num_users, config.federated.num_users))
    clustering_matrix = np.zeros((config.federated.num_users, config.federated.num_users))

    for idx0 in idxs_users:
        for idx1 in idxs_users:
            c0 = centers[idx0]
            c1 = centers[idx1]
        
            distance = min_matching_distance(c0, c1)
            
            clustering_matrix_soft[idx0][idx1] = distance
        
            if distance < 1:
                clustering_matrix[idx0][idx1] = 1
            else:
                clustering_matrix[idx0][idx1] = 0

    return clustering_matrix, clustering_matrix_soft, centers, embedding_matrix

def clustering_pca_kmeans(config, dict_users, cluster, dataset_train):
    idxs_users = np.random.choice(config.federated.num_users, config.federated.num_users, replace=False)
    
    centers = np.empty((0, config.model.latent_dim), dtype=int)
    center_dict = {}
    embedding_matrix = np.zeros((len(dict_users[0])*config.federated.num_users, config.model.latent_dim))
    
    for user_id in tqdm(idxs_users, desc='Clustering in progress ...'):
        user_dataset_train = DatasetSplit(dataset_train, dict_users[user_id])
        
        user_data_np = np.squeeze(np.array([item[0].view(1, -1).numpy() for item in user_dataset_train]))
        if config.model.latent_dim > len(user_dataset_train):
            user_data_np = np.repeat(user_data_np, np.ceil(config.model.latent_dim/len(user_dataset_train)),axis=0) 
        pca = PCA(n_components=config.model.latent_dim)
        embedding = pca.fit_transform(user_data_np)
        
        kmeans = KMeans(n_clusters=5, random_state=43).fit(embedding)
        centers = np.vstack((centers, kmeans.cluster_centers_))
        
        center_dict[user_id] = kmeans.cluster_centers_
    
    clustering_matrix_soft = np.zeros((config.federated.num_users, config.federated.num_users))
    clustering_matrix = np.zeros((config.federated.num_users, config.federated.num_users))

    c_dict = center_dict

    for idx0 in tqdm(idxs_users, desc='Creating clustering matrix'):
        c0 = c_dict[idx0]
        for idx1 in idxs_users:
            c0 = c_dict[idx0]
            c1 = c_dict[idx1]
        
            distance = min_matching_distance(c0, c1)
            
            clustering_matrix_soft[idx0][idx1] = distance
        
            if distance < 1.2:
                clustering_matrix[idx0][idx1] = 1
            else:
                clustering_matrix[idx0][idx1] = 0
                
    return clustering_matrix, clustering_matrix_soft, centers, c_dict

def clustering_umap_central(config, dict_users, cluster, dataset_train, ae_model, device):

    # idxs_users = np.random.shuffle(np.arange(num_users))
    idxs_users = np.random.choice(config.federated.num_users, config.federated.num_users, replace=False)
    
    #centers = np.zeros((num_users, max_num_center, 128)) # AE latent size going to be hyperparamter
    centers = np.empty((0, config.model.latent_dim), dtype=int)
    center_dict = {}
    embedding_matrix = np.zeros((len(dict_users[0])*config.federated.num_users, config.model.latent_dim))
    
    for user_id in tqdm(idxs_users, desc='Clustering in progress ...'):
        local_train = DataLoader(DatasetSplit(dataset_train, dict_users[user_id]), 
                            batch_size=config.dataset.train_batch_size, 
                            shuffle=True)

        embeddings = []
        for batch_idx, (images, labels) in enumerate(local_train):
            images = images.to(device)
            _, x_comp = ae_model(images)
            embeddings.append(x_comp.cpu().detach().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        
        # ----------------------------------
        # use Kmeans to cluster the data into 2 clusters
        #embedding_matrix[user_id*len(dict_users[0]): len(dict_users[0])*(user_id + 1),:] = embedding
        if config.dataset.name == 'FEMNIST':
            num_center = config.federated.nr_of_embedding_clusters
        else:
            cluster_size = cluster.shape[0]
            nr_in_clusters = config.federated.num_users // cluster_size
            cluster_index = (user_id//nr_in_clusters)
            class_index_range = np.where(cluster[cluster_index] != -1)[0]
            num_center = len(class_index_range)

        kmeans = KMeans(n_clusters=num_center, random_state=43).fit(embeddings)
        centers = np.vstack((centers, kmeans.cluster_centers_))
        
        center_dict[user_id] = kmeans.cluster_centers_
    
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    umap_reducer.fit(np.reshape(centers, (-1, config.model.latent_dim)))
    
    clustering_matrix_soft = np.zeros((config.federated.num_users, config.federated.num_users))
    clustering_matrix = np.zeros((config.federated.num_users, config.federated.num_users))

    c_dict = {}
    for idx in idxs_users:
        c_dict[idx] = umap_reducer.transform(center_dict[idx])

    for idx0 in tqdm(idxs_users):
        c0 = c_dict[idx0]
        for idx1 in idxs_users:
            c0 = c_dict[idx0]
            c1 = c_dict[idx1]
        
            distance = min_matching_distance(c0, c1)
            
            clustering_matrix_soft[idx0][idx1] = distance
        
            if distance < 1:
                clustering_matrix[idx0][idx1] = 1
            else:
                clustering_matrix[idx0][idx1] = 0
                
    return clustering_matrix, clustering_matrix_soft, centers, embedding_matrix, c_dict

def extract_clustering(config, dict_users, dataset_train, cluster, iter, device):

    logger.info(f'Extracting clusters')

    export_path = Path(config.project.path + '/scenario' + str(config.federated.scenario)).joinpath(config.project.experiment_name)
    export_path = export_path.joinpath('plots')
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True)

    logger.info(f'Clustering method: {config.federated.clustering_method}')

    if config.federated.clustering_method == 'single':
        clustering_matrix = clustering_single(config.federated.num_users)
        
    elif config.federated.clustering_method == 'local':
        clustering_matrix = clustering_seperate(config.federated.num_users)

    elif config.federated.clustering_method == 'perfect':
        clustering_matrix = clustering_perfect(config, dict_users, dataset_train, cluster)

        fig_path = export_path.joinpath(f'clust_perfect_nr_users-{config.federated.num_users}_nr_clusters_{config.federated.nr_of_embedding_clusters}_ep_{config.trainer.rounds}_itr_{iter}.png')
        plt.figure()
        plt.matshow(clustering_matrix)
        plt.savefig(fig_path)
        plt.close()

    elif config.federated.clustering_method == 'umap':
        clustering_matrix, _, _ = clustering_umap(config, dict_users, dataset_train, device)

    elif config.federated.clustering_method == 'encoder':
        ae_model = get_extractor(config, device)
        ae_model = ae_model.to(device)

        clustering_matrix, _, _, _ =\
            clustering_encoder(config, dict_users, dataset_train, ae_model, device)

    elif config.federated.clustering_method == 'umap_central':
        ae_model = get_extractor(config, device)
        ae_model = ae_model.to(device)
        clustering_matrix, clustering_matrix_soft, _, _, _ =\
                clustering_umap_central(config, dict_users, cluster, dataset_train, ae_model, device)

        fig_path = export_path.joinpath(f'clust_umapcentral_nr_users-{config.federated.num_users}_nr_clusters_{config.federated.nr_of_embedding_clusters}_ep_{config.trainer.rounds}_itr_{iter}.png')
        plt.figure()
        plt.matshow(clustering_matrix,origin='lower')
        plt.savefig(fig_path)
        plt.close()

    elif config.federated.clustering_method == 'kmeans':
        clustering_matrix, clustering_matrix_soft, _, _ =\
                clustering_pca_kmeans(config, dict_users, cluster, dataset_train)
        
        fig_path = export_path.joinpath(f'clust_umapcentral_nr_users-{config.federated.num_users}_nr_clusters_{config.federated.nr_of_embedding_clusters}_ep_{config.trainer.rounds}_itr_{iter}.png')
        plt.figure()
        plt.matshow(clustering_matrix,origin='lower')
        plt.savefig(fig_path)
        plt.close()
        
    return clustering_matrix

def partition_clusters(config, clustering_matrix, metric='euclidean', plotting=False):
    """
    Creates nr_clusters clusters of users based on the clustering_matrix (adjacency matrix).

    Arguments:
        config (namespace): general arguments
        clustering_matrix (ndarray): adjacency matrix

    Returns:
        cluster_user_dict (dict(int,list)): cluster ids mapped to user ids

    By: Attila Szabo
    """
    export_path = Path(config.project.path + '/scenario' + str(config.federated.scenario)).joinpath(config.project.experiment_name)
    export_path = export_path.joinpath('plots')
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True)

    # clustering with linkage
    method = config.federated.partition_method
    nr_clusters = config.federated.nr_of_partition_clusters
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
    # gives back linkage matrix after hierarchical clustering
    Y = sch.linkage(clustering_matrix, 
                    method=method,
                    metric=metric)
    # creates dendogram for plotting and flattening
    Z = sch.dendrogram(Y, orientation='left')
    ax1.set_xticks([])
    ax1.set_yticks([])
    # calculate cluster membership
    # fcluster flattens out dendograms to the specified nr_clusters
    cluster_memberships = sch.fcluster(Y, t=nr_clusters, criterion='maxclust') # ith element in this array is the cluster for i
    idx = np.array(Z['leaves']) # idx ordered in cluster
    
    ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
    Z2 = sch.dendrogram(Y)
    ax2.set_xticks([])
    ax2.set_yticks([])

    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])

    clustering_matrix = clustering_matrix[idx,:]
    clustering_matrix = clustering_matrix[:,idx]
    im = axmatrix.matshow(clustering_matrix, aspect='auto', origin='lower', cmap=plt.cm.YlGnBu)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
    plt.colorbar(im, cax=axcolor)
    if plotting:
        fig_path = export_path.joinpath(f'clust_{config.federated.clustering_method}_nr_users-{config.federated.num_users}_nr_of_partition_clusters_{nr_clusters}_method_{method}_reconstructed.png')
        fig.savefig(fig_path)

    # Plot filtered
    canvas = np.zeros_like(clustering_matrix)
    for i in range(1,nr_clusters+1):
        mask = np.ones_like(clustering_matrix)
        mask[cluster_memberships[idx]!=i,:] = 0
        mask[:,cluster_memberships[idx]!=i] = 0
        canvas+=clustering_matrix*mask
    fig = plt.figure()
    plt.matshow(canvas,origin='lower')
    if plotting:
        fig_path = export_path.joinpath(f'clust_{config.federated.clustering_method}_nr_users-{config.federated.num_users}_nr_of_partition_clusters_{nr_clusters}_method_{method}_filtered.png')
        fig.savefig(fig_path)

    d_error = np.sum(clustering_matrix-canvas)
    logger.info(f'Decompostion error: {d_error}, {d_error/np.sum(clustering_matrix)}')

    # build cluster id to client id user dict
    cluster_user_dict = { i : idx[cluster_memberships==i] for i in range(1,nr_clusters+1)}

    # Test overlaps within clusters
    collected = []
    for i, cluster_members_a in cluster_user_dict.items():
        for j, cluster_members_b in cluster_user_dict.items():
            assert np.all(cluster_members_a != cluster_members_b) or set(cluster_members_a).intersection(set(cluster_members_b)) != {}, f'clusters {i} and {j} are not disjoint'
        collected.extend(cluster_members_a)
    assert np.all(np.arange(0,len(clustering_matrix),1) == np.sort(np.array(collected)))

    return cluster_user_dict

def clustering_multi_center(config, net_local_list, multi_center_initialization_flag, 
                            est_multi_center, iter):
    
    def get_model_params_length(model):

        lst = [list(model[k].cpu().numpy().flatten()) for  k in model.keys()]
        flat_list = [item for sublist in lst for item in sublist]

        return len(flat_list)
    
    export_path = Path(config.project.path + '/scenario' + str(config.federated.scenario)).joinpath(config.project.experiment_name)
    export_path = export_path.joinpath('plots')
    if not Path.exists(export_path):
        Path.mkdir(export_path, exist_ok=True)

    num_users = config.federated.num_users
    model_params_length = get_model_params_length(net_local_list[0].state_dict())
    models_parameter_list = np.zeros((num_users, model_params_length))

    for i in range(num_users):
        model = net_local_list[i].state_dict()
        lst = [list(model[k].cpu().numpy().flatten()) for  k in model.keys()]
        flat_list = [item for sublist in lst for item in sublist]

        models_parameter_list[i] = np.array(flat_list).reshape(1,model_params_length)

    if multi_center_initialization_flag:                
        kmeans = KMeans(n_clusters=config.federated.nr_of_embedding_clusters, n_init=20).fit(models_parameter_list)

    else:
        kmeans = KMeans(n_clusters=config.federated.nr_of_embedding_clusters, init=est_multi_center, n_init=1).fit(models_parameter_list)#TODO: remove the best
    
    ind_center = kmeans.fit_predict(models_parameter_list)

    est_multi_center_new = kmeans.cluster_centers_  
    clustering_matrix = np.zeros((num_users, num_users))

    for ii in range(len(ind_center)):
        ind_inter_cluster = np.where(ind_center == ind_center[ii])[0]
        clustering_matrix[ii,ind_inter_cluster] = 1

    plt.figure()
    fig_path = export_path.joinpath(f'clust_multicenter_nr_users-{config.federated.num_users}_nr_clusters_{config.federated.nr_of_embedding_clusters}_ep_{config.trainer.rounds}_itr_{iter}.png')
    plt.imshow(clustering_matrix)
    plt.savefig(fig_path)
    plt.close()

    return clustering_matrix, est_multi_center_new

def filter_cluster_partition(cluster_user_dict, net_local_list):
    """
    Creates cluster_dict structure for FedAvg, containing ie. weights
    
    Arguments:
        cluster_user_dict (dict(int,list)): cluster ids mapped to user ids
        net_local_list (list): list of weights
    
    Returns:
        cluster_dict (dict(int,tuple)): dictionary containing weights, 
            adjacency matrix and members for a cluster

    By: Attila Szabo
    """
    cluster_dict = defaultdict(tuple)

    for i, cluster_members in cluster_user_dict.items():
        cluster_dict[i] = (net_local_list[cluster_members], 
                            np.ones((len(cluster_members), len(cluster_members))),
                            cluster_members)
    return cluster_dict