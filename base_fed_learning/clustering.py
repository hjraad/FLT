"""
Created on Mon Nov 23 19:44:39 2020

@author: Mohammad Abdizadeh & Hadi Jamali-Rad
@email(s):{moh.abdizadeh, h.jamali.rad}@gmail.com
"""
import sys
sys.path.append("./../")
sys.path.append("./../../")
sys.path.append("./")

import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from utils.sampling import mnist_iid, mnist_noniid, mnist_noniid_cluster
from utils.sampling import cifar_iid, cifar_noniid_cluster, emnist_noniid_cluster
from utils.options import args_parser
from utils.utils import extract_model_name
from models.Update import LocalUpdate
import pickle
from sklearn.cluster import KMeans
import itertools
import copy
import umap
from collections import defaultdict

from tqdm import tqdm
import scipy
import scipy.cluster.hierarchy as sch

from manifold_approximation.models.convAE_128D import ConvAutoencoder
from manifold_approximation.models.convAE_cifar import ConvAutoencoderCIFAR
from manifold_approximation.models.convAE_cifar_residual import ConvAutoencoderCIFARResidual
from manifold_approximation.encoder import Encoder
from manifold_approximation.sequential_encoder import Sequential_Encoder
from manifold_approximation.utils.load_datasets import load_dataset

from sympy.utilities.iterables import multiset_permutations
# from torchsummary import summary
# ----------------------------------
# Reproducability
# ----------------------------------
torch.manual_seed(123)
np.random.seed(321)
umap_random_state=42

def encoder_model_capsul(args):
    '''
    encapsulates encoder model components
    '''    
    if args.target_dataset in ['CIFAR10', 'CIFAR100', 'CIFAR110']:
        # ae_model = ConvAutoencoderCIFAR(latent_size).to(args.device)
        args.num_hiddens = 128
        args.num_residual_hiddens = 32
        args.num_residual_layers = 2
        args.latent_dim = 64
        
        ae_model = ConvAutoencoderCIFARResidual(args.num_hiddens, args.num_residual_layers, 
                                                args.num_residual_hiddens, args.latent_dim).to(args.device)
        
        # loss
        criterion = nn.MSELoss()
        
    else:
        args.latent_dim = 128
        ae_model = ConvAutoencoder().to(args.device)
        # loss
        criterion = nn.BCELoss()
    print(ae_model)
    # summary(ae_model)
    

    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    ae_optimizer = optim.Adam(ae_model.parameters(), lr=0.001)

    # Decay LR by a factor of x*gamma every step_size epochs
    exp_lr_scheduler = lr_scheduler.StepLR(ae_optimizer, step_size=10, gamma=0.5)

    # Load the model ckpt
    checkpoint = torch.load(f'{args.model_root_dir}/{args.ae_model_name}_best.pt',  map_location=torch.device(args.device))
    ae_model.load_state_dict(checkpoint['model_state_dict']) 
    
    ae_model_dict = {
        'model':ae_model,
        'name': args.ae_model_name,
        'opt':ae_optimizer,
        'criterion':criterion,
        'scheduler':exp_lr_scheduler
    }
    return ae_model_dict
    

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
        for p in multiset_permutations(s):
            summation = 0

            for i in range(len(center_small)):
                summation = summation + (np.linalg.norm(center_small[i] - center_big[p][i])**2)

            dist = np.sqrt(summation)/len(center_small)
            if dist < distance:
                distance = dist
    
    return distance

def gen_data(iid, dataset_type, data_root_dir, transforms_dict, num_users, cluster, dataset_split=''):
    # load dataset 
    _, image_datasets, dataset_sizes, class_names =\
            load_dataset(dataset_type, data_root_dir, transforms_dict, batch_size=8, shuffle_flag=False, dataset_split=dataset_split)
    
    dataset_train = image_datasets['train']
    dataset_test = image_datasets['test']
    
    if dataset_type in ['mnist', 'MNIST']:
        # sample users
        if iid:
            dict_users = mnist_iid(dataset_train, num_users)
        else:
            dict_users = mnist_noniid_cluster(dataset_train, num_users, cluster)
    #
    elif dataset_type in ['emnist', 'EMNIST']:     
        if not iid:
            dict_users = emnist_noniid_cluster(dataset_train, num_users, cluster, 
                                               random_shuffle=True)
    #       
    elif dataset_type in ['cifar', 'CIFAR10']:
        if iid:
            dict_users = cifar_iid(dataset_train, num_users)
        else:
            dict_users = cifar_noniid_cluster(dataset_train, num_users, cluster)
    #
    else:
        exit('Error: unrecognized dataset')

    return dataset_train, dataset_test, dict_users

def clustering_single(num_users):
    clustering_matrix = np.ones((num_users, num_users))
                
    return clustering_matrix

def clustering_seperate(num_users):
    clustering_matrix = np.eye(num_users)
                
    return clustering_matrix

def clustering_perfect(num_users, dict_users, dataset_train, cluster, args):
    idxs_users = np.arange(num_users)
    ar_label = np.zeros((args.num_users, args.num_classes))-1
    for idx in idxs_users:
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
        label_matrix = np.empty(0, dtype=int)
        for batch_idx, (images, labels) in enumerate(local.ldr_train):
            label_matrix = np.concatenate((label_matrix, labels.numpy()), axis=0)
        label_matrix = np.unique(label_matrix)
        ar_label[idx][0:len(label_matrix)] = label_matrix
    
    clustering_matrix = np.zeros((num_users, num_users))
    for idx in idxs_users:
        for idx0 in idxs_users:
            if np.all(ar_label[idx0][0:len(cluster.T)] == ar_label[idx][0:len(cluster.T)]):
                clustering_matrix[idx][idx0] = 1
                
    return clustering_matrix

def clustering_umap(num_users, dict_users, dataset_train, args):
    reducer_loaded = pickle.load( open( f'{args.model_root_dir}/umap_reducer_EMNIST.p', "rb" ) )
    reducer = reducer_loaded

    idxs_users = np.arange(num_users)
    
    input_dim = dataset_train[0][0].shape[-1]
    channel_dim = dataset_train[0][0].shape[0]
    
    centers = np.zeros((num_users, 2, 2))
    for idx in tqdm(idxs_users, desc='Clustering progress'):
        images_matrix = np.empty((0, channel_dim*input_dim*input_dim))
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
        for batch_idx, (images, labels) in enumerate(local.ldr_train):#TODO: concatenate the matrices
            # if batch_idx == 3:# TODO: abalation test
            #     break
            ne = images.numpy().flatten().T.reshape((len(labels), channel_dim*input_dim*input_dim))
            images_matrix = np.vstack((images_matrix, ne))
        embedding1 = reducer.transform(images_matrix)
        X = list(embedding1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(X))
        centers[idx,:,:] = kmeans.cluster_centers_
    
    clustering_matrix_soft = np.zeros((num_users, num_users))
    clustering_matrix = np.zeros((num_users, num_users))

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

def clustering_encoder(dict_users, dataset_train, ae_model_dict, args):

    idxs_users = np.arange(args.num_users)

    centers = np.zeros((args.num_users, 2, 2))
    embedding_matrix = np.zeros((len(dict_users[0])*args.num_users, 2))
    for user_id in tqdm(idxs_users, desc='Custering in progress ...'):
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user_id])
        
        user_dataset_train = local.ldr_train.dataset
            
        encoder = Encoder(ae_model_dict['model'], ae_model_dict['name'], 
                          args.model_root_dir, args.manifold_dim, 
                          user_dataset_train, user_id, device=args.device)
         
        encoder.autoencoder()
        #encoder.manifold_approximation_umap()
        #reducer = encoder.umap_reducer
        # embedding1 = encoder.umap_embedding
        embedding1 = encoder.ae_embedding_np
        
        # ----------------------------------
        # use Kmeans to cluster the data into 2 clusters
        X = list(embedding1)
        embedding_matrix[user_id*len(dict_users[0]): len(dict_users[0])*(user_id + 1),:] = embedding1
        kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(X))
        centers[user_id,:,:] = kmeans.cluster_centers_
    
    clustering_matrix_soft = np.zeros((num_users, num_users))
    clustering_matrix = np.zeros((num_users, num_users))

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

def clustering_umap_central(dict_users, cluster, dataset_train, ae_model_dict, args):

    # idxs_users = np.random.shuffle(np.arange(num_users))
    idxs_users = np.random.choice(args.num_users, args.num_users, replace=False)
    
    #centers = np.zeros((num_users, max_num_center, 128)) # AE latent size going to be hyperparamter
    centers = np.empty((0, args.latent_dim), dtype=int)
    center_dict = {}
    embedding_matrix = np.zeros((len(dict_users[0])*args.num_users, args.latent_dim))
    
    for user_id in tqdm(idxs_users, desc='Clustering in progress ...'):
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user_id])
        
        user_dataset_train = local.ldr_train.dataset
            
        encoder = Sequential_Encoder(ae_model_dict['model'], ae_model_dict['opt'], 
                                     ae_model_dict['criterion'], ae_model_dict['scheduler'], 
                                     args.nr_epochs_sequential_training, ae_model_dict['name'],
                                     args.model_root_dir, args.log_root_dir, args.manifold_dim, user_dataset_train, 
                                     user_id, args.pre_trained_dataset, dataset_name=args.target_dataset, 
                                     train_umap=False, use_AE=True, device=args.device)
        encoder.autoencoder()
        # encoder.manifold_approximation_umap()
        embedding = encoder.ae_embedding_np 
        # reducer = encoder.umap_reducer
        # embedding = encoder.umap_embedding
        # ae_model_name = encoder.new_model_name
        
        # ----------------------------------
        # use Kmeans to cluster the data into 2 clusters
        #embedding_matrix[user_id*len(dict_users[0]): len(dict_users[0])*(user_id + 1),:] = embedding
        if args.target_dataset == 'FEMNIST':
            num_center = args.nr_of_embedding_clusters
        else:
            cluster_size = cluster.shape[0]
            nr_in_clusters = args.num_users // cluster_size
            cluster_index = (user_id//nr_in_clusters)
            class_index_range = np. where(cluster[cluster_index] != -1)[0]
            num_center = len(class_index_range)

        kmeans = KMeans(n_clusters=num_center, random_state=43).fit(embedding)
        centers = np.vstack((centers, kmeans.cluster_centers_))
        
        center_dict[user_id] = kmeans.cluster_centers_
    
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    umap_reducer.fit(np.reshape(centers, (-1, args.latent_dim)))
    
    clustering_matrix_soft = np.zeros((args.num_users, args.num_users))
    clustering_matrix = np.zeros((args.num_users, args.num_users))

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

def filter_cluster_partition(cluster_user_dict, net_local_list):
    cluster_dict = defaultdict(tuple)

    for i, cluster_members in cluster_user_dict.items():
        cluster_dict[i] = (net_local_list[cluster_members], 
                            np.ones((len(cluster_members), len(cluster_members))),
                            cluster_members)
    return cluster_dict

def partition_clusters(clustering_matrix, args, nr_clusters=5, method='complete', metric='euclidean'):
    # clustering
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
    Y = sch.linkage(clustering_matrix, method=method,metric=metric)
    Z = sch.dendrogram(Y, orientation='left')
    ax1.set_xticks([])
    ax1.set_yticks([])
    # calculate cluster membership
    cluster_memberships = sch.fcluster(Y, t=nr_clusters, criterion='maxclust') # ith element in this array is the cluster for i
    idx = np.array(Z['leaves']) # idx ordered in cluster
    
    ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
    Z2 = sch.dendrogram(Y)
    ax2.set_xticks([])
    ax2.set_yticks([])
    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])

    clustering_matrix = clustering_matrix[idx,:]
    clustering_matrix = clustering_matrix[:,idx]
    im = axmatrix.matshow(clustering_matrix, aspect='auto', origin='lower', cmap=plt.cm.YlGnBu)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
    plt.colorbar(im, cax=axcolor)
    fig.savefig(f'{args.results_root_dir}/clust_umapcentral_nr_users-{args.num_users}_nr_of_partition_clusters_{nr_clusters}_method_{method}_reconstructed.png')

    # Plot filtered
    canvas = np.zeros_like(clustering_matrix)
    for i in range(1,nr_clusters+1):
        mask = np.ones_like(clustering_matrix)
        mask[cluster_memberships[idx]!=i,:] = 0
        mask[:,cluster_memberships[idx]!=i] = 0
        canvas+=clustering_matrix*mask
    fig = plt.figure()
    plt.imshow(canvas)
    fig.savefig(f'{args.results_root_dir}/clust_umapcentral_nr_users-{args.num_users}_nr_of_partition_clusters_{nr_clusters}_method_{method}_filtered.png')

    d_error = np.sum(clustering_matrix-canvas)
    print(f'Decompostion error: {d_error}, {d_error/np.sum(clustering_matrix)}')
    exit()
    cluster_user_dict = { i : idx[cluster_memberships==i] for i in range(1,nr_clusters+1)}

    # Test
    collected = []
    for i, cluster_members_a in cluster_user_dict.items():
        for j, cluster_members_b in cluster_user_dict.items():
            assert np.all(cluster_members_a != cluster_members_b) or set(cluster_members_a).intersection(set(cluster_members_b)) != {}, f'clusters {i} and {j} are not disjoint'
        collected.extend(cluster_members_a)
    assert np.all(np.arange(0,len(clustering_matrix),1) == np.sort(np.array(collected)))

    return cluster_user_dict

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


    # args.num_users = 20
    # args.ae_model_name = "model-1607623811-epoch40-latent128"
    # args.pre_trained_dataset = 'FMNIST'

    args.num_users = 100
    args.num_classes = 10
    args.target_dataset = 'CIFAR10'
    args.dataset_split = 'balanced'
    
    if args.target_dataset in ['CIFAR10', 'CIFAR100', 'CIFAR110']:
        transforms_dict = {    
        'train': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'test': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        }
    else:  
        transforms_dict = {
            'train': transforms.Compose([transforms.ToTensor()]),
            'test': transforms.Compose([transforms.ToTensor()])
        }
    
    args.pre_trained_dataset = 'CIFAR100'
    
    # find the model name automatically
    args.ae_model_name = extract_model_name(args.model_root_dir, args.pre_trained_dataset)
    
    args.iid = False

    # ----------------------------------
    # generate cluster settings    
    args.nr_of_embedding_clusters = 5
    cluster_length = args.num_users // args.nr_of_embedding_clusters
    
    if args.target_dataset == 'EMNIST': 
        n_1 = 47 // (args.nr_of_embedding_clusters - 1)
        n_2 = 47 % n_1
        cluster = np.zeros((args.nr_of_embedding_clusters, n_1), dtype='int64')
        # cluster_array = np.random.choice(47, 47, replace=False)
        cluster_array = np.arange(47)
        for i in range(args.nr_of_embedding_clusters - 1):
            cluster[i] = cluster_array[i*n_1: i*n_1 + n_1]
        cluster[args.nr_of_embedding_clusters - 1][0:n_2] = cluster_array[-n_2:] 
        
    else:
        cluster = np.zeros((args.nr_of_embedding_clusters, 2), dtype='int64')
        cluster_array = np.random.choice(10, 10, replace=False)
        for i in range(args.nr_of_embedding_clusters):
            cluster[i] = cluster_array[i*2: i*2 + 2] 
        # for i in range(args.nr_of_embedding_clusters):
    #     cluster[i] = np.random.choice(10, 2, replace=False)
    
    # ---------------------------------- 
    ae_model_dict = encoder_model_capsul(args)
    args.nr_epochs_sequential_training = 5
    
    # ----------------------------------       
    args.clustering_method = 'umap_central'    # umap, encoder, sequential_encoder, umap_central
    
    # ----------------------------------
    # generate clustered data
    dataset_train, dataset_test, dict_users = gen_data(args.iid, args.target_dataset, args.data_root_dir, 
                                                       transforms_dict, args.num_users, cluster, dataset_split=args.dataset_split)
    
    # ----------------------------------    
    clustering_matrix = clustering_perfect(args.num_users, dict_users, dataset_train, cluster, args)
    
    if args.clustering_method == 'umap':
        clustering_matrix0, clustering_matrix0_soft, centers = clustering_umap(args.num_users, dict_users, dataset_train, args)
    #
    elif args.clustering_method == 'encoder':
        clustering_matrix0, clustering_matrix0_soft, centers, embedding_matrix =\
            clustering_encoder(dict_users, dataset_train, ae_model_dict, args)
    #
    elif args.clustering_method == 'umap_central':
        clustering_matrix0, clustering_matrix0_soft, centers, embedding_matrix, c_dict =\
            clustering_umap_central(dict_users, cluster, dataset_train, ae_model_dict, args)
    
    # ----------------------------------    
    # plot results
    plt.close('all')
    # ----------------------------------
    plt.figure(1)
    plt.imshow(clustering_matrix,cmap=plt.cm.viridis)
    plt.savefig(f'{args.results_root_dir}/Clustering/clustMat_perfect_nrclust_nrusers-{args.num_users}-{args.nr_of_embedding_clusters}_from-{args.pre_trained_dataset}_to-{args.target_dataset}.jpg')
    
    plt.figure(2)
    plt.imshow(clustering_matrix0,cmap=plt.cm.viridis)
    plt.savefig(f'{args.results_root_dir}/Clustering/clustMat_{args.clustering_method}_nrusers-{args.num_users}_nrclust-{args.nr_of_embedding_clusters}_from-{args.pre_trained_dataset}_to-{args.target_dataset}.jpg')
    
    plt.figure(3)
    plt.imshow(-1*clustering_matrix0_soft,cmap=plt.cm.viridis)
    plt.savefig(f'{args.results_root_dir}/Clustering/softClustMat_{args.clustering_method}_nrusers-{args.num_users}_nrclust-{args.nr_of_embedding_clusters}_from-{args.pre_trained_dataset}_to-{args.target_dataset}.jpg')
    
    # nr_of_centers = 2*cluster_length
    # colors = itertools.cycle(["r"] * nr_of_centers +["b"]*nr_of_centers+["g"]*nr_of_centers+["k"]*nr_of_centers+["y"]*nr_of_centers)
    # plt.figure(4)
    # for i in range(0,args.num_users):
    #     plt.scatter(centers[i][0][0],centers[i][0][1], color=next(colors))
    #     plt.scatter(centers[i][1][0],centers[i][1][1], color=next(colors))
    # plt.savefig(f'{args.results_root_dir}/Clustering/centers_{args.clustering_method}_nrclust-{args.nr_of_embedding_clusters}_from-{args.pre_trained_dataset}_to-{args.target_dataset}.jpg')
    
        # ----------------------------------    
    # plot embedding results in 3D
    centers_embeding = np.reshape(centers, (args.num_users*2, args.latent_dim))

    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    # PCA decomposition
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(centers_embeding)


    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection='3d')
    nr_of_centers = len(c_dict[0])*cluster_length
    colors = ['r']*nr_of_centers + ['b']*nr_of_centers +['g']*nr_of_centers+['k']*nr_of_centers+['y']*nr_of_centers
    ax.scatter(principalComponents[:,0]
               , principalComponents[:,1]
               , principalComponents[:,2]
               , s=30
               , c = colors)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    plt.savefig(f'{args.results_root_dir}/Clustering/ecnoder_pca_plot3d_{args.clustering_method}_nrusers-{args.num_users}_nrclust-{args.nr_of_embedding_clusters}_from-{args.pre_trained_dataset}_to-{args.target_dataset}.jpg')


    if args.clustering_method not in ['umap_central', 'umap']:
        plt.figure(5)
        nr_of_centers = len(dict_users[0])*cluster_length
        colors = itertools.cycle(["r"]*1 + ["b"]*1 + ["g"]*1 + ["k"]*1 + ["y"]*1)
        for i in range(args.nr_of_embedding_clusters):
            plt.scatter(embedding_matrix[i*nr_of_centers:(i+1)*nr_of_centers, 0], embedding_matrix[i*nr_of_centers:(i+1)*nr_of_centers:, 1], color=next(colors))
        plt.savefig(f'{args.results_root_dir}/Clustering/embeddingMat_{args.clustering_method}_nrusers-{args.num_users}_nrclust-{args.nr_of_embedding_clusters}_from-{args.pre_trained_dataset}_to-{args.target_dataset}.jpg')
    elif args.clustering_method == 'umap_central':
        plt.figure(5)
        nr_of_centers = len(c_dict[0])*cluster_length
        colors = itertools.cycle(["r"]*nr_of_centers + ["b"]*nr_of_centers + ["g"]*nr_of_centers + ["k"]*nr_of_centers + ["y"]*nr_of_centers)
        for i in range(args.num_users):
            plt.scatter(c_dict[i][0][0], c_dict[i][0][1], color=next(colors))
            plt.scatter(c_dict[i][1][0], c_dict[i][1][1], color=next(colors))
        #c_dict
        plt.savefig(f'{args.results_root_dir}/Clustering/umap_embeding_{args.clustering_method}_nrusers-{args.num_users}_nrclust-{args.nr_of_embedding_clusters}_from-{args.pre_trained_dataset}_to-{args.target_dataset}.jpg')
    plt.show()
