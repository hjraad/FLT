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

from tqdm import tqdm

from manifold_approximation.models.convAE_128D import ConvAutoencoder
from manifold_approximation.models.convAE_cifar import ConvAutoencoderCIFAR
from manifold_approximation.models.convAE_cifar_residual import ConvAutoencoderCIFARResidual
from manifold_approximation.encoder import Encoder
from manifold_approximation.sequential_encoder import Sequential_Encoder
from manifold_approximation.utils.load_datasets import load_dataset

from sympy.utilities.iterables import multiset_permutations

# ----------------------------------
# Reproducability
# ----------------------------------
torch.manual_seed(123)
np.random.seed(321)
umap_random_state=42

def gen_data(iid, dataset_type, data_root_dir, transforms_dict, num_users, cluster):
    # load dataset 
    _, image_datasets, dataset_sizes, class_names =\
            load_dataset(dataset_type, data_root_dir, transforms_dict, batch_size=8, shuffle_flag=False, dataset_split='')
    
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

def clustering_perfect(num_users, dict_users, dataset_train, args):
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
            set_1 = set(ar_label[idx0][np.where(ar_label[idx0] != -1)].astype(int))
            set_2 = set(ar_label[idx][np.where(ar_label[idx] != -1)].astype(int))
            if np.intersect1d(set_1, set_2):
                if len( np.intersect1d(set_1, set_2)[0] ) >= np.floor(0.6 * min(len(set_1), len(set_2))):   
                #if ar_label[idx][0] == ar_label[idx0][0] and ar_label[idx][1] == ar_label[idx0][1]:
                    clustering_matrix[idx][idx0] = 1
                
    return clustering_matrix

def clustering_umap(num_users, dict_users, dataset_train, args):
    reducer_loaded = pickle.load( open( "../model_weights/umap_reducer_EMNIST.p", "rb" ) )
    reducer = reducer_loaded

    idxs_users = np.arange(num_users)
    
    input_dim = dataset_train[0][0].shape[-1]
    channel_dim = dataset_train[0][0].shape[0]
    
    centers = np.zeros((num_users, 2, 2))
    for idx in tqdm(idxs_users, desc='Clustering progress'):
        images_matrix = np.empty((0, channel_dim*input_dim*input_dim))
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
        for batch_idx, (images, labels) in enumerate(local.ldr_train):#TODO: concatenate the matrices
            # print(batch_idx)
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

    for idx0 in idxs_users:
        for idx1 in idxs_users:
            c0 = centers[idx0]
            c1 = centers[idx1]

            if len(c0) < len(c1):
                c_small = c0
                c_big = c1
            else:
                c_small = c1
                c_big = c0

            distance = 1000000
            if len(c_small) > 0:
                s = set(range(len(c_big)))
                for p in multiset_permutations(s):
                    summation = 0

                    for i in range(len(c_small)):
                        summation = summation + (np.linalg.norm(c_small[i] - c_big[p][i])**2)

                    dist = summation/len(c_small)
                    if dist < distance:
                        distance = dist

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
                          user_dataset_train, user_id)
        
        encoder.autoencoder()
        encoder.manifold_approximation_umap()
        reducer = encoder.umap_reducerclustering_matrix

    for idx0 in idxs_users:
        for idx1 in idxs_users:
            c0 = centers[idx0]
            c1 = centers[idx1]
        
            dist0 = np.linalg.norm(c0[0] - c1[0])**2 + np.linalg.norm(c0[1] - c1[1])**2
            dist1 = np.linalg.norm(c0[0] - c1[1])**2 + np.linalg.norm(c0[1] - c1[0])**2
        
            distance = min([dist0, dist1])#min (max)
            clustering_matrix_soft[idx0][idx1] = distance
        
            if distance < 1:
                clustering_matrix[idx0][idx1] = 1
            else:
                clustering_matrix[idx0][idx1] = 0

    return clustering_matrix, clustering_matrix_soft, centers, embedding_matrix

def clustering_umap_central(dict_users, dataset_train, ae_model_dict, args):

    # idxs_users = np.random.shuffle(np.arange(num_users))
    idxs_users = np.random.choice(args.num_users, args.num_users, replace=False)
    centers = np.zeros((args.num_users, 2, args.latent_dim)) # AE latent size going to be hyperparamter
    embedding_matrix = np.zeros((len(dict_users[0])*args.num_users, args.latent_dim))
    
    for user_id in tqdm(idxs_users, desc='Custering in progress ...'):
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user_id])
        
        user_dataset_train = local.ldr_train.dataset
            
        encoder = Sequential_Encoder(ae_model_dict['model'], ae_model_dict['opt'], 
                                     ae_model_dict['criterion'], ae_model_dict['scheduler'], 
                                     args.nr_epochs_sequential_training, ae_model_dict['name'],
                                     args.model_root_dir, args.log_root_dir, args.manifold_dim, user_dataset_train, 
                                     user_id, args.pre_trained_dataset, dataset_name=args.target_dataset, 
                                     train_umap=False, use_AE=True)
        
        encoder.autoencoder()
        # encoder.manifold_approximation_umap()
        embedding = encoder.ae_embedding_np 
        # reducer = encoder.umap_reducer
        # embedding = encoder.umap_embedding
        # ae_model_name = encoder.new_model_name
        
        # ----------------------------------
        # use Kmeans to cluster the data into 2 clusters
        embedding_matrix[user_id*len(dict_users[0]): len(dict_users[0])*(user_id + 1),:] = embedding
        kmeans = KMeans(n_clusters=2, random_state=43).fit(embedding)
        centers[user_id,:,:] = kmeans.cluster_centers_
    
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    umap_embedding = umap_reducer.fit_transform(np.reshape(centers, (-1, args.latent_dim)))
    centers = np.reshape(umap_embedding, (args.num_users, -1, 2))
    
    clustering_matrix_soft = np.zeros((args.num_users, args.num_users))
    clustering_matrix = np.zeros((args.num_users, args.num_users))

    for idx0 in idxs_users:
        for idx1 in idxs_users:
            c0 = centers[idx0]
            c1 = centers[idx1]
        
            dist0 = np.linalg.norm(c0[0] - c1[0])**2 + np.linalg.norm(c0[1] - c1[1])**2
            dist1 = np.linalg.norm(c0[0] - c1[1])**2 + np.linalg.norm(c0[1] - c1[0])**2
        
            distance = min([dist0, dist1])#min (max)
            clustering_matrix_soft[idx0][idx1] = distance
        
            if distance < 1:
                clustering_matrix[idx0][idx1] = 1
            else:
                clustering_matrix[idx0][idx1] = 0
                
    return clustering_matrix, clustering_matrix_soft, centers, embedding_matrix


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # args.num_users = 20
    # args.ae_model_name = "model-1607623811-epoch40-latent128"
    # args.pre_trained_dataset = 'FMNIST'

    args.num_users = 20
    args.num_classes = 10
    args.target_dataset = 'CIFAR10'
    
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
    
    args.pre_trained_dataset = 'CIFAR10'
    
    # find the model name automatically
    args.ae_model_name = extract_model_name(args.model_root_dir, args.pre_trained_dataset)
    
    args.iid = False
    
    # ----------------------------------
    plt.close('all')
    
    # ----------------------------------
    # generate cluster settings    

    nr_of_clusters = 5
    cluster_length = args.num_users // nr_of_clusters
    cluster = np.zeros((nr_of_clusters, 2), dtype='int64')
    for i in range(nr_of_clusters):
        cluster[i] = np.random.choice(10, 2, replace=False)
        
    # cluster_array = np.random.choice(10, 10, replace=False)
    # for i in range(nr_of_clusters):
    #     cluster[i] = cluster_array[i*2: i*2 + 1]
    
    if args.target_dataset == 'EMNIST': 
        n_1 = 47 // (nr_of_clusters - 1)
        n_2 = 47 % n_1
        cluster = np.zeros((nr_of_clusters, n_1), dtype='int64')
        # cluster_array = np.random.choice(47, 47, replace=False)
        cluster_array = np.arange(47)
        for i in range(nr_of_clusters - 1):
            cluster[i] = cluster_array[i*n_1: i*n_1 + n_1]
        cluster[nr_of_clusters - 1][0:n_2] = cluster_array[-n_2:]  
    # ---------------------------------- 
    # model
    args.latent_dim = 64
    #
    if args.target_dataset in ['CIFAR10', 'CIFAR100', 'CIFAR110']:
        # ae_model = ConvAutoencoderCIFAR(latent_size).to(args.device)
        args.num_hiddens = 128
        args.num_residual_hiddens = 32
        args.num_residual_layers = 2
        
        ae_model = ConvAutoencoderCIFARResidual(args.num_hiddens, args.num_residual_layers, 
                                                args.num_residual_hiddens, args.latent_dim).to(args.device)
    else:
        ae_model = ConvAutoencoder().to(args.device)
    
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    ae_optimizer = optim.Adam(ae_model.parameters(), lr=0.001)

    # Decay LR by a factor of x*gamma every step_size epochs
    exp_lr_scheduler = lr_scheduler.StepLR(ae_optimizer, step_size=10, gamma=0.5)
    
    # loss
    criterion = nn.BCELoss()
    
    # Load the model ckpt
    checkpoint = torch.load(f'{args.model_root_dir}/{args.ae_model_name}_best.pt')
    ae_model.load_state_dict(checkpoint['model_state_dict']) 
    
    ae_model_dict = {
        'model':ae_model,
        'name': args.ae_model_name,
        'opt':ae_optimizer,
        'criterion':criterion,
        'scheduler':exp_lr_scheduler
    }
    
    args.nr_epochs_sequential_training = 0
    
    # ----------------------------------       
    clustering_method = 'umap_central'    # umap, encoder, sequential_encoder, umap_central
    
    # ----------------------------------
    # generate clustered data
    dataset_train, dataset_test, dict_users = gen_data(args.iid, args.target_dataset, args.data_root_dir, 
                                                       transforms_dict, args.num_users, cluster)
    
    # ----------------------------------    
    clustering_matrix = clustering_perfect(args.num_users, dict_users, dataset_train, args)
    
    if clustering_method == 'umap':
        clustering_matrix0, clustering_matrix0_soft, centers = clustering_umap(args.num_users, dict_users, dataset_train, args)
    #
    elif clustering_method == 'encoder':
        clustering_matrix0, clustering_matrix0_soft, centers, embedding_matrix =\
            clustering_encoder(dict_users, dataset_train, ae_model_dict, args)
    #
    elif clustering_method == 'umap_central':
        clustering_matrix0, clustering_matrix0_soft, centers, embedding_matrix =\
            clustering_umap_central(dict_users, dataset_train, ae_model_dict, args)
    
    # ----------------------------------    
    # plot results
    plt.figure(1)
    plt.imshow(clustering_matrix,cmap=plt.cm.viridis)
    plt.savefig(f'{args.results_root_dir}/Clustering/clustMat_perfect_nrclust-{nr_of_clusters}_from-{args.pre_trained_dataset}_to-{args.target_dataset}.jpg')
    
    plt.figure(2)
    plt.imshow(clustering_matrix0,cmap=plt.cm.viridis)
    plt.savefig(f'{args.results_root_dir}/Clustering/clustMat_{clustering_method}_nrclust-{args.nr_of_clusters}_from-{args.pre_trained_dataset}_to-{args.target_dataset}.jpg')
    
    plt.figure(3)
    plt.imshow(-1*clustering_matrix0_soft,cmap=plt.cm.viridis)
    plt.savefig(f'{args.results_root_dir}/Clustering/softClustMat_{clustering_method}_nrclust-{args.nr_of_clusters}_from-{args.pre_trained_dataset}_to-{args.target_dataset}.jpg')
    
    nr_of_centers = 2*cluster_length
    colors = itertools.cycle(["r"] * nr_of_centers +["b"]*nr_of_centers+["g"]*nr_of_centers+["k"]*nr_of_centers+["y"]*nr_of_centers)
    plt.figure(4)
    for i in range(0,args.num_users):
        plt.scatter(centers[i][0][0],centers[i][0][1], color=next(colors))
        plt.scatter(centers[i][1][0],centers[i][1][1], color=next(colors))
    plt.savefig(f'{args.results_root_dir}/Clustering/centers_{clustering_method}_nrclust-{args.nr_of_clusters}_from-{args.pre_trained_dataset}_to-{args.target_dataset}.jpg')
    
    if clustering_method not in ['umap_central', 'umap']:
        plt.figure(5)
        nr_of_centers = len(dict_users[0])*cluster_length
        colors = itertools.cycle(["r"]*1 + ["b"]*1 + ["g"]*1 + ["k"]*1 + ["y"]*1)
        for i in range(args.nr_of_clusters):
            plt.scatter(embedding_matrix[i*nr_of_centers:(i+1)*nr_of_centers, 0], embedding_matrix[i*nr_of_centers:(i+1)*nr_of_centers:, 1], color=next(colors))
        plt.savefig(f'{args.results_root_dir}/Clustering/embeddingMat_{clustering_method}_nrclust-{args.nr_of_clusters}_from-{args.pre_trained_dataset}_to-{args.target_dataset}.jpg')
    plt.show()
