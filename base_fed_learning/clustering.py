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

from utils.sampling import mnist_iid, mnist_noniid, mnist_noniid_cluster, cifar_iid, emnist_noniid_cluster
from utils.options import args_parser
from models.Update import LocalUpdate
import pickle
from sklearn.cluster import KMeans
import itertools
import copy
import umap

from tqdm import tqdm

from manifold_approximation.models.convAE_128D import ConvAutoencoder
from manifold_approximation.encoder import Encoder
from manifold_approximation.sequential_encoder import Sequential_Encoder
from sympy.utilities.iterables import multiset_permutations
# ----------------------------------
# Reproducability
# ----------------------------------
torch.manual_seed(123)
np.random.seed(321)
umap_random_state=42

def gen_data(iid, dataset_type, num_users, cluster):
    # load dataset and split users
    if dataset_type == 'mnist':
        # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trans_mnist = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if iid:
            dict_users = mnist_iid(dataset_train, num_users)
        else:
            dict_users = mnist_noniid_cluster(dataset_train, num_users, cluster)
    #
    elif dataset_type == 'EMNIST':
        dataset_train = datasets.EMNIST(root='../data', split=args.dataset_split, 
                                                train=True, download=True, 
                                                transform=transforms.Compose([
                                                lambda img: transforms.functional.rotate(img, -90),
                                                lambda img: transforms.functional.hflip(img),
                                                transforms.ToTensor()]))

        dataset_test = datasets.EMNIST(root='../data', split=args.dataset_split, 
                                                    train=False, download=True, 
                                                    transform= transforms.Compose([
                                                    lambda img: transforms.functional.rotate(img, -90),
                                                    lambda img: transforms.functional.hflip(img),
                                                    transforms.ToTensor()]))      
        if not iid:
            dict_users = emnist_noniid_cluster(dataset_train, num_users, cluster, 
                                               random_shuffle=True)
    #       
    elif dataset_type == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if iid:
            dict_users = cifar_iid(dataset_train, num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
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
    args.local_bs = 10
    idxs_users = np.arange(num_users)
    ar_label = np.zeros((args.num_users, args.num_classes))-1
    for idx in idxs_users:
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
        label_matrix = np.empty(0, dtype=int)
        for batch_idx, (images, labels) in enumerate(local.ldr_train):
            #print(batch_idx)
            #if batch_idx == 3:# TODO: abalation test
            #    break
            label_matrix = np.concatenate((label_matrix, labels.numpy()), axis=0)
        label_matrix = np.unique(label_matrix)
        ar_label[idx][0:len(label_matrix)] = label_matrix
    args.local_bs = 10
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
    reducer_loaded = pickle.load( open( "./model_weights/umap_reducer_EMNIST.p", "rb" ) )
    reducer = reducer_loaded

    idxs_users = np.arange(num_users)
    args.local_bs = 10
    centers = np.zeros((num_users, 2, 2))
    for idx in tqdm(idxs_users, desc='Clustering progress, embedding'):
        images_matrix = np.empty((0,28*28))
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
        for batch_idx, (images, labels) in enumerate(local.ldr_train):#TODO: concatenate the matrices
            #print(batch_idx)
            #if batch_idx == 3:# TODO: abalation test
            #    break
            ne = images.numpy().flatten().T.reshape((len(labels),28*28))
            images_matrix = np.vstack((images_matrix, ne))
        embedding1 = reducer.transform(images_matrix)
        X = list(embedding1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(X))
        centers[idx,:,:] = kmeans.cluster_centers_
    
    args.local_bs = 10
    clustering_matrix_soft = np.zeros((num_users, num_users))
    clustering_matrix = np.zeros((num_users, num_users))

    for idx0 in tqdm(idxs_users, desc='Clustering matrix generation'):
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

def clustering_encoder(num_users, dict_users, dataset_train, ae_model_name, 
                                                        model_root_dir, manifold_dim, args):

    # model
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

    idxs_users = np.arange(num_users)

    centers = np.zeros((num_users, 2, 2))
    embedding_matrix = np.zeros((len(dict_users[0])*num_users, 2))
    for user_id in tqdm(idxs_users, desc='Custering in progress ...'):
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user_id])
        
        user_dataset_train = local.ldr_train.dataset
            
        encoder = Encoder(ae_model, ae_model_name, model_root_dir, 
                                    manifold_dim, user_dataset_train, user_id)
        
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
 
def clustering_sequential_encoder(num_users, dict_users, dataset_train, ae_model_name, 
                                  nr_epochs_sequential_training, args):

    # model
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

    # idxs_users = np.random.shuffle(np.arange(num_users))
    idxs_users = np.random.choice(num_users, num_users, replace=False)
    centers = np.zeros((num_users, 2, 2))
    embedding_matrix = np.zeros((len(dict_users[0])*num_users, 2))

    for user_id in tqdm(idxs_users, desc='Custering in progress ...'):
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user_id])
        
        user_dataset_train = local.ldr_train.dataset
            
        encoder = Sequential_Encoder(ae_model, ae_optimizer, criterion, exp_lr_scheduler, nr_epochs_sequential_training, 
                                     ae_model_name, args.model_root_dir, args.log_root_dir, args.manifold_dim, user_dataset_train, 
                                     user_id, args.pre_trained_dataset)
        
        encoder.autoencoder()
        encoder.manifold_approximation_umap()
        # reducer = encoder.umap_reducer
        embedding = encoder.umap_embedding
        # ae_model_name = encoder.new_model_name
        
        # ----------------------------------
        # use Kmeans to cluster the data into 2 clusters
        X = list(embedding)
        embedding_matrix[user_id*len(dict_users[0]): len(dict_users[0])*(user_id + 1),:] = embedding
        kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(X))
        centers[user_id,:,:] = kmeans.cluster_centers_
    
    clustering_matrix_soft = np.zeros((num_users, num_users))
    clustering_matrix = np.zeros((num_users, num_users))

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

def clustering_umap_central(num_users, dict_users, dataset_train, ae_model_name, 
                            nr_epochs_sequential_training, args):

    # model
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

    # idxs_users = np.random.shuffle(np.arange(num_users))
    idxs_users = np.random.choice(num_users, num_users, replace=False)
    centers = np.zeros((num_users, 2, 128)) # AE latent size going to be hyperparamter
    embedding_matrix = np.zeros((len(dict_users[0])*num_users, 128))

    for user_id in tqdm(idxs_users, desc='Custering in progress ...'):
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user_id])
        
        user_dataset_train = local.ldr_train.dataset
            
        encoder = Sequential_Encoder(ae_model, ae_optimizer, criterion, exp_lr_scheduler, nr_epochs_sequential_training, 
                                     ae_model_name, args.model_root_dir, args.log_root_dir, args.manifold_dim, user_dataset_train, 
                                     user_id, args.pre_trained_dataset, train_umap=False, use_AE=True)
        
        encoder.autoencoder()
        # encoder.manifold_approximation_umap()
        embedding = encoder.ae_embedding_np 
        # reducer = encoder.umap_reducer
        # embedding = encoder.umap_embedding
        # ae_model_name = encoder.new_model_name
        
        # ----------------------------------
        # use Kmeans to cluster the data into 2 clusters
        #embedding_matrix[user_id*len(dict_users[0]): len(dict_users[0])*(user_id + 1),:] = embedding
        kmeans = KMeans(n_clusters=2, random_state=43).fit(embedding)
        centers[user_id,:,:] = kmeans.cluster_centers_
    
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    umap_embedding = umap_reducer.fit_transform(np.reshape(centers, (-1, 128)))
    centers = np.reshape(umap_embedding, (num_users, -1, 2))
    
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
                
    return clustering_matrix, clustering_matrix_soft, centers, embedding_matrix


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # args.num_users = 20
    # args.ae_model_name = "model-1607623811-epoch40-latent128"
    # args.pre_trained_dataset = 'FMNIST'

    args.num_users = 240
    args.num_classes = 47
    args.dataset = 'EMNIST'
    args.model_name = "model-1606927012-epoch40-latent128"
    args.pre_trained_dataset = 'EMNIST'
    args.iid = False
    
    # ----------------------------------
    plt.close('all')
    
    # ----------------------------------
    # generate cluster settings    

    nr_of_clusters = 10
    cluster_length = args.num_users // nr_of_clusters
    cluster = np.zeros((nr_of_clusters, 2), dtype='int64')
    for i in range(nr_of_clusters):
        cluster[i] = np.random.choice(10, 2, replace=False)
        
    # cluster_array = np.random.choice(10, 10, replace=False)
    # for i in range(nr_of_clusters):
    #     cluster[i] = cluster_array[i*2: i*2 + 1]
    
    if args.dataset == 'EMNIST': 
        n_1 = 47 // (nr_of_clusters - 1)
        n_2 = 47 % n_1
        cluster = np.zeros((nr_of_clusters, n_1), dtype='int64')
        # cluster_array = np.random.choice(47, 47, replace=False)
        cluster_array = np.arange(47)
        for i in range(nr_of_clusters - 1):
            cluster[i] = cluster_array[i*n_1: i*n_1 + n_1]
        cluster[nr_of_clusters - 1][0:n_2] = cluster_array[-n_2:]  
    # ----------------------------------       
    manifold_dim = 2
    nr_epochs_sequential_training = 5
    encoding_method = 'umap'    # umap, encoder, sequential_encoder, umap_central
    
    # ----------------------------------       
    clustering_method = 'umap_central'    # umap, encoder, sequential_encoder, umap_central

    # ----------------------------------
    # generate clustered data
    dataset_train, dataset_test, dict_users = gen_data(args.iid, args.dataset, args.num_users, cluster)
    
    # ----------------------------------    
    #average over clients in a same cluster
    clustering_matrix = clustering_perfect(args.num_users, dict_users, dataset_train, args)
    
    if clustering_method == 'umap':
        clustering_matrix0, clustering_matrix0_soft, centers = clustering_umap(args.num_users, dict_users, dataset_train, args)
    elif clustering_method == 'encoder':
        clustering_matrix0, clustering_matrix0_soft, centers, embedding_matrix =\
            clustering_encoder(args.num_users, dict_users, dataset_train, 
                               args.ae_model_name, args.model_root_dir, args.manifold_dim, args)
    elif clustering_method == 'sequential_encoder':
        clustering_matrix0, clustering_matrix0_soft, centers, embedding_matrix =\
            clustering_sequential_encoder(args.num_users, dict_users, dataset_train, args.ae_model_name, 
                                        args.nr_epochs_sequential_training, args)
    elif clustering_method == 'umap_central':
        clustering_matrix0, clustering_matrix0_soft, centers, embedding_matrix =\
            clustering_umap_central(args.num_users, dict_users, dataset_train, args.ae_model_name, 
                                        args.nr_epochs_sequential_training, args)
    
    # ----------------------------------    
    # plot results
    plt.figure(1)
    plt.imshow(clustering_matrix,cmap=plt.cm.viridis)
    plt.savefig(f'{args.results_root_dir}/Clustering/clustMat_perfect_nrclust-{nr_of_clusters}_from-{args.pre_trained_dataset}_to-{args.dataset}.jpg')
    
    plt.figure(2)
    plt.imshow(clustering_matrix0,cmap=plt.cm.viridis)
    plt.savefig(f'{args.results_root_dir}/Clustering/clustMat_{clustering_method}_nrclust-{args.nr_of_clusters}_from-{args.pre_trained_dataset}_to-{args.dataset}.jpg')
    
    plt.figure(3)
    plt.imshow(-1*clustering_matrix0_soft,cmap=plt.cm.viridis)
    plt.savefig(f'{args.results_root_dir}/Clustering/softClustMat_{clustering_method}_nrclust-{args.nr_of_clusters}_from-{args.pre_trained_dataset}_to-{args.dataset}.jpg')
    
    nr_of_centers = 2*cluster_length
    colors = itertools.cycle(["r"] * nr_of_centers +["b"]*nr_of_centers+["g"]*nr_of_centers+["k"]*nr_of_centers+["y"]*nr_of_centers)
    plt.figure(4)
    for i in range(0,args.num_users):
        plt.scatter(centers[i][0][0],centers[i][0][1], color=next(colors))
        plt.scatter(centers[i][1][0],centers[i][1][1], color=next(colors))
    plt.savefig(f'{args.results_root_dir}/Clustering/centers_{clustering_method}_nrclust-{args.nr_of_clusters}_from-{args.pre_trained_dataset}_to-{args.dataset}.jpg')
    
    if clustering_method not in ['umap_central', 'umap']:
        plt.figure(5)
        nr_of_centers = len(dict_users[0])*cluster_length
        colors = itertools.cycle(["r"]*1 + ["b"]*1 + ["g"]*1 + ["k"]*1 + ["y"]*1)
        for i in range(args.nr_of_clusters):
            plt.scatter(embedding_matrix[i*nr_of_centers:(i+1)*nr_of_centers, 0], embedding_matrix[i*nr_of_centers:(i+1)*nr_of_centers:, 1], color=next(colors))
        plt.savefig(f'{args.results_root_dir}/Clustering/embeddingMat_{clustering_method}_nrclust-{args.nr_of_clusters}_from-{args.pre_trained_dataset}_to-{args.dataset}.jpg')
    plt.show()
