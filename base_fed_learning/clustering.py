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
import torch.optim as optim

from utils.sampling import mnist_iid, mnist_noniid, mnist_noniid_cluster, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
import pickle
from sklearn.cluster import KMeans
import itertools

from tqdm import tqdm

from manifold_approximation.models.convAE_128D import ConvAutoencoder
from manifold_approximation.encoder import Encoder
from manifold_approximation.sequential_encoder import Sequential_Encoder

# ----------------------------------
# Reproducability
# ----------------------------------
torch.manual_seed(123)
np.random.seed(321)
umap_random_state=42

def gen_data(iid, dataset_type, num_users, cluster):
    # load dataset and split users
    if dataset_type == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if iid:
            dict_users = mnist_iid(dataset_train, num_users)
        else:
            dict_users = mnist_noniid_cluster(dataset_train, num_users, cluster)
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
            if ar_label[idx][0] == ar_label[idx0][0] and ar_label[idx][1] == ar_label[idx0][1]:
                clustering_matrix[idx][idx0] = 1
                
    return clustering_matrix

def clustering_umap(num_users, dict_users, dataset_train, args):
    reducer_loaded = pickle.load( open( "./model_weights/umap_reducer_EMNIST.p", "rb" ) )
    reducer = reducer_loaded

    idxs_users = np.arange(num_users)

    centers = np.zeros((num_users, 2, 2))
    for idx in tqdm(idxs_users, desc='Clustering progress'):
        images_matrix = np.empty((0,28*28))
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
        for batch_idx, (images, labels) in enumerate(local.ldr_train):#TODO: concatenate the matrices
            # if batch_idx == 3:# TODO: abalation test
            #     break
            ne = images.numpy().flatten().T.reshape((10,28*28))
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
        
            dist0 = np.linalg.norm(c0[0] - c1[0])**2 + np.linalg.norm(c0[1] - c1[1])**2
            dist1 = np.linalg.norm(c0[0] - c1[1])**2 + np.linalg.norm(c0[1] - c1[0])**2
        
            distance = min([dist0, dist1])#min (max)
            clustering_matrix_soft[idx0][idx1] = distance
        
            if distance < 1:
                clustering_matrix[idx0][idx1] = 1
            else:
                clustering_matrix[idx0][idx1] = 0

    return clustering_matrix, clustering_matrix_soft, centers

def clustering_encoder(num_users, dict_users, dataset_train, ae_model, ae_model_name, 
                                                        model_root_dir, manifold_dim, args):
    
    idxs_users = np.arange(num_users)

    centers = np.zeros((num_users, 2, 2))
    embedding_matrix = np.zeros((len(dict_users[0])*num_users, 2))
    for user_id in tqdm(idxs_users, desc='Custering in progress ...'):
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user_id])
        
        user_dataset_train = local.ldr_train.dataset
            
        # #TODO: Mo(k)h to review this! 
        encoder = Encoder(ae_model, ae_model_name, model_root_dir, 
                                    manifold_dim, user_dataset_train, user_id)
        
        encoder.autoencoder()
        encoder.manifold_approximation_umap()
        reducer = encoder.umap_reducer
        embedding1 = encoder.umap_embedding
        
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
    args.num_users = 20
    args.iid = False
    # ----------------------------------
    plt.close('all')
    
    # ----------------------------------
    # generate cluster settings    
    nr_of_clusters = 5
    cluster_length = args.num_users // nr_of_clusters
    cluster = np.zeros((nr_of_clusters,2), dtype='int64')
    for i in range(nr_of_clusters):
        cluster[i] = np.random.choice(10, 2, replace=False)
    
    # ----------------------------------       
    manifold_dim = 2
    
    # ----------------------------------       
    # model
    model = ConvAutoencoder().to(args.device)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # ----------------------------------
    # Load the model ckpt
    checkpoint = torch.load(f'{args.model_root_dir}/{args.model_name}_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']  
        
    # ----------------------------------
    # generate clustered data
    dataset_train, dataset_test, dict_users = gen_data(args.iid, args.dataset, args.num_users, cluster)
    
    # ----------------------------------    
    #average over clients in a same cluster
    clustering_matrix = clustering_perfect(args.num_users, dict_users, dataset_train, args)
    #clustering_matrix0, clustering_matrix0_soft, centers = clustering_umap(args.num_users, dict_users, dataset_train, args)
    clustering_matrix0, clustering_matrix0_soft, centers, embedding_matrix = clustering_encoder(args.num_users, dict_users, dataset_train, 
                                                                model, args.model_name, args.model_root_dir, manifold_dim, args)
    
    # ----------------------------------    
    # plot results
    plt.figure(1)
    plt.imshow(clustering_matrix,cmap=plt.cm.viridis)
    plt.figure(2)
    plt.imshow(clustering_matrix0,cmap=plt.cm.viridis)
    plt.figure(3)
    plt.imshow(-clustering_matrix0_soft,cmap=plt.cm.viridis)

    plt.show()

    nr_of_centers = 2*cluster_length
    colors = itertools.cycle(["r"] * nr_of_centers +["b"]*nr_of_centers+["g"]*nr_of_centers+["k"]*nr_of_centers+["y"]*nr_of_centers)
    plt.figure(4)
    for i in range(0,args.num_users):
        plt.scatter(centers[i][0][0],centers[i][0][1], color=next(colors))
        plt.scatter(centers[i][1][0],centers[i][1][1], color=next(colors))

    plt.show()

    plt.figure(5)
    nr_of_centers = len(dict_users[0])*cluster_length
    colors = itertools.cycle(["r"]*1 + ["b"]*1 + ["g"]*1 + ["k"]*1 + ["y"]*1)
    for i in range(nr_of_clusters):
        plt.scatter(embedding_matrix[i*nr_of_centers:(i+1)*nr_of_centers, 0], embedding_matrix[i*nr_of_centers:(i+1)*nr_of_centers:, 1], color=next(colors))

    plt.show()
