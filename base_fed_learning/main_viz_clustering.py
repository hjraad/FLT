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

from utils.sampling import mnist_iid, mnist_noniid, mnist_noniid_cluster, cifar_iid
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

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
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
    centers_embeding = np.zeros((num_users, 2, 128)) # AE latent size going to be hyperparamter
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
        embedding_matrix[user_id*len(dict_users[0]): len(dict_users[0])*(user_id + 1),:] = embedding
        kmeans = KMeans(n_clusters=2, random_state=43).fit(embedding)
        sorted_centers = kmeans.cluster_centers_
        sorted_centers.sort(axis=0)
        centers_embeding[user_id,:,:] = sorted_centers
    
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    umap_embedding = umap_reducer.fit_transform(np.reshape(centers_embeding, (-1, 128)))
    centers = np.reshape(umap_embedding, (num_users, -1, 2))
    
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
                
    return clustering_matrix, clustering_matrix_soft, centers, embedding_matrix, centers_embeding
import random

def plot_weighted_graph(W, index, cmap):
    "Plot a weighted graph"

    mm = np.max(np.max(W))
    
    #2. Add nodes
    G = nx.Graph() #Create a graph object called G
    node_list = [str(i) for i in index]

    for node in node_list:
        G.add_node(node)
 
    #Note: You can also try a spring_layout
    #pos=nx.shell_layout(G, shells) 
    pos=nx.shell_layout(G) 
    nx.draw_networkx_nodes(G,pos,node_color='grey',node_size=500)
 
    #3. If you want, add labels to the nodes
    labels = {}
    for node_name in node_list:
        labels[str(node_name)] =str(node_name)
    nx.draw_networkx_labels(G,pos,labels,font_size=16)
 
    #4. Add the edges (4C2 = 6 combinations)
    #NOTE: You usually read this data in from some source
    #To keep the example self contained, I typed this out
    for i in range(len(W)):
        for j in range(len(W)):
            if i!= j:
                weight = W[i,j]
                G.add_edge(node_list[i],node_list[j],color='b', weight = weight/10)#,weight=1/(weight+0.0000001)) #Kramnik vs Anand
                #G.add_edge(node_list[i],node_list[j],weight=weight) #Kramnik vs Anand
 
    all_weights = []
    #4 a. Iterate through the graph nodes to gather all the weights
    for (node1,node2,data) in G.edges(data=True):
        all_weights.append(data['weight']) #we'll use this when determining edge thickness
 
    #4 b. Get unique weights
    unique_weights = list(set(all_weights))
 
    #4 c. Plot the edges - one by one!
    for weight in unique_weights:
        #4 d. Form a filtered list with just the weight you want to draw
        weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in G.edges(data=True) if edge_attr['weight']==weight]
        #4 e. I think multiplying by [num_nodes/sum(all_weights)] makes the graphs edges look cleaner
        width = weight*len(node_list)*3.0/sum(all_weights)*3

        nx.draw_networkx_edges(G,pos,edgelist=weighted_edges,edge_color='navy', width=width)

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.num_users = 25
    args.ae_model_name = "model-1607623811-epoch40-latent128"
    args.pre_trained_dataset = 'FMNIST'
    args.iid = False
    
    # ----------------------------------
    plt.close('all')
    
    # ----------------------------------
    # generate cluster settings    
    cluster_length = args.num_users // args.nr_of_clusters
    cluster = np.zeros((args.nr_of_clusters, 2), dtype='int64')
    if False:#args.clustering_with_overlap:
        for i in range(args.nr_of_clusters):
            cluster[i] = np.random.choice(10, 2, replace=False)
    else:
        cluster_array = np.random.choice(10, 10, replace=False)
        for i in range(args.nr_of_clusters):
            cluster[i] = cluster_array[i*2: i*2 + 2]

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
        clustering_matrix0, clustering_matrix0_soft, centers, embedding_matrix, centers_embeding =\
            clustering_umap_central(args.num_users, dict_users, dataset_train, args.ae_model_name, 
                                        args.nr_epochs_sequential_training, args)

    # ----------------------------------    
    # randomize the clustering
    index_random = list(np.arange(len(clustering_matrix0)))
    random.shuffle(index_random)
    clustering_matrix_random = clustering_matrix.copy()
    clustering_matrix0_random = clustering_matrix0.copy()
    clustering_matrix0_soft_random = clustering_matrix0_soft.copy()
    for i in range(len(clustering_matrix)):
        for j in range(len(clustering_matrix)):
            clustering_matrix_random[i,j] = clustering_matrix[index_random[i], index_random[j]]
            clustering_matrix0_random[i,j] = clustering_matrix0[index_random[i], index_random[j]]
            clustering_matrix0_soft_random[i,j] = clustering_matrix0_soft[index_random[i], index_random[j]]

    # ---------------------------------- 
    # plot results
    plt.figure(1)
    plt.imshow(clustering_matrix,cmap=plt.cm.viridis)
    plt.savefig(f'{args.results_root_dir}/Clustering/clustMatRef_{clustering_method}_nrclust-{args.nr_of_clusters}_from-{args.pre_trained_dataset}_to-{args.dataset}.png')
    plt.axis('off')
    plt.figure(2)
    plt.imshow(clustering_matrix0_soft < 2.2,cmap=plt.cm.viridis)
    plt.axis('off')
    plt.savefig(f'{args.results_root_dir}/Clustering/clustMatEst_{clustering_method}_nrclust-{args.nr_of_clusters}_from-{args.pre_trained_dataset}_to-{args.dataset}.png')
    plt.figure(3)
    plt.imshow(-1*clustering_matrix0_soft,cmap=plt.cm.viridis)
    plt.axis('off')
    plt.savefig(f'{args.results_root_dir}/Clustering/ClustEstSoftMat_{clustering_method}_nrclust-{args.nr_of_clusters}_from-{args.pre_trained_dataset}_to-{args.dataset}.png')
    
    # ----------------------------------    
    # plot connectivity graphs
    scaling = 4
    plt.figure(4, figsize=(8, 6), dpi=80)
    cmap = plt.get_cmap('Blues')
    plot_weighted_graph(clustering_matrix0_soft_random < 2.2, index_random, cmap)
    #Plot the graph
    plt.axis('off')
    plt.savefig(f'{args.results_root_dir}/Clustering/ClustEstSoftRandomGraph_{clustering_method}_nrclust-{args.nr_of_clusters}_from-{args.pre_trained_dataset}_to-{args.dataset}.png')
    #plt.show() 

    plt.figure(5, figsize=(8, 6), dpi=80)
    plot_weighted_graph(clustering_matrix0_soft < 2.2, list(np.arange(len(clustering_matrix0))), cmap)
    #Plot the graph
    plt.axis('off')
    plt.savefig(f'{args.results_root_dir}/Clustering/ClustEstSoftGraph_{clustering_method}_nrclust-{args.nr_of_clusters}_from-{args.pre_trained_dataset}_to-{args.dataset}.png')
    #plt.show() 
    # ----------------------------------    
    # plot adjacency matrices
    labels = [str(i) for i in list(np.arange(len(clustering_matrix)))]
    fig = plt.figure(6)
    ax = fig.add_subplot(111)
    cax = ax.matshow(1./(clustering_matrix0_soft/scaling+1))
    #fig.colorbar(cax)
    plt.axis('off')
    plt.savefig(f'{args.results_root_dir}/Clustering/ClustEstSoftMatColorMap_{clustering_method}_nrclust-{args.nr_of_clusters}_from-{args.pre_trained_dataset}_to-{args.dataset}.png')

    fig = plt.figure(7)
    ax = fig.add_subplot(111)
    cax = ax.matshow(clustering_matrix0_soft_random < 2.2)
    #fig.colorbar(cax)
    plt.axis('off')
    plt.savefig(f'{args.results_root_dir}/Clustering/ClustEstSoftMatRandomColorMap_{clustering_method}_nrclust-{args.nr_of_clusters}_from-{args.pre_trained_dataset}_to-{args.dataset}.png')
    
    plt.show()