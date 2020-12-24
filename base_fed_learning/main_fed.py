'''
Base code forked from https://github.com/shaoxiongji/federated-learning
'''
import matplotlib
# matplotlib.use('Agg')
import sys
sys.path.append("./../")
sys.path.append("./../../")
sys.path.append("./")

import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import torchvision
from base_fed_learning.utils.sampling import mnist_iid, mnist_noniid, mnist_noniid_cluster, cifar_iid
from base_fed_learning.utils.options import args_parser
from base_fed_learning.models.Update import LocalUpdate
from base_fed_learning.models.Nets import MLP, CNNMnist, CNNCifar
from base_fed_learning.models.Fed import FedAvg
from base_fed_learning.models.test import test_img, test_img_classes
from clustering import clustering_single, clustering_perfect, clustering_umap, clustering_encoder, clustering_umap_central, clustering_sequential_encoder
from sklearn.cluster import KMeans

from manifold_approximation.models.convAE_128D import ConvAutoencoder

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
        trans_mnist = transforms.Compose([transforms.ToTensor()])# TODO: fix transform
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

def gen_model(dataset, dataset_train, num_users):
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    net_glob_list = [net_glob for i in range(num_users)]
    w_glob_list = [w_glob for i in range(num_users)]
    
    return net_glob, w_glob, net_glob_list, w_glob_list

def clustering_multi_center(num_users, multi_center_initialization_flag):
    clustering_matrix = np.zeros((num_users, num_users))

    return clustering_matrix
def FedMLAlgo(net_glob_list, w_glob_list, dataset_train, dict_users, num_users, clustering_matrix, multi_center_flag):
    # training
    loss_train = []
    if multi_center_flag:
        multi_center_initialization_flag = 1

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = w_glob_list
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * num_users), 1)
        idxs_users = np.random.choice(range(num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob_list[idx]).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        if multi_center_flag:
            #clustering_matrix = multi_center(w_locals)
            if multi_center_initialization_flag:
                ll = np.zeros((num_users, 156800+200+2000+10))
                for i in range(num_users):
                    aa = w_locals[i]
                    lis = []
                    for k in aa.keys():
                        #print(aa[k].numpy().flatten().shape)
                        lis = lis + list(aa[k].cpu().numpy().flatten())
                    ll[i] = np.array(lis).reshape(1,159010)
                
                kmeans = KMeans(n_clusters=args.nr_of_clusters, n_init=20).fit(ll)
                ind_center = kmeans.fit_predict(ll)
                est_multi_center = kmeans.cluster_centers_
                multi_center_initialization_flag = False
                clustering_matrix = np.zeros((num_users, num_users))
                for ii in range(len(ind_center)):
                    for jj in range(len(ind_center)):
                        if ind_center[ii] == ind_center[jj]:
                            clustering_matrix[ii][jj] = 1
                    clustering_matrix[ii][ii] = 1
                #clustering_matrix0 = clustering_multi_center(num_users, multi_center_initialization_flag)
                        
            else:
                ll = np.zeros((num_users, 156800+200+2000+10))
                for i in range(num_users):
                    aa = w_locals[i]
                    lis = []
                    for k in aa.keys():
                        #print(aa[k].numpy().flatten().shape)
                        lis = lis + list(aa[k].cpu().numpy().flatten())
                    ll[i] = np.array(lis).reshape(1,159010)
                kmeans = KMeans(n_clusters=args.nr_of_clusters, init=est_multi_center, n_init=1).fit(ll)#TODO: remove the best
                ind_center = kmeans.fit_predict(ll)
                est_multi_center = kmeans.cluster_centers_
                clustering_matrix = np.zeros((num_users, num_users))
                for ii in range(len(ind_center)):
                    for jj in range(len(ind_center)):
                        if ind_center[ii] == ind_center[jj]:
                            clustering_matrix[ii][jj] = 1
                    clustering_matrix[ii][ii] = 1
        if multi_center_flag:
            plt.figure()
            plt.imshow(clustering_matrix)
            plt.savefig(f'{args.results_root_dir}/Clustering/multi_center_nrclust-{args.nr_of_clusters}_num_users-{args.num_users}_{args.epochs}_epoch-{iter}.jpg')
            #plt.show()
        w_glob_list = FedAvg(w_locals, clustering_matrix)

        # copy weight to net_glob
        for idx in np.arange(num_users): #TODO: fix this
            net_glob_list0 = copy.deepcopy(net_glob_list[0])
            net_glob_list0.load_state_dict(w_glob_list[idx])
            net_glob_list[idx] = net_glob_list0

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
    
    return loss_train, net_glob_list, clustering_matrix

def evaluate_performance(net_glob_list, dataset_train, dataset_test, cluster, cluster_length, evaluation_index_range, args, outputFile):
    # evaluate the performance of the models on train and test datasets
    acc_train_final = np.zeros(args.num_users)
    loss_train_final = np.zeros(args.num_users)
    acc_test_final = np.zeros(args.num_users)
    loss_test_final = np.zeros(args.num_users)

    for idx in evaluation_index_range:
        print("user under process: ", idx)
        acc_train_final[idx], loss_train_final[idx] = test_img_classes(net_glob_list[idx], dataset_train, cluster[idx//cluster_length], args)
        acc_test_final[idx], loss_test_final[idx] = test_img_classes(net_glob_list[idx], dataset_test, cluster[idx//cluster_length], args)
    print('Training accuracy: {:.2f}'.format(np.average(acc_train_final[evaluation_index_range])))
    print('Testing accuracy: {:.2f}'.format(np.average(acc_test_final[evaluation_index_range])))

    print('Training accuracy: {:.2f}'.format(np.average(acc_train_final[evaluation_index_range])), file = outputFile)
    print('Testing accuracy: {:.2f}'.format(np.average(acc_test_final[evaluation_index_range])), file = outputFile)

    return

def gen_cluster(args):
    # setting the clustering format
    if args.iid == True:
        nr_of_clusters = 1
    
        cluster_length = args.num_users // nr_of_clusters
        cluster = np.zeros((nr_of_clusters,10), dtype='int64')
        for i in range(nr_of_clusters):
            # TODO: should it be np.random.choice(10, 2, replace=False) for a fairer comparison?
            cluster[i] = np.random.choice(10, 10, replace=False)
    else:
        cluster_length = args.num_users // args.nr_of_clusters
        # generate cluster settings    
        cluster = np.zeros((args.nr_of_clusters, 2), dtype='int64')
        if args.flag_with_overlap:
            for i in range(args.nr_of_clusters):
                cluster[i] = np.random.choice(10, 2, replace=False)
        else:
            cluster_array = np.random.choice(10, 10, replace=False)
            for i in range(args.nr_of_clusters):
                cluster[i] = cluster_array[i*2: i*2 + 2]

    return cluster, cluster_length

def extract_clustering(dict_users, dataset_train, args):

    if args.clustering_method == 'single':
        clustering_matrix = clustering_single(args.num_users)
    if args.clustering_method == 'umap':
        clustering_matrix, _, _ = clustering_umap(args.num_users, dict_users, dataset_train, args)
    elif args.clustering_method == 'encoder':
        clustering_matrix, _, _, _ =\
            clustering_encoder(args.num_users, dict_users, dataset_train, 
                               args.ae_model_name, args.model_root_dir, args.manifold_dim, args)
    elif args.clustering_method == 'sequential_encoder':
        clustering_matrix, _, _, _ =\
            clustering_sequential_encoder(args.num_users, dict_users, dataset_train, args.ae_model_name, 
                                        args.nr_epochs_sequential_training, args)
    elif args.clustering_method == 'umap_central':
        clustering_matrix, _, _, _ =\
            clustering_umap_central(args.num_users, dict_users, dataset_train, args.ae_model_name, 
                                        args.nr_epochs_sequential_training, args)
    
    return clustering_matrix

def run(args, outputFile):
    # setting the clustering format
    cluster, cluster_length = gen_cluster(args)

    dataset_train, dataset_test, dict_users = gen_data(args.iid, args.dataset, args.num_users, cluster)

    # clustering the clients
    clustering_matrix = extract_clustering(dict_users, dataset_train, args)

    net_glob, w_glob, net_glob_list, w_glob_list = gen_model(args.dataset, dataset_train, args.num_users)
    loss_train, net_glob_list, clustering_matrix = FedMLAlgo(net_glob_list, w_glob_list, dataset_train, dict_users, args.num_users, clustering_matrix, args.multi_center)

    # ----------------------------------
    # testing: average over all clients
    if args.iid == True:
        evaluation_index_step = 1
        evaluation_index_max = 1
    elif args.clustering_method == 'single' and args.multi_center == False:
        evaluation_index_step = args.num_users // args.nr_of_clusters# clustering_length
        evaluation_index_max = args.num_users-1
    else:
        evaluation_index_step = 1
        evaluation_index_max = args.num_users-1

    evaluation_index_range = np.arange(0, evaluation_index_max, evaluation_index_step)

    evaluate_performance(net_glob_list, dataset_train, dataset_test, cluster, cluster_length, evaluation_index_range, args, outputFile)

    return loss_train

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # ----------------------------------
    plt.close('all')
    
    # open the output file to write the results to
    outputFile = open(f'{args.results_root_dir}/main_fed/results_numusers_{args.num_users}_{args.clustering_method}_epoch_{args.epochs}.txt', 'w')
    
    # ----------------------------------
    # case 1: N clients with labeled from all the images --> iid
    # ----------------------------------
    print('case 1: 100 clients with labeled from all the images --> iid')
    print('case 1: 100 clients with labeled from all the images --> iid', file = outputFile)
    args.iid=True
    args.clustering_method = 'single'
    args.multi_center = False
  
    loss_train_iid = run(args, outputFile)
    # ----------------------------------
    # case 2: N clients with labeled from only two image labelss for each client --> noniid
    # ----------------------------------
    print('case 2: 100 clients with labeled from only two images for eahc client --> noniid')
    print('case 2: 100 clients with labeled from only two images for eahc client --> noniid', file = outputFile)
    args.iid = False
    args.clustering_method = 'single'
    args.multi_center = False
     
    loss_train_noniid_noclustering = run(args, outputFile)
    # ----------------------------------
    # case 3: N clients with labeled from only two image labelss for each client --> noniid --> adding clustered fed average
    # ----------------------------------
    print('case 3: 100 clients with labeled from only two images for each client --> noniid --> adding clustered fed average')
    print('case 3: 100 clients with labeled from only two images for each client --> noniid --> adding clustered fed average', file = outputFile)
    args.iid=False
    args.clustering_method = 'umap_central'
    args.multi_center = False
   
    loss_train_noniid_clustering = run(args, outputFile)
    # ----------------------------------
    # case 4: N clients with labeled from only two image labelss for each client --> noniid and multicenter clustering
    # ----------------------------------
    print('case 4: 100 clients with labeled from only two images for eahc client --> noniid')
    print('case 4: 100 clients with labeled from only two images for eahc client --> noniid', file = outputFile)
    args.iid=False
    args.clustering_method = 'single'
    args.multi_center = True
    evaluation_index_step = 1

    loss_train_noniid_multicenter = run(args, outputFile)
    # ----------------------------------
    # Create plots with pre-defined labels.
    fig, ax = plt.subplots()

    ax.plot(loss_train_iid, 'r', label=f'FedAvg, iid data {args.num_users}  clients')# no clustered data, no clustering algo
    ax.plot(loss_train_noniid_noclustering, 'g', label=f'FedAvg, non-iid data, no clustering algo {args.num_users} clients')
    ax.plot(loss_train_noniid_clustering, 'b', label=f'clustered data, {args.clustering_method} clustering algo {args.num_users} clients')

    legend = ax.legend(loc='upper center', fontsize='x-large')
    plt.savefig(f'{args.results_root_dir}/training_accuracy_{args.num_users}_{args.clustering_method}_{args.epochs}.png')
    plt.show()

    outputFile.close()
