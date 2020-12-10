'''
Base code forked from https://github.com/shaoxiongji/federated-learning
'''
import matplotlib
# matplotlib.use('Agg')
import sys
sys.path.append("./../")
sys.path.append("./../../")
sys.path.append("./")
sys.path.append("../")

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
from clustering import clustering_dummy, clustering_perfect, clustering_umap, clustering_encoder

from manifold_approximation.models.convAE_128D import ConvAutoencoder

def gen_data(iid, dataset_type, num_users, cluster, cluster_num):
    # load dataset and split users
    if dataset_type == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if iid:
            #TODO: cluster_number doesn't have to be passed in below
            dict_users = mnist_noniid_cluster(dataset_train, num_users, cluster, cluster_num)
        else:
            dict_users = mnist_iid(dataset_train, num_users)
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

def net_gen(dataset, dataset_train, num_users):
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

def FedMLAlgo(net_glob_list, w_glob_list, dataset_train, dict_users, num_users, ar_related):
    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

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
        w_glob_list = FedAvg(w_locals, ar_related)

        # copy weight to net_glob
        for idx in np.arange(num_users): #TODO: fix this
            net_glob_list0 = copy.deepcopy(net_glob_list[0])
            net_glob_list0.load_state_dict(w_glob_list[idx])
            net_glob_list[idx] = net_glob_list0

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
    
    return loss_train, net_glob_list

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.gpu = 0
    args.all_clients = True
    args.iid=True
    args.frac=0.1
    args.lr=0.01
    args.num_users=20#100
    args.seed=1
    args.epochs=10
    args.num_classes = 10
    # ----------------------------------
    num_users=40
    manifold_dim = 2
    model_name = "model-1606927012-epoch40-latent128"
    data_root_dir = '../data'
    model_root_dir = './model_weights'
        
    # Load the model ckpt
    model = ConvAutoencoder().to(args.device)

    # Load the model ckpt
    checkpoint = torch.load(f'{model_root_dir}/{model_name}_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # TODO: clean up umap_mo & umap
    clustering_method = 'encoder'#'perfect','umap_mo','umap','encoder'
    results_root_dir = './results/main_fed'
    
    outputFile = open(f'{results_root_dir}/results_{num_users}_{clustering_method}.txt', 'w')
    
    # ----------------------------------
    # case 1: N clients with labeled from all the images --> iid
    #TODO: fix the naming of iid here (seems like it has to be removed)
    iid=False# change the naming!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    num_users=40
    
    # TODO: cluster_num => nr_of_clusters, cluster_length => cluster_size
    cluster_num = 1
    cluster_length = num_users // cluster_num
    cluster = np.zeros((cluster_num,10), dtype='int64')
    for i in range(cluster_num):
        # TODO: should it be np.random.choice(10, 2, replace=False) for a fairer comparison?
        cluster[i] = np.random.choice(10, 10, replace=False)
        
    dataset='mnist'
    dataset_train, dataset_test, dict_users = gen_data(iid, dataset, num_users, cluster, cluster_num)

    #average over all clients
    clustering_matrix = clustering_dummy(num_users)

    net_glob, w_glob, net_glob_list, w_glob_list = net_gen(dataset, dataset_train, num_users)
    loss_train, net_glob_list = FedMLAlgo(net_glob_list, w_glob_list, dataset_train, dict_users, num_users, clustering_matrix)

    # testing
    acc_train_final = np.zeros(num_users)
    loss_train_final = np.zeros(num_users)
    acc_test_final = np.zeros(num_users)
    loss_test_final = np.zeros(num_users)
    if True:#idx in np.arange(1):#TODO: no need to loop over users
        idx = 0
        print(idx)
        #net_glob_list[idx].eval()
        acc_train_final[idx], loss_train_final[idx] = test_img_classes(net_glob_list[idx], dataset_train, cluster[idx//cluster_length], args)
        acc_test_final[idx], loss_test_final[idx] = test_img_classes(net_glob_list[idx], dataset_test, cluster[idx//cluster_length], args)
    print('Training accuracy: {:.2f}'.format(acc_train_final[0]))
    print('Testing accuracy: {:.2f}'.format(acc_test_final[0]))

    print('case 1: 100 clients with labeled from all the images --> iid', file = outputFile)
    print('Training accuracy: {:.2f}'.format(acc_train_final[0]), file = outputFile)
    print('Testing accuracy: {:.2f}'.format(acc_test_final[0]), file = outputFile)
    
    loss_train_iid = loss_train
    # ----------------------------------
    # case 2: N clients with labeled from only two image labelss for each client --> noniid
    # ----------------------------------
    iid=True
    num_users=40
    
    cluster_num = 5
    cluster_length = num_users // cluster_num
    cluster = np.zeros((cluster_num,2), dtype='int64')
    for i in range(cluster_num):
        cluster[i] = np.random.choice(10, 2, replace=False)

    dataset='mnist'
    dataset_train, dataset_test, dict_users = gen_data(iid, dataset, num_users, cluster, cluster_num)
    
    #average over all clients
    # TODO: optimize this
    clustering_matrix = clustering_dummy(num_users)
    
    net_glob, w_glob, net_glob_list, w_glob_list = net_gen(dataset, dataset_train, num_users)
    loss_train, net_glob_list = FedMLAlgo(net_glob_list, w_glob_list, dataset_train, dict_users, num_users, clustering_matrix)

    # testing
    acc_train_final = np.zeros(num_users)
    loss_train_final = np.zeros(num_users)
    acc_test_final = np.zeros(num_users)
    loss_test_final = np.zeros(num_users)
    for idx in np.arange(0, num_users-1, cluster_length):#TODO: no need to loop over all users
        print(idx)
        #net_glob_list[idx].eval()
        acc_train_final[idx], loss_train_final[idx] = test_img_classes(net_glob_list[0], dataset_train, cluster[idx//cluster_length], args)
        acc_test_final[idx], loss_test_final[idx] = test_img_classes(net_glob_list[0], dataset_test, cluster[idx//cluster_length], args)
    print('Training accuracy: {:.2f}'.format(np.average(acc_train_final[np.arange(0, num_users-1, cluster_length)])))
    print('Testing accuracy: {:.2f}'.format(np.average(acc_test_final[np.arange(0, num_users-1, cluster_length)])))

    print('case 2: 100 clients with labeled from only two images for eahc client --> noniid', file = outputFile)
    print('Training accuracy: {:.2f}'.format(np.average(acc_train_final[np.arange(0, num_users-1, cluster_length)])), file = outputFile)
    print('Testing accuracy: {:.2f}'.format(np.average(acc_test_final[np.arange(0, num_users-1, cluster_length)])), file = outputFile)
       
    loss_train_noniid_noclustering = loss_train
    # ----------------------------------
    # case 3: N clients with labeled from only two image labelss for each client --> noniid --> adding clustered fed average
    # ----------------------------------
    iid=True
    num_users=40
    
    cluster_num = 5
    cluster_length = num_users // cluster_num
    
    #average over clients in a same cluster
    if clustering_method == 'perfect':#'perfect','umap_mo','umap','encoder'
        clustering_matrix = clustering_perfect(num_users, dict_users, dataset_train, args)
    elif clustering_method == 'umap_mo':
        clustering_matrix, _, _ = clustering_umap(num_users, dict_users, dataset_train, args)
    elif clustering_method == 'encoder':
        clustering_matrix, _, _, _ = clustering_encoder(num_users, dict_users, dataset_train, model, 
                                                 model_name, model_root_dir, manifold_dim, args)
            
    net_glob, w_glob, net_glob_list, w_glob_list = net_gen(dataset, dataset_train, num_users)
    loss_train, net_glob_list = FedMLAlgo(net_glob_list, w_glob_list, dataset_train, dict_users, num_users, clustering_matrix)

    # testing
    acc_train_final = np.zeros(num_users)
    loss_train_final = np.zeros(num_users)
    acc_test_final = np.zeros(num_users)
    loss_test_final = np.zeros(num_users)
    for idx in np.arange(0,num_users-1,cluster_length):#TODO: no need to loop over all the users!
        print("cluster under process: ", idx//cluster_length)
        net_glob_list[idx].eval()
        acc_train_final[idx], loss_train_final[idx] = test_img_classes(net_glob_list[idx], dataset_train, cluster[idx//cluster_length], args)
        acc_test_final[idx], loss_test_final[idx] = test_img_classes(net_glob_list[idx], dataset_test, cluster[idx//cluster_length], args)
    print('Training accuracy: {:.2f}'.format(np.average(acc_train_final[np.arange(0,num_users-1,cluster_length)])))
    print('Testing accuracy: {:.2f}'.format(np.average(acc_test_final[np.arange(0,num_users-1,cluster_length)])))

    print('case 3: 100 clients with labeled from only two images for each client --> noniid --> adding clustered fed average', file = outputFile)
    print('Training accuracy: {:.2f}'.format(np.average(acc_train_final[np.arange(0,num_users-1,cluster_length)])), file = outputFile)
    print('Testing accuracy: {:.2f}'.format(np.average(acc_test_final[np.arange(0,num_users-1,cluster_length)])), file = outputFile)
    
    loss_train_noniid_clustering = loss_train
    # Create plots with pre-defined labels.
    fig, ax = plt.subplots()

    ax.plot(loss_train_iid, 'r', label=f'FedAvg, iid data {num_users}  clients')# no clustered data, no clustering algo
    ax.plot(loss_train_noniid_noclustering, 'g', label=f'FedAvg, non-iid data, no clustering algo {num_users} clients')
    ax.plot(loss_train_noniid_clustering, 'b', label=f'clustered data, {clustering_method} clustering algo {num_users} clients')

    legend = ax.legend(loc='upper center', fontsize='x-large')
    plt.savefig(f'{results_root_dir}/training_accuracy_{num_users}_{clustering_method}.png')
    plt.show()

    outputFile.close()
