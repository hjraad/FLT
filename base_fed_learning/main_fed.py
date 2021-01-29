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
from clustering import clustering_single, clustering_perfect, clustering_umap, clustering_encoder, clustering_umap_central

from manifold_approximation.models.convAE_128D import ConvAutoencoder

import os
import argparse

from utils.args import parse_args
from utils.model_utils import read_data
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.mnist import MNIST
import warnings
from PIL import Image

# ----------------------------------
# Reproducability
# ----------------------------------
torch.manual_seed(123)
np.random.seed(321)
umap_random_state=42
from torchvision.datasets.utils import download_url, download_and_extract_archive, extract_archive, \
    verify_str_arg
from torchvision.datasets import MNIST, utils
from PIL import Image
import os.path
import torch
from torchvision.datasets.mnist import read_image_file, read_label_file
class FEMNIST(VisionDataset):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``FEMNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    resources = [
        ('https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz',
         '59c65cec646fc57fe92d27d83afdf0ed')
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes =  ['0',  '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9',
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                    'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',  'Z',
                    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(FEMNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets, _ = torch.load(os.path.join(self.processed_folder, data_file))

        train_data_dir = os.path.join('..', 'data', 'femnist', 'FEMNIST', 'train')
        test_data_dir = os.path.join('..', 'data', 'femnist', 'FEMNIST', 'test')  
        self.dict_users = {}

        if self.train == True:
            self.users, groups, self.data = read_data(train_data_dir, test_data_dir, train_flag = True)
        else:
            self.users, groups, self.data = read_data(train_data_dir, test_data_dir, train_flag = False)



        counter = 0        
        for i in range(len(self.users)):
            lst = list(counter + np.arange(len(self.data[self.users[i]]['y'])))
            self.dict_users.update({i: set(lst)})
            counter = lst[-1] + 1#+= len(self.data[self.users[i]]['y'])
        print(len(self.dict_users))
        self.dict_index = {}
        sum = 0
        for i in range(len(self.users)):
            sum += len(self.data[self.users[i]]['y'])
        print(sum)

        counter = 0
        for i in range(len(self.users)):
            for j in range(len(self.data[self.users[i]]['y'])):
                self.dict_index[counter] = [i, j]
                counter += 1
        self.length_data = counter
        self.num_classes = 100
        self.n_classes = 100

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img0, target0 = self.data[index], int(self.targets[index])
        [i, j] = self.dict_index[index]
        img, target = self.data[self.users[i]]['x'][j], int(self.data[self.users[i]]['y'][j])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img0 = Image.fromarray(img0.numpy(), mode='L')
        img = Image.fromarray(np.array(img).reshape(28,28), mode='L')

        if self.transform is not None:
            #img0 = self.transform(img0)
            img = self.transform(img)

        if self.target_transform is not None:
            #target0 = self.target_transform(target0)
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length_data

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')
        """
        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)
        """
        os.replace(os.path.join(self.raw_folder, 'training.pt'), os.path.join(self.processed_folder, 'training.pt'))
        os.replace(os.path.join(self.raw_folder, 'test.pt'), os.path.join(self.processed_folder, 'test.pt'))

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

def gen_data(iid, dataset_type, num_users, cluster):
    # load dataset and split users
    if dataset_type == 'mnist':
        # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trans_mnist = transforms.Compose([transforms.ToTensor()])
        dataset_train = MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if iid:
            dict_users = mnist_iid(dataset_train, num_users)
        else:
            dict_users = mnist_noniid_cluster(dataset_train, num_users, cluster)
    elif dataset_type == 'femnist':
        # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trans_mnist = transforms.Compose([transforms.ToTensor()])
        dataset_train = FEMNIST('../data/femnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = FEMNIST('../data/femnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        dict_users_train = dataset_train.dict_users
        dict_users_test = dataset_test.dict_users
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

    return dataset_train, dataset_test, dict_users_train, dict_users_test

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

def FedMLAlgo(net_glob_list, w_glob_list, dataset_train, dict_users, num_users, clustering_matrix):
    # training
    loss_train = []

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
    
    return loss_train, net_glob_list

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.num_classes = 100# TODO: fix this
    # ----------------------------------
    plt.close('all')
    
    # open the output file to write the results to
    outputFile = open(f'{args.results_root_dir}/main_fed/results_{args.num_users}_{args.clustering_method}.txt', 'w')
    
    # ----------------------------------
    # case 1: N clients with labeled from all the images --> iid
    # ----------------------------------
    args.iid=True
    
    # setting the clustering format
    nr_of_clusters = 1
    cluster_length = args.num_users // nr_of_clusters
    cluster = np.zeros((nr_of_clusters,10), dtype='int64')
    for i in range(nr_of_clusters):
        # TODO: should it be np.random.choice(10, 2, replace=False) for a fairer comparison?
        cluster[i] = np.random.choice(10, 10, replace=False)
    
    # ----------------------------------
    # case 2: N clients with labeled from all the images --> iid
    # ----------------------------------
    #dataset_train0, dataset_test0, dict_users0 = gen_data(False, 'femnist', 100, [])
        
    dataset_train, dataset_test, dict_users_train, dict_users_test = gen_data(args.iid, 'femnist', args.num_users, cluster)
    args.num_users = len(dict_users_train)#
    #dd = FEMNIST()
    #print(dataset_train[100])

    # clustering the clients
    clustering_matrix = clustering_single(args.num_users)

    net_glob, w_glob, net_glob_list, w_glob_list = gen_model(args.dataset, dataset_train, args.num_users)
    loss_train, net_glob_list = FedMLAlgo(net_glob_list, w_glob_list, dataset_train, dict_users_train, args.num_users, clustering_matrix)

    # testing: average over all clients
    # testing: average over clients in a same cluster
    acc_train_final = np.zeros(args.num_users)
    loss_train_final = np.zeros(args.num_users)
    acc_test_final = np.zeros(args.num_users)
    loss_test_final = np.zeros(args.num_users)
    for idx in np.arange(0,args.num_users-1):#TODO: no need to loop over all the users!
        print("user under process: ", idx)
        #print(list(dict_users_train[idx]))
        net_glob_list[idx].eval()
        acc_train_final[idx], loss_train_final[idx] = test_img_classes(net_glob_list[idx], dataset_train, list(dict_users_train[idx]), args)
        acc_test_final[idx], loss_test_final[idx] = test_img_classes(net_glob_list[idx], dataset_test, list(dict_users_test[idx]), args)
    print('Training accuracy: {:.2f}'.format(np.average(acc_train_final[np.arange(0,args.num_users-1,cluster_length)])))
    print('Testing accuracy: {:.2f}'.format(np.average(acc_test_final[np.arange(0,args.num_users-1,cluster_length)])))

