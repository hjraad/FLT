#----------------------------------------------------------------------------
# Created By  : Mohammad Abdizadeh & Hadi Jamali-Rad
# Created Date: 23-Nov-2020
# 
# Refactored By: Sayak Mukherjee
# Last Update: 27-Oct-2023
# ---------------------------------------------------------------------------
# File contains the code for FLT.
# ---------------------------------------------------------------------------

import torch
import copy
import logging
import numpy as np
import torch.nn.functional as F

from omegaconf import DictConfig
from datasets.load_dataset import load_dataset
from datasets.utils import DatasetSplit
from torch.utils.data import DataLoader
from comm.fedavg import FedAvg
from datasets import sampling
from pathlib import Path
from models.nets import CNNCifar, CNNLeaf, CNNMnist, MLP
from utils.cluster import extract_clustering, partition_clusters, clustering_multi_center, filter_cluster_partition

logger = logging.getLogger(__name__)

class FLT:

    def __init__(self, config: DictConfig, device) -> None:

        self.config = config
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)

        self.trainset, self.testset, self.dict_train_users, self.dict_test_users, self.cluster = self.init_dataset()
        self.net_list = self.gen_model()

        self.fedMLAlgo()
        
    def init_dataset(self):

        cluster, cluster_length = self.gen_cluster()

        datasets, dataset_sizes, class_names = load_dataset(self.config.dataset.name.upper(), 
                                                            self.config.dataset.path)
        
        dataset_train = datasets['train']
        dataset_test = datasets['test']

        logger.info(f'Configuring {self.config.dataset.name} dataset.')
        
        if self.config.dataset.name in ['mnist', 'MNIST']:
            # sample users
            if self.config.federated.iid:
                dict_train_users = sampling.mnist_iid(dataset_train, 
                                                      self.config.federated.num_users)
                dict_test_users = sampling.cluster_testdata_dict(dataset_test, 
                                                                 self.config.dataset.name, 
                                                                 self.config.federated.num_users, 
                                                                 cluster)
            else:
                dict_train_users = sampling.mnist_noniid_cluster(dataset_train, 
                                                                 self.config.federated.num_users, 
                                                                 cluster)
                dict_test_users = sampling.cluster_testdata_dict(dataset_test, 
                                                                 self.config.dataset.name, 
                                                                 self.config.federated.num_users, 
                                                                 cluster)
        #
        elif self.config.dataset.name in ['emnist', 'EMNIST']:     
            if not self.config.federated.iid:
                dict_train_users = sampling.emnist_noniid_cluster(dataset_train, 
                                                                  self.config.federated.num_users, 
                                                                  cluster, 
                                                                  random_shuffle=True)
                dict_test_users = sampling.cluster_testdata_dict(dataset_test, 
                                                                 self.config.dataset.name, 
                                                                 self.config.federated.num_users, 
                                                                 cluster)
        #       
        elif self.config.dataset.name in ['cifar', 'CIFAR10']:
            if self.config.federated.iid:
                dict_train_users = sampling.cifar_iid(dataset_train, 
                                                      self.config.federated.num_users)
                dict_test_users = sampling.cluster_testdata_dict(dataset_test, 
                                                                 self.config.dataset.name, 
                                                                 self.config.federated.num_users, 
                                                                 cluster)
            else:
                dict_train_users = sampling.cifar_noniid_cluster(dataset_train, 
                                                                 self.config.federated.num_users, 
                                                                 cluster)
                dict_test_users = sampling.cluster_testdata_dict(dataset_test, 
                                                                 self.config.dataset.name, 
                                                                 self.config.federated.num_users, 
                                                                 cluster)
        #
        elif self.config.dataset.name in ['femnist', 'FEMNIST']:
            dict_train_users = dataset_train.dict_users
            dict_test_users = dataset_test.dict_users
        #

        logger.info('Dataset configured.')

        return dataset_train, dataset_test, dict_train_users, dict_test_users, cluster

    def gen_cluster(self):

        # setting the clustering format
        if self.config.federated.iid:
            nr_of_clusters = 1
        
            cluster_length = self.config.federated.num_users // nr_of_clusters
            cluster = np.zeros((nr_of_clusters,10), dtype='int64')
            for i in range(nr_of_clusters):
                # TODO: should it be np.random.choice(10, 2, replace=False) for a fairer comparison?
                cluster[i] = np.random.choice(10, 10, replace=False)

        elif self.config.dataset.name == 'EMNIST':
            nr_of_clusters = self.config.federated.nr_of_embedding_clusters
            cluster_length = self.config.federated.num_users // nr_of_clusters
            n_1 = 47 // (nr_of_clusters - 1)
            n_2 = 47 % n_1
            cluster = np.zeros((nr_of_clusters, n_1), dtype='int64')
            # cluster_array = np.random.choice(47, 47, replace=False)
            cluster_array = np.arange(47)
            for i in range(nr_of_clusters - 1):
                cluster[i] = cluster_array[i*n_1: i*n_1 + n_1]
            cluster[nr_of_clusters - 1][0:n_2] = cluster_array[-n_2:]

        elif self.config.federated.scenario == 2:
            assert self.config.federated.nr_of_embedding_clusters == 5
            cluster_length = self.config.federated.num_users // self.config.federated.nr_of_embedding_clusters
            # generate cluster settings   
            if self.config.federated.flag_with_overlap:
                cluster = np.zeros((self.config.federated.nr_of_embedding_clusters, 3), dtype='int64')
                lst = np.random.choice(10, 10, replace=False) # what is this?
                cluster[0] = lst[0:3]
                cluster[1] = lst[2:5]
                cluster[2] = lst[4:7]
                cluster[3] = lst[6:9]
                cluster[4] = [lst[-2], lst[-1], lst[0]]

            else:
                cluster = np.zeros((self.config.federated.nr_of_embedding_clusters, 2), dtype='int64')
                cluster_array = np.random.choice(10, 10, replace=False)
                for i in range(self.config.federated.nr_of_embedding_clusters):
                    cluster[i] = cluster_array[i*2: i*2 + 2]

        elif self.config.federated.scenario == 3:
            # scenario 3
            assert self.config.federated.nr_of_embedding_clusters == 2
            cluster_length = self.config.federated.num_users // self.config.federated.nr_of_embedding_clusters
            cluster = np.zeros((self.config.federated.nr_of_embedding_clusters, 5), dtype='int64')
            cluster_array = np.random.choice(10, 10, replace=False)
            if self.config.federated.cluster_overlap == 0:
                cluster[0] = cluster_array[0:5]
                cluster[1] = cluster_array[5:]
            elif self.config.federated.cluster_overlap == 20:
                cluster[0] = cluster_array[0:5]
                cluster[1] = cluster_array[4:9]
            elif self.config.federated.cluster_overlap == 40:
                cluster[0] = cluster_array[0:5]
                cluster[1] = cluster_array[3:8]
            elif self.config.federated.cluster_overlap == 60:
                cluster[0] = cluster_array[0:5]
                cluster[1] = cluster_array[2:7]
            elif self.config.federated.cluster_overlap == 80:
                cluster[0] = cluster_array[0:5]
                cluster[1] = cluster_array[1:6]
            elif self.config.federated.cluster_overlap == 100:
                cluster[0] = cluster_array[0:5]
                cluster[1] = cluster_array[0:5]

        elif self.config.dataset.name == 'FEMNIST':
            cluster_length = self.config.federated.num_users
            cluster = list(np.arange(self.config.dataset.num_classes))

        return cluster, cluster_length

    def extract_evaluation_range(self):
        if self.config.federated.iid == True:
            evaluation_index_step = 1
            evaluation_index_max = 1
        elif self.config.dataset.name in ['FEMNIST', 'EMNIST']:
            evaluation_index_step = 1
            evaluation_index_max = self.config.federated.num_users
        elif self.config.federated.clustering_method == 'single' and self.config.federated.multi_center == False:
            evaluation_index_step = self.config.federated.num_users // self.config.federated.nr_of_embedding_clusters# clustering_length
            evaluation_index_max = self.config.federated.num_users
        else:
            evaluation_index_step = 1
            evaluation_index_max = self.config.federated.num_users

        evaluation_index_range = np.arange(0, evaluation_index_max, evaluation_index_step)

        return evaluation_index_range

    def gen_model(self):

        img_size = self.trainset[0][0].shape

        # build model
        if self.config.model.name == 'cnn' and (self.config.dataset.name in ['cifar10', 'CIFAR10']):
            net = CNNCifar(self.config).to(self.device)
        elif self.config.model.name == 'cnn' and (self.config.dataset.name in ['mnist', 'MNIST', 'FEMNIST']):
            net = CNNMnist(self.config).to(self.device)
        elif self.config.model.name == 'mlp':
            len_in = 1
            for x in img_size:
                len_in *= x
            net = MLP(dim_in=len_in, dim_hidden=200, dim_out=self.config.dataset.num_classes).to(self.device)
        elif self.config.model.name == 'cnn_leaf':
            net = CNNLeaf(self.config).to(self.device)
        else:
            exit('Error: unrecognized model')

        net.train()

        # copy weights
        net_list = [copy.deepcopy(net) for i in range(self.config.federated.num_users)]

        return net_list

    def localUpdate(self, user_idx):

        ldr_train = DataLoader(DatasetSplit(self.trainset, self.dict_train_users[user_idx]), 
                               batch_size=self.config.dataset.train_batch_size, 
                               shuffle=True)
        
        net = copy.deepcopy(self.net_list[user_idx])
        net.train()

        optimizer = torch.optim.SGD(net.parameters(), 
                                    lr=self.config.model.lr, 
                                    momentum=self.config.model.momentum)
        loss_func = torch.nn.CrossEntropyLoss()

        logger.info(f'Local training for user {user_idx}')

        epoch_loss = []
        for iter in range(self.config.trainer.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                loss = loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.config.project.verbose and batch_idx % 10 == 0:
                    logger.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(ldr_train.dataset),
                               100. * batch_idx / len(ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net, sum(epoch_loss) / len(epoch_loss)

    def fedMLAlgo(self):

        export_path = Path(self.config.project.path).joinpath(self.config.project.experiment_name)
        if not Path.exists(export_path):
            Path.mkdir(export_path, exist_ok=True)

        outputFile = open(export_path.joinpath('results.csv'), 'w')

        outputFile_log = open(export_path.joinpath('results_allmodels.csv'), 'w')

        print('iteration,training_average_loss,training_accuracy,test_accuracy,training_variance,test_variance', file = outputFile)
    
        print('0, ', end = '', file = outputFile_log)
        evaluation_user_index_range = self.extract_evaluation_range()
        for idx in evaluation_user_index_range:
            print('{:.2f}, '.format(idx), end = '', file = outputFile_log)
        for idx in evaluation_user_index_range:
            print('{:.2f}, '.format(idx), end = '', file = outputFile_log)
        print('', file = outputFile_log)

        clustering_matrix = extract_clustering(self.config, self.dict_train_users, self.trainset, self.cluster, 
                                               0, self.device)

        # training
        loss_train = []
        if self.config.federated.multi_center:
            multi_center_initialization_flag = True
            est_multi_center = []

        if self.config.federated.partition_clusters_flag:
            # hierarchical clustering
            cluster_user_dict = partition_clusters(self.config, clustering_matrix)

        for round in range(self.config.trainer.rounds):

            loss_locals = []
            net_local_list = []
            m = max(int(self.config.federated.frac *  self.config.federated.num_users), 1)
            idxs_users = np.random.choice(range(self.config.federated.num_users), m, replace=False)
            logger.info(f"Local update started for {len(idxs_users)} users")

            for idx in idxs_users:
                
                net, loss = self.localUpdate(user_idx=idx)
                net_local_list.append(copy.deepcopy(net))
                loss_locals.append(copy.deepcopy(loss))

            logger.info(f"Local update finished for {len(idxs_users)} users")

            # update global weights
            if self.config.federated.multi_center:
                clustering_matrix, est_multi_center = clustering_multi_center(self.config, net_local_list, multi_center_initialization_flag, est_multi_center, iter=round+1)
                multi_center_initialization_flag = False
            
            if self.config.federated.partition_clusters_flag:
                # cluster information
                cluster_partitions = filter_cluster_partition(cluster_user_dict, net_local_list)
            else:
                # 1 big cluster for all for compatibility
                cluster_partitions = {1 : (net_local_list, clustering_matrix, np.arange(0,self.config.federated.num_users,1))}

            for cluster_idx, (net_cluster_list, filtered_clustering_matrix, cluster_users) in cluster_partitions.items():
                logger.info(f'FedAvg over cluster {cluster_idx} with {len(net_cluster_list)} users')
                net_cluster_list = FedAvg(net_cluster_list, filtered_clustering_matrix, self.dict_train_users)

                # copy weights to net_glob indexed globally, from averaged models, indexed locally
                for local_idx, global_idx in enumerate(cluster_users):
                    self.net_list[global_idx] = copy.deepcopy(net_cluster_list[local_idx])

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            logger.info(f'Round {round}, Average loss {loss_avg}')
            if (round % self.config.project.iter_to_iter_results) == 0 or (round == self.config.federated.rounds - 1):
                print(f'{round}, {loss_avg}, ', end = '', file = outputFile)
                print(f'{round}, ', end = '', file = outputFile_log)

            loss_train.append(loss_avg)

            if self.config.federated.change_dataset_flag == True:
                if round == (self.config.federated.change_dataset_epoch-1):
                    self.config.federated.flag_with_overlap = True

                    self.trainset, self.testset, self.dict_train_users, self.dict_test_users, self.cluster = self.init_dataset()

                    # clustering the clients
                    clustering_matrix = extract_clustering(self.config, self.dict_train_users, self.trainset, self.cluster, round + 1, self.device)

            if (round % self.config.project.iter_to_iter_results) == 0 or (round == self.config.federated.rounds - 1):
                print(f'iteration under process: {round}')
                self.evaluate_performance(evaluation_user_index_range, outputFile, outputFile_log)

    def localTest(self, user_idx, on_trainset=False):

        if on_trainset:
            data_loader = DataLoader(DatasetSplit(self.trainset, self.dict_train_users[user_idx]), 
                                    batch_size=self.config.dataset.train_batch_size, 
                                    shuffle=True)
        else:
            data_loader = DataLoader(DatasetSplit(self.testset, self.dict_test_users[user_idx]), 
                                    batch_size=self.config.dataset.train_batch_size, 
                                    shuffle=True)
        
        net = copy.deepcopy(self.net_list[user_idx])
        net.eval()

        test_loss = 0
        correct = 0

        l = len(data_loader)
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device) 
            log_probs = net(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
        if self.config.project.verbose:
            logger.info('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(data_loader.dataset), accuracy))
        return accuracy, test_loss

    def evaluate_performance(self, evaluation_user_index_range, outputFile, outputFile_log):

        # evaluate the performance of the models on train and test datasets
        acc_train_final = np.zeros(self.config.federated.num_users)
        loss_train_final = np.zeros(self.config.federated.num_users)
        acc_test_final = np.zeros(self.config.federated.num_users)
        loss_test_final = np.zeros(self.config.federated.num_users)

        sum_weight_training = 0
        sum_weight_test = 0

        # ----------------------------------
        # testing: average over all clients
        for idx in evaluation_user_index_range:
            if idx % (len(evaluation_user_index_range) // 10) == 0:  
                logger.info(f'user under process: {idx}')
            acc_train_final[idx], loss_train_final[idx] = self.localTest(idx, on_trainset=True)
            acc_test_final[idx], loss_test_final[idx] = self.localTest(idx) 
            
            if self.config.federated.weithed_evaluation == True:
                sum_weight_training += len(self.dict_train_users[idx])
                acc_train_final[idx] = acc_train_final[idx] * len(self.dict_train_users[idx])
            
                sum_weight_test += len(self.dict_test_users[idx])
                acc_test_final[idx] = acc_test_final[idx] * len(self.dict_test_users[idx])
                
        if self.config.federated.weithed_evaluation == True:
            training_accuracy = np.sum(acc_train_final[evaluation_user_index_range]) / sum_weight_training
            test_accuracy = np.sum(acc_test_final[evaluation_user_index_range]) / sum_weight_test

            training_variance = np.var(acc_train_final[evaluation_user_index_range]) / sum_weight_training
            test_variance = np.var(acc_test_final[evaluation_user_index_range]) / sum_weight_test
        else:
            training_accuracy = np.mean(acc_train_final[evaluation_user_index_range])
            test_accuracy = np.mean(acc_test_final[evaluation_user_index_range])

            training_variance = np.var(acc_train_final[evaluation_user_index_range])
            test_variance = np.var(acc_test_final[evaluation_user_index_range])

        logger.info('Training accuracy: {:.2f}'.format(training_accuracy))
        logger.info('Testing accuracy: {:.2f}'.format(test_accuracy))

        print('{:.2f}, '.format(training_accuracy), end = '', file = outputFile)
        print('{:.2f}, '.format(test_accuracy), end = '', file = outputFile)
        print('{:.2f}, '.format(training_variance), end = '', file = outputFile)
        print('{:.2f}'.format(test_variance), file = outputFile)

        for idx in evaluation_user_index_range:
            print('{:.2f}, '.format(acc_train_final[idx]), end = '', file = outputFile_log)
        for idx in evaluation_user_index_range:
            print('{:.2f}, '.format(acc_test_final[idx]), end = '', file = outputFile_log)
        print('', file = outputFile_log)

        return