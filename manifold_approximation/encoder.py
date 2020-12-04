'''
Encoder Class (AE plus manifold approximation)
@Author: Hadi Jamali-Rad
@e-mail: h.jamali.rad@gmail.com
'''

from __future__ import print_function, division
import sys
sys.path.append("./../")
sys.path.append("./../../")

import numpy as np
import pandas as pd
import os
import time
import copy
import sys
import pickle

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import umap
from manifold_approximation.utils.load_datasets import load_dataset

from tqdm import tqdm
from manifold_approximation.utils.train_AE import train_model
from manifold_approximation.utils.vis_tools import create_acc_loss_graph
from manifold_approximation.models.convAE_128D import ConvAutoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder():
    '''
    Encoder class for clustering 
    '''
    def __init__(self, ae_model, ae_model_name, model_root_dir,
                                    manifold_dim, image_dataset, client_name, dataset_name='MNIST'):
        self.ae_model = ae_model
        self.ae_model_name = ae_model_name
        self.model_root_dir = model_root_dir
        self.manifold_dim = manifold_dim
        self.image_dataset = image_dataset
        self.client_name = client_name
        self.dataset_name = dataset_name
    
    #TODO: make it static method
    def autoencoder(self):
        # model definition
        model = self.ae_model

        # extract the AE embedding of test data
        # if not os.path.exists(f'{self.model_root_dir}/AE_embedding_{self.dataset_name}_{self.ae_model_name}_client{self.client_name}.p'):
        embedding_list = []
        labels_list = []
        with torch.no_grad():
            for _, (image, label) in enumerate(tqdm(self.image_dataset, 
                                                    desc=f'Extracting AE embedding of client_{self.client_name}')):
                    image = image.to(device)
                    labels_list.append(label) 
                    _, embedding = model(image.unsqueeze(0))
                    embedding_list.append(embedding.cpu().detach().numpy())
        self.ae_embedding_np = np.concatenate(embedding_list, axis=0)
        self.ae_labels_np = np.array(labels_list)
        # pickle.dump((self.ae_embedding_np, self.ae_labels_np), 
        #             open(f'{self.model_root_dir}/AE_embedding_{self.dataset_name}_{self.ae_model_name}_client{self.client_name}.p', 'wb'))
        # else: 
        #     self.ae_embedding_np, self.ae_labels_np = pickle.load(open(f'{self.model_root_dir}/AE_embedding_{self.dataset_name}_{self.ae_model_name}_client{self.client_name}.p', 'rb'))
        #     print(f'AE embedding for client{self.client_name} is loaded!')

        # ae_embedding_dim = ae_embedding_np.shape[1]

    #TODO: 1) make this static method
    #TODO: 2) consider tSNE and other manifold approximation methods
    def manifold_approximation_umap(self):
        # check if manifold approximation is needed
        if self.manifold_dim == self.ae_embedding_np.shape[1]:
            raise AssertionError("We need need manifold learning, AE dim = 2 !")

        data_list = [data[0] for data in self.image_dataset]
        data_tensor = torch.cat(data_list, dim=0)
        data_2D_np = torch.reshape(data_tensor, (data_tensor.shape[0], -1)).numpy()
        labels_np = np.array([data[1] for data in self.image_dataset])

        if (labels_np != self.ae_labels_np).all():
            raise AssertionError('Order of data samples is shuffled!')
    
        # if not os.path.exists(f'{self.model_root_dir}/umap_embedding_{self.dataset_name}_{self.ae_model_name}_{self.manifold_dim}D.p'):
        self.umap_reducer = umap.UMAP(n_components=self.manifold_dim, random_state=42)
        self.umap_embedding = self.umap_reducer.fit_transform(self.ae_embedding_np)
        print(f'UMAP embedding for client_{self.client_name} is extracted.')
            # pickle.dump(self.umap_embedding, open(f'{self.model_root_dir}/umap_embedding_{self.dataset_name}_{self.ae_model_name}_client{self.client_name}_{self.manifold_dim}D.p', 'wb'))
            # pickle.dump(self.umap_reducer, open(f'{self.model_root_dir}/umap_reducer_{self.dataset_name}_{self.ae_model_name}_client{self.client_name}_{self.manifold_dim}D.p', 'wb'))
        # else:
        #     self.umap_embedding = pickle.load(open(f'{self.model_root_dir}/umap_embedding_{self.dataset_name}_{self.ae_model_name}_client{self.client_name}_{self.manifold_dim}D.p', 'rb'))
        #     self.umap_reducer = pickle.load(open(f'{self.model_root_dir}/umap_reducer_{self.dataset_name}_{self.ae_model_name}_client{self.client_name}_{self.manifold_dim}D.p', 'rb'))
        #     print(f'UMAP embedding/reducer for client{self.client_name} is loaded.')

# unit test
if __name__ == '__main__':
    #
    batch_size = 1
    manifold_dim = 2
    model_name = "model-1606927012-epoch40-latent128"
    dataset_name = 'MNIST'
    data_root_dir = '../data'
    model_root_dir = '../model_weights'
    client_name = 1
    dataset_split = 'balanced'
    
    # Load sample data 
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.ToTensor()
        ]),
    }

    dataloaders, image_datasets, dataset_sizes, class_names = load_dataset(dataset_name, data_root_dir, data_transforms, 
                                                                        batch_size=batch_size, shuffle_flag=False, 
                                                                        dataset_split=dataset_split)

    image_dataset = image_datasets['train']
    
    if dataset_name == 'EMNIST' and dataset_split == 'balanced':    
        class_names = ['0',  '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9',
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                    'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',  'Z',
                    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

    # model
    model = ConvAutoencoder().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Load the model ckpt
    checkpoint = torch.load(f'{model_root_dir}/{model_name}_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']


    encoder = Encoder(model, model_name, model_root_dir, 
                                manifold_dim, image_dataset, client_name)
    
    encoder.autoencoder()
    encoder.manifold_approximation_umap()
    reducer = encoder.umap_reducer
    