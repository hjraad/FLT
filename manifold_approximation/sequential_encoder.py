'''
Sequential Encoder (AE + manifold approximation)
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
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import umap
from manifold_approximation.utils.load_datasets import load_dataset
from  manifold_approximation.utils.train_AE import train_model
from manifold_approximation.models.convAE_128D import ConvAutoencoder

from tqdm import tqdm


class Sequential_Encoder():
    '''
    Sequential Encoder class for clustering 
    This takes a pre-trained model and further fine tunes it
    on input data. 
    '''
    def __init__(self, ae_model, ae_opt, criterion, scheduler, nr_epochs, ae_model_name, model_root_dir, log_root_dir,
                                    manifold_dim, image_dataset, client_name, pre_trained_dataset, batch_size=64,  
                                    dataset_name='', train_umap=False, use_AE=True, device='cpu'):
        self.ae_model = ae_model
        self.ae_opt = ae_opt
        self.criterion = criterion
        self.lr_scheduler = scheduler
        self.nr_epochs = nr_epochs
        self.ae_model_name = ae_model_name
        self.model_root_dir = model_root_dir
        self.log_root_dir = log_root_dir
        self.manifold_dim = manifold_dim
        self.image_dataset = image_dataset
        self.client_name = client_name
        self.pre_trained_dataset = pre_trained_dataset
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.train_umap = train_umap
        self.use_AE = use_AE
        self.device = device
    
    #TODO: make it static method
    def autoencoder(self):
        # load the model
        # checkpoint = torch.load(f'{self.model_root_dir}/{self.ae_model_name}_last.pt')
        # self.ae_model.load_state_dict(checkpoint['model_state_dict'])
        # self.ae_opt.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        
        self.new_model_name = f"model-{self.dataset_name}-{int(time.time())}-epoch{self.nr_epochs}-client{self.client_name}" # use time to make the name unique

        # make a dataloader from the image dataset for training
        dataloaders = {'train': DataLoader(self.image_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)}
        dataset_sizes = {'train': len(self.image_dataset)}
        phases = ['train']
        
        if self.nr_epochs == 0:
            model = self.ae_model
        else:
            _, model_l = train_model(self.ae_model, self.new_model_name, dataloaders, dataset_sizes, phases, 
                                        self.criterion, self.ae_opt, self.lr_scheduler, num_epochs=self.nr_epochs, 
                                        model_save_dir=self.model_root_dir, log_save_dir=self.log_root_dir, save_flag=False, device=self.device)
            model = model_l
        
        # extract the AE embedding of test data
        # if not os.path.exists(f'{self.model_root_dir}/AE_embedding_{self.dataset_name}_{self.ae_model_name}_client{self.client_name}.p'):
        if self.use_AE:
            embedding_list = []
            labels_list = []
            with torch.no_grad():
                for _, (image, label) in enumerate(tqdm(self.image_dataset, desc=f'Extracting AE embedding of client_{self.client_name}')):
                        image = image.to(self.device)
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
        # self.ae_embedding_np, self.ae_labels_np = pickle.load(open(f'{self.model_root_dir}/AE_embedding_{self.dataset_name}_{self.ae_model_name}.p', 'rb'))

        # ae_embedding_dim = ae_embedding_np.shape[1]
        
    def manifold_approximation_umap(self):
        # check if manifold approximation is needed
        if self.use_AE and self.manifold_dim == self.ae_embedding_np.shape[1]:
            raise AssertionError("We don't need manifold learning, AE dim = 2 !")
        
        #TODO: do this once up there, and function input!    
        data_list = [data[0] for data in self.image_dataset]
        data_tensor = torch.cat(data_list, dim=0)
        data_2D_np = torch.reshape(data_tensor, (data_tensor.shape[0], -1)).numpy()
        labels_np = np.array([data[1] for data in self.image_dataset])
        
        if self.use_AE: 
            if (labels_np != self.ae_labels_np).all():
                raise AssertionError('Order of data samples is shuffled!')
            
            print('Using AE for E2E encoding ...')
            umap_data = self.ae_embedding_np
            umap_model_address = f'{self.model_root_dir}/umap_reducer_{self.pre_trained_dataset}_{self.ae_model_name}.p'
        else:
            print('AE not used in this scenario ...')
            umap_data = data_2D_np
            umap_model_address = f'{self.model_root_dir}/umap_reducer_{self.pre_trained_dataset}.p'
            
        # if not os.path.exists(f'{self.model_root_dir}/umap_embedding_{self.dataset_name}_{self.ae_model_name}_{self.manifold_dim}D.p'):
        if self.train_umap:
            print('Training UMAP on AE embedding ...')
            self.umap_reducer = umap.UMAP(n_components=self.manifold_dim, random_state=42)
            self.umap_embedding = self.umap_reducer.fit_transform(umap_data)
            print(f'UMAP embedding for client_{self.client_name} is extracted.')
            # pickle.dump(self.umap_embedding, open(f'{self.model_root_dir}/umap_embedding_{self.dataset_name}_{self.ae_model_name}_client{self.client_name}_{self.manifold_dim}D.p', 'wb'))
            # pickle.dump(self.umap_reducer, open(f'{self.model_root_dir}/umap_reducer_{self.dataset_name}_{self.ae_model_name}_client{self.client_name}_{self.manifold_dim}D.p', 'wb'))
        else:
            self.umap_reducer = pickle.load(open(umap_model_address, 'rb'))
            print('Loading UMAP embedding ...')
            # only load the transfrom and use it as a function
            self.umap_embedding = self.umap_reducer.transform(umap_data)   
            print(f'UMAP embedding/reducer for client_{self.client_name} is loaded.')
    
# unit test
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    batch_size = 1
    manifold_dim = 2
    model_name = "model-1607623811-epoch40-latent128"
    pre_trained_dataset = 'FMNIST'  
    dataset_name = 'MNIST' # testing on MNIST
    data_root_dir = '../data'
    results_root_dir = '../results/Encoder'
    model_root_dir = '../model_weights'
    log_root_dir = '../logs'
    client_name = 1
    nr_epochs = 3
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
    
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Decay LR by a factor of x*gamma every step_size epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # loss
    criterion = nn.BCELoss()

    # sequential encoding
    encoder = Sequential_Encoder(model, optimizer, criterion, exp_lr_scheduler, nr_epochs, model_name, 
                                 model_root_dir, log_root_dir, manifold_dim, image_dataset, client_name, pre_trained_dataset)
    
    
    encoder.autoencoder()
    encoder.manifold_approximation_umap()
    reducer = encoder.umap_reducer
    umap_embedding = encoder.umap_embedding
    
    fig, ax = plt.subplots(figsize=(12, 10))
    labels_np = np.array([data[1] for data in image_datasets['train']])
    # df = pd.DataFrame({'x':umap_embedding[:, 0], 'y':umap_embedding[:, 1]})
    # sns.scatterplot(x='x', y='y', hue=labels_np, palette=sns.color_palette("hls", len(class_names)), 
    #                 data=df, legend="full", alpha=0.3)
    color = labels_np.astype(int)
    plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=color, cmap='Spectral', s=0.1)
    plt.setp(ax, xticks=[], yticks=[])
    plt.savefig(f'{results_root_dir}/umap_scatter_{dataset_name}_{model_name}.jpg')
    plt.show()
    
    
    