'''
E2E Encoding and Clustering 
@Author: Hadi Jamali-Rad
@e-mail: h.jamali.rad@gmail.com
'''

from __future__ import print_function, division

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
import umap
from utils.load_datasets import load_dataset

from tqdm import tqdm
from utils.train_AE import train_model
from utils.vis_tools import create_acc_loss_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------
# Initialization
# ----------------------------------
TRAIN_FLAG = False  # train or not?
manifold_FLAG = both  # tsne, umpa, both

MODEL_NAME = "model-1606927012-epoch40-latent128"
 
batch_size = 20

dataset_name = 'MNIST'
# dataset_split = 'balanced'
# train_val_split = (100000, 12800)

data_root_dir = '../data'
model_root_dir = "./model_weights/"
results_root_dir = '../results/Encoder'
log_root_dir = './logs/'

# which model to use? 
from models.convAE_128D import ConvAutoencoder

# ----------------------------------
# Reproducability
# ----------------------------------
torch.manual_seed(123)
np.random.seed(321)
umap_random_state=123



checkpoint = torch.load(model_root_dir + MODEL_NAME + '_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
    
    # Load embddings of the contractie AE
    ae_embeddings_np, ae_labels_np = pickle.load(open(f'{model_root_dir}/AE_embedding_{dataset_name}_{MODEL_NAME}_best.p', 'rb'))



# ----------------------------------
# Manifold learning
# ---------------------------------- 
test_data_list = [data[0] for data in image_datasets['test']]
test_data_tensor = torch.cat(test_data_list, dim=0)
test_data_tensor_2D_np = torch.reshape(test_data_tensor, (test_data_tensor.shape[0], -1)).numpy()
test_labels_np = np.array([data[1] for data in image_datasets['test']])

# extract the AE embeddings of test data
embeddings_list = []
labels_list = []
with torch.no_grad():
    for i, (images, labels) in enumerate(tqdm(dataloaders['test'], desc='')):
            images = images.to(device)
            labels_list.append(labels.cpu().numpy()) 
            _, embeddings = model(images)
            embeddings_list.append(embeddings.cpu().detach().numpy())
embeddings_np = np.concatenate(embeddings_list, axis=0)
test_labels = np.concatenate(labels_list)

if (test_labels != test_labels_np).all():
    raise AssertionError('dataloader is shuffling at random - set Shuffle=False')
    
if TSNE_FLAG:
    # ------------- Use tSNE for dimensionality reduction  ------------
    if latent_size != 2:
        embeddings_tsne = TSNE(n_components=2).fit_transform(embeddings_np)
        # plt.figure()
        # plt.scatter(X_embedded[:,0], X_embedded[:,1], c=test_labels_tensor_np, 
        #             s=8, cmap='tab10', label=classes)
        # plt.legend()
        # plt.savefig(f'{results_root_dir}/scatter_{MODEL_NAME}.jpg')
        
        plt.figure()
        df = pd.DataFrame({'x':embeddings_tsne[:,0], 'y':embeddings_tsne[:,1]})
        sns.scatterplot(x='x', y='y', hue=test_labels, 
                        palette=sns.color_palette("hls", len(class_names)), 
                        data=df, legend="full", alpha=0.3)
        plt.savefig(f'{results_root_dir}/tsne_scatter_{dataset_name}_{MODEL_NAME}.jpg')
    elif latent_size == 2:
        print('Latent dim = 2, no need for dimensionality reduction!')
        
if UMAP_FLAG:
    # ------------- Use Umap for dimensionality reduction  ------------
    if latent_size != 2:
        if os.path.exists(f'{model_root_dir}/umap_reducer_{dataset_name}.p'):
            # load the reducer and calculate the embedding of the test data
            reducer = pickle.load(open(f'{model_root_dir}/umap_reducer_{dataset_name}.p', 'rb'))
            #TODO: we need to train the umap on embeddings of the AE to be fair here
            embeddings_umap = reducer.transform(test_data_tensor_2D_np)
            
            plt.figure()
            df = pd.DataFrame({'x':embeddings_umap[:,0], 'y':embeddings_umap[:,1]})
            sns.scatterplot(x='x', y='y', hue=test_labels_np, 
                            palette=sns.color_palette("hls", len(class_names)), 
                            data=df, legend="full", alpha=0.3)
            plt.savefig(f'{results_root_dir}/umap_scatter_{dataset_name}_{MODEL_NAME}.jpg')
            
        else:
            print(f"The reducer for {dataset_name} doesn't exist, train the UMAP separately!")
            print('-'*30)
            
    elif latent_size == 2:
        print('Latent dim = 2, no need for dimensionality reduction!')
        
    
