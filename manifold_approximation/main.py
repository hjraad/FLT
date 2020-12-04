'''
E2E Encoding 
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
from sklearn.cluster import KMeans
import umap
from utils.load_datasets import load_dataset

from tqdm import tqdm
from utils.train_AE import train_model
from utils.vis_tools import create_acc_loss_graph
from models.convAE_128D import ConvAutoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------
# Initialization
# ----------------------------------
batch_size = 1
manifold_learning_methods = ['umap']  # tsne, umap, both

manifold_dim = 2

MODEL_NAME = "model-1606927012-epoch40-latent128"
dataset_name = 'MNIST'
dataset_split = 'balanced'

data_root_dir = '../data'
model_root_dir = '../model_weights'
results_root_dir = '../results/Encoder'
log_root_dir = './logs/'

# ----------------------------------
# Reproducability
# ----------------------------------
torch.manual_seed(123)
np.random.seed(321)
umap_random_state=123

# ---------------------
# Check GPU
# ---------------------
torch.cuda.is_available()
device = torch.device("cuda:0")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('runing on GPU')
else:
    device = torch.device("cpu")
    print('runing on CPU')
    
torch.cuda.device_count()

# ----------------------------------
# Load data 
# ----------------------------------
# For now both have no special transformation 
#TODO: test the imapct of transformation later
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

if dataset_name == 'EMNIST' and dataset_split == 'balanced':    
    class_names = ['0',  '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',  'Z',
                'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

# ----------------------------------
# Load AE and extract embedding
# ----------------------------------
model = ConvAutoencoder().to(device)

# specify loss function
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(model_root_dir + MODEL_NAME + '_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# extract the AE embedding of test data
if not os.path.exists(f'{model_root_dir}/AE_embedding_{dataset_name}_{MODEL_NAME}_best.p'):
    embedding_list = []
    labels_list = []
    with torch.no_grad():
        for _, (image, label) in enumerate(tqdm(image_datasets['train'], desc='Extracting AE embedding')):
                image = image.to(device)
                labels_list.append(label) 
                _, embedding = model(image.unsqueeze(0))
                embedding_list.append(embedding.cpu().detach().numpy())
    ae_embedding_np = np.concatenate(embedding_list, axis=0)
    ae_labels_np = np.array(labels_list)
    pickle.dump((ae_embedding_np, ae_labels_np), open(f'{model_root_dir}/AE_embedding_{dataset_name}_{MODEL_NAME}_best.p', 'wb'))
else: 
    ae_embedding_np, ae_labels_np = pickle.load(open(f'{model_root_dir}/AE_embedding_{dataset_name}_{MODEL_NAME}_best.p', 'rb'))

ae_embedding_dim = ae_embedding_np.shape[1]

if manifold_dim == ae_embedding_dim:
    raise AssertionError("We need need manifold learning, AE dim = 2 !")

# ----------------------------------
# Manifold learning (tSNE, UMAP)
# ---------------------------------- 
data_list = [data[0] for data in image_datasets['train']]
data_tensor = torch.cat(data_list, dim=0)
data_2D_np = torch.reshape(data_tensor, (data_tensor.shape[0], -1)).numpy()
labels_np = np.array([data[1] for data in image_datasets['train']])

if (labels_np != ae_labels_np).all():
    raise AssertionError('Order of data samples is shuffled!')
    
if 'tsne' in manifold_learning_methods:
    if not os.path.exists(f'{model_root_dir}/tsne_embedding_{dataset_name}_{MODEL_NAME}_dim{manifold_dim}.p'):
        tsne_embedding = TSNE(n_components=manifold_dim).fit_transform(ae_embedding_np)
        pickle.dump(tsne_embedding, open(f'{model_root_dir}/tsne_embedding_{dataset_name}_{MODEL_NAME}_dim{manifold_dim}.p', 'wb'))
        print('tSNE embedding is extracted.')
    else:
        pickle.load(open(f'{model_root_dir}/tsne_embedding_{dataset_name}_{MODEL_NAME}_dim{manifold_dim}.p', 'rb'))
        print('tSNE embedding is loaded.')
 
if 'umap' in manifold_learning_methods:
    if not os.path.exists(f'{model_root_dir}/umap_embedding_{dataset_name}_{MODEL_NAME}_dim{manifold_dim}.p'):
        umap_reducer = umap.UMAP(n_components=manifold_dim, random_state=umap_random_state)
        umap_embedding = umap_reducer.fit_transform(ae_embedding_np)
        print('UMAP embedding is extracted.')
        pickle.dump(umap_embedding, open(f'{model_root_dir}/umap_embedding_{dataset_name}_{MODEL_NAME}_dim{manifold_dim}.p', 'wb'))
        pickle.dump(umap_reducer, open(f'{model_root_dir}/umap_reducer_{dataset_name}_{MODEL_NAME}_dim{manifold_dim}.p', 'wb'))
    else:
        umap_embedding = pickle.load(open(f'{model_root_dir}/umap_embedding_{dataset_name}_{MODEL_NAME}_dim{manifold_dim}.p', 'rb'))
        print('UMAP embedding is loaded.')
        
if manifold_dim == 2:
    for method in manifold_learning_methods:
        plt.figure()
        df = pd.DataFrame({'x':eval(method + '_embedding[:, 0]'), 'y':eval(method + '_embedding[:, 1]')})
        sns.scatterplot(x='x', y='y', hue=labels_np, palette=sns.color_palette("hls", len(class_names)), 
                        data=df, legend="full", alpha=0.3)
        plt.savefig(f'{results_root_dir}/{method}_scatter_{dataset_name}_{MODEL_NAME}.jpg')
