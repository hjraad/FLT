
'''
Train Umap based on the daset of choice
See also => #https://umap-learn.readthedocs.io/en/latest/transform.html
'''

from __future__ import division, print_function

import sys

sys.path.append("./../")
sys.path.append("./../../")

import copy
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import umap
from matplotlib import style
from sklearn.manifold import TSNE
from torch.optim import lr_scheduler
from torchvision import transforms
from tqdm import tqdm

from manifold_approximation.utils.load_datasets import load_dataset
from manifold_approximation.utils.train_AE import train_model
from manifold_approximation.models.convAE_128D import ConvAutoencoder
from manifold_approximation.models.convAE_cifar import ConvAutoencoderCIFAR

from base_fed_learning.utils.utils import extract_model_name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------
# Initialization
# ----------------------------------
batch_size = 1
num_workers = 0

train_on_AE_embedding = False

dataset_name = 'CIFAR110'
dataset_split = 'balanced'

data_root_dir = '../data'
model_root_dir = '../model_weights'
results_root_dir = '../results/UMAP'

# find the model name automatically
MODEL_NAME = extract_model_name(model_root_dir, dataset_name)
# MODEL_NAME = 'model-1607623811-epoch40-latent128'

# ----------------------------------
# Reproducability
# ----------------------------------
torch.manual_seed(123)
np.random.seed(321)
umap_random_state=42

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

# ---------------------
# Load data
# ---------------------
# For now both have no special transformation 
#TODO: test the imapct of transformation later
data_transforms = {'train': transforms.Compose([transforms.ToTensor()]),
    'test': transforms.Compose([transforms.ToTensor()])
    }

dataloaders, image_datasets, dataset_sizes, class_names = load_dataset(dataset_name, data_root_dir, data_transforms, 
                                                                    batch_size=batch_size, dataset_split=dataset_split)

# ----------------------------------
# Load data or embedding 
# ----------------------------------
if train_on_AE_embedding:
    if not os.path.exists(f'{model_root_dir}/AE_embedding_{MODEL_NAME}.p'):
        # load the model
        model = ConvAutoencoder().to(device)
        checkpoint = torch.load(f'{model_root_dir}/{MODEL_NAME}_best.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        # extract embeddings based on the frozen AE model 
        embedding_list = []
        labels_list = []
        with torch.no_grad():
            for _, (image, label) in enumerate(tqdm(image_datasets['train'], desc='Inferencing training embedding')):
                    image = image.to(device)
                    labels_list.append(label) 
                    _, embedding = model(image.unsqueeze(0))
                    embedding_list.append(embedding.cpu().detach().numpy())
        ae_embedding_np = np.concatenate(embedding_list, axis=0)
        ae_labels_np = np.array(labels_list)
        pickle.dump((ae_embedding_np, ae_labels_np), open(f'{model_root_dir}/AE_embedding_{MODEL_NAME}.p', 'wb'))
        train_data_2D_np, train_labels_np = ae_embedding_np, ae_labels_np
    else:
        train_data_2D_np, train_labels_np = pickle.load(open(f'{model_root_dir}/AE_embedding_{MODEL_NAME}.p', 'rb'))
        print('AE embedding loaded.')
else:
    # If train on the whole dataset (not embedding)
    # This is a semi-supervised setting and labels are of no immediate use except for visualization
    train_data_list = [data[0].view(1, -1) for data in image_datasets['train']]
    train_data_tensor = torch.cat(train_data_list, dim=0)
    train_data_2D_np = train_data_tensor.numpy()
    # train_data_2D_np = torch.reshape(train_data_tensor, (train_data_tensor.shape[0], -1)).numpy()
    train_labels_np = np.array([data[1] for data in image_datasets['train']])

# -------------------------------------------
# Train UMAP based on data or embedding
# -------------------------------------------
# train or load the upmap reducer
if train_on_AE_embedding:
    if not os.path.exists(f'{model_root_dir}/umap_embedding_{MODEL_NAME}.p'):
        print('Training on AE embedding ...')
        reducer = umap.UMAP(random_state=umap_random_state)
        embedding = reducer.fit_transform(train_data_2D_np)
        pickle.dump(embedding, open(f'{model_root_dir}/umap_embedding_{MODEL_NAME}.p', 'wb'))
        pickle.dump(reducer, open(f'{model_root_dir}/umap_reducer_{MODEL_NAME}.p', 'wb'))  
    else:
        # embedding = pickle.load(open(f'{model_root_dir}/umap_embedding_{MODEL_NAME}.p', 'rb'))
        reducer = pickle.load(open(f'{model_root_dir}/umap_reducer_{MODEL_NAME}.p', 'rb'))
        embedding = reducer.transform(train_data_2D_np)
        print('Model trained on AE embedding is loaded.')
else: 
    if not os.path.exists(f'{model_root_dir}/umap_embedding_{dataset_name}.p'):
        print(f'Training on full {dataset_name} ...')
        reducer = umap.UMAP(random_state=umap_random_state)
        embedding = reducer.fit_transform(train_data_2D_np)
        pickle.dump(embedding, open(f'{model_root_dir}/umap_embedding_{dataset_name}.p', 'wb'))
        pickle.dump(reducer, open(f'{model_root_dir}/umap_reducer_{dataset_name}.p', 'wb'))
    else:    
        embedding = pickle.load(open(f'{model_root_dir}/umap_embedding_{dataset_name}.p', 'rb'))
        print(f'Model trained full {dataset_name} is loaded.')
    
# sns.set(context='paper', style='white')
fig, ax = plt.subplots(figsize=(12, 10))
color = train_labels_np.astype(int)
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap='Spectral', s=0.1)
plt.setp(ax, xticks=[], yticks=[])

if train_on_AE_embedding:
    plt.title(f'{dataset_name} dimensionality reduction by UMAP with AE', fontsize=18)
    plt.savefig(f'{results_root_dir}/umap_embedding_{MODEL_NAME}.jpg')
else:
    plt.title(f'{dataset_name} dimensionality reduction by UMAP', fontsize=18)
    plt.savefig(f'{results_root_dir}/umap_embedding_{dataset_name}.jpg')

plt.show()