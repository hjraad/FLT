
'''
Train Umap based on the daset of choice
@Author(s): Mohammad Abdizadeh & Hadi Jamali-Rad
@email(s): {moh.abdizadeh. h.jamali.rad}@gmail.com
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

from utils.load_datasets import load_dataset
from utils.train_AE import train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------
# Initialization
# ----------------------------------
batch_size = 1
num_workers = 0

dataset_name = 'EMNIST'
dataset_split = 'balanced'
# train_val_split = (100000, 12800)

data_root_dir = '../../data'
model_root_dir = '../model_weights'

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

dataloaders, image_datasets, dataset_sizes, class_names = load_dataset('EMNIST', data_root_dir, data_transforms, 
                                                       batch_size=batch_size, dataset_split=dataset_split)

if dataset_name == 'EMNIST' and dataset_split == 'balanced':    
    class_names = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',  'Z',
                'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']


# This is a semi-supervised setting and labels are of no immediate use except for FL part
train_data_list = [data[0] for data in image_datasets['train']]
train_data_tensor = torch.cat(train_data_list, dim=0)
train_data_tensor_2D_np = torch.reshape(train_data_tensor, (train_data_tensor.shape[0], -1)).numpy()
train_labels_np = np.array([data[1] for data in image_datasets['train']])

# ----------------------------------
# Train UMAP based on the train data
# ----------------------------------
# train or load the upmap reducer and embedding
if not os.path.exists(f'{model_root_dir}/umap_embedding_{dataset_name}.p')\
    and not os.path.exists(f'{model_root_dir}/umap_embedding_{dataset_name}.p'):
    reducer = umap.UMAP(random_state=umap_random_state)
    embedding = reducer.fit_transform(train_data_tensor_2D_np)
    pickle.dump(embedding, open(f'{model_root_dir}/umap_embedding_{dataset_name}.p', 'wb'))
    pickle.dump(reducer, open(f'{model_root_dir}/umap_reducer_{dataset_name}.p', 'wb'))
else:
    embedding = pickle.load(open(f'{model_root_dir}/umap_embedding_{dataset_name}.p', 'rb'))
    reducer = pickle.load(open(f'{model_root_dir}/umap_reducer_{dataset_name}.p', 'rb'))
    
sns.set(context="paper", style="white")
fig, ax = plt.subplots(figsize=(12, 10))
color = train_labels_np.astype(int)
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1)
plt.setp(ax, xticks=[], yticks=[])
plt.title("EMNIST data dimensionality reduction by UMAP", fontsize=18)
plt.show()