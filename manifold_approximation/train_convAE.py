'''
Build a convolutional autoencoder

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
from  manifold_approximation.utils.load_datasets import load_dataset, MySubset
from  manifold_approximation.utils.train_AE import train_model
from manifold_approximation.utils.vis_tools import create_acc_loss_graph

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------
# Initialization
# ----------------------------------
TRAIN_FLAG = True  # train or not?
 

#TODO eval_interval = every how many epochs to evlaute 
batch_size = 64
nr_epochs = 25

dataset_name = 'CIFAR100'
dataset_split = 'balanced'
# train_val_split = (100000, 12800)

data_root_dir = '../data'
model_root_dir = "../model_weights"
results_root_dir = '../results/AE'
log_root_dir = '../logs'
phases = ['train', 'test']

latent_size = 64
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

# which model to use? 
from models.convAE_128D import ConvAutoencoder
from models.convAE_cifar import ConvAutoencoderCIFAR
from models import convAE_cifar_residual
from models.convAE_cifar_residual import ConvAutoencoderCIFARResidual

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
# ToTensor() automatically converts everything to [0, 1]
#if dataset_name in ['MNIST', 'FMNIST', 'EMNIST', 'CIFAR10', 'CINIC10','CIFAR100']:
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        #transforms.Normalize((0.1307,), (0.3081,))
        ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        #transforms.Normalize((0.1307,), (0.3081,))
        ])
    }
# elif dataset_name in ['CIFAR10', 'cifar10']:
#     data_transforms = {
#         'train': transforms.Compose(
#         [transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ]),
#         'test': transforms.Compose(
#         [transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])                
#     }

dataloaders, image_datasets, dataset_sizes, class_names = load_dataset(dataset_name, data_root_dir, data_transforms, 
                                                                       batch_size=batch_size, shuffle_flag=True, 
                                                                       dataset_split=dataset_split)

if dataset_name == 'EMNIST' and dataset_split == 'balanced':    
    class_names = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',  'Z',
                'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']


# ----------------------------------
# Visualize data 
# ----------------------------------
# helper function to un-normalize and display an image
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

if dataset_name in ['CIFAR10', 'CIFAR100', 'CIFAR110']:
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        # npimg = img.numpy()
        plt.imshow(np.transpose(img, (1, 2, 0)))
else:
    def imshow(img):
        img = np.squeeze(img, axis=0) 
        plt.imshow(img)  # convert from Tensor image
    
# get some training images
plot_sample_size = 40
for dataset_type in ['train', 'test']: 
    dataiter = iter(dataloaders[dataset_type])
    images, labels = dataiter.next()
    # images = images.numpy() # convert images to numpy for display
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display 20 images
    for idx in np.arange(plot_sample_size):
        ax = fig.add_subplot(2, plot_sample_size/2, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(class_names[labels[idx]])
    plt.savefig(f'{results_root_dir}/{dataset_type}_data_samples_{dataset_name}.jpg')
        
# ----------------------------------
# Initialize the model
# ----------------------------------
if dataset_name in ['CIFAR10', 'CIFAR100', 'CIFAR110']:
    # model = ConvAutoencoderCIFAR(latent_size).to(device)
    model = ConvAutoencoderCIFARResidual(num_hiddens, num_residual_layers, 
                                         num_residual_hiddens, latent_size).to(device)
    
else:
    model = ConvAutoencoder().to(device)
print(model)

# ----------------------------------
# Train the AE
# ----------------------------------
if not os.path.exists(model_root_dir):
    os.makedirs(model_root_dir)

# specify loss citerion
# only use BCE for data in range [0, 1]
# else best to use sum of squared differences
# see => https://www.youtube.com/watch?v=xTU79Zs4XKY


# criterion = nn.BCELoss()
criterion = nn.MSELoss()

# specify loss function
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=False)

# Decay LR by a factor of x*gamma every step_size epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

if TRAIN_FLAG:
    # generate a model name
    MODEL_NAME = f'model-{dataset_name}-{int(time.time())}-epoch{nr_epochs}-latent{latent_size}' # use time to make the name unique
    model_b, model_l = train_model(model, MODEL_NAME, dataloaders, dataset_sizes, phases, criterion, 
                                    optimizer, exp_lr_scheduler, num_epochs=nr_epochs,
                                    model_save_dir=model_root_dir, log_save_dir=log_root_dir)
    
    if 'test' in phases:
        model = model_b
        # Visualize training (train vs test)
        create_acc_loss_graph(MODEL_NAME, dataset_name, log_root_dir, results_root_dir)
    else:
        model = model_l
    
    # also pickle dump the embedding from the best model
    # this is for the Umap to pick up
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

# load the model (inference or to continue training)
if not TRAIN_FLAG:
    # load the model 
    MODEL_NAME = ''
    if os.path.exists(f'{model_root_dir}/{MODEL_NAME}_best.pt'):
        checkpoint = torch.load(f'{model_root_dir}/{MODEL_NAME}_best.pt')
    elif os.path.exists(f'{model_root_dir}/{MODEL_NAME}_last.pt'):
        checkpoint = torch.load(f'{model_root_dir}/{MODEL_NAME}_last.pt')
    else:
        raise FileNotFoundError('No relevant model file found!')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    # Load embddings of the contractie AE
    ae_embedding_np, ae_labels_np = pickle.load(open(f'{model_root_dir}/AE_embedding_{MODEL_NAME}.p', 'rb'))
    
# ----------------------------------
# Test the AE on test data
# ---------------------------------- 
# obtain one batch of test images
dataiter = iter(dataloaders['test'])
images, labels = dataiter.next()
images = images.to(device)
labels = labels.to(device)

# get sample outputs
output, latent = model(images)
# prep images for display

# back to cpu for numpy manipulations
output = output.cpu()
latent = latent.cpu()

# output is resized into a batch of images
output = output.view(batch_size, 3, images.shape[-1], images.shape[-1])
# use detach when it's an output that requires_grad
output = output.detach().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(output[idx])
    ax.set_title(class_names[labels[idx]])
plt.savefig(f'{results_root_dir}/reconstructed_test_samples_{MODEL_NAME}.jpg')