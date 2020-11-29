'''
Build a convolutional autoencoder
'''

from __future__ import print_function, division

import numpy as np
import pandas as pd
import os
import time
import copy
import sys

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms

import matplotlib.pyplot as plt
from matplotlib import style

import seaborn as sns
from sklearn.manifold import TSNE
from utils.load_datasets import load_dataset

from tqdm import tqdm
from utils.train_AE import train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------
# Initialization
# ----------------------------------
latent_size = 128
from models.convAE_128D import ConvAutoencoder
TRAIN_FLAG = True
eval_interval = 3 # epochs
batch_size = 20
nr_epochs = 40

dataset_name = 'EMNIST'
dataset_split = 'balanced'
# train_val_split = (100000, 12800)

data_root_dir = '../data'
model_root_dir = "./model_weights/"
results_root_dir = '../results/AE'
log_root_dir = './logs/'

# ----------------------------------
# Reproducability
# ----------------------------------
# torch.manual_seed(1230)
# np.random.seed(3210)

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

dataloaders, dataset_sizes, class_names = load_dataset('EMNIST', data_root_dir, data_transforms, 
                                                       batch_size=batch_size, dataset_split=dataset_split)

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

def imshow(img):
    img = np.squeeze(img, axis=0) 
    plt.imshow(img)  # convert from Tensor image
    
# get some training images
for dataset_type in ['train', 'test']: 
    dataiter = iter(dataloaders[dataset_type])
    images, labels = dataiter.next()
    # images = images.numpy() # convert images to numpy for display
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display 20 images
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(2, batch_size/2, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(class_names[labels[idx]])
    plt.savefig(f'{results_root_dir}/{dataset_type}_data_samples_{dataset_name}.jpg')
        
# ----------------------------------
# Initialize the model
# ----------------------------------
model = ConvAutoencoder().to(device)
print(model)

# ----------------------------------
# Train the AE
# ----------------------------------
if not os.path.exists(model_root_dir):
    os.makedirs(model_root_dir)

# specify loss citerion
criterion = nn.BCELoss()

# specify loss function
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of x*gamma every step_size epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

if TRAIN_FLAG:
    # generate a model name
    MODEL_NAME = f"model-{int(time.time())}-epoch{nr_epochs}-latent{latent_size}" # use time to make the name unique
    model, MODEL_NAME = train_model(model, MODEL_NAME, dataloaders, dataset_sizes, criterion, 
                                    optimizer, exp_lr_scheduler, num_epochs=nr_epochs,
                                    model_save_dir=model_root_dir, log_save_dir=log_root_dir)

# load the model (inference or to continue training)
if not TRAIN_FLAG:
    MODEL_NAME = "model-1606648577-epoch40-latent128"
    checkpoint = torch.load(model_root_dir + MODEL_NAME)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
# ----------------------------------
# Visualize training (train vs test)
# ----------------------------------
style.use("ggplot")

def create_acc_loss_graph(model_name):
    contents = open(log_root_dir + model_name + ".log", "r").read().split("\n")

    times = []
    train_losses = []
    test_losses = []
    epochs = []

    for c in contents:
        if model_name in c:
            name, timestamp, _, train_loss, _, test_loss, epoch = c.split(",")

            times.append(float(timestamp))
            train_losses.append(float(train_loss))
            test_losses.append(float(test_loss))
            epochs.append(float(epoch))

    fig = plt.figure()

    ax1 = plt.subplot2grid((2,1), (0,0))
    ax1.plot(epochs,train_losses, label="train_loss")
    ax1.plot(epochs,test_losses, label="test_loss")
    ax1.legend(loc=3) # loc=3 means lower left 
    plt.savefig(f'{results_root_dir}/train_test_graph_{dataset_name}_{model_name}.jpg')

create_acc_loss_graph(MODEL_NAME)

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
output = output.view(batch_size, 1, 28, 28)
# use detach when it's an output that requires_grad
output = output.detach().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(output[idx])
    ax.set_title(class_names[labels[idx]])
plt.savefig(f'{results_root_dir}/reconstructed_test_samples_{dataset_name}_{MODEL_NAME}.jpg')

# ----------------------------------
# Visualize the latent vector
# ----------------------------------
embeddings_list = []
labels_list = []
with torch.no_grad():
    for i, (images, labels) in enumerate(dataloaders['test']):
            images = images.to(device)
            labels_list.append(labels.cpu().numpy()) 
            _, embeddings = model(images)
            embeddings_list.append(embeddings.cpu().detach().numpy())
            
embeddings_np = np.concatenate(embeddings_list, axis=0)
test_labels = np.concatenate(labels_list)

if latent_size == 2:
    plt.figure()
    classes = [str(nr) for nr in range(0,10)]
    plt.scatter(embeddings_np[:,0], embeddings_np[:,1], c=test_labels, 
                s=8, cmap='tab10', label=classes)
    plt.legend()
    plt.savefig(f'{results_root_dir}/scatter_{MODEL_NAME}.jpg')

if latent_size != 2:
    X = embeddings_np
    X_embedded = TSNE(n_components=2).fit_transform(X)
    # plt.figure()
    # plt.scatter(X_embedded[:,0], X_embedded[:,1], c=test_labels_tensor_np, 
    #             s=8, cmap='tab10', label=classes)
    # plt.legend()
    # plt.savefig(f'{results_root_dir}/scatter_{MODEL_NAME}.jpg')
    
    plt.figure()
    df = pd.DataFrame({'x':X_embedded[:,0], 'y':X_embedded[:,1]})
    sns.scatterplot(x='x', y='y', hue=test_labels, 
                    palette=sns.color_palette("hls", len(class_names)), 
                    data=df, legend="full", alpha=0.3)
    plt.savefig(f'{results_root_dir}/sns_scatter_{dataset_name}_{MODEL_NAME}.jpg')
    

