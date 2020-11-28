import numpy as np
import pandas as pd
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------
# Initialization
# ----------------------------------
latent_size = 128
from models.convAE_128D import ConvAutoencoder
TRAIN_FLAG = False
eval_interval = 3 # epochs

dataset_name = 'EMNIST'
train_val_split = (100000, 12800)

data_root_dir = '../data'
model_path_root = "./model_files/"
results_root_dir = '../results/AE'

# ----------------------------------
# Reproducability
# ----------------------------------
torch.manual_seed(123)
np.random.seed(321)

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
# Prepare data 
# ----------------------------------
# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# download the training and test datasets
dataset = datasets.__dict__[dataset_name](root=data_root_dir, split='balanced', 
                                          train=True,download=True, transform=transform)

train_data, val_data = torch.utils.data.random_split(dataset, [train_val_split[0], train_val_split[1]])

test_data = datasets.EMNIST(root=data_root_dir, split='balanced', 
                            train=False, download=True, transform=transform)

# Create training and test dataloaders
num_workers = 0
batch_size = 20

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, 
                                         num_workers=num_workers, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                          num_workers=num_workers, shuffle=False)
# specify the image classes
classes = ['0', '1', '2', '3', '4',
           '5', '6', '7', '8', '9']

# This is a semi-supervised setting and labels are of no immediate use except for FL part
train_data_list = [data[0] for data in train_loader]
train_data_tensor = torch.cat(train_data_list, dim=0)
train_labels_tensor = dataset.targets[train_data.indices]

val_data_list = [data[0] for data in val_loader]
val_data_tensor = torch.cat(val_data_list, dim=0)
val_labels_tensor = dataset.targets[val_data.indices]

test_data_list = [data[0] for data in test_loader]
test_data_tensor = torch.cat(test_data_list, dim=0)
test_labels_tensor = test_data.targets

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
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display
# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])
plt.savefig(f'{results_root_dir}/train_data_samples.jpg')
            
# ----------------------------------
# Initialize the model
# ----------------------------------
model = ConvAutoencoder().to(device)
print(model)

# ----------------------------------
# Train the AE
# ----------------------------------
if not os.path.exists(model_path_root):
    os.makedirs(model_path_root)

# specify loss citerion
criterion = nn.BCELoss()

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# number of epochs to train the model
n_epochs = 20





if TRAIN_FLAG:
    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0
        eval_loss = 0.0
        
        # train the model 
        for data in train_loader:
            # _ stands in for labels, here
            # no need to flatten images
            images, _ = data
            images = images.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs, latent = model(images)
            # calculate the loss
            loss = criterion(outputs, images)
            
            
            
            
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()
            
        
                
        
                
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    # save model name
    MODEL_NAME = f"model-{int(time.time())}-epoch{epoch}-latent{latent_size}" # use time to make the name unique
    print(f'Finished training for {MODEL_NAME} ...')
    
    # save the model
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_path_root + MODEL_NAME)


# load the model (inference or to continue training)
if not TRAIN_FLAG:
    MODEL_NAME = "model-1606000037-epoch20-latent128"
checkpoint = torch.load(model_path_root + MODEL_NAME)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# ----------------------------------
# Test the AE on test data
# ---------------------------------- 
# obtain one batch of test images
dataiter = iter(test_loader)
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
    ax.set_title(classes[labels[idx]])
plt.savefig(f'{results_root_dir}/reconstructed_test_samples_{MODEL_NAME}.jpg')

# ----------------------------------
# Visualize the latent vector
# ----------------------------------
_, embeddings = model(test_data_tensor.to(device))

# back to cpu for numpy manipulation
embeddings = embeddings.cpu()

embeddings_np = embeddings.detach().numpy()
test_labels_tensor_np = test_labels_tensor.numpy()

if latent_size == 2:
    plt.figure()
    classes = [str(nr) for nr in range(0,10)]
    plt.scatter(embeddings_np[:,0], embeddings_np[:,1], c=test_labels_tensor_np, 
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
    sns.scatterplot(x='x', y='y', hue=test_labels_tensor_np, 
                    palette=sns.color_palette("hls", 10), 
                    data=df, legend="full", alpha=0.3)
    plt.savefig(f'{results_root_dir}/sns_scatter_{MODEL_NAME}.jpg')