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
import seaborn as sns
from sklearn.manifold import TSNE
from utils.load_datasets import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------
# Initialization
# ----------------------------------
latent_size = 128
from models.convAE_128D import ConvAutoencoder
TRAIN_FLAG = True
eval_interval = 3 # epochs
batch_size = 60
nr_epochs = 5

dataset_name = 'EMNIST'
dataset_split = 'digits'
# train_val_split = (100000, 12800)

data_root_dir = '../data'
model_path_root = "./model_weights/"
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
                                                       batch_size=batch_size, dataset_split='balanced')

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
dataiter = iter(dataloaders['train'])
images, labels = dataiter.next()
# images = images.numpy() # convert images to numpy for display
# plot the images in the batch, along with the corresponding labels

class_names = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
            'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',  'Z',
            'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(batch_size):
    ax = fig.add_subplot(2, batch_size/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(class_names[labels[idx]])
plt.savefig(f'{results_root_dir}/train_data_samples_{dataset_name}.jpg')

# ----------------------------------
# Define model training procedure
# ----------------------------------
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    least_loss = np.Inf
    
    # generate a model name
    MODEL_NAME = f"model-{int(time.time())}-epoch{num_epochs}-latent{latent_size}" # use time to make the name unique
    
    with open(log_root_dir + MODEL_NAME + ".log", "a") as f:
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            temp_loss = {'train':0, 'test':0}
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0

                # Iterate over data.
                # semi-supervised => labels are unimportant
                for images, _ in dataloaders[phase]:
                    images = images.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, _ = model(images)
                        loss = criterion(outputs, images)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * images.size(0)
                    
                if phase == 'train':
                    scheduler.step()
                
                epoch_loss = running_loss / dataset_sizes[phase]
                temp_loss[phase] = epoch_loss

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                # deep copy the model
                if phase == 'test' and epoch_loss < least_loss:
                    least_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    
            # write a line for this epoch's loss values     
            f.write(f"{MODEL_NAME},{round(time.time(),3)}, train_loss, {round(float(temp_loss['train']),4)}, test_loss, {round(float(temp_loss['test']),4)},{epoch}\n")
            
            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Least test Acc: {:4f}, best epoch:{}'.format(least_loss, best_epoch))

    # save the model
    torch.save({
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(best_model_wts),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_path_root + MODEL_NAME)
    
    return model, MODEL_NAME
        
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
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

if TRAIN_FLAG:
    model, MODEL_NAME = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=nr_epochs)


# load the model (inference or to continue training)
if not TRAIN_FLAG:
    MODEL_NAME = "model-1606574353-epoch9-latent128"
    checkpoint = torch.load(model_path_root + MODEL_NAME)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

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
plt.savefig(f'{results_root_dir}/reconstructed_test_samples_{MODEL_NAME}.jpg')

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
            
embeddings_np = np.concatenate(embeddings_list, axis=0)[0:1000,:]
test_labels = np.concatenate(labels_list)[0:1000]

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
    plt.savefig(f'{results_root_dir}/sns_scatter_{MODEL_NAME}.jpg')