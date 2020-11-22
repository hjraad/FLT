import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sklearn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# load the training and test datasets
train_data = datasets.MNIST(root='../data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='../data', train=False,
                                  download=True, transform=transform)

# Create training and test dataloaders

num_workers = 0
# how many samples per batch to load
batch_size = 20

latent_size = 2

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# ----------------------------------
# Visualize data 
# ----------------------------------
import matplotlib.pyplot as plt
# %matplotlib inline

# helper function to un-normalize and display an image
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
    
def imshow(img):
    img = np.squeeze(img, axis=0) 
    plt.imshow(img)  # convert from Tensor image
    
# specify the image classes
classes = ['0', '1', '2', '3', '4',
           '5', '6', '7', '8', '9']

# obtain one batch of training images
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
# plt.show()
            
# ----------------------------------
# Build the autoencoder (AE)
# ----------------------------------
import torch.nn as nn
import torch.nn.functional as F

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        # dense layers
        self.fc1 = nn.Linear(7*7*4, 64) #flattening (input should be calculated by a forward pass - stupidity of Pytorch)
        self.fc2 = nn.Linear(64, 2) # 2 Dim visualization
        
        ## decoder layers ##
        # decoding dense layer
        self.dec_linear_2 = nn.Linear(2, 64)
        self.dec_linear_1 = nn.Linear(64, 7*7*4)
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x, return_comp=True):
        ## ==== encode ==== ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  
        # flatten and apply dense layer
        x = x.view(-1, 7*7*4)
        x = self.fc1(x) # compressed layer
        x_comp = self.fc2(x) # compressed layer
        
        ## ==== decode ==== ##
        x = self.dec_linear_2(x_comp)
        x = self.dec_linear_1(x)
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x.view(-1, 4, 7, 7)))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))
                
        if return_comp:
            return x, x_comp
        else:
            return x

# initialize the NN
model = ConvAutoencoder()
print(model)

# ----------------------------------
# Train the AE
# ----------------------------------
TRAIN_FLAG = False
model_path_root = "./model_files/"

import os
if not os.path.exists(model_path_root):
    os.makedirs(model_path_root)

# specify loss function
criterion = nn.BCELoss()

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# number of epochs to train the model
n_epochs = 20

if TRAIN_FLAG:
    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0
        
        # train the model 
        for data in train_loader:
            # _ stands in for labels, here
            # no need to flatten images
            images, _ = data
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
            train_loss += loss.item()*images.size(0)
                
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    MODEL_NAME = f"model-{int(time.time())}-epoch{epoch}-latent{latent_size}" # use time to make the name unique
    print(MODEL_NAME)
    
    # save model
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_path_root + MODEL_NAME)


# Model class must be defined somewhere
if not TRAIN_FLAG:
    MODEL_NAME = "model-1606041368-epoch20-latent2"
checkpoint = torch.load(model_path_root + MODEL_NAME)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# ----------------------------------
# Test the AE
# ---------------------------------- 
# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()

# get sample outputs
output, latent = model(images)
# prep images for display
images = images.numpy()

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

# ----------------------------------
# Visualize the latent vector
# ----------------------------------
full_train_data = [data[0] for data in train_loader]
full_train_data_tensor = torch.cat(full_train_data, dim=0)
full_train_labels_tensor = train_data.targets

full_test_data = [data[0] for data in test_loader]
full_test_data_tensor = torch.cat(full_test_data, dim=0)
full_test_labels_tensor = test_data.targets

_, embeddings = model(full_test_data_tensor)

embeddings_np = embeddings.detach().numpy()
full_test_labels_tensor_np = full_test_labels_tensor.numpy()

plt.figure()
classes = [str(nr) for nr in range(0,10)]
plt.scatter(embeddings_np[:,0], embeddings_np[:,1], c=full_test_labels_tensor_np, 
            s=8, cmap='tab10', label=classes)
plt.legend()
plt.show() 


